from __future__ import annotations

import csv
import json
import os
import random
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_JUDGE_MODEL = "openrouter/free"
DEFAULT_JUDGE_MAX_TOKENS = 4000
DEFAULT_JUDGE_RETRY_ATTEMPTS = 3
DEFAULT_JUDGE_TIMEOUT = 150
DEFAULT_JUDGE_RETRY_SLEEP = 2.5
DEFAULT_JUDGE_LONG_RETRY_SLEEP = 60.0
DEFAULT_OUTPUT_DIR = Path("data") / "judged"
DEFAULT_FLUENCY_OUTPUT_DIR = Path("data") / "judged_fluency"
DEFAULT_JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_API_URL = f"{DEFAULT_JUDGE_BASE_URL}/chat/completions"
OPENROUTER_URL = DEFAULT_JUDGE_API_URL
PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "judge_speeches.txt"
FLUENCY_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "judge_fluency.txt"

# Language code -> human-readable name used in the fluency prompt.
LANGUAGE_NAMES = {
    "en": "English",
    "cs": "Czech",
    "sk": "Slovak",
    "hsb": "Upper Sorbian",
}

# Dataset -> language code of the original entities. The generation prompt keeps
# proper named entities in their source form, so these entities can appear in a
# different language than the target sentence (e.g. Czech names in an English
# sentence). The fluency judge is told not to penalise such untranslated names.
DATASET_ENTITY_LANGUAGE = {
    "cs-qa": "cs",
    "sk-qa": "sk",
    "webnlg": "en",
}

# Fallback phrase when the source dataset (and thus its entity language) is unknown.
DEFAULT_ENTITY_LANGUAGE_NAME = "the original source language"


class JudgeRequestError(RuntimeError):
    """Raised when the judge API request fails in a user-actionable way."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class EntryData:
    eid: str
    category: str | None
    shape: str | None
    shape_type: str | None
    size: int
    modified_triples: list[tuple[str, str, str]]


@dataclass(frozen=True)
class SourceSpec:
    label: str
    csv_path: Path
    source_id: str


def source_namespace(source_path: str | Path) -> str | None:
    path = Path(source_path).resolve()
    parts = path.parts
    if "generated" in parts:
        index = parts.index("generated")
        if index + 1 < len(parts):
            return sanitize_identifier(parts[index + 1])
    return None


def source_identity(source_path: str | Path, *, label: str | None = None) -> str:
    path = Path(source_path).resolve()
    base = label or path.stem
    namespace = source_namespace(path)
    if namespace:
        return sanitize_identifier(f"{namespace}__{base}")
    return sanitize_identifier(base)


def source_dataset_variant(source_path: str | Path) -> tuple[str | None, str | None]:
    stem = Path(source_path).stem
    if stem.startswith("sentences_"):
        stem = stem.removeprefix("sentences_")

    parts = stem.split("_")
    if len(parts) < 2:
        return None, None

    dataset = sanitize_identifier(parts[0])
    variant = sanitize_identifier(parts[1])
    return dataset, variant


def parse_triple(text: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in text.split("|", 2)]
    if len(parts) != 3:
        raise ValueError(f"Malformed triple: {text!r}")
    return parts[0], parts[1], parts[2]


def prompt_text(value: object) -> str:
    return str(value).replace("_", " ")


def display_triple(triple: tuple[str, str, str]) -> tuple[str, str, str]:
    subject, predicate, obj = triple
    return prompt_text(subject), prompt_text(predicate), prompt_text(obj)


def format_triple(triple: tuple[str, str, str], *, for_prompt: bool = False) -> str:
    subject, predicate, obj = triple
    if for_prompt:
        subject, predicate, obj = display_triple(triple)
    return f"{subject} | {predicate} | {obj}"


def format_triples(triples: list[tuple[str, str, str]], *, for_prompt: bool = False) -> str:
    return "\n".join(f"- {format_triple(triple, for_prompt=for_prompt)}" for triple in triples)


def sanitize_identifier(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    clean = clean.strip("_")
    return clean or "source"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_env_defaults() -> None:
    candidates = [
        repo_root() / ".env.local",
        Path.cwd() / ".env.local",
        Path(__file__).resolve().parent / ".env.local",
    ]
    for candidate in candidates:
        values = _load_env_file(candidate)
        for key, value in values.items():
            os.environ.setdefault(key, value)


def normalize_judge_api_url(value: str | None = None) -> str:
    raw_value = (value or "").strip().rstrip("/")
    if not raw_value:
        return DEFAULT_JUDGE_API_URL
    if raw_value.endswith("/chat/completions"):
        return raw_value
    return f"{raw_value}/chat/completions"


def judge_api_url_from_env() -> str:
    return normalize_judge_api_url(
        os.getenv("JUDGE_API_URL") or os.getenv("JUDGE_BASE_URL") or os.getenv("OPENROUTER_URL")
    )


def parse_xml_entries(
    xml_path: str | Path,
    *,
    triple_xpath: str = "modifiedtripleset/mtriple",
) -> dict[str, EntryData]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    entries: dict[str, EntryData] = {}
    for entry in root.findall("entries/entry"):
        eid = entry.get("eid") or ""
        meta = {
            "category": entry.get("category"),
            "shape": entry.get("shape"),
            "shape_type": entry.get("shape_type"),
            "size": int(entry.get("size", 0)),
        }

        modified_triples: list[tuple[str, str, str]] = []
        for triple_elem in entry.findall(triple_xpath):
            if triple_elem.text:
                modified_triples.append(parse_triple(triple_elem.text))

        entries[eid] = EntryData(
            eid=eid,
            category=meta["category"],
            shape=meta["shape"],
            shape_type=meta["shape_type"],
            size=meta["size"],
            modified_triples=modified_triples,
        )

    return entries


def infer_xml_path(sentences_csv: str | Path) -> Path:
    csv_path = Path(sentences_csv).resolve()
    stem = csv_path.stem.lower()
    dataset, variant = source_dataset_variant(csv_path)

    if dataset in {"cs-qa", "sk-qa"} and variant in {"cf", "fa", "fi"}:
        candidates = [
            repo_root() / "data" / dataset / f"{variant}.csv",
            csv_path.parent.parent.parent / dataset / f"{variant}.csv",
            csv_path.parent.parent / dataset / f"{variant}.csv",
        ]
        for csv_source_path in candidates:
            if csv_source_path.exists():
                return csv_source_path

        raise FileNotFoundError(candidates[0])

    if "webnlg_cf" in stem:
        candidates = [
            repo_root() / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-CFA_WebNLG_CounterFactual.xml",
            csv_path.parent / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-CFA_WebNLG_CounterFactual.xml",
            csv_path.parent.parent / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-CFA_WebNLG_CounterFactual.xml",
        ]
    elif "webnlg_fa" in stem:
        candidates = [
            repo_root() / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-FA_WebNLG_Factual.xml",
            csv_path.parent / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-FA_WebNLG_Factual.xml",
            csv_path.parent.parent / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-FA_WebNLG_Factual.xml",
        ]
    elif "webnlg_fi" in stem:
        candidates = [
            repo_root() / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-FI_WebNLG_Fictional.xml",
            csv_path.parent / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-FI_WebNLG_Fictional.xml",
            csv_path.parent.parent / "data" / "GEM-v2-D2T-SharedTask" / "D2T-1-FI_WebNLG_Fictional.xml",
        ]
    else:
        raise ValueError(
            f"Cannot infer XML path from {csv_path.name!r}; pass an XML path explicitly."
        )
    for xml_path in candidates:
        if xml_path.exists():
            return xml_path

    raise FileNotFoundError(candidates[0])


def infer_triple_xpath(source_path: str | Path) -> str:
    path = Path(source_path)
    stem = path.stem.lower()
    name = path.name.lower()
    path_parts = {part.lower() for part in path.parts}
    if (
        "cs-qa" in stem
        or "sk-qa" in stem
        or {"cs-qa", "sk-qa"} & path_parts
        or name in {"factual-triples.xml", "counterfactual-triples.xml"}
    ):
        return "originaltripleset/otriple"
    return "modifiedtripleset/mtriple"


def infer_default_xml_path() -> Path:
    return (
        repo_root()
        / "data"
        / "GEM-v2-D2T-SharedTask"
        / "D2T-1-CFA_WebNLG_CounterFactual.xml"
    )


def infer_default_flat_csv_path() -> Path:
    return (
        repo_root()
        / "data"
        / "webnlg_cf.csv"
    )


def load_entry_table_from_flat_csv(
    dataset_csv: str | Path,
    *,
    kind_value: str = "modified",
) -> pd.DataFrame:
    frame = pd.read_csv(dataset_csv)
    if frame.empty:
        return pd.DataFrame()

    required = {"eid", "category", "shape", "shape_type", "size", "kind", "subject", "predicate", "object"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{Path(dataset_csv).name} is missing required columns: {', '.join(missing)}"
        )

    subset = frame[frame["kind"].astype(str) == kind_value].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "eid",
                "category",
                "shape",
                "shape_type",
                "size",
                "modified_triples",
                "modified_triples_json",
                "num_modified_triples",
                "xml_path",
            ]
        )

    rows: list[dict[str, Any]] = []
    for eid, group in subset.groupby("eid", sort=False):
        triples = [
            (str(row["subject"]), str(row["predicate"]), str(row["object"]))
            for _, row in group.iterrows()
        ]
        first = group.iloc[0]
        rows.append(
            {
                "eid": str(eid),
                "category": first.get("category"),
                "shape": first.get("shape"),
                "shape_type": first.get("shape_type"),
                "size": int(first.get("size", 0)),
                "modified_triples": format_triples(triples, for_prompt=True),
                "modified_triples_json": [format_triple(triple, for_prompt=True) for triple in triples],
                "num_modified_triples": len(triples),
                "xml_path": str(dataset_csv),
            }
        )

    return pd.DataFrame(rows).sort_values("eid").reset_index(drop=True)


def load_entry_table(
    xml_path: str | Path,
    *,
    triple_xpath: str = "modifiedtripleset/mtriple",
) -> pd.DataFrame:
    entries = parse_xml_entries(xml_path, triple_xpath=triple_xpath)
    rows = []
    for entry in entries.values():
        rows.append(
            {
                "eid": entry.eid,
                "category": entry.category,
                "shape": entry.shape,
                "shape_type": entry.shape_type,
                "size": entry.size,
                "modified_triples": format_triples(entry.modified_triples, for_prompt=True),
                "modified_triples_json": [format_triple(triple, for_prompt=True) for triple in entry.modified_triples],
                "num_modified_triples": len(entry.modified_triples),
                "xml_path": str(xml_path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("eid").reset_index(drop=True)


def load_sentence_table(sentences_csv: str | Path) -> pd.DataFrame:
    with open(sentences_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required_columns = {"eid", "sentence"}
    missing_columns = sorted(required_columns - set(fieldnames))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{Path(sentences_csv).name} is missing required columns: {missing}")

    return pd.DataFrame(rows)


def enrich_sentences(sentences_csv: str | Path, xml_path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(sentences_csv)
    resolved_xml_path = Path(xml_path) if xml_path is not None else infer_xml_path(csv_path)
    triple_xpath = infer_triple_xpath(resolved_xml_path if xml_path is not None else csv_path)

    sentence_df = load_sentence_table(csv_path)
    if resolved_xml_path.suffix.lower() == ".csv":
        kind_value = "original" if triple_xpath == "originaltripleset/otriple" else "modified"
        entries = load_entry_table_from_flat_csv(resolved_xml_path, kind_value=kind_value)
    else:
        entries = load_entry_table(resolved_xml_path, triple_xpath=triple_xpath)
    if entries.empty:
        return entries

    merged = sentence_df.merge(
        entries,
        how="inner",
        on=["eid"],
        suffixes=("", "_xml"),
    )
    return merged


def parse_source_specs(raw_text: str) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    used_source_ids: dict[str, int] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        label: str | None = None
        path_text = line
        if "::" in line:
            maybe_label, maybe_path = line.split("::", 1)
            label = maybe_label.strip() or None
            path_text = maybe_path.strip()

        csv_path = Path(path_text).expanduser()
        resolved = csv_path.resolve()
        source_label = label or resolved.stem
        base_source_id = source_identity(resolved, label=source_label)
        source_id_count = used_source_ids.get(base_source_id, 0)
        used_source_ids[base_source_id] = source_id_count + 1
        source_id = base_source_id if source_id_count == 0 else f"{base_source_id}_{source_id_count + 1}"
        specs.append(
            SourceSpec(
                label=source_label,
                csv_path=resolved,
                source_id=source_id,
            )
        )
    return specs


def _parse_formatted_triple(text: str) -> tuple[str, str, str] | None:
    line = text.strip()
    if line.startswith("-"):
        line = line[1:].strip()
    parts = [part.strip() for part in line.split("|", 2)]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _row_triples(row: pd.Series | dict[str, Any]) -> list[tuple[str, str, str]]:
    triples_json = row.get("modified_triples_json") if hasattr(row, "get") else None
    if isinstance(triples_json, str):
        try:
            triples_json = json.loads(triples_json)
        except json.JSONDecodeError:
            triples_json = None

    triples: list[tuple[str, str, str]] = []
    if isinstance(triples_json, list):
        for item in triples_json:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                triples.append((str(item[0]), str(item[1]), str(item[2])))
            elif isinstance(item, str):
                parsed = _parse_formatted_triple(item)
                if parsed:
                    triples.append(parsed)

    if triples:
        return triples

    formatted = row.get("modified_triples") if hasattr(row, "get") else None
    if isinstance(formatted, str):
        for line in formatted.splitlines():
            parsed = _parse_formatted_triple(line)
            if parsed:
                triples.append(parsed)
    return triples


def lexical_terms_from_row(row: pd.Series | dict[str, Any]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for subject, _predicate, obj in _row_triples(row):
        for value in (subject, obj):
            term = prompt_text(value).strip()
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return terms


def format_lexical_terms(terms: list[str]) -> str:
    if not terms:
        return "(none provided)"
    return "\n".join(f"- {term}" for term in terms)


def load_output_source(spec: SourceSpec) -> pd.DataFrame:
    table = load_sentence_table(spec.csv_path)
    table = table.copy()
    table["eid"] = table["eid"].astype(str)
    table["source_label"] = spec.label
    table["source_path"] = str(spec.csv_path)
    table["source_stem"] = spec.csv_path.stem
    table["source_id"] = spec.source_id
    table["source_namespace"] = source_namespace(spec.csv_path)
    return table


def load_output_sources(specs: list[SourceSpec]) -> pd.DataFrame:
    frames = [load_output_source(spec) for spec in specs]
    if not frames:
        return pd.DataFrame(
            columns=["eid", "sentence", "source_label", "source_path", "source_stem", "source_id"]
        )
    return pd.concat(frames, ignore_index=True)


def build_prompt_payload(
    *,
    eid: str,
    category: str | None,
    sentence: str,
    modified_triples: str,
    target_language: str | None = None,
) -> dict[str, str]:
    return {
        "eid": eid,
        "category": category or "",
        "sentence": prompt_text(sentence),
        "modified_triples": prompt_text(modified_triples),
        "target_language": target_language or "the target language",
    }


def load_judge_prompt_template() -> str:
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_judge_prompt(row: pd.Series | dict[str, Any]) -> str:
    payload = build_prompt_payload(
        eid=str(row["eid"]),
        category=row.get("category"),
        sentence=str(row["sentence"]),
        modified_triples=str(row["modified_triples"]),
        target_language=str(row.get("target_language") or ""),
    )
    return load_judge_prompt_template().format(**payload)


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])

    raise ValueError(f"Judge response did not contain complete JSON: {text}")


def normalize_usage_payload(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
        "is_byok": usage.get("is_byok"),
    }


def judge_output_path(source_path: str | Path, judge_model: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    safe_model = judge_model.replace("/", "_")
    path = Path(source_path)
    filename = f"judge_{path.stem}_{safe_model}.jsonl"
    namespace = source_namespace(path)
    output_path = Path(output_dir)
    if namespace:
        output_path = output_path / namespace
    return output_path / filename


def adhoc_output_path(dataset_name: str, judge_model: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    safe_dataset = sanitize_identifier(dataset_name)
    safe_model = judge_model.replace("/", "_")
    return Path(output_dir) / f"judge_adhoc_{safe_dataset}_{safe_model}.jsonl"


def request_judge(
    *,
    prompt: str,
    judge_model: str,
    auth_token: str,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    retry_attempts: int = DEFAULT_JUDGE_RETRY_ATTEMPTS,
    retry_sleep: float = DEFAULT_JUDGE_RETRY_SLEEP,
    long_retry_sleep: float = DEFAULT_JUDGE_LONG_RETRY_SLEEP,
    api_url: str | None = None,
    timeout: int = DEFAULT_JUDGE_TIMEOUT,
    request_jitter_min: float = 0.1,
    request_jitter_max: float = 0.5,
    reasoning: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    api_url = normalize_judge_api_url(api_url or judge_api_url_from_env())
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    response = None
    last_http_error: requests.HTTPError | None = None
    used_long_retry_sleep = False
    if request_jitter_min < 0 or request_jitter_max < 0 or request_jitter_min > request_jitter_max:
        raise ValueError("request jitter bounds must be non-negative and min <= max")
    for attempt in range(1, retry_attempts + 1):
        try:
            if request_jitter_max > 0:
                time.sleep(random.uniform(request_jitter_min, request_jitter_max))
            request_payload: dict[str, Any] = {
                "model": judge_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluator. Return only valid JSON. "
                            "Do not overthink it."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if reasoning is not None:
                request_payload["reasoning"] = reasoning
            else:
                request_payload["chat_template_kwargs"] = {"thinking": "true"}
            response = requests.post(
                api_url,
                headers=headers,
                json=request_payload,
                timeout=timeout,
            )
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            last_http_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            body = exc.response.text.strip() if exc.response is not None and exc.response.text else ""
            retryable = status_code == 429 or (status_code is not None and 500 <= status_code < 600)
            if retryable and attempt < retry_attempts:
                if not used_long_retry_sleep and long_retry_sleep > 0:
                    time.sleep(long_retry_sleep)
                    used_long_retry_sleep = True
                else:
                    time.sleep(retry_sleep)
                continue

            message = f"Judge API returned HTTP {status_code if status_code is not None else 'error'}."
            if body:
                message = f"{message} Response body: {body[:1000]}"
            raise JudgeRequestError(
                message,
                details={
                    "response_body": body,
                    "api_url": api_url,
                    "attempt": attempt,
                    "retry_attempts": retry_attempts,
                },
            ) from exc
        except requests.RequestException as exc:
            if attempt < retry_attempts:
                if not used_long_retry_sleep and long_retry_sleep > 0:
                    time.sleep(long_retry_sleep)
                    used_long_retry_sleep = True
                else:
                    time.sleep(retry_sleep)
                continue
            raise JudgeRequestError(
                f"Judge API request failed: {exc}",
                details={"api_url": api_url, "attempt": attempt, "retry_attempts": retry_attempts},
            ) from exc

    if response is None:
        raise JudgeRequestError(
            "Judge API request failed without a response.",
            details={"api_url": api_url, "retry_attempts": retry_attempts},
        ) from last_http_error

    try:
        payload = response.json()
    except ValueError as exc:
        raise JudgeRequestError(
            "Judge API returned invalid JSON.",
            details={"response_text": response.text[:2000], "api_url": api_url},
        ) from exc

    try:
        raw = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeRequestError(
            "Judge API response did not include choices[0].message.content.",
            details={"response_payload": payload, "api_url": api_url},
        ) from exc

    if raw is None:
        raise JudgeRequestError(
            "Judge API returned null content.",
            details={"response_payload": payload, "api_url": api_url},
        )
    if not isinstance(raw, str):
        raise JudgeRequestError(
            f"Judge API returned non-string content of type {type(raw).__name__}.",
            details={"response_payload": payload, "api_url": api_url},
        )

    try:
        parsed = extract_json(raw)
    except Exception as exc:
        raise JudgeRequestError(
            "Judge model response did not contain valid JSON in the expected schema.",
            details={"raw_response": raw, "response_payload": payload, "api_url": api_url},
        ) from exc

    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    usage = normalize_usage_payload(usage)
    response_meta = {
        "response_model": payload.get("model") if isinstance(payload, dict) else None,
        "provider": payload.get("provider") if isinstance(payload, dict) else None,
        "api_url": api_url,
        "requested_reasoning": reasoning,
    }
    return raw, parsed, usage, response_meta


def build_judge_record(
    *,
    row: pd.Series | dict[str, Any],
    source_label: str,
    source_path: str,
    source_id: str,
    judge_model: str,
    api_url: str,
    raw_response: str | None,
    parsed: dict[str, Any] | None,
    usage: dict[str, Any] | None = None,
    response_meta: dict[str, Any] | None = None,
    token_env_var: str | None = None,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = build_judge_prompt(row)
    prompt_preview = prompt.splitlines()[0] if prompt else ""
    usage = usage or {}
    response_meta = response_meta or {}
    return {
        "eid": str(row["eid"]),
        "source_label": source_label,
        "source_path": source_path,
        "source_id": source_id,
        "judge_model": response_meta.get("response_model") or judge_model,
        "requested_judge_model": judge_model,
        "requested_judge_api_url": normalize_judge_api_url(api_url),
        "requested_token_env_var": token_env_var,
        "requested_reasoning": reasoning,
        "provider": response_meta.get("provider"),
        "request_cost": usage.get("cost"),
        "usage": usage or None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentence": str(row["sentence"]),
        "modified_triples": str(row.get("modified_triples", "")),
        "modified_triples_json": row.get("modified_triples_json"),
        "num_modified_triples": row.get("num_modified_triples"),
        "prompt": prompt_preview,
        "raw_response": raw_response,
        "parsed": parsed,
    }


def judge_row(
    row: pd.Series | dict[str, Any],
    *,
    source_label: str,
    source_path: str,
    source_id: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    auth_token: str | None = None,
    token_env_var: str | None = None,
    api_url: str | None = None,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    timeout: int = DEFAULT_JUDGE_TIMEOUT,
    retry_attempts: int = DEFAULT_JUDGE_RETRY_ATTEMPTS,
    retry_sleep: float = DEFAULT_JUDGE_RETRY_SLEEP,
    long_retry_sleep: float = DEFAULT_JUDGE_LONG_RETRY_SLEEP,
    request_jitter_min: float = 0.1,
    request_jitter_max: float = 0.5,
    reasoning: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    prompt = build_judge_prompt(row)
    resolved_api_url = normalize_judge_api_url(api_url or judge_api_url_from_env())
    if dry_run:
        return build_judge_record(
            row=row,
            source_label=source_label,
            source_path=source_path,
            source_id=source_id,
            judge_model=judge_model,
            api_url=resolved_api_url,
            raw_response=None,
            parsed=None,
            usage=None,
            response_meta=None,
            token_env_var=token_env_var,
            reasoning=reasoning,
        )

    token = auth_token or os.getenv("AUTH_TOKEN")
    if not token:
        raise RuntimeError("AUTH_TOKEN is missing; set it in .env.local or the app settings.")

    raw, parsed, usage, response_meta = request_judge(
        prompt=prompt,
        judge_model=judge_model,
        auth_token=token,
        api_url=resolved_api_url,
        max_tokens=max_tokens,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_sleep=retry_sleep,
        long_retry_sleep=long_retry_sleep,
        request_jitter_min=request_jitter_min,
        request_jitter_max=request_jitter_max,
        reasoning=reasoning,
    )
    return build_judge_record(
        row=row,
        source_label=source_label,
        source_path=source_path,
        source_id=source_id,
        judge_model=judge_model,
        api_url=resolved_api_url,
        raw_response=raw,
        parsed=parsed,
        usage=usage,
        response_meta=response_meta,
        token_env_var=token_env_var,
        reasoning=reasoning,
    )


def language_code_from_source(source_path: str | Path) -> str | None:
    """Infer the two/three-letter language code from a sentences CSV filename.

    Generated files follow the ``{dataset}_{variant}_{language}`` convention
    (optionally prefixed with ``sentences_``), so the language is the last
    underscore-separated part that we recognise.
    """
    stem = Path(source_path).stem
    if stem.startswith("sentences_"):
        stem = stem.removeprefix("sentences_")
    for part in reversed(stem.split("_")):
        code = sanitize_identifier(part)
        if code in LANGUAGE_NAMES:
            return code
    return None


def resolve_language(
    row: pd.Series | dict[str, Any] | None,
    source_path: str | Path,
    *,
    override: str | None = None,
) -> tuple[str | None, str]:
    """Return ``(code, name)`` for the fluency prompt.

    Preference order: explicit override, a ``language`` column on the row
    (for future generation that stores it), then the filename suffix.
    """
    code = override
    if not code and row is not None:
        row_language = row.get("language") if hasattr(row, "get") else None
        if isinstance(row_language, str) and row_language.strip():
            code = row_language.strip()
    if not code:
        code = language_code_from_source(source_path)
    if code:
        code = sanitize_identifier(code)
    name = LANGUAGE_NAMES.get(code or "", "the target language")
    return code, name


def entity_language_name_from_source(source_path: str | Path) -> str:
    """Human-readable language of the original (untranslated) entities.

    Derived from the source dataset (e.g. cs-qa -> Czech, sk-qa -> Slovak) so
    the fluency judge knows which language proper names may still be written in.
    Falls back to a generic phrase when the dataset is unknown.
    """
    dataset, _ = source_dataset_variant(source_path)
    code = DATASET_ENTITY_LANGUAGE.get(dataset or "")
    if code:
        return LANGUAGE_NAMES.get(code, DEFAULT_ENTITY_LANGUAGE_NAME)
    return DEFAULT_ENTITY_LANGUAGE_NAME


def load_fluency_prompt_template() -> str:
    return FLUENCY_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_fluency_prompt(
    sentence: str,
    language_name: str,
    entity_language_name: str = DEFAULT_ENTITY_LANGUAGE_NAME,
    lexical_terms: list[str] | None = None,
) -> str:
    return load_fluency_prompt_template().format(
        language=language_name,
        entity_language=entity_language_name,
        lexical_terms=format_lexical_terms(lexical_terms or []),
        sentence=prompt_text(sentence),
    )


def build_fluency_record(
    *,
    row: pd.Series | dict[str, Any],
    source_label: str,
    source_path: str,
    source_id: str,
    judge_model: str,
    api_url: str,
    language_code: str | None,
    language_name: str,
    lexical_terms: list[str],
    prompt: str,
    raw_response: str | None,
    parsed: dict[str, Any] | None,
    usage: dict[str, Any] | None = None,
    response_meta: dict[str, Any] | None = None,
    token_env_var: str | None = None,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_preview = prompt.splitlines()[0] if prompt else ""
    usage = usage or {}
    response_meta = response_meta or {}
    return {
        "eid": str(row["eid"]),
        "source_label": source_label,
        "source_path": source_path,
        "source_id": source_id,
        "judge_model": response_meta.get("response_model") or judge_model,
        "requested_judge_model": judge_model,
        "requested_judge_api_url": normalize_judge_api_url(api_url),
        "requested_token_env_var": token_env_var,
        "requested_reasoning": reasoning,
        "provider": response_meta.get("provider"),
        "request_cost": usage.get("cost"),
        "usage": usage or None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentence": str(row["sentence"]),
        "language": language_code,
        "language_name": language_name,
        "source_lexical_terms": lexical_terms,
        "prompt": prompt_preview,
        "raw_response": raw_response,
        "parsed": parsed,
    }


def judge_fluency_row(
    row: pd.Series | dict[str, Any],
    *,
    source_label: str,
    source_path: str,
    source_id: str,
    language_code: str | None,
    language_name: str,
    entity_language_name: str = DEFAULT_ENTITY_LANGUAGE_NAME,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    auth_token: str | None = None,
    token_env_var: str | None = None,
    api_url: str | None = None,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    timeout: int = DEFAULT_JUDGE_TIMEOUT,
    retry_attempts: int = DEFAULT_JUDGE_RETRY_ATTEMPTS,
    retry_sleep: float = DEFAULT_JUDGE_RETRY_SLEEP,
    long_retry_sleep: float = DEFAULT_JUDGE_LONG_RETRY_SLEEP,
    request_jitter_min: float = 0.1,
    request_jitter_max: float = 0.5,
    reasoning: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    lexical_terms = lexical_terms_from_row(row)
    prompt = build_fluency_prompt(
        str(row["sentence"]),
        language_name,
        entity_language_name,
        lexical_terms=lexical_terms,
    )
    resolved_api_url = normalize_judge_api_url(api_url or judge_api_url_from_env())

    def make_record(
        *,
        raw_response: str | None,
        parsed: dict[str, Any] | None,
        usage: dict[str, Any] | None,
        response_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return build_fluency_record(
            row=row,
            source_label=source_label,
            source_path=source_path,
            source_id=source_id,
            judge_model=judge_model,
            api_url=resolved_api_url,
            language_code=language_code,
            language_name=language_name,
            lexical_terms=lexical_terms,
            prompt=prompt,
            raw_response=raw_response,
            parsed=parsed,
            usage=usage,
            response_meta=response_meta,
            token_env_var=token_env_var,
            reasoning=reasoning,
        )

    if dry_run:
        return make_record(raw_response=None, parsed=None, usage=None, response_meta=None)

    token = auth_token or os.getenv("AUTH_TOKEN")
    if not token:
        raise RuntimeError("AUTH_TOKEN is missing; set it in .env.local or the app settings.")

    raw, parsed, usage, response_meta = request_judge(
        prompt=prompt,
        judge_model=judge_model,
        auth_token=token,
        api_url=resolved_api_url,
        max_tokens=max_tokens,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_sleep=retry_sleep,
        long_retry_sleep=long_retry_sleep,
        request_jitter_min=request_jitter_min,
        request_jitter_max=request_jitter_max,
        reasoning=reasoning,
    )
    return make_record(raw_response=raw, parsed=parsed, usage=usage, response_meta=response_meta)


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                records.append(
                    {
                        "_load_error": f"Invalid JSON on line {line_number}",
                        "_path": str(jsonl_path),
                    }
                )
                continue

            payload.setdefault("_path", str(jsonl_path))
            records.append(payload)
    return records


def records_to_annotation_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        parsed = record.get("parsed") or {}
        request_cost = record.get("request_cost")
        if request_cost is None:
            request_cost = 0
        incorrect_information = parsed.get("incorrect_information")
        if incorrect_information is None and parsed.get("unsupported_facts"):
            incorrect_information = [
                {
                    "info_used": item,
                    "correct_info": "",
                    "comment": "",
                }
                for item in parsed.get("unsupported_facts", [])
            ]
        normalized_score = parsed.get("faithfulness_score")
        if normalized_score is None:
            normalized_score = parsed.get("score")
        rows.append(
            {
                "eid": str(record.get("eid", "")),
                "source_label": record.get("source_label"),
                "source_path": record.get("source_path"),
                "source_id": record.get("source_id"),
                "judge_model": record.get("judge_model"),
                "requested_judge_model": record.get("requested_judge_model"),
                "provider": record.get("provider"),
                "request_cost": request_cost,
                "usage": record.get("usage"),
                "timestamp": record.get("timestamp"),
                "sentence": record.get("sentence"),
                "prompt": record.get("prompt"),
                "raw_response": record.get("raw_response"),
                "score": normalized_score,
                "faithfulness_score": normalized_score,
                "label": parsed.get("label"),
                "unsupported_facts": parsed.get("unsupported_facts"),
                "supported_facts": parsed.get("supported_facts"),
                "rationale": parsed.get("rationale"),
                "incorrect_information": incorrect_information,
                "style_comment": parsed.get("style_comment") or parsed.get("rationale"),
                "parsed": record.get("parsed"),
                "_path": record.get("_path"),
                "_load_error": record.get("_load_error"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "eid",
                "source_label",
                "source_path",
                "source_id",
                "judge_model",
                "requested_judge_model",
                "provider",
                "request_cost",
                "usage",
                "timestamp",
                "sentence",
                "prompt",
                "raw_response",
                "score",
                "faithfulness_score",
                "label",
                "unsupported_facts",
                "supported_facts",
                "rationale",
                "incorrect_information",
                "style_comment",
                "parsed",
                "_path",
                "_load_error",
            ]
        )
    return pd.DataFrame(rows)


def load_annotations_for_sources(
    *,
    output_dir: str | Path,
    source_specs: list[SourceSpec],
    judge_model: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    output_root = Path(output_dir)
    for spec in source_specs:
        namespace = source_namespace(spec.csv_path)
        pattern = f"judge_{spec.csv_path.stem}_*.jsonl"
        candidate_paths: list[Path] = []
        if namespace:
            namespaced_root = output_root / namespace
            if namespaced_root.exists():
                candidate_paths = sorted(namespaced_root.rglob(pattern))
        if not candidate_paths:
            candidate_paths = sorted(output_root.rglob(pattern))

        for path in candidate_paths:
            frame = records_to_annotation_frame(load_jsonl_records(path))
            if frame.empty:
                continue
            frame["source_label"] = frame["source_label"].fillna(spec.label)
            frame["source_path"] = frame["source_path"].fillna(str(spec.csv_path))
            frame["source_id"] = frame["source_id"].fillna(spec.source_id)
            if frame["judge_model"].isna().all():
                frame["judge_model"] = path.stem.removeprefix(f"judge_{spec.csv_path.stem}_").replace("_", "/")
            frames.append(frame)

    if not frames:
        return records_to_annotation_frame([])

    combined = pd.concat(frames, ignore_index=True)
    if judge_model:
        combined = combined[combined["judge_model"] == judge_model]
    return combined.reset_index(drop=True)


def load_adhoc_annotations(
    *,
    output_dir: str | Path,
    dataset_name: str,
    judge_model: str | None = None,
) -> pd.DataFrame:
    output_root = Path(output_dir)
    pattern = f"judge_adhoc_{sanitize_identifier(dataset_name)}_*.jsonl"
    frames: list[pd.DataFrame] = []
    for path in sorted(output_root.glob(pattern)):
        frame = records_to_annotation_frame(load_jsonl_records(path))
        if frame.empty:
            continue
        if frame["judge_model"].isna().all():
            frame["judge_model"] = path.stem.removeprefix(
                f"judge_adhoc_{sanitize_identifier(dataset_name)}_"
            ).replace("_", "/")
        frames.append(frame)

    if not frames:
        return records_to_annotation_frame([])

    combined = pd.concat(frames, ignore_index=True)
    if judge_model:
        combined = combined[combined["judge_model"] == judge_model]
    return combined.reset_index(drop=True)


def annotation_lookup_key(eid: str, source_id: str) -> tuple[str, str]:
    return str(eid), source_id


def latest_annotation_map(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if frame.empty:
        return {}

    normalized = frame.copy()
    normalized["timestamp"] = normalized["timestamp"].fillna("")
    normalized = normalized.sort_values(["eid", "source_id", "timestamp"])

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in normalized.iterrows():
        key = annotation_lookup_key(row["eid"], row["source_id"])
        latest[key] = row.to_dict()
    return latest


def write_judge_records(
    *,
    path: str | Path,
    records: list[dict[str, Any]],
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
        model = record.get("requested_judge_model") or record.get("judge_model") or ""
        api_url = record.get("requested_judge_api_url") or DEFAULT_JUDGE_API_URL
        return (
            str(record.get("eid", "")),
            str(record.get("source_id", "")),
            str(model),
            normalize_judge_api_url(str(api_url)),
        )

    existing = load_jsonl_records(output_path)
    keyed_existing: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    ordered_existing: list[dict[str, Any]] = []
    for record in existing:
        if str(record.get("sentence", "")).strip():
            keyed_existing[record_key(record)] = record
        ordered_existing.append(record)

    new_by_key = {record_key(record): record for record in records}

    if overwrite:
        merged_by_key = keyed_existing | new_by_key
        merged_records = list(merged_by_key.values())
        with output_path.open("w", encoding="utf-8") as handle:
            for record in merged_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    replace_empty_keys = {
        record_key(record)
        for record in ordered_existing
        if not str(record.get("sentence", "")).strip() and record_key(record) in new_by_key
    }
    if replace_empty_keys:
        with output_path.open("w", encoding="utf-8") as handle:
            for record in ordered_existing:
                if record_key(record) in replace_empty_keys:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            for key, record in new_by_key.items():
                if key in keyed_existing:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    with output_path.open("a", encoding="utf-8") as handle:
        for key, record in new_by_key.items():
            if key in keyed_existing:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_annotation_summary(
    *,
    outputs_frame: pd.DataFrame,
    annotations_frame: pd.DataFrame,
    filtered_eids: list[str],
) -> pd.DataFrame:
    empty_summary = pd.DataFrame(
        columns=["source_label", "rows", "judged", "avg_score", "avg_issues", "label_counts"]
    )
    if outputs_frame.empty:
        return empty_summary

    scoped_outputs = outputs_frame[outputs_frame["eid"].isin(filtered_eids)].copy()
    scoped_annotations = annotations_frame[annotations_frame["eid"].isin(filtered_eids)].copy()
    if scoped_outputs.empty:
        return empty_summary

    latest = latest_annotation_map(scoped_annotations)

    rows: list[dict[str, Any]] = []
    for source_label, group in scoped_outputs.groupby("source_label", sort=True):
        source_id = group["source_id"].iloc[0]
        judged_records = [
            latest.get(annotation_lookup_key(eid, source_id))
            for eid in group["eid"].astype(str).tolist()
        ]
        judged_records = [record for record in judged_records if record]
        scores = []
        for record in judged_records:
            score = record.get("faithfulness_score")
            if score is None:
                score = record.get("score")
            if score is not None:
                scores.append(score)
        labels = [record.get("label") for record in judged_records if record.get("label")]

        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        rows.append(
            {
                "source_label": source_label,
                "rows": int(len(group)),
                "judged": int(len(judged_records)),
                "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
                "avg_issues": round(
                    sum(len(record.get("incorrect_information") or []) for record in judged_records) / len(judged_records),
                    2,
                ) if judged_records else None,
                "label_counts": ", ".join(f"{label}:{count}" for label, count in sorted(counts.items())) or "None",
            }
        )

    if not rows:
        return empty_summary

    return pd.DataFrame(rows).sort_values("source_label").reset_index(drop=True)
