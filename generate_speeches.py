from __future__ import annotations

import json
import os
import queue
import random
import re
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from api_caller import NO_AUTH_TOKEN_NAME, call_api

parser = argparse.ArgumentParser()

TASK_PROMPTS = {
    "generated": "prompts/generate_speeches.json",
    "classified": "prompts/classify_speeches.json",
}

parser.add_argument("--model", default="qwen3.5-122b", type=str, help="Model name.")
parser.add_argument("--output-model-name", default="", type=str, help="Optional model name to use in output paths.")
parser.add_argument("--dataset", default="webnlg", type=str, help="Dataset name (e.g. webnlg, cs-qa, sk-qa).")
parser.add_argument("--variant", default="cf", choices=["cf", "fa", "fi"], type=str, help="Dataset variant: cf=counterfactual, fa=factual, fi=fictional.")
parser.add_argument("--kind", default="modified", type=str, help="Value of the 'kind' column to filter on.")
parser.add_argument("--language", default="en", type=str, help="Prompt language (e.g. en, cs, sk).")
parser.add_argument("--token-env-vars", default="", type=str, help="Comma-separated API token env var names.")
parser.add_argument("--concurrency-per-key", default=4, type=int, help="Concurrent requests per token env var.")
parser.add_argument("--retry-attempts", default=3, type=int, help="Maximum attempts per request.")
parser.add_argument("--retry-sleep", default=2.0, type=float, help="Short sleep between retry attempts.")
parser.add_argument("--long-retry-sleep", default=60.0, type=float, help="One longer cooldown before retrying a transient request failure; use 0 to disable.")
parser.add_argument("--task", default="generated", choices=list(TASK_PROMPTS), type=str, help="Task type: determines output folder and prompt file.")
parser.add_argument("--limit", default=None, type=int, help="Optional number of entries to process from the start of the dataset.")
parser.add_argument("--output-root", default="data", type=str, help="Root directory for generated/classified output folders.")
parser.add_argument("--skip-existing", action="store_true", help="Skip this run if the output CSV already exists (resume mode); otherwise it is overwritten.")
parser.add_argument("--resume-missing", action="store_true", help="Reuse non-empty rows from an existing output CSV and request only missing/empty eids.")
parser.add_argument("--repeats", default=1, type=int, help="Number of independent API calls per entry.")
parser.add_argument("--temperature", default=0.0, type=float, help="Sampling temperature for API calls.")
parser.add_argument("--request-jitter-min", default=0.1, type=float, help="Minimum random sleep before each API request.")
parser.add_argument("--request-jitter-max", default=0.5, type=float, help="Maximum random sleep before each API request.")
parser.add_argument("--max-tokens", default=2048, type=int, help="Max tokens per API completion (raise for reasoning models that need room to emit content).")
parser.add_argument("--reasoning-effort", default="", type=str, help="Reasoning effort for reasoning models (e.g. low, medium, high). Empty omits the field.")
parser.add_argument("--disable-thinking", action="store_true", help="Ask compatible local/vLLM reasoning models to disable thinking traces.")


@dataclass(frozen=True)
class TokenSlot:
    env_var: str


@dataclass(frozen=True)
class LLMResult:
    content: str
    reasoning: str | None = None
    raw_content: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    response_model: str | None = None


@dataclass(frozen=True)
class GenerationBatch:
    rows: pd.DataFrame
    failures: list[dict]


class LLMGenerationError(RuntimeError):
    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


_LABEL_RE = re.compile(r"\b(CFA|CF|FA|FI)\b", re.IGNORECASE)
INVALID_CLASSIFICATION_LABEL = "INVALID"
_CLASSIFICATION_LABEL_MAP = {
    "cf": "CFA",
    "cfa": "CFA",
    "fa": "FA",
    "fi": "FI",
}
_CLASSIFICATION_WORD_LABELS = (
    ("CFA", re.compile(r"\bcounterfactual\b|\bkontrafaktu[aá]ln", re.IGNORECASE)),
    ("FI", re.compile(r"\bfictional\b|\bfiktivn|\bfiktívn", re.IGNORECASE)),
    ("FA", re.compile(r"\bfactual\b|\bfaktick", re.IGNORECASE)),
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s*)+")
_CODE_FENCE_RE = re.compile(r"^\s*```")
_META_LINE_RE = re.compile(
    r"^\s*(?:"
    r"len(?:\s+vety?|\s+vetu/vety)?|"
    r"length|lengva|"
    r"vet[auy]|"
    r"d(?:ĺ|l)žka(?:\s+vety)?|"
    r"lo(?:ž|z)ka"
    r")\s*[:：]",
    re.IGNORECASE,
)
_TRAILING_META_RE = re.compile(
    r"(?:"
    r"\s*\((?:len(?:\s+vety?|\s+vetu/vety)?|length|lengva|vet[auy]|d(?:ĺ|l)žka|lo(?:ž|z)ka)[^)]*\)"
    r"|"
    r"\s+(?:len(?:\s+vety?|\s+vetu/vety)?|length|lengva|vet[auy]|d(?:ĺ|l)žka|lo(?:ž|z)ka)\s*[:：].*$"
    r")",
    re.IGNORECASE,
)


def sanitize_classification_content(content: str) -> str:
    """Reduce a classification response to a single-line value.

    The classify prompt asks for only an FA/CFA/FI label, but some models append
    multi-line reasoning, which spills a single entry across many physical CSV
    lines. Accept exactly one label token or label word, mark multi-label
    responses as invalid, and otherwise collapse whitespace so the response
    stays inspectable.
    """
    if not content:
        return content
    words = re.findall(r"[A-Za-z]+", content)
    boundary_labels = [
        _CLASSIFICATION_LABEL_MAP.get(word.lower())
        for word in ((words[:1] or []) + (words[-1:] if len(words) > 1 else []))
    ]
    boundary_labels = [label for label in boundary_labels if label]
    distinct_boundary_labels = list(dict.fromkeys(boundary_labels))
    if len(distinct_boundary_labels) == 1:
        return distinct_boundary_labels[0]
    if len(distinct_boundary_labels) > 1:
        return INVALID_CLASSIFICATION_LABEL

    token_labels = [
        _CLASSIFICATION_LABEL_MAP[match.group(1).lower()]
        for match in _LABEL_RE.finditer(content)
    ]
    word_labels = [label for label, pattern in _CLASSIFICATION_WORD_LABELS if pattern.search(content)]
    labels = token_labels + word_labels
    distinct_labels = list(dict.fromkeys(labels))
    if len(distinct_labels) == 1:
        return distinct_labels[0]
    if labels:
        return INVALID_CLASSIFICATION_LABEL
    return " ".join(content.split())


def strip_wrapping_quotes(content: str) -> str:
    stripped = content.strip()
    quote_pairs = [('"', '"'), ("'", "'"), ("`", "`"), ("“", "”"), ("„", "“")]
    changed = True
    while changed and len(stripped) >= 2:
        changed = False
        for start, end in quote_pairs:
            if stripped.startswith(start) and stripped.endswith(end):
                stripped = stripped[len(start):-len(end)].strip()
                changed = True
                break
    return stripped


def sanitize_generated_content(content: str) -> str:
    """Keep generated text in one physical CSV row and trim common model chatter."""
    if not content:
        return content
    lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or _CODE_FENCE_RE.match(line) or _META_LINE_RE.match(line):
            continue
        line = _LIST_MARKER_RE.sub("", line).strip()
        if line:
            lines.append(line)
    text = strip_wrapping_quotes(" ".join(lines))
    previous = None
    while previous != text:
        previous = text
        text = _TRAILING_META_RE.sub("", text).strip()
        text = strip_wrapping_quotes(text)
    return " ".join(text.split())


def collapse_response_text(content: str | None) -> str:
    if not content:
        return ""
    return " ".join(content.split())


def displayed_raw_response(result: LLMResult) -> str:
    raw_response = collapse_response_text(result.raw_content)
    if raw_response.strip().lower() == result.content.strip().lower():
        return ""
    return raw_response


def compact_payload(payload: object, *, max_chars: int = 4000) -> object:
    if payload is None:
        return None
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return payload
    return {"_truncated_json": text[:max_chars], "_original_length": len(text)}


def response_details(payload: dict | None, *, attempt: int, retry_attempts: int) -> dict:
    choice = None
    message = None
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else None
        if isinstance(choice, dict):
            message = choice.get("message")
            if not isinstance(message, dict):
                message = None
    return {
        "attempt": attempt,
        "retry_attempts": retry_attempts,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "usage": payload.get("usage") if isinstance(payload, dict) else None,
        "response_model": payload.get("model") if isinstance(payload, dict) else None,
        "raw_content": message.get("content") if isinstance(message, dict) else None,
        "response_payload": compact_payload(payload),
    }


def exception_details(exc: Exception, *, attempt_count: int | None = None) -> dict:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        return details
    response = getattr(exc, "response", None)
    if response is not None:
        return {
            "status_code": getattr(response, "status_code", None),
            "response_body": (getattr(response, "text", "") or "")[:2000],
            "attempt_count": attempt_count,
        }
    return {"attempt_count": attempt_count}


def split_inline_reasoning(content: str) -> tuple[str, str | None]:
    """Remove inline <think> traces from content while preserving them separately."""
    reasoning_parts = [match.strip() for match in _THINK_RE.findall(content) if match.strip()]
    cleaned = _THINK_RE.sub("", content).strip()
    return cleaned, "\n\n".join(reasoning_parts) if reasoning_parts else None


def merge_reasoning(*parts: str | None) -> str | None:
    merged = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return "\n\n".join(merged) if merged else None


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_token_slots(token_env_vars: str, concurrency_per_key: int) -> list[TokenSlot]:
    if concurrency_per_key <= 0:
        raise ValueError("--concurrency-per-key must be positive")
    names = parse_csv_list(token_env_vars)
    names = list(dict.fromkeys(names))
    if not names:
        return [TokenSlot(env_var=NO_AUTH_TOKEN_NAME) for _ in range(concurrency_per_key)]

    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing token environment variable(s): {', '.join(missing)}")
    return [TokenSlot(env_var=name) for name in names for _ in range(concurrency_per_key)]


def is_retryable_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return True
    return status_code == 429 or 500 <= status_code < 600


def prompt_text(value: object) -> str:
    return str(value).replace("_", " ")


def build_prompt(entry_df: pd.DataFrame, prompts: dict, language: str = "en") -> str:
    category = entry_df["category"].iloc[0]
    size = entry_df["size"].iloc[0]

    triples = []
    for _, row in entry_df.iterrows():
        triples.append(
            f"  - {prompt_text(row['subject'])} | "
            f"{prompt_text(row['predicate'])} | "
            f"{prompt_text(row['object'])}"
        )

    triples_str = "\n".join(triples)

    template = prompts[language]
    return template.format(size=size, category=category, triples_str=triples_str)


def source_meta_for_eids(subset: pd.DataFrame, eids: set[str] | None = None) -> pd.DataFrame:
    meta_cols = ["eid", "category", "shape", "shape_type", "size"]
    meta = subset[meta_cols].drop_duplicates("eid").copy()
    meta["eid"] = meta["eid"].astype(str)
    if eids is not None:
        meta = meta[meta["eid"].isin(eids)]
    return meta


def make_failure_record(
    *,
    eid: str,
    meta_by_eid: dict[str, dict],
    task: str,
    dataset: str,
    variant: str,
    language: str,
    model: str,
    output_model_name: str,
    token_env_var: str | None,
    error_type: str,
    error: str,
    details: dict | None = None,
) -> dict:
    meta = meta_by_eid.get(str(eid), {})
    details = details or {}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eid": str(eid),
        "task": task,
        "dataset": dataset,
        "variant": variant,
        "language": language,
        "model": model,
        "output_model_name": output_model_name,
        "category": meta.get("category"),
        "shape": meta.get("shape"),
        "shape_type": meta.get("shape_type"),
        "size": meta.get("size"),
        "token_env_var": token_env_var,
        "error_type": error_type,
        "error": error,
        "finish_reason": details.get("finish_reason"),
        "usage": details.get("usage"),
        "response_model": details.get("response_model"),
        "attempt_count": details.get("attempt") or details.get("attempt_count"),
        "retry_attempts": details.get("retry_attempts"),
        "raw_content": collapse_response_text(details.get("raw_content")),
        "details": details,
    }


def failure_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}.failures.jsonl")


def write_jsonl_records(path: str | Path, records: list[dict]) -> None:
    if not records:
        return
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def nonempty_sentence_frame(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(csv_path)
    if "eid" not in frame.columns or "sentence" not in frame.columns:
        return pd.DataFrame()
    frame["eid"] = frame["eid"].astype(str)
    sentence = frame["sentence"].fillna("").astype(str).str.strip()
    frame = frame[sentence != ""].copy()
    return frame.drop_duplicates("eid", keep="last")


def repeat_sentence_column(repeat_index: int) -> str:
    return "sentence" if repeat_index == 1 else f"sentence_{repeat_index}"


def repeat_raw_response_column(repeat_index: int) -> str:
    return "raw_response" if repeat_index == 1 else f"raw_response_{repeat_index}"


def normalize_repeat_columns(frame: pd.DataFrame, repeats: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    if repeats > 1:
        if "sentence" in frame.columns and "sentence_1" not in frame.columns:
            frame["sentence_1"] = frame["sentence"]
        if "raw_response" in frame.columns and "raw_response_1" not in frame.columns:
            frame["raw_response_1"] = frame["raw_response"]
    return frame


def nonempty_value(value: object) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def missing_repeats_by_eid(frame: pd.DataFrame, expected_eids: set[str], repeats: int) -> dict[str, set[int]]:
    missing: dict[str, set[int]] = {eid: set(range(1, repeats + 1)) for eid in expected_eids}
    if frame.empty or "eid" not in frame.columns:
        return missing

    for _, row in frame.iterrows():
        eid = str(row.get("eid", ""))
        if eid not in missing:
            continue
        for repeat_index in range(1, repeats + 1):
            column = repeat_sentence_column(repeat_index)
            fallback_column = "sentence" if repeat_index == 1 else ""
            if column in frame.columns and nonempty_value(row.get(column)):
                missing[eid].discard(repeat_index)
            elif fallback_column and fallback_column in frame.columns and nonempty_value(row.get(fallback_column)):
                missing[eid].discard(repeat_index)
    return {eid: repeat_set for eid, repeat_set in missing.items() if repeat_set}


def merge_resume_frames(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new
    if new.empty:
        return existing

    existing = existing.copy()
    new = new.copy()
    existing["eid"] = existing["eid"].astype(str)
    new["eid"] = new["eid"].astype(str)
    all_columns = list(dict.fromkeys([*existing.columns, *new.columns]))
    existing = existing.reindex(columns=all_columns).astype(object)
    new = new.reindex(columns=all_columns).astype(object)

    merged = existing.set_index("eid", drop=False)
    for _, row in new.iterrows():
        eid = str(row["eid"])
        if eid not in merged.index:
            merged.loc[eid, all_columns] = row
            continue
        for column in all_columns:
            value = row.get(column)
            if nonempty_value(value):
                merged.at[eid, column] = value
    return merged.reset_index(drop=True)


def expected_eids_for_run(df: pd.DataFrame, *, kind: str, limit: int | None) -> list[str]:
    subset = df[df["kind"] == kind] if "kind" in df.columns else df
    eids = [str(eid) for eid in subset.groupby("eid", observed=True).groups.keys()]
    if limit is not None:
        return eids[:limit]
    return eids


def call_llm(
    prompt: str,
    model_id: str,
    token_name: str = "",
    *,
    retry_attempts: int = 3,
    retry_sleep: float = 2.5,
    long_retry_sleep: float = 60.0,
    temperature: float = 0.0,
) -> str:
    """Call the API with retry, extract content from response."""
    return call_llm_result(
        prompt,
        model_id,
        token_name=token_name,
        retry_attempts=retry_attempts,
        retry_sleep=retry_sleep,
        long_retry_sleep=long_retry_sleep,
        temperature=temperature,
    ).content


def extract_reasoning(message: dict) -> str | None:
    for key in ("reasoning", "reasoning_content", "reasoning_text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts = []
        for item in details:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    return None


def call_llm_result(
    prompt: str,
    model_id: str,
    token_name: str = "",
    *,
    retry_attempts: int = 3,
    retry_sleep: float = 2.5,
    long_retry_sleep: float = 60.0,
    temperature: float = 0.0,
    request_jitter_min: float = 0.1,
    request_jitter_max: float = 0.5,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    disable_thinking: bool = False,
) -> LLMResult:
    """Call the API with retry, extract content from response."""
    used_long_retry_sleep = False
    last_empty_details: dict | None = None
    if request_jitter_min < 0 or request_jitter_max < 0 or request_jitter_min > request_jitter_max:
        raise ValueError("request jitter bounds must be non-negative and min <= max")
    for attempt in range(1, retry_attempts + 1):
        try:
            if request_jitter_max > 0:
                time.sleep(random.uniform(request_jitter_min, request_jitter_max))
            response = call_api(
                prompt,
                model_id,
                token_name=token_name,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                disable_thinking=disable_thinking,
            )
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if content:
                cleaned_content, inline_reasoning = split_inline_reasoning(content)
                return LLMResult(
                    content=cleaned_content,
                    reasoning=merge_reasoning(extract_reasoning(message), inline_reasoning),
                    raw_content=content,
                    finish_reason=choice.get("finish_reason"),
                    usage=response.get("usage"),
                    response_model=response.get("model"),
                )
            last_empty_details = response_details(response, attempt=attempt, retry_attempts=retry_attempts)
            if attempt < retry_attempts:
                sleep_for = retry_sleep
                if not used_long_retry_sleep and long_retry_sleep > 0:
                    sleep_for = long_retry_sleep
                    used_long_retry_sleep = True
                print(
                    f"Empty content from {token_name or 'AUTH_TOKEN'}"
                    f" (finish_reason={last_empty_details.get('finish_reason')})."
                    f" Retrying in {sleep_for}s...",
                    flush=True,
                )
                time.sleep(sleep_for)
        except Exception as e:
            if not is_retryable_error(e) or attempt >= retry_attempts:
                raise
            sleep_for = retry_sleep
            if not used_long_retry_sleep and long_retry_sleep > 0:
                sleep_for = long_retry_sleep
                used_long_retry_sleep = True
            print(f"Error with {token_name or 'AUTH_TOKEN'}: {e}. Retrying in {sleep_for}s...", flush=True)
            time.sleep(sleep_for)
    raise LLMGenerationError("LLM response did not contain content.", details=last_empty_details)


def generate_sentences(
    df: pd.DataFrame,
    model_id: str,
    prompts: dict,
    *,
    task: str = "generated",
    dataset: str = "",
    variant: str = "",
    output_model_name: str = "",
    kind: str = "modified",
    language: str = "en",
    token_env_vars: str = "",
    concurrency_per_key: int = 4,
    retry_attempts: int = 3,
    retry_sleep: float = 2.5,
    long_retry_sleep: float = 60.0,
    limit: int | None = None,
    repeats: int = 1,
    temperature: float = 0.0,
    request_jitter_min: float = 0.1,
    request_jitter_max: float = 0.5,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    content_transform: Callable[[str], str] | None = None,
    include_raw_content: bool = False,
    disable_thinking: bool = False,
    skip_eids: set[str] | None = None,
    repeat_indices_by_eid: dict[str, set[int]] | None = None,
    strict_nonempty: bool = False,
) -> GenerationBatch:
    """Generate one sentence per entry."""
    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    transform = content_transform or (lambda content: content)

    subset = df[df["kind"] == kind] if "kind" in df.columns else df
    grouped = subset.groupby("eid", observed=True)

    tasks = {str(eid): build_prompt(group, prompts, language) for eid, group in grouped}
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        tasks = dict(list(tasks.items())[:limit])
        subset = subset[subset["eid"].isin(tasks)]
    if skip_eids:
        tasks = {eid: prompt for eid, prompt in tasks.items() if str(eid) not in skip_eids}
        subset = subset[subset["eid"].astype(str).isin(tasks)]
    if repeat_indices_by_eid is not None:
        tasks = {
            eid: prompt
            for eid, prompt in tasks.items()
            if repeat_indices_by_eid.get(str(eid))
        }
        subset = subset[subset["eid"].astype(str).isin(tasks)]

    meta_by_eid = {
        str(row["eid"]): row
        for row in source_meta_for_eids(subset).to_dict(orient="records")
    }
    results: dict[str, dict[int, LLMResult]] = {}
    failures: list[dict] = []
    if not tasks:
        empty_rows = source_meta_for_eids(subset, set())
        empty_rows["sentence"] = []
        return GenerationBatch(rows=empty_rows, failures=failures)

    token_slots = build_token_slots(token_env_vars, concurrency_per_key)
    slot_queue: queue.Queue[TokenSlot] = queue.Queue()
    for slot in token_slots:
        slot_queue.put(slot)

    def call_with_slot(prompt: str) -> tuple[LLMResult, str]:
        slot = slot_queue.get()
        try:
            result = call_llm_result(
                prompt,
                model_id,
                token_name=slot.env_var,
                retry_attempts=retry_attempts,
                retry_sleep=retry_sleep,
                long_retry_sleep=long_retry_sleep,
                temperature=temperature,
                request_jitter_min=request_jitter_min,
                request_jitter_max=request_jitter_max,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                disable_thinking=disable_thinking,
            )
            return result, slot.env_var
        except Exception as exc:
            setattr(exc, "token_env_var", slot.env_var)
            raise
        finally:
            slot_queue.put(slot)

    with ThreadPoolExecutor(max_workers=len(token_slots)) as pool:
        all_repeats = set(range(1, repeats + 1))
        futures = {
            pool.submit(call_with_slot, prompt): (eid, repeat_index)
            for eid, prompt in tasks.items()
            for repeat_index in sorted(
                repeat_indices_by_eid.get(str(eid), all_repeats)
                if repeat_indices_by_eid is not None
                else all_repeats
            )
        }
        total_calls = len(futures)
        failed_calls = 0
        for i, future in enumerate(as_completed(futures), 1):
            eid, repeat_index = futures[future]
            try:
                result, token_env_var = future.result()
                transformed_content = transform(result.content)
                if strict_nonempty and not transformed_content.strip():
                    failures.append(
                        make_failure_record(
                            eid=eid,
                            meta_by_eid=meta_by_eid,
                            task=task,
                            dataset=dataset,
                            variant=variant,
                            language=language,
                            model=model_id,
                            output_model_name=output_model_name or model_id,
                            token_env_var=token_env_var,
                            error_type="EmptyGeneratedContent",
                            error="Generated content was empty after response cleanup.",
                            details={
                                "finish_reason": result.finish_reason,
                                "usage": result.usage,
                                "response_model": result.response_model,
                                "raw_content": result.raw_content,
                            },
                        )
                    )
                    failed_calls += 1
                else:
                    results.setdefault(eid, {})[repeat_index] = LLMResult(
                        content=transformed_content,
                        reasoning=result.reasoning,
                        raw_content=result.raw_content,
                        finish_reason=result.finish_reason,
                        usage=result.usage,
                        response_model=result.response_model,
                    )
            except Exception as e:
                failed_calls += 1
                print(f"  [warn] eid={eid} repeat={repeat_index} failed, storing empty: {e}", flush=True)
                failures.append(
                    make_failure_record(
                        eid=eid,
                        meta_by_eid=meta_by_eid,
                        task=task,
                        dataset=dataset,
                        variant=variant,
                        language=language,
                        model=model_id,
                        output_model_name=output_model_name or model_id,
                        token_env_var=getattr(e, "token_env_var", None),
                        error_type=type(e).__name__,
                        error=str(e),
                        details=exception_details(e, attempt_count=retry_attempts),
                    )
                )
                if not strict_nonempty:
                    results.setdefault(eid, {})[repeat_index] = LLMResult(content="")
            if i % 50 == 0:
                print(f"  {i}/{total_calls} calls done")
        if failed_calls:
            if strict_nonempty:
                print(f"  [warn] {failed_calls}/{total_calls} calls failed or were empty; rows omitted from CSV", flush=True)
            else:
                print(f"  [warn] {failed_calls}/{total_calls} calls failed; rows kept with empty content", flush=True)

    meta_cols = ["eid", "category", "shape", "shape_type", "size"]
    successful_eids = {eid for eid, per_eid in results.items() if per_eid}
    meta = source_meta_for_eids(subset, successful_eids).set_index("eid")
    meta["sentence"] = meta.index.map(
        lambda eid: (results.get(eid, {}).get(1) or LLMResult(content="")).content
    )
    if repeats > 1:
        for repeat_index in range(1, repeats + 1):
            meta[f"sentence_{repeat_index}"] = meta.index.map(
                lambda eid, index=repeat_index: (results.get(eid, {}).get(index) or LLMResult(content="")).content
            )
    if include_raw_content:
        meta["raw_response"] = meta.index.map(
            lambda eid: displayed_raw_response(results.get(eid, {}).get(1) or LLMResult(content=""))
        )
        if repeats > 1:
            for repeat_index in range(1, repeats + 1):
                meta[f"raw_response_{repeat_index}"] = meta.index.map(
                    lambda eid, index=repeat_index: displayed_raw_response(
                        results.get(eid, {}).get(index) or LLMResult(content="")
                    )
                )
    has_reasoning = any(
        result.reasoning
        for per_eid in results.values()
        for result in per_eid.values()
    )
    if has_reasoning:
        meta["reasoning"] = meta.index.map(
            lambda eid: collapse_response_text(
                (results.get(eid, {}).get(1) or LLMResult(content="")).reasoning
            )
        )
        if repeats > 1:
            for repeat_index in range(1, repeats + 1):
                meta[f"reasoning_{repeat_index}"] = meta.index.map(
                    lambda eid, index=repeat_index: collapse_response_text(
                        (results.get(eid, {}).get(index) or LLMResult(content="")).reasoning
                    )
                )
    return GenerationBatch(rows=meta.reset_index(), failures=failures)


def main(args: argparse.Namespace) -> None:
    output_model_name = args.output_model_name or args.model
    output_dir = os.path.join(args.output_root, args.task, output_model_name)
    output_filename = f"{args.dataset}_{args.variant}_{args.language}.csv"
    output_path = os.path.join(output_dir, output_filename)
    fail_path = failure_output_path(output_path)
    if args.skip_existing and os.path.exists(output_path):
        print(f"[skip] {output_path} already exists; skipping (resume mode).")
        return

    with open(TASK_PROMPTS[args.task], encoding="utf-8") as f:
        prompts = json.load(f)

    input_path = os.path.join("data", args.dataset, f"{args.variant}.csv")
    df = pd.read_csv(input_path)

    n_total = df[df["kind"] == args.kind]["eid"].nunique() if "kind" in df.columns else df["eid"].nunique()
    n_target = min(args.limit, n_total) if args.limit is not None else n_total
    expected_eids = set(expected_eids_for_run(df, kind=args.kind, limit=args.limit))
    print(
        f"Processing {n_target}/{n_total} entries for task={args.task} "
        f"with repeats={args.repeats} temperature={args.temperature}..."
    )
    existing_successes = nonempty_sentence_frame(output_path) if args.resume_missing else pd.DataFrame()
    if not existing_successes.empty:
        existing_successes = existing_successes[existing_successes["eid"].astype(str).isin(expected_eids)].copy()
        existing_successes = normalize_repeat_columns(existing_successes, args.repeats)
    missing_repeat_map = (
        missing_repeats_by_eid(existing_successes, expected_eids, args.repeats)
        if args.resume_missing
        else None
    )
    skip_eids = (
        expected_eids - set(missing_repeat_map)
        if args.resume_missing and missing_repeat_map is not None
        else set()
    )
    if skip_eids:
        print(f"[resume] Reusing {len(skip_eids)} complete existing rows from {output_path}")
    if args.resume_missing and missing_repeat_map:
        missing_calls = sum(len(repeat_set) for repeat_set in missing_repeat_map.values())
        print(
            f"[resume] Requesting {missing_calls} missing repeat calls "
            f"for {len(missing_repeat_map)} eids from {output_path}"
        )

    batch = generate_sentences(
        df,
        args.model,
        prompts,
        task=args.task,
        dataset=args.dataset,
        variant=args.variant,
        output_model_name=output_model_name,
        kind=args.kind,
        language=args.language,
        token_env_vars=args.token_env_vars,
        concurrency_per_key=args.concurrency_per_key,
        retry_attempts=args.retry_attempts,
        retry_sleep=args.retry_sleep,
        long_retry_sleep=args.long_retry_sleep,
        limit=args.limit,
        repeats=args.repeats,
        temperature=args.temperature,
        request_jitter_min=args.request_jitter_min,
        request_jitter_max=args.request_jitter_max,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort or None,
        content_transform=(
            sanitize_classification_content if args.task == "classified" else sanitize_generated_content
        ),
        include_raw_content=args.task == "classified",
        disable_thinking=args.disable_thinking,
        skip_eids=skip_eids,
        repeat_indices_by_eid=missing_repeat_map,
        strict_nonempty=args.task == "generated",
    )

    os.makedirs(output_dir, exist_ok=True)

    result = batch.rows
    if args.resume_missing and not existing_successes.empty:
        result = merge_resume_frames(existing_successes, result)
        result["eid"] = result["eid"].astype(str)

    if args.task == "generated" and "sentence" in result.columns:
        sentence = result["sentence"].fillna("").astype(str).str.strip()
        result = result[sentence != ""].copy()

    write_jsonl_records(fail_path, batch.failures)
    print(result.head())
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} non-empty rows to {output_path}")
    if batch.failures:
        print(f"Wrote {len(batch.failures)} failures to {fail_path}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
