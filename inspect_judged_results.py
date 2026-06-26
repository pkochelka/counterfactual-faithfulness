#!/usr/bin/env python3
"""Inspect judged JSONL results and generate browsable faithfulness reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCORE_VALUES = (1, 2, 3, 4, 5)
LANGUAGES = {"en", "cs", "sk"}
CLASS_LABELS = ("CFA", "FA", "FI")
VARIANT_TO_CLASS_LABEL = {"cf": "CFA", "fa": "FA", "fi": "FI"}
ISSUE_PATTERNS = {
    "refusal": re.compile(
        r"\b("
        r"cannot|can't|unable to|impossible to|not possible to|"
        r"nonsensical|nonsense|invalid triple|erroneous data|"
        r"nelze|nemožn|není možné|nie je možné|nemůže|nemôže|"
        r"nedá sa|neexistuje|nesmysln|nezmyseln|"
        r"chybn(?:é|á|ým|ými)?|chybu|chybou|"
        r"faktick(?:é|ý|ú|ou)|faktografick(?:é|ú|ou)|"
        r"logick(?:é|ú|ou)|nesúlad|nesoulad"
        r")\b",
        re.I,
    ),
    "reverse_relation": re.compile(
        r"\b("
        r"revers(?:e|ed|es|ing)|"
        r"other way around|vice versa|"
        r"subject and object|object and subject|"
        r"direction(?:al|ality)?|direction is|direction of|"
        r"order is reversed|relationship direction|"
        r"not the other way around|"
        r"naopak|obrácen|obráceně|opačn"
        r")\b",
        re.I,
    ),
    "missing_information": re.compile(r"\b(missing|omits?|does not mention|nezmiňuje|chyb)\b", re.I),
    "unsupported_or_hallucinated": re.compile(
        r"\b(unsupported|hallucinat|adds?|not supported|navíc|nepodložen)\b", re.I
    ),
    "contradiction": re.compile(r"\b(contradict|conflict|opposite|rozpor|contradicts)\b", re.I),
    "wrong_entity": re.compile(r"\b(wrong entity|incorrect entity|entity|person|place|organization)\b", re.I),
    "wrong_relation": re.compile(r"\b(wrong relation|relationship|predicate|relation)\b", re.I),
    "wrong_number_date_unit": re.compile(r"\b(date|year|number|unit|měna|rok|datum|číslo)\b", re.I),
    "language_or_style": re.compile(r"\b(style|language|grammar|fluen|ungrammatical|jazyk|styl)\b", re.I),
    "ambiguous_or_unclear": re.compile(r"\b(ambiguous|unclear|vague|nejasn|ambivalent)\b", re.I),
}


@dataclass(frozen=True)
class SourceKey:
    generator_model: str
    dataset: str
    variant: str
    language: str


def sanitize_name(value: str) -> str:
    value = value.strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def display_model_name(name: str) -> str:
    return {"qwen3_5-122b": "qwen3.5-122b"}.get(name, name)


def model_dir_candidates(name: str) -> list[str]:
    candidates = [name]
    if "_" in name:
        candidates.append(name.replace("_", ".", 1))
    if "." in name:
        candidates.append(name.replace(".", "_", 1))
    return list(dict.fromkeys(candidates))


def parse_source_stem(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1] in LANGUAGES:
        return "_".join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) >= 2:
        return "_".join(parts[:-1]), parts[-1], "unknown"
    return stem, "unknown", "unknown"


def infer_key(path: Path, record: dict[str, Any]) -> SourceKey:
    generator_model = path.parent.name
    source_id = str(record.get("source_id") or "")
    if "__" in source_id:
        generator_model = source_id.split("__", 1)[0] or generator_model

    label = str(record.get("source_label") or "")
    source_path = str(record.get("source_path") or "")
    stem = label or Path(source_path.replace("\\", "/")).stem or path.stem
    stem = re.sub(r"^judge_", "", stem)
    stem = re.sub(r"_(?:openai_)?gpt-[^_]+$", "", stem)
    stem = re.sub(r"_glm-5$", "", stem)

    dataset, variant, language = parse_source_stem(stem)
    return SourceKey(generator_model, dataset, variant, language)


def extract_score(record: dict[str, Any]) -> int | None:
    parsed = record.get("parsed")
    candidates: list[Any] = []
    if isinstance(parsed, dict):
        candidates.extend([parsed.get("faithfulness_score"), parsed.get("score")])
    candidates.extend([record.get("faithfulness_score"), record.get("score")])
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, float) and candidate.is_integer():
            return int(candidate)
        if isinstance(candidate, str):
            match = re.search(r"\b([1-5])\b", candidate)
            if match:
                return int(match.group(1))
    return None


def normalize_comment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(normalize_comment(item) for item in value if normalize_comment(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def extract_incorrect_information(record: dict[str, Any]) -> str:
    parsed = record.get("parsed")
    values: list[Any] = []
    if isinstance(parsed, dict):
        values.extend(
            [
                parsed.get("incorrect_information"),
                parsed.get("unsupported_facts"),
                parsed.get("rationale"),
            ]
        )
    values.extend([record.get("incorrect_information"), record.get("unsupported_facts"), record.get("rationale")])
    return "; ".join(part for part in (normalize_comment(value) for value in values) if part)


def extract_style_comment(record: dict[str, Any]) -> str:
    parsed = record.get("parsed")
    values: list[Any] = []
    if isinstance(parsed, dict):
        values.append(parsed.get("style_comment"))
    values.append(record.get("style_comment"))
    return "; ".join(part for part in (normalize_comment(value) for value in values) if part)


def extract_judge_field(record: dict[str, Any], field_name: str) -> str:
    parsed = record.get("parsed")
    values: list[Any] = []
    if isinstance(parsed, dict):
        incorrect = parsed.get("incorrect_information")
        if isinstance(incorrect, list):
            for item in incorrect:
                if isinstance(item, dict):
                    values.append(item.get(field_name))
        elif isinstance(incorrect, dict):
            values.append(incorrect.get(field_name))
        values.append(parsed.get(field_name))
    return "; ".join(part for part in (normalize_comment(value) for value in values) if part)


def md_escape(value: Any) -> str:
    text = normalize_comment(value).replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def load_source_triples(repo_root: Path) -> tuple[dict[tuple[str, str, str], dict[str, dict[str, Any]]], list[dict[str, str]]]:
    sources: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    warnings: list[dict[str, str]] = []
    for source_csv in sorted((repo_root / "data").glob("*/*.csv")):
        rel_parts = source_csv.relative_to(repo_root / "data").parts
        if len(rel_parts) != 2:
            continue
        dataset = rel_parts[0]
        variant = source_csv.stem
        entries: dict[str, dict[str, Any]] = {}
        try:
            with source_csv.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    eid = row.get("eid", "")
                    if not eid:
                        continue
                    entry = entries.setdefault(
                        eid,
                        {
                            "eid": eid,
                            "category": row.get("category", ""),
                            "shape": row.get("shape", ""),
                            "shape_type": row.get("shape_type", ""),
                            "size": row.get("size", ""),
                            "triples_by_kind": defaultdict(list),
                        },
                    )
                    kind = row.get("kind") or "unknown"
                    triple = {
                        "subject": row.get("subject", ""),
                        "predicate": row.get("predicate", ""),
                        "object": row.get("object", ""),
                    }
                    entry["triples_by_kind"][kind].append(triple)
        except Exception as exc:  # noqa: BLE001
            warnings.append({"file": str(source_csv), "problem": f"source CSV load failed: {exc}"})
            continue

        for entry in entries.values():
            triples_by_kind = entry["triples_by_kind"]
            if triples_by_kind.get("modified"):
                selected_kind = "modified"
            elif triples_by_kind.get("original"):
                selected_kind = "original"
            else:
                selected_kind = next(iter(triples_by_kind), "unknown")
            triples = triples_by_kind.get(selected_kind, [])
            entry["triple_kind"] = selected_kind
            entry["triples"] = list(triples)
            entry["triple_count"] = len(triples)
            entry["triples_text"] = "; ".join(
                f"{t['subject']} | {t['predicate']} | {t['object']}" for t in triples
            )
            del entry["triples_by_kind"]
        sources[(dataset, variant, "any")] = entries
    return sources, warnings


def generated_source_path(repo_root: Path, key: SourceKey) -> Path | None:
    return generated_source_path_from_dirs(repo_root, key, [Path("data/generated"), Path("data/generated_v2")])


def generated_source_path_from_dirs(repo_root: Path, key: SourceKey, generated_dirs: list[Path]) -> Path | None:
    for model in model_dir_candidates(key.generator_model):
        for generated_dir in generated_dirs:
            base = generated_dir if generated_dir.is_absolute() else repo_root / generated_dir
            candidate = base / model / f"{key.dataset}_{key.variant}_{key.language}.csv"
            if candidate.exists():
                return candidate
    return None


def classify_untranslated_predicates(sentence: str, predicates: list[str]) -> list[str]:
    hits = []
    for predicate in sorted(set(predicates)):
        if not predicate:
            continue
        if not (re.search(r"[a-z][A-Z]", predicate) or re.search(r"^(?:has|is)[A-Z]", predicate)):
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(predicate)}(?![A-Za-z0-9_])")
        if pattern.search(sentence):
            hits.append(predicate)
    return hits


def load_expected_eids(path: Path | None) -> set[str]:
    if not path:
        return set()
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return {row.get("eid", "") for row in csv.DictReader(handle) if row.get("eid")}
    except Exception:
        return set()


def parse_classification_answer(value: Any) -> tuple[str, str]:
    text = normalize_comment(value)
    if not text:
        return "", "empty"
    if text in CLASS_LABELS:
        return text, "exact"

    compact = text.strip().strip("`*_ \t\r\n.:;-")
    upper = compact.upper()
    if upper in CLASS_LABELS:
        return upper, "recovered"

    start_match = re.match(r"^\s*[*_`#\s]*(CFA|FA|FI)\b", text, re.I)
    if start_match:
        return start_match.group(1).upper(), "recovered"

    labelled_match = re.search(r"\b(?:label|output|answer|classification|kategorie|výstup)\s*[:\-]\s*(CFA|FA|FI)\b", text, re.I)
    if labelled_match:
        return labelled_match.group(1).upper(), "recovered"

    word_map = [
        ("CFA", r"\bcounterfactual\b|\bkontrafaktu[aá]ln"),
        ("FI", r"\bfictional\b|\bfiktivn|\bfiktívn"),
        ("FA", r"\bfactual\b|\bfaktick"),
    ]
    found = re.findall(r"\b(CFA|FA|FI)\b", text, re.I)
    labels = [label.upper() for label in found]
    for label, pattern in word_map:
        if re.search(pattern, text, re.I):
            labels.append(label)
    distinct = list(dict.fromkeys(labels))
    if len(distinct) == 1:
        return distinct[0], "recovered"
    if len(distinct) > 1:
        return distinct[0], "multiple"
    return "", "unparsed"


def classification_file_key(classified_root: Path, path: Path) -> SourceKey | None:
    try:
        model = path.parent.name
        dataset, variant, language = parse_source_stem(path.stem)
        return SourceKey(model, dataset, variant, language)
    except Exception:
        return None


def load_classification_summaries(
    repo_root: Path,
    classified_dirs: list[Path],
) -> dict[SourceKey, dict[str, dict[str, Any]]]:
    summaries: dict[SourceKey, dict[str, dict[str, Any]]] = {}
    for classified_dir in classified_dirs:
        base = classified_dir if classified_dir.is_absolute() else repo_root / classified_dir
        if not base.exists():
            continue
        for csv_path in sorted(base.glob("*/*.csv")):
            key = classification_file_key(base, csv_path)
            if key is None:
                continue
            key_aliases = [key]
            for model in model_dir_candidates(key.generator_model):
                key_aliases.append(SourceKey(model, key.dataset, key.variant, key.language))
            key_aliases = list(dict.fromkeys(key_aliases))
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                answer_fields = [name for name in (reader.fieldnames or []) if name.startswith("sentence")]
                for row in reader:
                    eid = row.get("eid", "")
                    if not eid:
                        continue
                    parsed: list[str] = []
                    statuses: list[str] = []
                    non_exact_examples: list[str] = []
                    for field in answer_fields:
                        raw = row.get(field, "")
                        label, status = parse_classification_answer(raw)
                        if label:
                            parsed.append(label)
                        statuses.append(status)
                        if status != "exact" and raw and len(non_exact_examples) < 3:
                            non_exact_examples.append(normalize_comment(raw)[:400])
                    votes = Counter(parsed)
                    majority = ""
                    if votes:
                        majority = sorted(votes.items(), key=lambda item: (-item[1], CLASS_LABELS.index(item[0]) if item[0] in CLASS_LABELS else 99))[0][0]
                    expected = VARIANT_TO_CLASS_LABEL.get(key.variant, "")
                    parsed_unanimous = len(parsed) == len(answer_fields) and len(set(parsed)) == 1
                    raw_answers = [normalize_comment(row.get(field, "")) for field in answer_fields]
                    raw_unanimous = len(set(raw_answers)) == 1
                    summary = {
                        "classification_expected_label": expected,
                        "classification_vote_label": majority,
                        "classification_vote_count": votes.get(majority, 0) if majority else 0,
                        "classification_answer_count": len(answer_fields),
                        "classification_unanimous_vote": "yes" if parsed_unanimous else "no",
                        "classification_raw_unanimous": "yes" if raw_unanimous else "no",
                        "classification_exact_count": statuses.count("exact"),
                        "classification_recovered_count": statuses.count("recovered") + statuses.count("multiple"),
                        "classification_multiple_count": statuses.count("multiple"),
                        "classification_unparsed_count": statuses.count("unparsed") + statuses.count("empty"),
                        "classification_non_exact_count": sum(1 for status in statuses if status != "exact"),
                        "classification_correct_vote": "yes" if expected and majority == expected else "no" if expected and majority else "",
                        "classification_raw_examples": " || ".join(non_exact_examples),
                        "classification_file": str(csv_path.relative_to(repo_root)),
                    }
                    for key_alias in key_aliases:
                        summaries.setdefault(key_alias, {})[eid] = summary
    return summaries


def classify_issue(sentence: str, explanation: str) -> str:
    if ISSUE_PATTERNS["refusal"].search(" ".join([sentence, explanation])):
        return "refusal"
    for name, pattern in ISSUE_PATTERNS.items():
        if name == "refusal":
            continue
        if pattern.search(explanation or ""):
            return name
    return ""


def read_judged_records(
    input_dir: Path,
    repo_root: Path,
    generated_dirs: list[Path],
    classified_dirs: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_triples, warnings = load_source_triples(repo_root)
    classification_summaries = load_classification_summaries(repo_root, classified_dirs)
    expected_cache: dict[SourceKey, set[str]] = {}
    rows: list[dict[str, Any]] = []
    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(
                        {
                            "file": str(jsonl_path),
                            "line": str(line_number),
                            "problem": f"malformed JSON: {exc}",
                        }
                    )
                    continue
                key = infer_key(jsonl_path, record)
                score = extract_score(record)
                incorrect = extract_incorrect_information(record)
                style = extract_style_comment(record)
                judge_correct_info = extract_judge_field(record, "correct_info")
                judge_info_used = extract_judge_field(record, "info_used")
                source_entry = source_triples.get((key.dataset, key.variant, "any"), {}).get(str(record.get("eid", "")), {})
                expected_path = generated_source_path_from_dirs(repo_root, key, generated_dirs)
                if key not in expected_cache:
                    expected_cache[key] = load_expected_eids(expected_path)
                issue = classify_issue(str(record.get("sentence", "")), " ".join([incorrect, style]))
                source_predicates = [t.get("predicate", "") for t in source_entry.get("triples", []) if isinstance(t, dict)]
                untranslated_predicates = classify_untranslated_predicates(str(record.get("sentence", "")), source_predicates)
                classification_summary = classification_summaries.get(key, {}).get(str(record.get("eid", "")), {})
                row = {
                    "generator_model": key.generator_model,
                    "dataset": key.dataset,
                    "variant": key.variant,
                    "language": key.language,
                    "eid": record.get("eid", ""),
                    "faithfulness_score": score if score is not None else "",
                    "sentence": record.get("sentence", ""),
                    "incorrect_information": incorrect,
                    "judge_correct_info": judge_correct_info,
                    "judge_info_used": judge_info_used,
                    "style_comment": style,
                    "issue_category": issue,
                    "judge_model": record.get("requested_judge_model") or record.get("judge_model") or "",
                    "judge_endpoint": record.get("requested_judge_api_url") or "",
                    "timestamp": record.get("timestamp", ""),
                    "source_label": record.get("source_label") or "",
                    "source_id": record.get("source_id") or "",
                    "source_path": record.get("source_path") or "",
                    "judged_file": str(jsonl_path.relative_to(repo_root)),
                    "judged_line": line_number,
                    "source_csv": str(expected_path.relative_to(repo_root)) if expected_path else "",
                    "source_csv_found": "yes" if expected_path else "no",
                    "expected_source_rows": len(expected_cache[key]),
                    "triple_kind": source_entry.get("triple_kind", ""),
                    "triple_count": source_entry.get("triple_count", ""),
                    "triples": source_entry.get("triples_text", ""),
                    "source_predicates": "; ".join(source_predicates),
                    "untranslated_predicates": "; ".join(untranslated_predicates),
                    "untranslated_predicate_count": len(untranslated_predicates),
                    "category": source_entry.get("category", ""),
                    "shape": source_entry.get("shape", ""),
                    "shape_type": source_entry.get("shape_type", ""),
                    "size": source_entry.get("size", ""),
                    "raw_record": json.dumps(record, ensure_ascii=False, sort_keys=True),
                }
                row.update(classification_summary)
                rows.append(row)
                if score not in SCORE_VALUES:
                    warnings.append(
                        {
                            "file": str(jsonl_path),
                            "line": str(line_number),
                            "problem": f"invalid or missing faithfulness_score: {score}",
                        }
                    )
    return rows, warnings


def score_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [int(row["faithfulness_score"]) for row in rows if row.get("faithfulness_score") in SCORE_VALUES]
    counts = Counter(scores)
    total = len(rows)
    valid = len(scores)
    stats: dict[str, Any] = {
        "records": total,
        "valid_scored_records": valid,
        "invalid_score_records": total - valid,
        "unique_eids": len({row.get("eid") for row in rows if row.get("eid")}),
    }
    for score in SCORE_VALUES:
        stats[f"score_{score}_count"] = counts[score]
        stats[f"score_{score}_pct"] = round((counts[score] / valid * 100), 2) if valid else 0.0
    if scores:
        stats["mean_score"] = round(statistics.mean(scores), 4)
        stats["median_score"] = round(statistics.median(scores), 4)
        stats["stddev_score"] = round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0
        stats["min_score"] = min(scores)
        stats["max_score"] = max(scores)
        stats["strict_success_pct_score_5"] = round(counts[5] / valid * 100, 2)
        stats["lenient_success_pct_score_4_5"] = round((counts[4] + counts[5]) / valid * 100, 2)
        stats["failure_pct_score_1_2"] = round((counts[1] + counts[2]) / valid * 100, 2)
        stats["mixed_pct_score_3"] = round(counts[3] / valid * 100, 2)
    else:
        stats.update(
            {
                "mean_score": "",
                "median_score": "",
                "stddev_score": "",
                "min_score": "",
                "max_score": "",
                "strict_success_pct_score_5": 0.0,
                "lenient_success_pct_score_4_5": 0.0,
                "failure_pct_score_1_2": 0.0,
                "mixed_pct_score_3": 0.0,
            }
        )
    return stats


def group_stats(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for values, group_rows in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, values)}
        item.update(score_stats(group_rows))
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def stats_table_md(rows: list[dict[str, Any]], label_keys: list[str], limit: int = 50) -> str:
    cols = label_keys + [
        "records",
        "mean_score",
        "score_1_count",
        "score_2_count",
        "score_3_count",
        "score_4_count",
        "score_5_count",
        "failure_pct_score_1_2",
        "lenient_success_pct_score_4_5",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(md_escape(row.get(col, "")) for col in cols) + " |")
    if len(rows) > limit:
        lines.append(f"\nShowing {limit} of {len(rows)} rows. See adjacent CSV for all rows.")
    return "\n".join(lines)


def write_score_cases(base_dir: Path, rows: list[dict[str, Any]], sample_size: int) -> None:
    score_dir = base_dir / "score_cases"
    case_fields = [
        "generator_model",
        "dataset",
        "variant",
        "language",
        "eid",
        "faithfulness_score",
        "sentence",
        "triples",
        "source_predicates",
        "untranslated_predicates",
        "untranslated_predicate_count",
        "triple_kind",
        "triple_count",
        "category",
        "shape_type",
        "size",
        "incorrect_information",
        "judge_correct_info",
        "judge_info_used",
        "style_comment",
        "issue_category",
        "classification_expected_label",
        "classification_vote_label",
        "classification_correct_vote",
        "classification_non_exact_count",
        "classification_unparsed_count",
        "judge_model",
        "judge_endpoint",
        "source_csv",
        "judged_file",
        "judged_line",
    ]
    for score in SCORE_VALUES:
        cases = [row for row in rows if row.get("faithfulness_score") == score]
        cases.sort(key=lambda row: (row.get("generator_model", ""), row.get("dataset", ""), row.get("eid", "")))
        write_csv(score_dir / f"score_{score}_cases.csv", cases, case_fields)
        sample = cases[:sample_size]
        lines = [
            f"# Score {score} Cases",
            "",
            f"This Markdown file shows a sample of up to {sample_size} cases. "
            "The adjacent CSV contains all matching cases.",
            "",
            f"Total cases: {len(cases)}",
            "",
        ]
        if sample:
            lines.extend(
                [
                    "| model | dataset | variant | lang | eid | sentence | local source triples | judge correct info | incorrect information |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for row in sample:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            md_escape(row.get("generator_model")),
                            md_escape(row.get("dataset")),
                            md_escape(row.get("variant")),
                            md_escape(row.get("language")),
                            md_escape(row.get("eid")),
                            md_escape(row.get("sentence")),
                            md_escape(row.get("triples")),
                            md_escape(row.get("judge_correct_info")),
                            md_escape(row.get("incorrect_information")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("No cases.")
        (score_dir / f"score_{score}_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_score_cases(base_dir: Path, rows: list[dict[str, Any]], sample_size: int) -> None:
    case_fields = [
        "faithfulness_score",
        "generator_model",
        "dataset",
        "variant",
        "language",
        "eid",
        "sentence",
        "triples",
        "source_predicates",
        "untranslated_predicates",
        "untranslated_predicate_count",
        "triple_kind",
        "triple_count",
        "category",
        "shape_type",
        "size",
        "incorrect_information",
        "judge_correct_info",
        "judge_info_used",
        "style_comment",
        "issue_category",
        "classification_expected_label",
        "classification_vote_label",
        "classification_correct_vote",
        "classification_non_exact_count",
        "classification_unparsed_count",
        "judge_model",
        "judge_endpoint",
        "source_csv",
        "judged_file",
        "judged_line",
    ]
    cases = sorted(
        rows,
        key=lambda row: (
            row.get("faithfulness_score", ""),
            row.get("language", ""),
            row.get("eid", ""),
            row.get("generator_model", ""),
        ),
    )
    write_csv(base_dir / "score_cases.csv", cases, case_fields)

    sample = cases[:sample_size]
    score_counts = Counter(row.get("faithfulness_score") for row in cases)
    lines = [
        "# Score Cases",
        "",
        f"This Markdown file shows a sample of up to {sample_size} cases. "
        "The adjacent CSV contains all matching cases across all scores and languages.",
        "",
        f"Total cases: {len(cases)}",
        "",
        "| score | count |",
        "| --- | ---: |",
    ]
    for score in SCORE_VALUES:
        lines.append(f"| {score} | {score_counts.get(score, 0)} |")
    lines.append("")
    if sample:
        lines.extend(
            [
                "| score | lang | eid | sentence | local source triples | judge correct info | incorrect information |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in sample:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_escape(row.get("faithfulness_score")),
                        md_escape(row.get("language")),
                        md_escape(row.get("eid")),
                        md_escape(row.get("sentence")),
                        md_escape(row.get("triples")),
                        md_escape(row.get("judge_correct_info")),
                        md_escape(row.get("incorrect_information")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No cases.")
    (base_dir / "score_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, title: str, rows: list[dict[str, Any]], extra_sections: list[str] | None = None) -> None:
    stats = score_stats(rows)
    lines = [
        f"# {title}",
        "",
        f"Records: {stats['records']}",
        f"Valid scored records: {stats['valid_scored_records']}",
        f"Unique eids: {stats['unique_eids']}",
        "",
        "## Score Summary",
        "",
        f"- Mean score: {stats['mean_score']}",
        f"- Median score: {stats['median_score']}",
        f"- Failure rate, score 1-2: {stats['failure_pct_score_1_2']}%",
        f"- Mixed rate, score 3: {stats['mixed_pct_score_3']}%",
        f"- Lenient success rate, score 4-5: {stats['lenient_success_pct_score_4_5']}%",
        f"- Strict success rate, score 5: {stats['strict_success_pct_score_5']}%",
        "",
        "| score | count | percent |",
        "| --- | ---: | ---: |",
    ]
    for score in SCORE_VALUES:
        lines.append(f"| {score} | {stats[f'score_{score}_count']} | {stats[f'score_{score}_pct']}% |")
    if extra_sections:
        lines.extend(["", *extra_sections])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_issue_example_reports(base_dir: Path, rows: list[dict[str, Any]], sample_size: int) -> None:
    issue_dir = base_dir / "issue_examples"
    issue_fields = [
        "issue_category",
        "faithfulness_score",
        "generator_model",
        "dataset",
        "variant",
        "language",
        "eid",
        "sentence",
        "triples",
        "incorrect_information",
        "judge_correct_info",
        "style_comment",
        "untranslated_predicates",
        "classification_expected_label",
        "classification_vote_label",
        "classification_correct_vote",
        "classification_raw_examples",
        "judged_file",
        "judged_line",
    ]
    issue_rows = [row for row in rows if row.get("issue_category")]
    issue_rows.sort(
        key=lambda row: (
            row.get("issue_category", ""),
            row.get("faithfulness_score", 99),
            row.get("generator_model", ""),
            row.get("eid", ""),
        )
    )
    write_csv(issue_dir / "all_issue_examples.csv", issue_rows, issue_fields)

    counts = Counter(str(row.get("issue_category", "")) for row in issue_rows if row.get("issue_category"))
    lines = [
        "# Issue Examples",
        "",
        "Rows are assigned at most one primary issue category using judge explanations and refusal cues in the generated sentence.",
        "",
        "| issue | count |",
        "| --- | ---: |",
    ]
    for issue, count in counts.most_common():
        lines.append(f"| {md_escape(issue)} | {count} |")
    for issue in sorted(counts):
        examples = [row for row in issue_rows if row.get("issue_category") == issue][:sample_size]
        lines.extend(
            [
                "",
                f"## {issue}",
                "",
                "| score | model | dataset | variant | lang | eid | sentence | source triples | judge explanation |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in examples:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_escape(row.get("faithfulness_score")),
                        md_escape(row.get("generator_model")),
                        md_escape(row.get("dataset")),
                        md_escape(row.get("variant")),
                        md_escape(row.get("language")),
                        md_escape(row.get("eid")),
                        md_escape(row.get("sentence")),
                        md_escape(row.get("triples")),
                        md_escape(row.get("incorrect_information")),
                    ]
                )
                + " |"
            )
    (issue_dir / "issue_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predicate_leakage_reports(base_dir: Path, rows: list[dict[str, Any]], sample_size: int) -> None:
    leakage_rows = [row for row in rows if int(row.get("untranslated_predicate_count") or 0) > 0]
    summary_rows = []
    for group_keys in (["generator_model"], ["generator_model", "language"], ["generator_model", "dataset", "variant", "language"]):
        for stats_row in group_stats(rows, group_keys):
            matching = [
                row
                for row in rows
                if all(row.get(key, "") == stats_row.get(key, "") for key in group_keys)
                and int(row.get("untranslated_predicate_count") or 0) > 0
            ]
            out = {key: stats_row.get(key, "") for key in group_keys}
            out["grouping"] = "+".join(group_keys)
            out["records"] = stats_row["records"]
            out["predicate_leak_rows"] = len(matching)
            out["predicate_leak_row_pct"] = round(len(matching) / stats_row["records"] * 100, 2) if stats_row["records"] else 0.0
            out["predicate_leak_mentions"] = sum(int(row.get("untranslated_predicate_count") or 0) for row in matching)
            summary_rows.append(out)
    write_csv(base_dir / "predicate_leakage_summary.csv", summary_rows)

    predicate_counts: Counter[str] = Counter()
    for row in leakage_rows:
        for predicate in str(row.get("untranslated_predicates", "")).split("; "):
            if predicate:
                predicate_counts[predicate] += 1
    predicate_rows = [{"predicate": predicate, "mentions": count} for predicate, count in predicate_counts.most_common()]
    write_csv(base_dir / "predicate_leakage_predicates.csv", predicate_rows)

    example_fields = [
        "generator_model",
        "dataset",
        "variant",
        "language",
        "eid",
        "faithfulness_score",
        "untranslated_predicates",
        "sentence",
        "triples",
        "incorrect_information",
        "judged_file",
        "judged_line",
    ]
    leakage_rows.sort(
        key=lambda row: (
            -int(row.get("untranslated_predicate_count") or 0),
            row.get("generator_model", ""),
            row.get("eid", ""),
        )
    )
    write_csv(base_dir / "predicate_leakage_examples.csv", leakage_rows, example_fields)

    lines = [
        "# Source Predicate Leakage",
        "",
        "This flags generated sentences that copy an exact camelCase/source predicate from that row's local triples, such as `hasChild`, `isType`, or `completionYear`.",
        "",
        f"Rows with copied source predicates: {len(leakage_rows)} / {len(rows)}",
        "",
        "## Most Copied Predicates",
        "",
        "| predicate | mentions |",
        "| --- | ---: |",
    ]
    for predicate, count in predicate_counts.most_common(30):
        lines.append(f"| {md_escape(predicate)} | {count} |")
    lines.extend(["", "## Examples", "", "| model | dataset | variant | lang | eid | predicates | sentence |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in leakage_rows[:sample_size]:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.get("generator_model")),
                    md_escape(row.get("dataset")),
                    md_escape(row.get("variant")),
                    md_escape(row.get("language")),
                    md_escape(row.get("eid")),
                    md_escape(row.get("untranslated_predicates")),
                    md_escape(row.get("sentence")),
                ]
            )
            + " |"
        )
    (base_dir / "predicate_leakage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_classification_reports(base_dir: Path, rows: list[dict[str, Any]], sample_size: int) -> None:
    classified_rows = [row for row in rows if row.get("classification_answer_count")]
    if not classified_rows:
        return
    summary_rows = []
    for group_keys in (["generator_model"], ["generator_model", "dataset", "variant", "language"]):
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in classified_rows:
            groups[tuple(row.get(key, "") for key in group_keys)].append(row)
        for values, group in sorted(groups.items()):
            total_answers = sum(int(row.get("classification_answer_count") or 0) for row in group)
            non_exact = sum(int(row.get("classification_non_exact_count") or 0) for row in group)
            unparsed = sum(int(row.get("classification_unparsed_count") or 0) for row in group)
            multiple = sum(int(row.get("classification_multiple_count") or 0) for row in group)
            not_unanimous = sum(1 for row in group if row.get("classification_unanimous_vote") == "no")
            raw_not_unanimous = sum(1 for row in group if row.get("classification_raw_unanimous") == "no")
            correct_votes = sum(1 for row in group if row.get("classification_correct_vote") == "yes")
            out = {key: value for key, value in zip(group_keys, values)}
            out["grouping"] = "+".join(group_keys)
            out["records"] = len(group)
            out["classification_answers"] = total_answers
            out["non_exact_answers"] = non_exact
            out["non_exact_answer_pct"] = round(non_exact / total_answers * 100, 2) if total_answers else 0.0
            out["unparsed_answers"] = unparsed
            out["multiple_label_answers"] = multiple
            out["vote_not_unanimous_rows"] = not_unanimous
            out["vote_not_unanimous_pct"] = round(not_unanimous / len(group) * 100, 2) if group else 0.0
            out["raw_not_unanimous_rows"] = raw_not_unanimous
            out["raw_not_unanimous_pct"] = round(raw_not_unanimous / len(group) * 100, 2) if group else 0.0
            out["majority_vote_correct_rows"] = correct_votes
            out["majority_vote_incorrect_rows"] = len(group) - correct_votes
            out["majority_vote_correct_pct"] = round(correct_votes / len(group) * 100, 2) if group else 0.0
            summary_rows.append(out)
    write_csv(base_dir / "classification_format_summary.csv", summary_rows)

    example_rows = [
        row
        for row in classified_rows
        if int(row.get("classification_non_exact_count") or 0) > 0 or int(row.get("classification_unparsed_count") or 0) > 0
    ]
    example_rows.sort(
        key=lambda row: (
            -int(row.get("classification_unparsed_count") or 0),
            -int(row.get("classification_non_exact_count") or 0),
            row.get("generator_model", ""),
            row.get("eid", ""),
        )
    )
    write_csv(
        base_dir / "classification_format_examples.csv",
        example_rows,
        [
            "generator_model",
            "dataset",
            "variant",
            "language",
            "eid",
            "classification_expected_label",
            "classification_vote_label",
            "classification_correct_vote",
            "classification_answer_count",
            "classification_exact_count",
            "classification_recovered_count",
            "classification_multiple_count",
            "classification_unparsed_count",
            "classification_non_exact_count",
            "classification_raw_examples",
            "classification_file",
            "sentence",
        ],
    )
    lines = [
        "# Classification Output Format",
        "",
        "Strict answers are exactly `CFA`, `FI`, or `FA`. Recovered answers are parsed from punctuation, Markdown, labels with explanations, or longer text; unparsed answers contain no recoverable class.",
        "",
        "| grouping | model | dataset | variant | lang | records | answers | non-exact % | vote not unanimous % | majority incorrect | majority correct % |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows[:80]:
        lines.append(
            "| "
            + " | ".join(
                md_escape(row.get(col, ""))
                for col in [
                    "grouping",
                    "generator_model",
                    "dataset",
                    "variant",
                    "language",
                    "records",
                    "classification_answers",
                    "non_exact_answer_pct",
                    "vote_not_unanimous_pct",
                    "majority_vote_incorrect_rows",
                    "majority_vote_correct_pct",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Non-Exact Examples", "", "| model | dataset | variant | lang | eid | expected | vote | raw examples |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in example_rows[:sample_size]:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.get("generator_model")),
                    md_escape(row.get("dataset")),
                    md_escape(row.get("variant")),
                    md_escape(row.get("language")),
                    md_escape(row.get("eid")),
                    md_escape(row.get("classification_expected_label")),
                    md_escape(row.get("classification_vote_label")),
                    md_escape(row.get("classification_raw_examples")),
                ]
            )
            + " |"
        )
    (base_dir / "classification_format.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def classification_stats_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified_rows = [row for row in rows if row.get("classification_answer_count")]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in classified_rows:
        groups[str(row.get("generator_model", ""))].append(row)

    output = []
    for model, group in sorted(groups.items()):
        total_answers = sum(int(row.get("classification_answer_count") or 0) for row in group)
        non_exact = sum(int(row.get("classification_non_exact_count") or 0) for row in group)
        recovered = sum(int(row.get("classification_recovered_count") or 0) for row in group)
        unparsed = sum(int(row.get("classification_unparsed_count") or 0) for row in group)
        multiple = sum(int(row.get("classification_multiple_count") or 0) for row in group)
        not_unanimous = sum(1 for row in group if row.get("classification_unanimous_vote") == "no")
        raw_not_unanimous = sum(1 for row in group if row.get("classification_raw_unanimous") == "no")
        correct_votes = sum(1 for row in group if row.get("classification_correct_vote") == "yes")
        output.append(
            {
                "generator_model": model,
                "records": len(group),
                "classification_answers": total_answers,
                "non_exact_answers": non_exact,
                "non_exact_answer_pct": round(non_exact / total_answers * 100, 2) if total_answers else 0.0,
                "recovered_answers": recovered,
                "unparsed_answers": unparsed,
                "multiple_label_answers": multiple,
                "vote_not_unanimous_rows": not_unanimous,
                "vote_not_unanimous_pct": round(not_unanimous / len(group) * 100, 2) if group else 0.0,
                "raw_not_unanimous_rows": raw_not_unanimous,
                "raw_not_unanimous_pct": round(raw_not_unanimous / len(group) * 100, 2) if group else 0.0,
                "majority_vote_correct_rows": correct_votes,
                "majority_vote_incorrect_rows": len(group) - correct_votes,
                "majority_vote_correct_pct": round(correct_votes / len(group) * 100, 2) if group else 0.0,
            }
        )
    return output


def write_report_tree(output_dir: Path, rows: list[dict[str, Any]], warnings: list[dict[str, str]], sample_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "overall_statistics.json", score_stats(rows))
    all_groupings = {
        "datasets": ["dataset"],
        "variants": ["variant"],
        "languages": ["language"],
        "judge_models": ["judge_model"],
        "combinations": ["generator_model", "dataset", "variant", "language"],
    }
    for name, keys in all_groupings.items():
        stats_rows = group_stats(rows, keys)
        write_csv(output_dir / "comparisons" / f"{name}.csv", stats_rows)
        (output_dir / "comparisons" / f"{name}.md").write_text(
            f"# {name.replace('_', ' ').title()}\n\n"
            + stats_table_md(stats_rows, keys)
            + "\n",
            encoding="utf-8",
        )
    write_csv(output_dir / "overall_statistics.csv", group_stats(rows, ["generator_model", "dataset", "variant", "language"]))
    write_score_cases(output_dir, rows, sample_size)

    top_bad = sorted(group_stats(rows, ["generator_model", "dataset", "variant", "language"]), key=lambda r: (-float(r["failure_pct_score_1_2"]), -int(r["records"])))[:15]
    top_good = sorted(group_stats(rows, ["generator_model", "dataset", "variant", "language"]), key=lambda r: (-float(r["lenient_success_pct_score_4_5"]), -float(r["mean_score"] or 0)))[:15]
    issue_counts = Counter()
    for row in rows:
        issue = str(row.get("issue_category", ""))
        if issue:
            issue_counts[issue] += 1
    issue_classified_records = sum(issue_counts.values())
    predicate_leak_rows = sum(1 for row in rows if int(row.get("untranslated_predicate_count") or 0) > 0)
    classified_rows = [row for row in rows if row.get("classification_answer_count")]
    non_exact_classification_answers = sum(int(row.get("classification_non_exact_count") or 0) for row in classified_rows)
    total_classification_answers = sum(int(row.get("classification_answer_count") or 0) for row in classified_rows)
    extra = [
        "## Worst Combinations By Failure Rate",
        "",
        stats_table_md(top_bad, ["generator_model", "dataset", "variant", "language"], 15),
        "",
        "## Best Combinations By Lenient Success",
        "",
        stats_table_md(top_good, ["generator_model", "dataset", "variant", "language"], 15),
        "",
        "## Issue Category Counts",
        "",
        "Each row receives at most one primary issue category. Counts sum to the number of rows with a detected issue, not to all judged rows.",
        "",
        f"Rows with detected issue: {issue_classified_records}",
        "",
        "| issue | count |",
        "| --- | ---: |",
    ]
    extra.extend(f"| {md_escape(k)} | {v} |" for k, v in issue_counts.most_common())
    extra.extend(
        [
            "",
            "## Predicate Leakage",
            "",
            f"Rows with exact copied source predicates: {predicate_leak_rows} / {len(rows)}",
            "",
            "See `predicate_leakage.md` and adjacent CSV files for per-model, per-language, and per-predicate details.",
        ]
    )
    if classified_rows:
        classification_model_stats = classification_stats_by_model(rows)
        extra.extend(
            [
                "",
                "## Classification Output Format",
                "",
                f"Rows with classifier outputs: {len(classified_rows)}",
                f"Non-exact classifier answers: {non_exact_classification_answers} / {total_classification_answers}",
                "",
                "| model | records | answers | non-exact % | vote not unanimous | vote not unanimous % | raw not same | majority incorrect | majority correct % |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        extra.extend(
            "| "
            + " | ".join(
                [
                    md_escape(row.get("generator_model")),
                    md_escape(row.get("records")),
                    md_escape(row.get("classification_answers")),
                    md_escape(row.get("non_exact_answer_pct")),
                    md_escape(row.get("vote_not_unanimous_rows")),
                    md_escape(row.get("vote_not_unanimous_pct")),
                    md_escape(row.get("raw_not_unanimous_rows")),
                    md_escape(row.get("majority_vote_incorrect_rows")),
                    md_escape(row.get("majority_vote_correct_pct")),
                ]
            )
            + " |"
            for row in classification_model_stats
        )
        extra.extend(
            [
                "",
                "See `classification_format.md` for strict-vs-parsed label statistics and examples.",
            ]
        )
    write_summary(output_dir / "overall_report.md", "Overall Judged Results Report", rows, extra)
    write_issue_example_reports(output_dir, rows, sample_size)
    write_predicate_leakage_reports(output_dir, rows, sample_size)
    write_classification_reports(output_dir, rows, sample_size)
    (output_dir / "index.md").write_text(
        "\n".join(
            [
                "# Judged Results Reports",
                "",
                "- `overall_report.md`: full high-level report.",
                "- `overall_statistics.csv` / `.json`: machine-readable overall statistics.",
                "- `score_cases/`: all examples by faithfulness score in CSV, with sampled Markdown views.",
                "- `issue_examples/`: examples grouped by detected fail type such as reverse relation, hallucination, refusal, missing information, and wrong entity/relation.",
                "- `predicate_leakage.md` / `.csv`: exact source-predicate copies left untranslated in generated sentences.",
                "- `classification_format.md` / `.csv`: strict-vs-parsed `CFA` / `FI` / `FA` classifier output statistics, when matching classified CSVs are available.",
                "- `comparisons/`: dataset, variant, language, judge model, and combination tables.",
                "- `models/`: browsable report tree by generator model, dataset, and variant.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for model, model_rows in group_by(rows, "generator_model").items():
        model_dir = output_dir / "models" / sanitize_name(model)
        write_json(model_dir / "statistics.json", score_stats(model_rows))
        write_csv(model_dir / "statistics.csv", group_stats(model_rows, ["dataset", "variant", "language"]))
        write_summary(model_dir / "summary.md", f"Model: {display_model_name(model)}", model_rows)
        write_score_cases(model_dir, model_rows, sample_size)
        for dataset, dataset_rows in group_by(model_rows, "dataset").items():
            dataset_dir = model_dir / "datasets" / sanitize_name(dataset)
            write_csv(dataset_dir / "statistics.csv", group_stats(dataset_rows, ["variant", "language"]))
            write_summary(dataset_dir / "summary.md", f"{display_model_name(model)} / {dataset}", dataset_rows)
            for variant, variant_rows in group_by(dataset_rows, "variant").items():
                variant_dir = dataset_dir / sanitize_name(variant)
                write_csv(variant_dir / "statistics.csv", group_stats(variant_rows, ["language"]))
                write_json(variant_dir / "statistics.json", score_stats(variant_rows))
                write_summary(variant_dir / "summary.md", f"{display_model_name(model)} / {dataset} / {variant}", variant_rows)
                write_combined_score_cases(variant_dir, variant_rows, sample_size)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return dict(sorted(groups.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/judged", type=Path, help="Directory containing judged JSONL files.")
    parser.add_argument("--output", default="reports/judged", type=Path, help="Directory where reports will be written.")
    parser.add_argument("--sample-size", default=50, type=int, help="Max cases shown in each Markdown score file.")
    parser.add_argument("--repo-root", default=Path("."), type=Path, help="Repository root for enrichment lookups.")
    parser.add_argument(
        "--generated-dir",
        action="append",
        type=Path,
        help="Generated CSV directory to use for coverage lookups. Can be passed multiple times.",
    )
    parser.add_argument(
        "--classified-dir",
        action="append",
        type=Path,
        help="Classifier-output CSV directory to parse for CFA/FI/FA format reports. Can be passed multiple times.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    input_dir = (repo_root / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    output_dir = (repo_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    generated_dirs = args.generated_dir or [Path("data/generated"), Path("data/generated_v2")]
    classified_dirs = args.classified_dir or [Path("data/classified"), Path("data/classified_v2")]
    rows, warnings = read_judged_records(input_dir, repo_root, generated_dirs, classified_dirs)
    write_report_tree(output_dir, rows, warnings, args.sample_size)
    stats = score_stats(rows)
    print(f"Read {stats['records']} judged records from {input_dir}")
    print(f"Valid scored records: {stats['valid_scored_records']}")
    print(f"Mean score: {stats['mean_score']}")
    print(f"Failure rate score 1-2: {stats['failure_pct_score_1_2']}%")
    print(f"Lenient success score 4-5: {stats['lenient_success_pct_score_4_5']}%")
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
