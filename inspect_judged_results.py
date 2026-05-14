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
ISSUE_PATTERNS = {
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
    for model in model_dir_candidates(key.generator_model):
        candidate = repo_root / "data" / "generated" / model / f"{key.dataset}_{key.variant}_{key.language}.csv"
        if candidate.exists():
            return candidate
    return None


def load_expected_eids(path: Path | None) -> set[str]:
    if not path:
        return set()
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return {row.get("eid", "") for row in csv.DictReader(handle) if row.get("eid")}
    except Exception:
        return set()


def detect_issues(text: str) -> list[str]:
    return [name for name, pattern in ISSUE_PATTERNS.items() if pattern.search(text or "")]


def read_judged_records(input_dir: Path, repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_triples, warnings = load_source_triples(repo_root)
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
                expected_path = generated_source_path(repo_root, key)
                if key not in expected_cache:
                    expected_cache[key] = load_expected_eids(expected_path)
                issues = detect_issues(" ".join([incorrect, style]))
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
                    "issue_categories": "; ".join(issues),
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
                    "category": source_entry.get("category", ""),
                    "shape": source_entry.get("shape", ""),
                    "shape_type": source_entry.get("shape_type", ""),
                    "size": source_entry.get("size", ""),
                    "raw_record": json.dumps(record, ensure_ascii=False, sort_keys=True),
                }
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
        "triple_kind",
        "triple_count",
        "category",
        "shape_type",
        "size",
        "incorrect_information",
        "judge_correct_info",
        "judge_info_used",
        "style_comment",
        "issue_categories",
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
        "triple_kind",
        "triple_count",
        "category",
        "shape_type",
        "size",
        "incorrect_information",
        "judge_correct_info",
        "judge_info_used",
        "style_comment",
        "issue_categories",
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
        for issue in str(row.get("issue_categories", "")).split("; "):
            if issue:
                issue_counts[issue] += 1
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
        "| issue | count |",
        "| --- | ---: |",
    ]
    extra.extend(f"| {md_escape(k)} | {v} |" for k, v in issue_counts.most_common())
    write_summary(output_dir / "overall_report.md", "Overall Judged Results Report", rows, extra)
    (output_dir / "index.md").write_text(
        "\n".join(
            [
                "# Judged Results Reports",
                "",
                "- `overall_report.md`: full high-level report.",
                "- `overall_statistics.csv` / `.json`: machine-readable overall statistics.",
                "- `score_cases/`: all examples by faithfulness score in CSV, with sampled Markdown views.",
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
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    input_dir = (repo_root / args.input).resolve() if not args.input.is_absolute() else args.input.resolve()
    output_dir = (repo_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    rows, warnings = read_judged_records(input_dir, repo_root)
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
