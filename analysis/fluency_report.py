"""Fluency low-score cases report (the fluency analogue of the score_cases/ tree
that inspect_judged_results.py writes for faithfulness).

For each fluency score 1..5 it writes score_<n>_cases.{md,csv} listing the actual
sentences the judge gave that score, together with the judge's fluency_comment —
so the low-scoring (poorly-written) sentences are easy to inspect. It also writes
overall_statistics.csv grouped by (model, dataset, variant, language).

Reads judged_fluency records: parsed.fluency_score (1..5) + parsed.fluency_comment,
plus sentence and language. Skips *.failures.jsonl sidecars. hsb is handled like any
other language; models without hsb simply contribute no hsb rows.

Config mirrors the fluency plot:
  --input   judged_fluency dir (default data/judged_fluency, or $JUDGED_DIR)
  --output  report dir         (default data/reports/fluency)
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

from judged_io import iter_judged_records

MAX_SCORE = 5
SCORE_VALUES = [1, 2, 3, 4, 5]


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("\r", " ").replace("|", "\\|").strip()


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in iter_judged_records(input_dir):
        parsed = r.record.get("parsed") or {}
        score = parsed.get("fluency_score")
        if not isinstance(score, int) or score not in SCORE_VALUES:
            continue
        rows.append({
            "generator_model": r.model,
            "dataset": r.dataset,
            "variant": r.variant,
            "language": r.record.get("language", r.language),
            "eid": r.record.get("eid", ""),
            "fluency_score": score,
            "sentence": r.record.get("sentence", ""),
            "fluency_comment": parsed.get("fluency_comment", ""),
        })
    return rows


CASE_COLUMNS = ["generator_model", "dataset", "variant", "language", "eid",
                "fluency_score", "sentence", "fluency_comment"]
MD_HEADERS = ["model", "dataset", "variant", "lang", "eid", "sentence", "fluency comment"]
MD_FIELDS = ["generator_model", "dataset", "variant", "language", "eid", "sentence", "fluency_comment"]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_score_cases(output_dir: Path, rows: list[dict[str, Any]], sample_size: int) -> None:
    cases_dir = output_dir / "score_cases"
    for score in SCORE_VALUES:
        cases = sorted(
            (r for r in rows if r["fluency_score"] == score),
            key=lambda r: (r["generator_model"], r["dataset"], r["variant"], r["language"], str(r["eid"])),
        )
        write_csv(cases_dir / f"score_{score}_cases.csv", cases, CASE_COLUMNS)

        lines = [
            f"# Fluency Score {score} Cases",
            "",
            f"This Markdown file shows a sample of up to {sample_size} cases. "
            "The adjacent CSV contains all matching cases across all languages.",
            "",
            f"Total cases: {len(cases)}",
            "",
            "| " + " | ".join(MD_HEADERS) + " |",
            "| " + " | ".join(["---"] * len(MD_HEADERS)) + " |",
        ]
        for row in cases[:sample_size]:
            lines.append("| " + " | ".join(md_escape(row.get(f)) for f in MD_FIELDS) + " |")
        (cases_dir / f"score_{score}_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for row in rows:
        key = (row["generator_model"], row["dataset"], row["variant"], row["language"])
        groups.setdefault(key, []).append(row["fluency_score"])
    out: list[dict[str, Any]] = []
    for key, scores in sorted(groups.items()):
        n = len(scores)
        counts = Counter(scores)
        out.append({
            "generator_model": key[0],
            "dataset": key[1],
            "variant": key[2],
            "language": key[3],
            "records": n,
            "mean_score": round(sum(scores) / n, 4),
            "pct_score_1": round(counts[1] / n * 100, 2),
            "pct_score_2": round(counts[2] / n * 100, 2),
            "pct_score_3": round(counts[3] / n * 100, 2),
            "pct_score_4": round(counts[4] / n * 100, 2),
            "pct_score_5": round(counts[5] / n * 100, 2),
            "low_pct_score_1_2": round((counts[1] + counts[2]) / n * 100, 2),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = os.environ.get("JUDGED_DIR", str(Path(__file__).parent.parent / "data" / "judged_fluency"))
    parser.add_argument("--input", default=Path(default_input), type=Path)
    parser.add_argument("--output", default=Path(__file__).parent.parent / "data" / "reports" / "fluency", type=Path)
    parser.add_argument("--sample-size", default=50, type=int)
    args = parser.parse_args()

    rows = load_records(args.input.resolve())
    if not rows:
        raise SystemExit(f"No fluency records found under {args.input}")

    output_dir = args.output.resolve()
    write_score_cases(output_dir, rows, args.sample_size)
    stats = group_stats(rows)
    write_csv(output_dir / "overall_statistics.csv", stats, list(stats[0].keys()))

    low = [r for r in rows if r["fluency_score"] <= 2]
    print(f"Read {len(rows)} fluency records from {args.input}")
    print(f"Low (score 1-2) cases: {len(low)} ({len(low) / len(rows) * 100:.2f}%)")
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
