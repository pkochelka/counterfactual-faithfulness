from __future__ import annotations

import argparse
import sys
from pathlib import Path

from webnlg_utils import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_OUTPUT_DIR,
    JudgeRequestError,
    SourceSpec,
    enrich_sentences,
    judge_output_path,
    judge_row,
    load_env_defaults,
    source_identity,
    write_judge_records,
)


DEFAULT_SAMPLE_SIZE = 10


def parse_sample_size(value: str) -> int | str:
    lowered = value.strip().lower()
    if lowered == "all":
        return "all"

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample size must be an integer or 'all'") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample size must be positive or 'all'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge on generated sentence CSVs.")
    parser.add_argument("csv", type=str, help="Path to sentences CSV")
    parser.add_argument("--xml", type=str, default=None, help="Path to matching XML or flat CSV with source triples")
    parser.add_argument(
        "--sample-size",
        type=parse_sample_size,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to score for a smoke test, or 'all' for the whole CSV",
    )
    parser.add_argument("--head", action="store_true", help="Take the first sample-size rows in file order instead of a random sample")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed")
    parser.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL, help="OpenRouter model id")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS, help="Maximum completion tokens for each judge request")
    parser.add_argument("--limit", type=int, default=None, help="Upper bound on rows after sampling")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for outputs")
    parser.add_argument("--label", type=str, default=None, help="Optional source label override")
    parser.add_argument("--force", action="store_true", help="Overwrite existing annotation rows for this source/model")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without calling the API")
    return parser.parse_args()


def main() -> None:
    load_env_defaults()
    args = parse_args()

    csv_path = Path(args.csv).resolve()
    source_label = args.label or csv_path.stem
    source_id = source_identity(csv_path, label=source_label)
    source = SourceSpec(label=source_label, csv_path=csv_path, source_id=source_id)

    enriched = enrich_sentences(csv_path, args.xml)
    if enriched.empty:
        sys.exit("No rows loaded from input CSV.")

    if args.sample_size == "all":
        sampled = enriched
    else:
        sample_size = min(args.sample_size, len(enriched))
        if args.head:
            sampled = enriched.head(sample_size)
        else:
            sampled = enriched.sample(n=sample_size, random_state=args.seed)
    if args.limit is not None:
        sampled = sampled.head(args.limit)

    out_path = judge_output_path(source.csv_path, args.model, args.output_dir)
    if args.force and out_path.exists():
        out_path.unlink()

    failures = 0
    for _, row in sampled.iterrows():
        try:
            result = judge_row(
                row,
                source_label=source.label,
                source_path=str(source.csv_path),
                source_id=source.source_id,
                judge_model=args.model,
                max_tokens=args.max_tokens,
                dry_run=args.dry_run,
            )
            write_judge_records(path=out_path, records=[result], overwrite=False)
            print(f"judged {row['eid']}")
        except JudgeRequestError as exc:
            failures += 1
            print(f"failed {row['eid']}: {exc}", file=sys.stderr)
            continue

    if failures:
        print(f"completed with {failures} failures; wrote judged rows to {out_path}", file=sys.stderr)
    else:
        print(f"wrote {len(sampled)} judged rows to {out_path}")


if __name__ == "__main__":
    main()
