from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from webnlg_utils import (
    DEFAULT_JUDGE_API_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_OUTPUT_DIR,
    JudgeRequestError,
    SourceSpec,
    enrich_sentences,
    judge_output_path,
    judge_row,
    load_env_defaults,
    load_jsonl_records,
    judge_api_url_from_env,
    normalize_judge_api_url,
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
    parser.add_argument("--model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model id")
    parser.add_argument(
        "--judge-base-url",
        "--judge-api-url",
        dest="judge_api_url",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible judge endpoint. Accepts either a base URL such as "
            f"https://api.openai.com/v1 or a full /chat/completions URL. Default: {DEFAULT_JUDGE_API_URL}"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS, help="Maximum completion tokens for each judge request")
    parser.add_argument("--limit", type=int, default=None, help="Upper bound on rows after sampling")
    parser.add_argument("--concurrency", type=int, default=1, help="How many rows to judge in parallel within one CSV")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for outputs")
    parser.add_argument("--label", type=str, default=None, help="Optional source label override")
    parser.add_argument("--force", action="store_true", help="Overwrite existing annotation rows for this source/model")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without calling the API")
    return parser.parse_args()


def main() -> None:
    load_env_defaults()
    args = parse_args()
    judge_api_url = normalize_judge_api_url(args.judge_api_url or judge_api_url_from_env())

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

    existing_keys = set()
    if not args.force:
        for record in load_jsonl_records(out_path):
            key_model = record.get("requested_judge_model") or record.get("judge_model") or ""
            key_api_url = normalize_judge_api_url(
                record.get("requested_judge_api_url") or (DEFAULT_JUDGE_API_URL if judge_api_url == DEFAULT_JUDGE_API_URL else "")
            )
            existing_keys.add((str(record.get("eid", "")), str(record.get("source_id", "")), str(key_model), key_api_url))

    pending_rows = []
    skipped = 0
    for _, row in sampled.iterrows():
        key = (str(row["eid"]), source.source_id, args.model, judge_api_url)
        if not args.force and key in existing_keys:
            skipped += 1
            continue
        pending_rows.append(row)

    failures = 0
    rows = pending_rows

    def worker(row):
        return judge_row(
            row,
            source_label=source.label,
            source_path=str(source.csv_path),
            source_id=source.source_id,
            judge_model=args.model,
            api_url=judge_api_url,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
    )

    max_workers = max(1, int(args.concurrency))
    judged = 0
    if max_workers == 1:
        for row in rows:
            try:
                result = worker(row)
                write_judge_records(path=out_path, records=[result], overwrite=False)
                judged += 1
                print(f"judged {row['eid']}")
            except JudgeRequestError as exc:
                failures += 1
                print(f"failed {row['eid']}: {exc}", file=sys.stderr)
            except Exception as exc:
                failures += 1
                print(f"failed {row['eid']}: {exc}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(worker, row): row for row in rows}
            for future in as_completed(future_map):
                row = future_map[future]
                try:
                    result = future.result()
                    write_judge_records(path=out_path, records=[result], overwrite=False)
                    judged += 1
                    print(f"judged {row['eid']}")
                except JudgeRequestError as exc:
                    failures += 1
                    print(f"failed {row['eid']}: {exc}", file=sys.stderr)
                except Exception as exc:
                    failures += 1
                    print(f"failed {row['eid']}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"completed with {failures} failures and skipped {skipped} existing rows; wrote judged rows to {out_path}",
            file=sys.stderr,
        )
    else:
        print(f"wrote {judged} judged rows to {out_path} (skipped {skipped} existing rows)")


if __name__ == "__main__":
    main()
