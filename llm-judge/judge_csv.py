from __future__ import annotations

import argparse
import json
import os
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

from webnlg_utils import (
    DEFAULT_JUDGE_API_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_RETRY_ATTEMPTS,
    DEFAULT_JUDGE_RETRY_SLEEP,
    DEFAULT_JUDGE_LONG_RETRY_SLEEP,
    DEFAULT_JUDGE_TIMEOUT,
    DEFAULT_FLUENCY_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    JudgeRequestError,
    LANGUAGE_NAMES,
    SourceSpec,
    enrich_sentences,
    entity_language_name_from_source,
    judge_fluency_row,
    judge_output_path,
    judge_row,
    load_env_defaults,
    load_jsonl_records,
    judge_api_url_from_env,
    normalize_judge_api_url,
    resolve_language,
    source_identity,
    write_judge_records,
)


DEFAULT_SAMPLE_SIZE = 10
REASONING_EFFORT_CHOICES = ("max", "xhigh", "high", "medium", "low", "minimal", "none")
PROGRESS_INTERVAL = 50


@dataclass(frozen=True)
class TokenSlot:
    env_var: str | None
    token: str | None


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


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge on generated sentence CSVs.")
    parser.add_argument("csv", type=str, help="Path to sentences CSV")
    parser.add_argument("--xml", type=str, default=None, help="Path to matching XML or flat CSV with source triples")
    parser.add_argument(
        "--fluency",
        action="store_true",
        help=(
            "Judge linguistic fluency instead of faithfulness. Uses matching source triples "
            "only to provide a lexical term set, uses the fluency prompt, and defaults "
            f"the output directory to {DEFAULT_FLUENCY_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Target-language code for judging (e.g. en, cs, sk, hsb). "
            "When omitted, it is read from a 'language' column or inferred from the CSV filename suffix."
        ),
    )
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
    parser.add_argument("--timeout", type=int, default=DEFAULT_JUDGE_TIMEOUT, help="HTTP read timeout in seconds for each judge request")
    parser.add_argument("--retry-attempts", type=positive_int, default=DEFAULT_JUDGE_RETRY_ATTEMPTS, help="Maximum attempts per judge request")
    parser.add_argument("--retry-sleep", type=nonnegative_float, default=DEFAULT_JUDGE_RETRY_SLEEP, help="Short sleep between retry attempts")
    parser.add_argument(
        "--long-retry-sleep",
        type=nonnegative_float,
        default=DEFAULT_JUDGE_LONG_RETRY_SLEEP,
        help="One longer cooldown before retrying a transient request failure; use 0 to disable",
    )
    parser.add_argument("--request-jitter-min", type=nonnegative_float, default=0.1, help="Minimum random sleep before each judge API request")
    parser.add_argument("--request-jitter-max", type=nonnegative_float, default=0.5, help="Maximum random sleep before each judge API request")
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=None,
        help="OpenRouter reasoning effort to request. When omitted, the legacy thinking=true payload is used.",
    )
    parser.add_argument(
        "--reasoning-exclude",
        action="store_true",
        help="Ask the provider to omit reasoning content from the response when --reasoning-effort is set.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Upper bound on rows after sampling")
    parser.add_argument(
        "--concurrency-per-key",
        type=positive_int,
        default=1,
        help="How many concurrent judge requests to allow for each token env var in --token-env-vars",
    )
    parser.add_argument(
        "--token-env-vars",
        type=str,
        default=None,
        help="Comma-separated token environment variable names. Total concurrency is keys * --concurrency-per-key.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for outputs")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Exact JSONL output path. Overrides the model-derived path.",
    )
    parser.add_argument(
        "--skip-existing-eids",
        action="store_true",
        help="Skip any eid/source_id already present in the output file, regardless of judge model or endpoint.",
    )
    parser.add_argument(
        "--failure-output-path",
        type=str,
        default=None,
        help="Exact JSONL path for failed judge rows. Defaults to <output>.failures.jsonl.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write failed rows to the failure JSONL and exit 0 so they can be retried later.",
    )
    parser.add_argument("--label", type=str, default=None, help="Optional source label override")
    parser.add_argument("--force", action="store_true", help="Overwrite existing annotation rows for this source/model")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without calling the API")
    return parser.parse_args()


def reasoning_config_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.reasoning_effort:
        return None
    config: dict[str, object] = {"effort": args.reasoning_effort}
    if args.reasoning_exclude:
        config["exclude"] = True
    return config


def build_token_slots(args: argparse.Namespace) -> list[TokenSlot]:
    token_env_vars = parse_csv_list(args.token_env_vars)
    token_env_vars = list(dict.fromkeys(token_env_vars))

    if not token_env_vars:
        if args.dry_run:
            return [TokenSlot(env_var=None, token=None)]
        raise RuntimeError("Set --token-env-vars to one or more token environment variable names.")

    slots: list[TokenSlot] = []
    missing = []
    for env_var in token_env_vars:
        token = os.getenv(env_var)
        if not token and not args.dry_run:
            missing.append(env_var)
            continue
        slots.extend(TokenSlot(env_var=env_var, token=token) for _ in range(args.concurrency_per_key))
    if missing:
        raise RuntimeError(f"Missing token environment variable(s): {', '.join(missing)}")
    return slots


def failure_output_path(out_path: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return out_path.with_name(f"{out_path.stem}.failures.jsonl")


def write_failure_record(
    *,
    path: Path,
    row,
    source: SourceSpec,
    judge_model: str,
    judge_api_url: str,
    token_env_var: str | None,
    exc: Exception,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    details = getattr(exc, "details", None)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eid": str(row["eid"]),
        "source_label": source.label,
        "source_path": str(source.csv_path),
        "source_id": source.source_id,
        "requested_judge_model": judge_model,
        "requested_judge_api_url": judge_api_url,
        "requested_token_env_var": token_env_var,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "details": details if isinstance(details, dict) else None,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    load_env_defaults()
    args = parse_args()
    judge_api_url = normalize_judge_api_url(args.judge_api_url or judge_api_url_from_env())
    reasoning_config = reasoning_config_from_args(args)
    token_slots = build_token_slots(args)

    csv_path = Path(args.csv).resolve()
    source_label = args.label or csv_path.stem
    source_id = source_identity(csv_path, label=source_label)
    source = SourceSpec(label=source_label, csv_path=csv_path, source_id=source_id)

    enriched = enrich_sentences(csv_path, args.xml)
    if enriched.empty:
        sys.exit("No rows loaded from input CSV.")
    sentence_text = enriched["sentence"].fillna("").astype(str).str.strip()
    empty_sentences = int((sentence_text == "").sum())
    if empty_sentences:
        print(f"[skip] {empty_sentences} rows have empty sentence text; they will not be judged.", file=sys.stderr)
        enriched = enriched[sentence_text != ""].copy()
    if enriched.empty:
        sys.exit("No non-empty sentence rows loaded from input CSV.")

    language_code, language_name = resolve_language(
        enriched.iloc[0], csv_path, override=args.language
    )
    if language_code is None:
        sys.exit(
            "Could not determine the target language. "
            f"Pass --language with one of: {', '.join(LANGUAGE_NAMES)}."
        )
    enriched["target_language"] = language_name

    entity_language_name = None
    if args.fluency:
        entity_language_name = entity_language_name_from_source(csv_path)

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

    output_dir = args.output_dir
    if args.fluency and not args.output_path and Path(output_dir) == DEFAULT_OUTPUT_DIR:
        output_dir = str(DEFAULT_FLUENCY_OUTPUT_DIR)
    out_path = Path(args.output_path) if args.output_path else judge_output_path(source.csv_path, args.model, output_dir)
    fail_path = failure_output_path(out_path, args.failure_output_path)
    if args.force and out_path.exists():
        out_path.unlink()

    existing_keys = set()
    existing_eid_source_keys = set()
    if not args.force:
        for record in load_jsonl_records(out_path):
            if not str(record.get("sentence", "")).strip():
                continue
            eid_source_key = (str(record.get("eid", "")), str(record.get("source_id", "")))
            existing_eid_source_keys.add(eid_source_key)
            key_model = record.get("requested_judge_model") or record.get("judge_model") or ""
            # Trimmed/legacy records don't store the endpoint; count them as judged at the current one.
            key_api_url = normalize_judge_api_url(record.get("requested_judge_api_url") or judge_api_url)
            existing_keys.add((str(record.get("eid", "")), str(record.get("source_id", "")), str(key_model), key_api_url))

    pending_rows = []
    skipped = 0
    for _, row in sampled.iterrows():
        key = (str(row["eid"]), source.source_id, args.model, judge_api_url)
        eid_source_key = (str(row["eid"]), source.source_id)
        if not args.force and (key in existing_keys or (args.skip_existing_eids and eid_source_key in existing_eid_source_keys)):
            skipped += 1
            continue
        pending_rows.append(row)

    failures = 0
    rows = pending_rows

    def timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def print_judged_progress(judged_count: int) -> None:
        if judged_count % PROGRESS_INTERVAL == 0:
            print(f"[{timestamp()}] wrote {judged_count} judged rows to {out_path}", flush=True)

    slot_queue: queue.Queue[TokenSlot] = queue.Queue()
    for slot in token_slots:
        slot_queue.put(slot)

    def worker_with_slot(row):
        slot = slot_queue.get()
        try:
            common_kwargs = dict(
                source_label=source.label,
                source_path=str(source.csv_path),
                source_id=source.source_id,
                judge_model=args.model,
                auth_token=slot.token,
                token_env_var=slot.env_var,
                api_url=judge_api_url,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retry_attempts=args.retry_attempts,
                retry_sleep=args.retry_sleep,
                long_retry_sleep=args.long_retry_sleep,
                request_jitter_min=args.request_jitter_min,
                request_jitter_max=args.request_jitter_max,
                reasoning=reasoning_config,
                dry_run=args.dry_run,
            )
            if args.fluency:
                result = judge_fluency_row(
                    row,
                    language_code=language_code,
                    language_name=language_name,
                    entity_language_name=entity_language_name,
                    **common_kwargs,
                )
            else:
                result = judge_row(row, **common_kwargs)
            return result, slot.env_var
        except Exception as exc:
            setattr(exc, "token_env_var", slot.env_var)
            raise
        finally:
            slot_queue.put(slot)

    max_workers = len(token_slots)
    judged = 0
    if max_workers == 1:
        for row in rows:
            try:
                result = worker_with_slot(row)
                record, _token_env_var = result
                write_judge_records(path=out_path, records=[record], overwrite=False)
                judged += 1
                print_judged_progress(judged)
            except JudgeRequestError as exc:
                failures += 1
                write_failure_record(
                    path=fail_path,
                    row=row,
                    source=source,
                    judge_model=args.model,
                    judge_api_url=judge_api_url,
                    token_env_var=getattr(exc, "token_env_var", None),
                    exc=exc,
                )
                print(f"failed {row['eid']}: {exc}", file=sys.stderr)
            except Exception as exc:
                failures += 1
                write_failure_record(
                    path=fail_path,
                    row=row,
                    source=source,
                    judge_model=args.model,
                    judge_api_url=judge_api_url,
                    token_env_var=getattr(exc, "token_env_var", None),
                    exc=exc,
                )
                print(f"failed {row['eid']}: {exc}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(worker_with_slot, row): row for row in rows}
            for future in as_completed(future_map):
                row = future_map[future]
                try:
                    result, token_env_var = future.result()
                    write_judge_records(path=out_path, records=[result], overwrite=False)
                    judged += 1
                    print_judged_progress(judged)
                except JudgeRequestError as exc:
                    failures += 1
                    write_failure_record(
                        path=fail_path,
                        row=row,
                        source=source,
                        judge_model=args.model,
                        judge_api_url=judge_api_url,
                        token_env_var=getattr(exc, "token_env_var", None),
                        exc=exc,
                    )
                    print(f"failed {row['eid']}: {exc}", file=sys.stderr)
                except Exception as exc:
                    failures += 1
                    write_failure_record(
                        path=fail_path,
                        row=row,
                        source=source,
                        judge_model=args.model,
                        judge_api_url=judge_api_url,
                        token_env_var=getattr(exc, "token_env_var", None),
                        exc=exc,
                    )
                    print(f"failed {row['eid']}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"[{timestamp()}] completed with {failures} failures and skipped {skipped} existing rows; wrote {judged} judged rows to {out_path}",
            file=sys.stderr,
        )
        print(f"wrote failed rows to {fail_path}", file=sys.stderr)
        if not args.allow_failures:
            sys.exit(1)
    else:
        print(f"[{timestamp()}] wrote {judged} judged rows to {out_path} (skipped {skipped} existing rows)")


if __name__ == "__main__":
    main()
