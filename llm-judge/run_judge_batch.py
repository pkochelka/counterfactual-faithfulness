from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_GENERATED_DIR = Path("data") / "generated"
DEFAULT_OUTPUT_DIR = Path("data") / "judged"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_MODEL = "glm-5"
DEFAULT_JUDGE_BASE_URL = "https://llm.ai.e-infra.cz/v1"
REASONING_EFFORT_CHOICES = ("max", "xhigh", "high", "medium", "low", "minimal", "none")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run judge_csv.py over generated CSVs with retry, per-key concurrency fallback, and logging."
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Directory containing generated sentence CSVs.",
    )
    parser.add_argument(
        "--language",
        choices=["en", "cs", "sk"],
        default=None,
        help="Optional language suffix filter, e.g. en selects *_en.csv.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern under generated-dir. Ignored when --language is set.",
    )
    parser.add_argument(
        "--token-env-vars",
        default=None,
        help="Comma-separated provider token environment variables. The token values are never logged.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model id.")
    parser.add_argument(
        "--judge-base-url",
        default=DEFAULT_JUDGE_BASE_URL,
        help="OpenAI-compatible judge base URL.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=None,
        help="Pass an OpenRouter reasoning effort to judge_csv.py.",
    )
    parser.add_argument(
        "--reasoning-exclude",
        action="store_true",
        help="Ask the provider to omit reasoning content from responses.",
    )
    parser.add_argument(
        "--concurrency-per-key",
        type=positive_int,
        default=4,
        help="Concurrent requests per token env var.",
    )
    parser.add_argument("--fallback-concurrency-per-key", type=positive_int, default=3)
    parser.add_argument(
        "--retries-before-fallback",
        type=int,
        default=1,
        help="How many same-per-key-concurrency retries to run after the first failed attempt.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path. Defaults to logs/llm-judge-batch-<timestamp>.log.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run judge_csv.py.",
    )
    parser.add_argument(
        "--judge-script",
        type=Path,
        default=Path("llm-judge") / "judge_csv.py",
        help="Path to judge_csv.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to judge_csv.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to judge_csv.py.",
    )
    return parser.parse_args()


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_log_file() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_LOG_DIR / f"llm-judge-batch-{stamp}.log"


def find_csvs(generated_dir: Path, *, language: str | None, pattern: str) -> list[Path]:
    if language:
        pattern = f"**/*_{language}.csv"
    else:
        pattern = f"**/{pattern}"
    return sorted(path for path in generated_dir.glob(pattern) if path.is_file())


def build_command(
    args: argparse.Namespace,
    csv_path: Path,
    concurrency_per_key: int,
    *,
    force: bool = False,
) -> list[str]:
    command = [
        args.python,
        str(args.judge_script),
        str(csv_path),
        "--sample-size",
        str(args.sample_size),
        "--model",
        args.model,
        "--judge-base-url",
        args.judge_base_url,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.token_env_vars:
        command.extend(["--token-env-vars", args.token_env_vars])
    command.extend(["--concurrency-per-key", str(concurrency_per_key)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    if args.max_tokens is not None:
        command.extend(["--max-tokens", str(args.max_tokens)])
    if args.reasoning_effort:
        command.extend(["--reasoning-effort", args.reasoning_effort])
    if args.reasoning_exclude:
        command.append("--reasoning-exclude")
    if args.dry_run:
        command.append("--dry-run")
    if force:
        command.append("--force")
    return command


def log(handle, message: str = "") -> None:
    handle.write(f"{message}\n")
    handle.flush()


def run_attempt(
    *,
    args: argparse.Namespace,
    csv_path: Path,
    concurrency_per_key: int,
    attempt_label: str,
    env: dict[str, str],
    log_handle,
    force: bool = False,
) -> bool:
    command = build_command(args, csv_path, concurrency_per_key, force=force)
    log(log_handle, f"[{utc_stamp()}] START {attempt_label} csv={csv_path} concurrency_per_key={concurrency_per_key}")
    log(log_handle, f"[{utc_stamp()}] CMD {' '.join(command)}")

    completed = subprocess.run(command, env=env, text=True, capture_output=True)

    if completed.stdout:
        log(log_handle, "[stdout]")
        log(log_handle, completed.stdout.rstrip())
    if completed.stderr:
        log(log_handle, "[stderr]")
        log(log_handle, completed.stderr.rstrip())

    if completed.returncode == 0:
        log(log_handle, f"[{utc_stamp()}] OK {attempt_label} csv={csv_path}")
        return True

    log(
        log_handle,
        f"[{utc_stamp()}] ERROR {attempt_label} csv={csv_path} returncode={completed.returncode}",
    )
    return False


def run_csv(args: argparse.Namespace, csv_path: Path, env: dict[str, str], log_handle) -> bool:
    attempt = 1
    if run_attempt(
        args=args,
        csv_path=csv_path,
        concurrency_per_key=args.concurrency_per_key,
        attempt_label=f"attempt-{attempt}",
        env=env,
        log_handle=log_handle,
        force=args.force,
    ):
        return True

    for _ in range(args.retries_before_fallback):
        attempt += 1
        log(log_handle, f"[{utc_stamp()}] RETRY csv={csv_path} concurrency_per_key={args.concurrency_per_key}")
        if run_attempt(
            args=args,
            csv_path=csv_path,
            concurrency_per_key=args.concurrency_per_key,
            attempt_label=f"attempt-{attempt}",
            env=env,
            log_handle=log_handle,
        ):
            return True

    attempt += 1
    log(
        log_handle,
        f"[{utc_stamp()}] FALLBACK csv={csv_path} concurrency_per_key={args.fallback_concurrency_per_key}",
    )
    return run_attempt(
        args=args,
        csv_path=csv_path,
        concurrency_per_key=args.fallback_concurrency_per_key,
        attempt_label=f"attempt-{attempt}-fallback",
        env=env,
        log_handle=log_handle,
    )


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    for key, value in load_env_file(Path(".env.local")).items():
        env.setdefault(key, value)

    token_env_vars = parse_csv_list(args.token_env_vars)
    token_env_vars = list(dict.fromkeys(token_env_vars))
    if not token_env_vars and not args.dry_run:
        sys.exit("Set --token-env-vars.")

    missing = [name for name in token_env_vars if not env.get(name)]
    if missing and not args.dry_run:
        sys.exit(f"Missing token environment variable(s): {', '.join(missing)}")

    csvs = find_csvs(args.generated_dir, language=args.language, pattern=args.pattern)
    if not csvs:
        sys.exit(f"No CSV files matched under {args.generated_dir}")

    log_file = args.log_file or default_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[Path] = []
    with log_file.open("a", encoding="utf-8") as log_handle:
        log(log_handle, f"[{utc_stamp()}] Batch started")
        log(log_handle, f"token_env_vars={','.join(token_env_vars) if token_env_vars else 'dry-run'}")
        log(log_handle, f"generated_dir={args.generated_dir}")
        log(log_handle, f"language={args.language or 'all'}")
        log(log_handle, f"concurrency_per_key={args.concurrency_per_key}")
        log(log_handle, f"fallback_concurrency_per_key={args.fallback_concurrency_per_key}")
        log(log_handle, f"files={len(csvs)}")

        for index, csv_path in enumerate(csvs, start=1):
            log(log_handle, "")
            log(log_handle, f"[{utc_stamp()}] FILE {index}/{len(csvs)} {csv_path}")
            print(f"[{index}/{len(csvs)}] judging {csv_path}")
            if not run_csv(args, csv_path, env, log_handle):
                failed.append(csv_path)
                print(f"failed after fallback: {csv_path}", file=sys.stderr)

        log(log_handle, "")
        if failed:
            log(log_handle, f"[{utc_stamp()}] Batch finished with {len(failed)} failed files")
            for path in failed:
                log(log_handle, f"FAILED {path}")
        else:
            log(log_handle, f"[{utc_stamp()}] Batch finished successfully")

    print(f"log: {log_file}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
