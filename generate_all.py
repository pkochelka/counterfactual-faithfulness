import os
import subprocess
import sys
import threading

DATASETS = ["webnlg", "cs-qa", "sk-qa"]
VARIANTS = ["cf", "fa", "fi"]
LANGUAGES = ["en", "cs", "sk"]
MODELS = ["gpt-oss-120b", "qwen3.5-122b"]
TASKS = ["generated", "classified"]
TOKEN_NAMES = ["AUTH_TOKEN", "AUTH_TOKEN2", "AUTH_TOKEN3"]

DATASET_CONFIG = {
    "cs-qa": {
        "excluded_variants": {"fi"},
        "extra_args": ["--kind", "original"],
    },
    "sk-qa": {
        "excluded_variants": {"fi"},
        "extra_args": ["--kind", "original"],
    },
}


def iter_dataset_jobs(task: str, model: str, dataset: str) -> tuple[list[tuple], int]:
    config = DATASET_CONFIG.get(dataset, {})
    excluded_variants = config.get("excluded_variants", set())
    extra_args = config.get("extra_args", [])
    jobs, skipped = [], 0
    for variant in VARIANTS:
        if variant in excluded_variants:
            continue
        for language in LANGUAGES:
            output_path = os.path.join(
                "data", task, model,
                f"{dataset}_{variant}_{language}.csv",
            )
            if os.path.exists(output_path):
                print(f"[skip] {output_path}")
                skipped += 1
            else:
                jobs.append((task, model, dataset, variant, language, extra_args))
    return jobs, skipped


def collect_jobs() -> tuple[list[tuple], int]:
    jobs, skipped = [], 0
    for task in TASKS:
        for model in MODELS:
            for dataset in DATASETS:
                new_jobs, new_skipped = iter_dataset_jobs(task, model, dataset)
                jobs.extend(new_jobs)
                skipped += new_skipped
    return jobs, skipped


def run_token_jobs(
    token_name: str,
    token_jobs: list[tuple],
    lock: threading.Lock,
    progress: list[int],
    total: int,
    failed: list[int],
) -> None:
    for task, model, dataset, variant, language, extra_args in token_jobs:
        label = f"{task}/{model}/{dataset}_{variant}_{language}"
        cmd = [
            sys.executable, "generate_speeches.py",
            "--model", model,
            "--dataset", dataset,
            "--variant", variant,
            "--language", language,
            "--token-env-vars", token_name,
            "--task", task,
            *extra_args,
        ]
        result = subprocess.run(cmd)
        with lock:
            progress[0] += 1
            status = "done" if result.returncode == 0 else "FAIL"
            print(f"[{status}] {label}  [{progress[0]}/{total}]")
            if result.returncode != 0:
                failed[0] += 1


def main() -> None:
    jobs, skipped = collect_jobs()
    total = len(jobs)
    print(f"\nQueued {total} jobs, skipped {skipped}\n")

    # Distribute jobs round-robin across tokens
    jobs_by_token: dict[str, list] = {name: [] for name in TOKEN_NAMES}
    for i, job in enumerate(jobs):
        jobs_by_token[TOKEN_NAMES[i % len(TOKEN_NAMES)]].append(job)

    lock = threading.Lock()
    progress = [0]
    failed = [0]

    threads = [
        threading.Thread(
            target=run_token_jobs,
            args=(token_name, token_jobs, lock, progress, total, failed),
        )
        for token_name, token_jobs in jobs_by_token.items()
        if token_jobs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\nDone: {total + skipped} total, {skipped} skipped, {failed[0]} failed.")


if __name__ == "__main__":
    main()
