#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
import csv
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS = [
    "llama-4-scout-17b-16e-instruct",
    "gemma4",
    "qwen3.5-122b",
]
DEFAULT_OUTPUT_MODEL_NAMES = {
    "gemma4": "gemma4-31B",
}
DEFAULT_KEYS = [
    "EINFRA_AP",
    "EINFRA_JR",
    "EINFRA_VD",
    "EINFRA_VK",
    "EINFRA_TS",
    "EINFRA_PK",
    "EINFRA_VS",
    "EINFRA_KD",
    "EINFRA_FK",
]


@dataclass(frozen=True)
class Task:
    kind: str
    model: str
    dataset: str
    variant: str
    language: str
    generator_model: str | None = None

    @property
    def label(self) -> str:
        if self.kind == "judged" and self.generator_model:
            return f"{self.kind}:{self.model}:{self.generator_model}:{self.dataset}:{self.variant}:{self.language}"
        return f"{self.kind}:{self.model}:{self.dataset}:{self.variant}:{self.language}"


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name, "")
    if not value.strip():
        return list(default)
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def output_model_name(model: str) -> str:
    return DEFAULT_OUTPUT_MODEL_NAMES.get(model, model)


def load_env_defaults(path: Path = Path(".env.local")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


class DynamicScheduler:
    def __init__(self) -> None:
        load_env_defaults()
        self.project_dir = Path(os.environ.get("PROJECT_DIR", os.getcwd())).resolve()
        os.chdir(self.project_dir)

        self.python = os.environ.get("PYTHON", "./.venv/bin/python")
        self.base_url = os.environ.get("BASE_URL", "https://llm.ai.e-infra.cz/v1")
        self.output_root = Path(os.environ.get("OUTPUT_ROOT", "data/einfra_run"))
        self.judge_output_dir = Path(os.environ.get("JUDGE_OUTPUT_DIR", str(self.output_root / "judged")))
        self.judge_model = os.environ.get("JUDGE_MODEL", "glm-5.2")

        self.models = env_list("MODELS", DEFAULT_MODELS)
        self.keys = env_list("EINFRA_KEYS", DEFAULT_KEYS)
        self.datasets = env_list("DATASETS", ["cs-qa", "sk-qa"])
        self.variants = env_list("VARIANTS", ["cf", "fa"])
        self.languages = env_list("LANGUAGES", ["en", "cs", "sk"])

        self.concurrency_per_key = env_int("CONCURRENCY_PER_KEY", 4)
        self.fallback_concurrency_per_key = env_int("FALLBACK_CONCURRENCY_PER_KEY", 3)
        self.max_keys_per_model = env_int("MAX_KEYS_PER_MODEL", 4)
        self.retry_attempts = env_int("RETRY_ATTEMPTS", 4)
        self.long_retry_sleep = float(os.environ.get("LONG_RETRY_SLEEP", "50"))
        self.request_jitter_min = float(os.environ.get("REQUEST_JITTER_MIN", "0.1"))
        self.request_jitter_max = float(os.environ.get("REQUEST_JITTER_MAX", "0.5"))
        self.classification_repeats = env_int("CLASSIFICATION_REPEATS", 5)
        self.classification_temperature = float(os.environ.get("CLASSIFICATION_TEMPERATURE", "1.0"))
        self.gen_max_tokens = env_int("GEN_MAX_TOKENS", 2048)
        self.reasoning_effort = os.environ.get("REASONING_EFFORT", "low").strip()
        self.judge_reasoning_effort = os.environ.get("JUDGE_REASONING_EFFORT", "low").strip()
        self.judge_reasoning_exclude = os.environ.get("JUDGE_REASONING_EXCLUDE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.limit = os.environ.get("LIMIT", "").strip()

        self.logs_dir = Path(os.environ.get("LOG_DIR", "logs"))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.judge_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "einfra-dynamic-pipeline.log"

        self.pending: deque[Task] = deque()
        self.active_by_model: Counter[str] = Counter()
        self.active_tasks = 0
        self.finished_tasks = 0
        self.failed = False
        self.max_active_total = 0
        self.max_active_by_model: Counter[str] = Counter()
        self.tasks_by_key: Counter[str] = Counter()
        self.tasks_by_model: Counter[str] = Counter()
        self.requests_by_key: Counter[str] = Counter()
        self.requests_by_model: Counter[str] = Counter()
        self.condition = threading.Condition()
        self.log_lock = threading.Lock()
        self.stats_lock = threading.Lock()

    def validate(self) -> None:
        if not Path(self.python).exists():
            raise SystemExit(
                f"Missing Python executable: {self.python}\n"
                "Create the venv first, then install pandas==2.3.3 requests==2.32.5."
            )
        missing = [key for key in self.keys if not os.getenv(key)]
        if missing:
            raise SystemExit("Missing token environment variable(s): " + ", ".join(missing))

    def seed_tasks(self) -> None:
        # Put generation before classification for each combination so judge work can start early.
        for dataset in self.datasets:
            for variant in self.variants:
                for language in self.languages:
                    for model in self.models:
                        self.pending.append(Task("generated", model, dataset, variant, language))
                    for model in self.models:
                        self.pending.append(Task("classified", model, dataset, variant, language))

    def generated_csv(self, task: Task) -> Path:
        return self.output_root / "generated" / output_model_name(task.model) / f"{task.dataset}_{task.variant}_{task.language}.csv"

    def classified_csv(self, task: Task) -> Path:
        return self.output_root / "classified" / output_model_name(task.model) / f"{task.dataset}_{task.variant}_{task.language}.csv"

    def output_exists(self, task: Task) -> bool:
        if task.kind == "generated":
            return self.generated_csv(task).is_file() and self.generated_csv(task).stat().st_size > 0
        if task.kind == "classified":
            return self.classified_csv(task).is_file() and self.classified_csv(task).stat().st_size > 0
        return False

    def log(self, message: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        with self.log_lock:
            print(stamped, flush=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(stamped + "\n")

    def pop_task_locked(self) -> Task | None:
        for _ in range(len(self.pending)):
            task = self.pending.popleft()
            if self.active_by_model[task.model] < self.max_keys_per_model:
                self.active_by_model[task.model] += 1
                self.active_tasks += 1
                self.max_active_total = max(self.max_active_total, self.active_tasks)
                self.max_active_by_model[task.model] = max(
                    self.max_active_by_model[task.model],
                    self.active_by_model[task.model],
                )
                return task
            self.pending.append(task)
        return None

    def count_input_entries(self, task: Task) -> int:
        path = Path("data") / task.dataset / f"{task.variant}.csv"
        if not path.exists():
            return 0
        kind = "original" if task.dataset in {"cs-qa", "sk-qa"} else "modified"
        eids: list[str] = []
        seen = set()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "kind" in row and row["kind"] != kind:
                    continue
                eid = row.get("eid", "")
                if eid and eid not in seen:
                    seen.add(eid)
                    eids.append(eid)
        if self.limit:
            return min(int(self.limit), len(eids))
        return len(eids)

    def count_csv_rows(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)

    def estimated_requests(self, task: Task) -> int:
        if task.kind == "generated":
            return self.count_input_entries(task)
        if task.kind == "classified":
            return self.count_input_entries(task) * self.classification_repeats
        if task.kind == "judged" and task.generator_model:
            csv_path = self.output_root / "generated" / output_model_name(task.generator_model) / f"{task.dataset}_{task.variant}_{task.language}.csv"
            return self.count_csv_rows(csv_path)
        return 0

    def record_task_stats(self, key: str, task: Task, request_count: int) -> None:
        with self.stats_lock:
            self.tasks_by_key[key] += 1
            self.tasks_by_model[task.model] += 1
            self.requests_by_key[key] += request_count
            self.requests_by_model[task.model] += request_count

    def log_stats(self) -> None:
        with self.stats_lock:
            self.log(f"[stats] max_active_total={self.max_active_total}")
            for model in sorted(set(self.tasks_by_model) | set(self.requests_by_model) | set(self.max_active_by_model)):
                self.log(
                    "[stats model] "
                    f"model={model} tasks={self.tasks_by_model[model]} "
                    f"estimated_requests={self.requests_by_model[model]} "
                    f"max_active_keys={self.max_active_by_model[model]} "
                    f"max_request_slots={self.max_active_by_model[model] * self.concurrency_per_key}"
                )
            for key in sorted(self.keys):
                self.log(
                    "[stats key] "
                    f"key={key} tasks={self.tasks_by_key[key]} "
                    f"estimated_requests={self.requests_by_key[key]}"
                )

    def complete_task(self, task: Task, ok: bool) -> None:
        with self.condition:
            self.active_by_model[task.model] -= 1
            if self.active_by_model[task.model] <= 0:
                del self.active_by_model[task.model]
            self.active_tasks -= 1
            self.finished_tasks += 1
            if not ok:
                self.failed = True
            self.condition.notify_all()

    def enqueue_judge(self, generated_task: Task) -> None:
        judge_task = Task(
            "judged",
            self.judge_model,
            generated_task.dataset,
            generated_task.variant,
            generated_task.language,
            generator_model=generated_task.model,
        )
        with self.condition:
            # Judge generated outputs promptly so judge keys overlap with the
            # remaining generation/classification queue instead of waiting
            # until the end of the run.
            self.pending.appendleft(judge_task)
            self.log(f"[enqueue judge] task={judge_task.label}")
            self.condition.notify_all()

    def run_command(self, key: str, task: Task, command: list[str]) -> bool:
        self.log(f"[start] key={key} task={task.label} cmd={shell_join(command)}")
        env = os.environ.copy()
        env["BASE_URL"] = self.base_url

        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log(f"[{key} {task.label}] {line.rstrip()}")
            return_code = proc.wait()

        if return_code == 0:
            self.log(f"[done] key={key} task={task.label}")
            return True
        self.log(f"[exit {return_code}] key={key} task={task.label}")
        return False

    def generate_command(self, key: str, task: Task, concurrency: int) -> list[str]:
        command = [
            self.python,
            "generate_speeches.py",
            "--model",
            task.model,
            "--output-model-name",
            output_model_name(task.model),
            "--dataset",
            task.dataset,
            "--variant",
            task.variant,
            "--language",
            task.language,
            "--task",
            task.kind,
            "--token-env-vars",
            key,
            "--concurrency-per-key",
            str(concurrency),
            "--retry-attempts",
            str(self.retry_attempts),
            "--long-retry-sleep",
            str(self.long_retry_sleep),
            "--request-jitter-min",
            str(self.request_jitter_min),
            "--request-jitter-max",
            str(self.request_jitter_max),
            "--output-root",
            str(self.output_root),
            "--max-tokens",
            str(self.gen_max_tokens),
        ]
        if self.reasoning_effort:
            command.extend(["--reasoning-effort", self.reasoning_effort])
        if task.dataset in {"cs-qa", "sk-qa"}:
            command.extend(["--kind", "original"])
        if task.kind == "classified":
            command.extend(
                [
                    "--repeats",
                    str(self.classification_repeats),
                    "--temperature",
                    str(self.classification_temperature),
                ]
            )
        else:
            command.extend(["--temperature", "0"])
        if self.limit:
            command.extend(["--limit", self.limit])
        return command

    def judge_command(self, key: str, task: Task, concurrency: int) -> list[str]:
        if not task.generator_model:
            raise ValueError("judge task is missing generator_model")
        generator_model = task.generator_model
        csv_path = self.output_root / "generated" / output_model_name(generator_model) / f"{task.dataset}_{task.variant}_{task.language}.csv"
        command = [
            self.python,
            "llm-judge/judge_csv.py",
            str(csv_path),
            "--sample-size",
            "all",
            "--model",
            self.judge_model,
            "--judge-base-url",
            self.base_url,
            "--token-env-vars",
            key,
            "--concurrency-per-key",
            str(concurrency),
            "--output-dir",
            str(self.judge_output_dir),
            "--retry-attempts",
            str(self.retry_attempts),
            "--long-retry-sleep",
            str(self.long_retry_sleep),
            "--request-jitter-min",
            str(self.request_jitter_min),
            "--request-jitter-max",
            str(self.request_jitter_max),
            "--allow-failures",
        ]
        if self.judge_reasoning_effort:
            command.extend(["--reasoning-effort", self.judge_reasoning_effort])
            if self.judge_reasoning_exclude:
                command.append("--reasoning-exclude")
        return command

    def run_task(self, key: str, task: Task) -> bool:
        request_count = self.estimated_requests(task)
        if task.kind in {"generated", "classified"} and self.output_exists(task):
            self.log(f"[skip existing] key={key} task={task.label}")
            self.record_task_stats(key, task, 0)
            if task.kind == "generated":
                self.enqueue_judge(task)
            return True

        command_builder = self.judge_command if task.kind == "judged" else self.generate_command
        command = command_builder(key, task, self.concurrency_per_key)
        if self.run_command(key, task, command):
            self.record_task_stats(key, task, request_count)
            if task.kind == "generated":
                self.enqueue_judge(task)
            return True

        self.log(f"[fallback] key={key} task={task.label} concurrency={self.fallback_concurrency_per_key}")
        fallback = command_builder(key, task, self.fallback_concurrency_per_key)
        ok = self.run_command(key, task, fallback)
        if ok:
            self.record_task_stats(key, task, request_count)
        if ok and task.kind == "generated":
            self.enqueue_judge(task)
        return ok

    def worker(self, key: str) -> None:
        while True:
            with self.condition:
                while not self.failed:
                    task = self.pop_task_locked()
                    if task is not None:
                        break
                    if not self.pending and self.active_tasks == 0:
                        return
                    self.condition.wait(timeout=5)
                else:
                    return

            ok = False
            try:
                ok = self.run_task(key, task)
            except Exception as exc:
                self.log(f"[error] key={key} task={task.label}: {exc}")
                ok = False
            finally:
                self.complete_task(task, ok)

    def run(self) -> int:
        start_time = time.monotonic()
        self.validate()
        self.seed_tasks()
        self.log(
            "[config] "
            f"keys={','.join(self.keys)} models={','.join(self.models)} "
            f"datasets={','.join(self.datasets)} variants={','.join(self.variants)} "
            f"languages={','.join(self.languages)} max_keys_per_model={self.max_keys_per_model} "
            f"concurrency_per_key={self.concurrency_per_key} fallback={self.fallback_concurrency_per_key} "
            f"long_retry_sleep={self.long_retry_sleep} "
            f"request_jitter={self.request_jitter_min}-{self.request_jitter_max} "
            f"reasoning_effort={self.reasoning_effort or '<omitted>'} "
            f"judge_reasoning_effort={self.judge_reasoning_effort or '<omitted>'}"
        )

        threads = [threading.Thread(target=self.worker, args=(key,), name=f"worker-{key}") for key in self.keys]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if self.failed:
            elapsed_seconds = time.monotonic() - start_time
            self.log_stats()
            self.log(f"[failed] at least one task failed elapsed_seconds={elapsed_seconds:.1f}")
            return 1
        elapsed_seconds = time.monotonic() - start_time
        self.log_stats()
        self.log(f"[done] finished_tasks={self.finished_tasks} outputs={self.output_root} elapsed_seconds={elapsed_seconds:.1f}")
        return 0


def main() -> None:
    scheduler = DynamicScheduler()
    sys.exit(scheduler.run())


if __name__ == "__main__":
    main()
