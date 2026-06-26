from __future__ import annotations

import json
import os
import queue
import random
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from dataclasses import dataclass

from api_caller import call_api

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
parser.add_argument("--token-name", default="", type=str, help="Env var name for the API token (default: AUTH_TOKEN).")
parser.add_argument("--token-env-vars", default="", type=str, help="Comma-separated API token env var names.")
parser.add_argument("--concurrency-per-key", default=None, type=int, help="Concurrent requests per token env var.")
parser.add_argument("--max-workers", default=4, type=int, help="Fallback total concurrency when using one token.")
parser.add_argument("--retry-attempts", default=3, type=int, help="Maximum attempts per request.")
parser.add_argument("--retry-sleep", default=2.5, type=float, help="Short sleep between retry attempts.")
parser.add_argument("--long-retry-sleep", default=60.0, type=float, help="One longer cooldown before retrying a transient request failure; use 0 to disable.")
parser.add_argument("--task", default="generated", choices=list(TASK_PROMPTS), type=str, help="Task type: determines output folder and prompt file.")
parser.add_argument("--limit", default=None, type=int, help="Optional number of entries to process from the start of the dataset.")
parser.add_argument("--output-root", default="data", type=str, help="Root directory for generated/classified output folders.")
parser.add_argument("--skip-existing", action="store_true", help="Skip this run if the output CSV already exists (resume mode); otherwise it is overwritten.")
parser.add_argument("--repeats", default=1, type=int, help="Number of independent API calls per entry.")
parser.add_argument("--temperature", default=0.0, type=float, help="Sampling temperature for API calls.")
parser.add_argument("--request-jitter-min", default=0.1, type=float, help="Minimum random sleep before each API request.")
parser.add_argument("--request-jitter-max", default=0.5, type=float, help="Maximum random sleep before each API request.")
parser.add_argument("--max-tokens", default=2048, type=int, help="Max tokens per API completion (raise for reasoning models that need room to emit content).")
parser.add_argument("--reasoning-effort", default="", type=str, help="Reasoning effort for reasoning models (e.g. low, medium, high). Empty omits the field.")


@dataclass(frozen=True)
class TokenSlot:
    env_var: str


@dataclass(frozen=True)
class LLMResult:
    content: str
    reasoning: str | None = None


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_token_slots(token_name: str, token_env_vars: str, concurrency_per_key: int | None, max_workers: int) -> list[TokenSlot]:
    names = parse_csv_list(token_env_vars)
    if token_name:
        names.append(token_name)
    names = list(dict.fromkeys(names))

    if names:
        missing = [name for name in names if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing token environment variable(s): {', '.join(missing)}")
        slots_per_key = concurrency_per_key or max(1, max_workers)
        slots = [TokenSlot(env_var=name) for name in names for _ in range(slots_per_key)]
    else:
        slots = [TokenSlot(env_var="AUTH_TOKEN") for _ in range(max(1, max_workers))]
    return slots


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
) -> LLMResult:
    """Call the API with retry, extract content from response."""
    used_long_retry_sleep = False
    if request_jitter_min < 0 or request_jitter_max < 0 or request_jitter_min > request_jitter_max:
        raise ValueError("request jitter bounds must be non-negative and min <= max")
    for attempt in range(1, retry_attempts + 1):
        try:
            if request_jitter_max > 0:
                time.sleep(random.uniform(request_jitter_min, request_jitter_max))
            response = call_api(prompt, model_id, token_name=token_name, temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
            message = response["choices"][0]["message"]
            content = message["content"]
            if content:
                return LLMResult(content=content, reasoning=extract_reasoning(message))
        except Exception as e:
            if not is_retryable_error(e) or attempt >= retry_attempts:
                raise
            sleep_for = retry_sleep
            if not used_long_retry_sleep and long_retry_sleep > 0:
                sleep_for = long_retry_sleep
                used_long_retry_sleep = True
            print(f"Error with {token_name or 'AUTH_TOKEN'}: {e}. Retrying in {sleep_for}s...", flush=True)
            time.sleep(sleep_for)
    raise RuntimeError("LLM response did not contain content.")


def generate_sentences(
    df: pd.DataFrame,
    model_id: str,
    prompts: dict,
    kind: str = "modified",
    language: str = "en",
    max_workers: int = 4,
    token_name: str = "",
    token_env_vars: str = "",
    concurrency_per_key: int | None = None,
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
) -> pd.DataFrame:
    """Generate one sentence per entry."""
    if repeats <= 0:
        raise ValueError("--repeats must be positive")

    subset = df[df["kind"] == kind] if "kind" in df.columns else df
    grouped = subset.groupby("eid", observed=True)

    tasks = {eid: build_prompt(group, prompts, language) for eid, group in grouped}
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        tasks = dict(list(tasks.items())[:limit])
        subset = subset[subset["eid"].isin(tasks)]
    results: dict[str, dict[int, LLMResult]] = {}
    token_slots = build_token_slots(token_name, token_env_vars, concurrency_per_key, max_workers)
    slot_queue: queue.Queue[TokenSlot] = queue.Queue()
    for slot in token_slots:
        slot_queue.put(slot)

    def call_with_slot(prompt: str) -> str:
        slot = slot_queue.get()
        try:
            return call_llm_result(
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
            )
        finally:
            slot_queue.put(slot)

    with ThreadPoolExecutor(max_workers=len(token_slots)) as pool:
        futures = {
            pool.submit(call_with_slot, prompt): (eid, repeat_index)
            for eid, prompt in tasks.items()
            for repeat_index in range(1, repeats + 1)
        }
        total_calls = len(futures)
        failed_calls = 0
        for i, future in enumerate(as_completed(futures), 1):
            eid, repeat_index = futures[future]
            try:
                results.setdefault(eid, {})[repeat_index] = future.result()
            except Exception as e:
                failed_calls += 1
                print(f"  [warn] eid={eid} repeat={repeat_index} failed, storing empty: {e}", flush=True)
                results.setdefault(eid, {})[repeat_index] = LLMResult(content="")
            if i % 50 == 0:
                print(f"  {i}/{total_calls} calls done")
        if failed_calls:
            print(f"  [warn] {failed_calls}/{total_calls} calls failed; rows kept with empty content", flush=True)

    meta_cols = ["eid", "category", "shape", "shape_type", "size"]
    meta = subset[meta_cols].drop_duplicates("eid").set_index("eid")
    meta["sentence"] = meta.index.map(
        lambda eid: (results.get(eid, {}).get(1) or LLMResult(content="")).content
    )
    if repeats > 1:
        for repeat_index in range(1, repeats + 1):
            meta[f"sentence_{repeat_index}"] = meta.index.map(
                lambda eid, index=repeat_index: (
                    results.get(eid, {}).get(index) or LLMResult(content="")
                ).content
            )
    has_reasoning = any(
        result.reasoning
        for per_eid in results.values()
        for result in per_eid.values()
    )
    if has_reasoning:
        meta["reasoning"] = meta.index.map(
            lambda eid: (results.get(eid, {}).get(1) or LLMResult(content="")).reasoning
        )
        if repeats > 1:
            for repeat_index in range(1, repeats + 1):
                meta[f"reasoning_{repeat_index}"] = meta.index.map(
                    lambda eid, index=repeat_index: (
                        results.get(eid, {}).get(index) or LLMResult(content="")
                    ).reasoning
                )
    return meta.reset_index()


def main(args: argparse.Namespace) -> None:
    output_model_name = args.output_model_name or args.model
    output_dir = os.path.join(args.output_root, args.task, output_model_name)
    output_filename = f"{args.dataset}_{args.variant}_{args.language}.csv"
    output_path = os.path.join(output_dir, output_filename)
    if args.skip_existing and os.path.exists(output_path):
        print(f"[skip] {output_path} already exists; skipping (resume mode).")
        return

    with open(TASK_PROMPTS[args.task], encoding="utf-8") as f:
        prompts = json.load(f)

    input_path = os.path.join("data", args.dataset, f"{args.variant}.csv")
    df = pd.read_csv(input_path)

    n_total = df[df["kind"] == args.kind]["eid"].nunique() if "kind" in df.columns else df["eid"].nunique()
    n_target = min(args.limit, n_total) if args.limit is not None else n_total
    print(
        f"Processing {n_target}/{n_total} entries for task={args.task} "
        f"with repeats={args.repeats} temperature={args.temperature}..."
    )
    result = generate_sentences(
        df,
        args.model,
        prompts,
        kind=args.kind,
        language=args.language,
        max_workers=args.max_workers,
        token_name=args.token_name,
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
    )

    os.makedirs(output_dir, exist_ok=True)

    print(result.head())
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} sentences to {output_path}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
