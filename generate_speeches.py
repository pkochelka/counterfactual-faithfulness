import json
import os
import queue
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


@dataclass(frozen=True)
class TokenSlot:
    env_var: str


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


def build_prompt(entry_df: pd.DataFrame, prompts: dict, language: str = "en") -> str:
    category = entry_df["category"].iloc[0]
    size = entry_df["size"].iloc[0]

    triples = []
    for _, row in entry_df.iterrows():
        triples.append(f"  - {row['subject']} | {row['predicate']} | {row['object']}")

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
) -> str:
    """Call the API with retry, extract content from response."""
    used_long_retry_sleep = False
    for attempt in range(1, retry_attempts + 1):
        try:
            response = call_api(prompt, model_id, token_name=token_name)
            content = response["choices"][0]["message"]["content"]
            if content:
                return content
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
) -> pd.DataFrame:
    """Generate one sentence per entry."""
    subset = df[df["kind"] == kind] if "kind" in df.columns else df
    grouped = subset.groupby("eid", observed=True)

    tasks = {eid: build_prompt(group, prompts, language) for eid, group in grouped}
    results = {}
    token_slots = build_token_slots(token_name, token_env_vars, concurrency_per_key, max_workers)
    slot_queue: queue.Queue[TokenSlot] = queue.Queue()
    for slot in token_slots:
        slot_queue.put(slot)

    def call_with_slot(prompt: str) -> str:
        slot = slot_queue.get()
        try:
            return call_llm(
                prompt,
                model_id,
                token_name=slot.env_var,
                retry_attempts=retry_attempts,
                retry_sleep=retry_sleep,
                long_retry_sleep=long_retry_sleep,
            )
        finally:
            slot_queue.put(slot)

    with ThreadPoolExecutor(max_workers=len(token_slots)) as pool:
        futures = {
            pool.submit(call_with_slot, prompt): eid
            for eid, prompt in tasks.items()
        }
        for i, future in enumerate(as_completed(futures), 1):
            eid = futures[future]
            results[eid] = future.result()
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} done")

    meta_cols = ["eid", "category", "shape", "shape_type", "size"]
    meta = subset[meta_cols].drop_duplicates("eid").set_index("eid")
    meta["sentence"] = meta.index.map(results)
    return meta.reset_index()


def main(args: argparse.Namespace) -> None:
    with open(TASK_PROMPTS[args.task], encoding="utf-8") as f:
        prompts = json.load(f)

    input_path = os.path.join("data", args.dataset, f"{args.variant}.csv")
    df = pd.read_csv(input_path)

    n = df[df['kind'] == args.kind]['eid'].nunique() if 'kind' in df.columns else df['eid'].nunique()
    print(f"Generating sentences for {n} entries...")
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
    )

    output_dir = os.path.join("data", args.task, args.model)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{args.dataset}_{args.variant}_{args.language}.csv"
    output_path = os.path.join(output_dir, output_filename)

    print(result.head())
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} sentences to {output_path}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
