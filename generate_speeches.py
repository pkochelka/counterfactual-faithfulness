from __future__ import annotations

import json
import os
import queue
import random
import re
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections.abc import Callable
from dataclasses import dataclass

from api_caller import NO_AUTH_TOKEN_NAME, call_api

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
parser.add_argument("--token-env-vars", default="", type=str, help="Comma-separated API token env var names.")
parser.add_argument("--concurrency-per-key", default=4, type=int, help="Concurrent requests per token env var.")
parser.add_argument("--retry-attempts", default=3, type=int, help="Maximum attempts per request.")
parser.add_argument("--retry-sleep", default=2.0, type=float, help="Short sleep between retry attempts.")
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
parser.add_argument("--disable-thinking", action="store_true", help="Ask compatible local/vLLM reasoning models to disable thinking traces.")


@dataclass(frozen=True)
class TokenSlot:
    env_var: str


@dataclass(frozen=True)
class LLMResult:
    content: str
    reasoning: str | None = None
    raw_content: str | None = None


_LABEL_RE = re.compile(r"\b(CFA|FA|FI)\b")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s*)+")
_CODE_FENCE_RE = re.compile(r"^\s*```")
_META_LINE_RE = re.compile(
    r"^\s*(?:"
    r"len(?:\s+vety?|\s+vetu/vety)?|"
    r"length|lengva|"
    r"vet[auy]|"
    r"d(?:ĺ|l)žka(?:\s+vety)?|"
    r"lo(?:ž|z)ka"
    r")\s*[:：]",
    re.IGNORECASE,
)
_TRAILING_META_RE = re.compile(
    r"(?:"
    r"\s*\((?:len(?:\s+vety?|\s+vetu/vety)?|length|lengva|vet[auy]|d(?:ĺ|l)žka|lo(?:ž|z)ka)[^)]*\)"
    r"|"
    r"\s+(?:len(?:\s+vety?|\s+vetu/vety)?|length|lengva|vet[auy]|d(?:ĺ|l)žka|lo(?:ž|z)ka)\s*[:：].*$"
    r")",
    re.IGNORECASE,
)


def sanitize_classification_content(content: str) -> str:
    """Reduce a classification response to a single-line value.

    The classify prompt asks for only an FA/CFA/FI label, but some models append
    multi-line reasoning, which spills a single entry across many physical CSV
    lines. Extract the first FA/CFA/FI label when present; otherwise collapse all
    whitespace so the raw response still stays on one line for inspection.
    """
    if not content:
        return content
    match = _LABEL_RE.search(content)
    if match:
        return match.group(1)
    return " ".join(content.split())


def strip_wrapping_quotes(content: str) -> str:
    stripped = content.strip()
    quote_pairs = [('"', '"'), ("'", "'"), ("`", "`"), ("“", "”"), ("„", "“")]
    changed = True
    while changed and len(stripped) >= 2:
        changed = False
        for start, end in quote_pairs:
            if stripped.startswith(start) and stripped.endswith(end):
                stripped = stripped[len(start):-len(end)].strip()
                changed = True
                break
    return stripped


def sanitize_generated_content(content: str) -> str:
    """Keep generated text in one physical CSV row and trim common model chatter."""
    if not content:
        return content
    lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or _CODE_FENCE_RE.match(line) or _META_LINE_RE.match(line):
            continue
        line = _LIST_MARKER_RE.sub("", line).strip()
        if line:
            lines.append(line)
    text = strip_wrapping_quotes(" ".join(lines))
    previous = None
    while previous != text:
        previous = text
        text = _TRAILING_META_RE.sub("", text).strip()
        text = strip_wrapping_quotes(text)
    return " ".join(text.split())


def collapse_response_text(content: str | None) -> str:
    if not content:
        return ""
    return " ".join(content.split())


def split_inline_reasoning(content: str) -> tuple[str, str | None]:
    """Remove inline <think> traces from content while preserving them separately."""
    reasoning_parts = [match.strip() for match in _THINK_RE.findall(content) if match.strip()]
    cleaned = _THINK_RE.sub("", content).strip()
    return cleaned, "\n\n".join(reasoning_parts) if reasoning_parts else None


def merge_reasoning(*parts: str | None) -> str | None:
    merged = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return "\n\n".join(merged) if merged else None


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_token_slots(token_env_vars: str, concurrency_per_key: int) -> list[TokenSlot]:
    if concurrency_per_key <= 0:
        raise ValueError("--concurrency-per-key must be positive")
    names = parse_csv_list(token_env_vars)
    names = list(dict.fromkeys(names))
    if not names:
        return [TokenSlot(env_var=NO_AUTH_TOKEN_NAME) for _ in range(concurrency_per_key)]

    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing token environment variable(s): {', '.join(missing)}")
    return [TokenSlot(env_var=name) for name in names for _ in range(concurrency_per_key)]


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
    disable_thinking: bool = False,
) -> LLMResult:
    """Call the API with retry, extract content from response."""
    used_long_retry_sleep = False
    if request_jitter_min < 0 or request_jitter_max < 0 or request_jitter_min > request_jitter_max:
        raise ValueError("request jitter bounds must be non-negative and min <= max")
    for attempt in range(1, retry_attempts + 1):
        try:
            if request_jitter_max > 0:
                time.sleep(random.uniform(request_jitter_min, request_jitter_max))
            response = call_api(
                prompt,
                model_id,
                token_name=token_name,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                disable_thinking=disable_thinking,
            )
            message = response["choices"][0]["message"]
            content = message["content"]
            if content:
                cleaned_content, inline_reasoning = split_inline_reasoning(content)
                return LLMResult(
                    content=cleaned_content,
                    reasoning=merge_reasoning(extract_reasoning(message), inline_reasoning),
                    raw_content=content,
                )
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
    token_env_vars: str = "",
    concurrency_per_key: int = 4,
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
    content_transform: Callable[[str], str] | None = None,
    include_raw_content: bool = False,
    disable_thinking: bool = False,
) -> pd.DataFrame:
    """Generate one sentence per entry."""
    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    transform = content_transform or (lambda content: content)

    subset = df[df["kind"] == kind] if "kind" in df.columns else df
    grouped = subset.groupby("eid", observed=True)

    tasks = {str(eid): build_prompt(group, prompts, language) for eid, group in grouped}
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        tasks = dict(list(tasks.items())[:limit])
        subset = subset[subset["eid"].isin(tasks)]
    results: dict[str, dict[int, LLMResult]] = {}
    token_slots = build_token_slots(token_env_vars, concurrency_per_key)
    slot_queue: queue.Queue[TokenSlot] = queue.Queue()
    for slot in token_slots:
        slot_queue.put(slot)

    def call_with_slot(prompt: str) -> LLMResult:
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
                disable_thinking=disable_thinking,
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
        lambda eid: transform((results.get(eid, {}).get(1) or LLMResult(content="")).content)
    )
    if repeats > 1:
        for repeat_index in range(1, repeats + 1):
            meta[f"sentence_{repeat_index}"] = meta.index.map(
                lambda eid, index=repeat_index: transform(
                    (results.get(eid, {}).get(index) or LLMResult(content="")).content
                )
            )
    if include_raw_content:
        meta["raw_response"] = meta.index.map(
            lambda eid: collapse_response_text(
                (results.get(eid, {}).get(1) or LLMResult(content="")).raw_content
            )
        )
        if repeats > 1:
            for repeat_index in range(1, repeats + 1):
                meta[f"raw_response_{repeat_index}"] = meta.index.map(
                    lambda eid, index=repeat_index: collapse_response_text(
                        (results.get(eid, {}).get(index) or LLMResult(content="")).raw_content
                    )
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
        content_transform=(
            sanitize_classification_content if args.task == "classified" else sanitize_generated_content
        ),
        include_raw_content=args.task == "classified",
        disable_thinking=args.disable_thinking,
    )

    os.makedirs(output_dir, exist_ok=True)

    print(result.head())
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} sentences to {output_path}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
