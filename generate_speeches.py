import json
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

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
parser.add_argument("--task", default="generated", choices=list(TASK_PROMPTS), type=str, help="Task type: determines output folder and prompt file.")


def build_prompt(entry_df: pd.DataFrame, prompts: dict, language: str = "en") -> str:
    category = entry_df["category"].iloc[0]
    size = entry_df["size"].iloc[0]

    triples = []
    for _, row in entry_df.iterrows():
        triples.append(f"  - {row['subject']} | {row['predicate']} | {row['object']}")

    triples_str = "\n".join(triples)

    template = prompts[language]
    return template.format(size=size, category=category, triples_str=triples_str)


def call_llm(prompt: str, model_id: str, token_name: str = "") -> str:
    """Call the API with retry, extract content from response."""
    while True:
        try:
            response = call_api(prompt, model_id, token_name=token_name)
            content = response["choices"][0]["message"]["content"]
            if content:
                return content
        except Exception as e:
            print(f"Error: {e}. Retrying...", flush=True)


def generate_sentences(
    df: pd.DataFrame,
    model_id: str,
    prompts: dict,
    kind: str = "modified",
    language: str = "en",
    max_workers: int = 4,
    token_name: str = "",
) -> pd.DataFrame:
    """Generate one sentence per entry."""
    subset = df[df["kind"] == kind] if "kind" in df.columns else df
    grouped = subset.groupby("eid", observed=True)

    tasks = {eid: build_prompt(group, prompts, language) for eid, group in grouped}
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(call_llm, prompt, model_id, token_name): eid
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
    result = generate_sentences(df, args.model, prompts, kind=args.kind, language=args.language, max_workers=4, token_name=args.token_name)

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