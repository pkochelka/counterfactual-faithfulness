import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUDGED_DIR = ROOT / "data" / "judged"
FLUENCY_DIR = ROOT / "data" / "judged_fluency"
OUTPUT_DIR = ROOT / "data" / "annotation"

SEED = 140
ROWS_PER_STRATUM = 5
STRATA = ("1", "2-4", "5")

MODEL_PARAMS_B = {
    "qwen3-1_7b": 1.7,
    "tiny-aya-global": 3.35,
    "gemma4-e2b": 5.1,
    "gemma4-e4b": 8.0,
    "qwen3_5-9b": 9.7,
    "llama4-scout-openrouter": 17.0,
    "gemma4-31B": 30.7,
    "gpt-oss-120b": 117.0,
    "qwen3_5-122b": 125.0,
}

BLIND_COLUMNS = [
    "uid", "prompt_language", "triples", "sentence",
    "human_faithfulness_score", "human_faithfulness_reason",
    "human_fluency_score", "human_fluency_reason",
]
KEY_COLUMNS = [
    "uid", "model", "params_b", "size_bucket", "dataset", "variant",
    "prompt_language", "eid", "faithfulness_stratum",
    "triples", "sentence",
    "judge_faithfulness_score", "judge_incorrect_information",
    "judge_fluency_score", "judge_fluency_comment",
]


def size_bucket(params_b):
    if params_b < 10:
        return "small"
    if params_b < 100:
        return "medium"
    return "large"


def faithfulness_stratum(score):
    return {1: "1", 5: "5"}.get(score, "2-4")


def load_judgments(directory):
    judgments = {}
    for model_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            if jsonl_path.name.endswith(".failures.jsonl"):
                continue
            _, dataset, variant, language, *_ = jsonl_path.stem.split("_")
            with jsonl_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (model_dir.name, dataset, variant, language, record["eid"])
                    judgments[key] = record
    return judgments


def build_rows():
    fluency_judgments = load_judgments(FLUENCY_DIR)
    rows = []
    for key, judged in load_judgments(JUDGED_DIR).items():
        model, dataset, variant, language, eid = key
        faithfulness = judged.get("parsed", {}).get("faithfulness_score")
        fluency = fluency_judgments.get(key, {}).get("parsed", {})
        if faithfulness is None or fluency.get("fluency_score") is None:
            continue
        rows.append({
            "uid": f"{model}__{dataset}_{variant}_{language}__{eid}",
            "model": model,
            "params_b": MODEL_PARAMS_B[model],
            "size_bucket": size_bucket(MODEL_PARAMS_B[model]),
            "dataset": dataset,
            "variant": variant,
            "prompt_language": language,
            "eid": eid,
            "faithfulness_stratum": faithfulness_stratum(faithfulness),
            "triples": judged["modified_triples"],
            "sentence": judged["sentence"],
            "judge_faithfulness_score": faithfulness,
            "judge_incorrect_information": json.dumps(
                judged["parsed"].get("incorrect_information", []), ensure_ascii=False),
            "judge_fluency_score": fluency["fluency_score"],
            "judge_fluency_comment": fluency.get("fluency_comment", ""),
        })
    return rows


def sample_split(candidates, quota, fluency_counts, rng):
    pool = candidates[:]
    rng.shuffle(pool)
    split_counts = Counter()
    picked = []
    while pool and len(picked) < quota:
        best = min(pool, key=lambda row: (
            split_counts["stratum", row["faithfulness_stratum"]],
            split_counts["model", row["model"]],
            split_counts["dataset", row["dataset"]],
            fluency_counts[row["judge_fluency_score"]],
        ))
        pool.remove(best)
        picked.append(best)
        split_counts["stratum", best["faithfulness_stratum"]] += 1
        split_counts["model", best["model"]] += 1
        split_counts["dataset", best["dataset"]] += 1
        fluency_counts[best["judge_fluency_score"]] += 1
    return picked


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    rng = random.Random(SEED)
    splits = {}
    for row in build_rows():
        split_key = (row["variant"], row["size_bucket"], row["prompt_language"])
        splits.setdefault(split_key, []).append(row)

    fluency_counts = Counter()
    sample = []
    for split_key in sorted(splits):
        quota = ROWS_PER_STRATUM * len(STRATA)
        sample.extend(sample_split(splits[split_key], quota, fluency_counts, rng))
    rng.shuffle(sample)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "annotation_sample.csv", BLIND_COLUMNS, sample)
    write_csv(OUTPUT_DIR / "annotation_key.csv", KEY_COLUMNS, sample)

    cell_counts = Counter(
        (row["variant"], row["size_bucket"], row["prompt_language"], row["faithfulness_stratum"])
        for row in sample)
    print("\nRows per (variant, size_bucket, prompt_language, faithfulness_stratum):")
    for (variant, bucket, language, stratum), count in sorted(cell_counts.items()):
        print(f"  {variant:<3} {bucket:<7} {language:<4} stratum {stratum:>3}: {count}")

    for field in ("model", "dataset", "prompt_language", "judge_fluency_score"):
        counts = Counter(row[field] for row in sample)
        print(f"\n{field} counts:")
        for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
            print(f"  {value}: {count}")


if __name__ == "__main__":
    main()
