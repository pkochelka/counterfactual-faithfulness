"""Rejudge the annotation_sample_old examples with the current judge prompt.

Steps (run "all" for the full pipeline, or each step separately):
  prepare  filter data/generated to the sampled eids under data/rejudge_annotation/
  judge    run llm-judge/judge_csv.py on every prepared CSV (API keys from .env.local)
  collect  match judgments to annotation_sample_old and write annotation_key_rejudged.csv

The judge step resumes by default (already-judged eids for the same judge model and
endpoint are skipped); pass --fresh after updating the judge prompt so every example
is actually rescored.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm-judge"))

from sample_for_annotation import MODEL_PARAMS_B, faithfulness_stratum, load_judgments, size_bucket

ANNOTATION_DIR = ROOT / "data" / "annotation"
GENERATED_DIR = ROOT / "data" / "generated"
SOURCE_CSV_DIR = ROOT / "data" / "source_csv"
REJUDGE_DIR = ROOT / "data" / "rejudge_annotation"
JUDGE_SCRIPT = ROOT / "llm-judge" / "judge_csv.py"

KEY_COLUMNS = [
    "uid", "model", "params_b", "size_bucket", "dataset", "variant",
    "prompt_language", "eid", "faithfulness_stratum",
    "triples", "sentence",
    "judge_faithfulness_score", "judge_incorrect_information",
]


def normalize(text):
    return (text or "").replace("\r\n", "\n").strip()


def load_sample_rows():
    with (ANNOTATION_DIR / "annotation_sample_old.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        model, combo, eid = row["uid"].split("__")
        dataset, variant, language = combo.split("_")
        row.update(model=model, dataset=dataset, variant=variant, language=language, eid=eid)
    return rows


def prepare():
    from webnlg_utils import sanitize_identifier

    generated_dirs = {
        sanitize_identifier(path.name): path
        for path in GENERATED_DIR.iterdir() if path.is_dir()
    }
    for name, path in list(generated_dirs.items()):
        generated_dirs.setdefault(name.removesuffix("-openrouter"), path)
    rows = load_sample_rows()
    groups = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["variant"], row["language"])
        groups.setdefault(key, set()).add(row["eid"])

    for dataset in ("cs-qa", "sk-qa"):
        (REJUDGE_DIR / dataset).mkdir(parents=True, exist_ok=True)
        for variant_file in ("cf.csv", "fa.csv", "fi.csv"):
            data = (SOURCE_CSV_DIR / dataset / variant_file).read_bytes()
            (REJUDGE_DIR / dataset / variant_file).write_bytes(data)

    total = 0
    for (model, dataset, variant, language), eids in sorted(groups.items()):
        source_path = generated_dirs[model] / f"{dataset}_{variant}_{language}.csv"
        with source_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            selected = [row for row in reader if row["eid"] in eids]
        missing = eids - {row["eid"] for row in selected}
        if missing:
            print(f"MISSING in {source_path.name} ({model}): {sorted(missing)}")
        out_path = REJUDGE_DIR / "generated" / model / source_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
        total += len(selected)
    print(f"Wrote {total} rows across {len(groups)} filtered CSVs under {REJUDGE_DIR / 'generated'}")
    verify(rows)


def verify(sample_rows):
    from webnlg_utils import enrich_sentences

    by_key = {
        (row["model"], row["dataset"], row["variant"], row["language"], row["eid"]): row
        for row in sample_rows
    }
    checked, mismatches = 0, 0
    for csv_path in sorted((REJUDGE_DIR / "generated").glob("*/*.csv")):
        model = csv_path.parent.name
        dataset, variant, language = csv_path.stem.split("_")
        for _, row in enrich_sentences(csv_path).iterrows():
            sample_row = by_key.get((model, dataset, variant, language, str(row["eid"])))
            if sample_row is None:
                continue
            checked += 1
            same = (normalize(row["modified_triples"]) == normalize(sample_row["triples"])
                    and normalize(row["sentence"]) == normalize(sample_row["sentence"]))
            if not same:
                mismatches += 1
                print("MISMATCH:", sample_row["uid"])
    print(f"Verified {checked}/{len(sample_rows)} rows against annotation_sample_old, mismatches: {mismatches}")


def judge(args):
    judged_dir = REJUDGE_DIR / (f"judged_{args.tag}" if args.tag else "judged")
    if judged_dir.exists():
        if args.fresh:
            shutil.rmtree(judged_dir)
        elif any(judged_dir.rglob("*.jsonl")):
            print(f"Resuming into {judged_dir}: eids already judged by {args.judge_model} "
                  "at this endpoint are skipped (use --fresh to rescore everything).")
    csv_paths = sorted((REJUDGE_DIR / "generated").glob("*/*.csv"))
    if not csv_paths:
        sys.exit(f"No prepared CSVs under {REJUDGE_DIR / 'generated'}; run the prepare step first.")
    failed = []
    for index, csv_path in enumerate(csv_paths, 1):
        print(f"[{index}/{len(csv_paths)}] judging {csv_path.relative_to(ROOT)}", flush=True)
        command = [
            sys.executable, str(JUDGE_SCRIPT), str(csv_path),
            "--sample-size", "all",
            "--model", args.judge_model,
            "--judge-base-url", args.judge_base_url,
            "--token-env-vars", args.token_env_vars,
            "--concurrency-per-key", str(args.concurrency_per_key),
            "--max-tokens", str(args.max_tokens),
            "--output-dir", str(judged_dir),
            "--allow-failures",
        ]
        if args.reasoning_effort:
            command += ["--reasoning-effort", args.reasoning_effort]
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            failed.append(csv_path.name)
    if failed:
        print(f"judge_csv.py failed for {len(failed)} files: {', '.join(failed)}", file=sys.stderr)


def collect(tag=None):
    judged_dir = REJUDGE_DIR / (f"judged_{tag}" if tag else "judged")
    sample_rows = load_sample_rows()
    judgments = load_judgments(judged_dir)
    key_rows, missing = [], []
    for row in sample_rows:
        record = judgments.get((row["model"], row["dataset"], row["variant"], row["language"], row["eid"]))
        parsed = (record or {}).get("parsed") or {}
        score = parsed.get("faithfulness_score")
        if score is None:
            missing.append(row["uid"])
            continue
        key_rows.append({
            "uid": row["uid"],
            "model": row["model"],
            "params_b": MODEL_PARAMS_B[row["model"]],
            "size_bucket": size_bucket(MODEL_PARAMS_B[row["model"]]),
            "dataset": row["dataset"],
            "variant": row["variant"],
            "prompt_language": row["language"],
            "eid": row["eid"],
            "faithfulness_stratum": faithfulness_stratum(score),
            "triples": record["modified_triples"],
            "sentence": record["sentence"],
            "judge_faithfulness_score": score,
            "judge_incorrect_information": json.dumps(parsed.get("incorrect_information", []), ensure_ascii=False),
        })
    output_path = ANNOTATION_DIR / (f"annotation_key_rejudged_{tag}.csv" if tag else "annotation_key_rejudged.csv")
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"Wrote {len(key_rows)} rows to {output_path}")
    if missing:
        print(f"Missing judgments for {len(missing)} uids:")
        for uid in missing:
            print(" ", uid)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "judge", "collect", "all"])
    parser.add_argument("--tag", default=None,
                        help="suffix for the judged dir and key CSV: judged_<tag>, annotation_key_rejudged_<tag>.csv")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-base-url", default="https://llm.ai.e-infra.cz/v1")
    parser.add_argument("--token-env-vars", default="KEY1",
                        help="comma-separated env-var names holding judge API keys (loaded from .env.local)")
    parser.add_argument("--concurrency-per-key", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["max", "xhigh", "high", "medium", "low", "minimal", "none"],
                        help="when omitted, judge_csv.py sends the legacy thinking=true payload")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the judged dir first so the updated judge rescores every example")
    args = parser.parse_args()
    if args.step in ("prepare", "all"):
        prepare()
    if args.step in ("judge", "all"):
        judge(args)
    if args.step in ("collect", "all"):
        collect(args.tag)


if __name__ == "__main__":
    main()