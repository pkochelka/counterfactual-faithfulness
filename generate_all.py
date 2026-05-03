import os
import subprocess
import sys

DATASETS = ["webnlg", "cus-qa"]
VARIANTS = ["cf", "fa", "fi"]
LANGUAGES = ["en", "cs", "sk"]
MODELS = ["gpt-oss-120b", "qwen3.5-122b"]

DATASET_CONFIG = {
    "cus-qa": {
        "excluded_variants": {"fi"},
        "extra_args": ["--kind", "original"],
    },
}

total = 0
skipped = 0
failed = 0

for model in MODELS:
    for dataset in DATASETS:
        config = DATASET_CONFIG.get(dataset, {})
        excluded_variants = config.get("excluded_variants", set())
        extra_args = config.get("extra_args", [])

        for variant in VARIANTS:
            if variant in excluded_variants:
                continue

            for language in LANGUAGES:
                output_path = os.path.join(
                    "data", "generated", model,
                    f"{dataset}_{variant}_{language}.csv"
                )
                if os.path.exists(output_path):
                    print(f"[skip] {output_path}")
                    skipped += 1
                    total += 1
                    continue

                cmd = [
                    sys.executable, "generate_speeches.py",
                    "--model", model,
                    "--dataset", dataset,
                    "--variant", variant,
                    "--language", language,
                    *extra_args,
                ]
                label = f"{model}/{dataset}_{variant}_{language}"
                print(f"[run]  {label}")
                result = subprocess.run(cmd)
                total += 1
                if result.returncode != 0:
                    print(f"[FAIL] {label} exited with code {result.returncode}")
                    failed += 1

print(f"\nDone: {total} total, {skipped} skipped, {failed} failed.")