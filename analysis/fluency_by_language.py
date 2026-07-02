"""Plot mean fluency (linguistic quality) score by prompt language and variant.

Fluency is the second LLM-judge dimension (see prompts/judge_fluency.txt): how well
a sentence is written in its target language, independent of faithfulness. Records
live under judged_fluency_v2/<model>/judge_<dataset>_<variant>_<language>_<judge>.jsonl
with parsed.fluency_score in 1..5.

Configuration mirrors analysis/faithfulness_by_language.py:
  JUDGED_DIR     override the input directory (default data/judged_fluency)
  OUTPUT_SUFFIX  appended to the output filename (e.g. "_v2")
  LANGUAGES      plotted languages, comma/space separated (default en,cs,sk,hsb)
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_DEFAULT_JUDGED_DIR = Path(__file__).parent.parent / "data" / "judged_fluency"
JUDGED_DIR = Path(os.environ.get("JUDGED_DIR", _DEFAULT_JUDGED_DIR))
_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")
OUTPUT_PATH = Path(__file__).parent / f"fluency_by_language{_SUFFIX}.png"

DEFAULT_LANGUAGE_ORDER = ["en", "cs", "sk", "hsb"]
LANGUAGE_ORDER = [
    lang
    for lang in os.environ.get("LANGUAGES", " ".join(DEFAULT_LANGUAGE_ORDER)).replace(",", " ").split()
]
VARIANT_ORDER = ["fa", "cf", "fi"]
VARIANT_STYLE = {
    "fa": {"color": "#2ca02c", "linestyle": "-",  "label": "fa  (factual)"},
    "cf": {"color": "#ff7f0e", "linestyle": "--", "label": "cf  (counterfactual)"},
    "fi": {"color": "#d62728", "linestyle": ":",  "label": "fi  (fictional)"},
}

MAX_SCORE = 5


def load_fluency_records(judged_dir: Path) -> pd.DataFrame:
    records = []
    for model_dir in sorted(judged_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            # Skip judge diagnostic sidecars (error records, no parsed score).
            if jsonl_path.name.endswith(".failures.jsonl"):
                continue
            _, dataset, variant, language, *_ = jsonl_path.stem.split("_")
            with jsonl_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    score = json.loads(line).get("parsed", {}).get("fluency_score")
                    if score is None:
                        continue
                    records.append(dict(model=model_dir.name, dataset=dataset,
                                        variant=variant, language=language,
                                        fluency_pct=score / MAX_SCORE * 100))
    return pd.DataFrame(records)


def plot_fluency(results: pd.DataFrame, output_path: Path) -> None:
    models = sorted(results["model"].unique())
    datasets = sorted(results["dataset"].unique())

    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(5 * len(datasets), 4 * len(models)),
        sharey=True, sharex=True,
        squeeze=False,
    )

    # Data-driven lower bound: fluency spans a wider range than faithfulness, so
    # a fixed near-100 window would clip low-resource languages. Floor to the
    # nearest 5 below the minimum observed mean (never above 90).
    means_by_group = (results
                      .groupby(["model", "dataset", "variant", "language"])["fluency_pct"]
                      .mean())
    lo = means_by_group.min() if not means_by_group.empty else 0.0
    y_lower = min(90.0, (lo // 5) * 5)

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            subset = results[(results["model"] == model) & (results["dataset"] == dataset)]
            present_variants = subset["variant"].unique()

            for variant in [v for v in VARIANT_ORDER if v in present_variants]:
                means = (subset[subset["variant"] == variant]
                         .groupby("language")["fluency_pct"]
                         .mean()
                         .reindex(LANGUAGE_ORDER))
                # reindex leaves NaN for languages this model wasn't judged on;
                # matplotlib renders those as gaps (no marker), so a model
                # missing hsb shows nothing at hsb instead of distorting the line.
                ax.plot(LANGUAGE_ORDER, means, marker="o", linewidth=2,
                        **VARIANT_STYLE[variant])

            ax.set_ylim(y_lower, 101)
            ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_title(f"{model} / {dataset}", fontsize=10)
            ax.set_xlabel("Prompt language" if row == len(models) - 1 else "")
            ax.set_ylabel("Mean fluency (%)" if col == 0 else "")

    handles = [plt.Line2D([0], [0], marker="o", linewidth=2, **s)
               for s in VARIANT_STYLE.values()]
    fig.legend(handles, [s["label"] for s in VARIANT_STYLE.values()],
               loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Mean fluency (linguistic quality) score by prompt language and variant",
        fontsize=12,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    results = load_fluency_records(JUDGED_DIR)
    if results.empty:
        raise SystemExit(f"No fluency records found under {JUDGED_DIR}")
    summary = (results
               .groupby(["model", "dataset", "variant", "language"])["fluency_pct"]
               .mean()
               .round(1))
    print(summary.to_string())
    plot_fluency(results, OUTPUT_PATH)
