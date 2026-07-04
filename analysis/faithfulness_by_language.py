import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_DEFAULT_JUDGED_DIR = Path(__file__).parent.parent / "data" / "judged"
JUDGED_DIR = Path(os.environ.get("JUDGED_DIR", _DEFAULT_JUDGED_DIR))
_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")
OUTPUT_PATH = Path(__file__).parent / f"faithfulness_by_language{_SUFFIX}.png"

DEFAULT_LANGUAGE_ORDER = ["en", "cs", "sk", "hsb"]
LANGUAGE_ORDER = [
    lang
    for lang in os.environ.get("LANGUAGES", " ".join(DEFAULT_LANGUAGE_ORDER)).replace(",", " ").split()
]
VARIANT_ORDER = ["fa", "cf", "fi"]
VARIANT_STYLE = {
    "fa": {"color": "#2ca02c", "linestyle": "-",  "label": "fa  (factual — upper bound)"},
    "cf": {"color": "#ff7f0e", "linestyle": "--", "label": "cf  (counterfactual)"},
    "fi": {"color": "#d62728", "linestyle": ":",  "label": "fi  (fictional)"},
}

MAX_SCORE = 5


def load_faithfulness_records(judged_dir: Path) -> pd.DataFrame:
    records = []
    for model_dir in sorted(judged_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            if jsonl_path.name.endswith(".failures.jsonl"):
                continue
            _, dataset, variant, language, *_ = jsonl_path.stem.split("_")
            with jsonl_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    score = json.loads(line).get("parsed", {}).get("faithfulness_score")
                    if score is None:
                        continue
                    records.append(dict(model=model_dir.name, dataset=dataset,
                                        variant=variant, language=language,
                                        faithfulness_pct=score / MAX_SCORE * 100))
    return pd.DataFrame(records)


def plot_faithfulness(results: pd.DataFrame, output_path: Path) -> None:
    models = sorted(results["model"].unique())
    datasets = sorted(results["dataset"].unique())

    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(5 * len(datasets), 4 * len(models)),
        sharey=True, sharex=True,
        squeeze=False,
    )

    # Data-driven lower bound so lower-scoring languages (e.g. hsb) are not
    # clipped by a fixed near-100 window. Floor to the nearest 5 below the
    # minimum observed group mean, but never show more than down to 90.
    means_by_group = (results
                      .groupby(["model", "dataset", "variant", "language"])["faithfulness_pct"]
                      .mean())
    lo = means_by_group.min() if not means_by_group.empty else 95.0
    y_lower = min(90.0, (lo // 5) * 5)

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            subset = results[(results["model"] == model) & (results["dataset"] == dataset)]
            present_variants = subset["variant"].unique()

            for variant in [v for v in VARIANT_ORDER if v in present_variants]:
                means = (subset[subset["variant"] == variant]
                         .groupby("language")["faithfulness_pct"]
                         .mean()
                         .reindex(LANGUAGE_ORDER))
                ax.plot(LANGUAGE_ORDER, means, marker="o", linewidth=2,
                        **VARIANT_STYLE[variant])

            ax.set_ylim(y_lower, 101)
            ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_title(f"{model} / {dataset}", fontsize=10)
            ax.set_xlabel("Prompt language" if row == len(models) - 1 else "")
            ax.set_ylabel("Mean faithfulness (%)" if col == 0 else "")

    handles = [plt.Line2D([0], [0], marker="o", linewidth=2, **s)
               for s in VARIANT_STYLE.values()]
    fig.legend(handles, [s["label"] for s in VARIANT_STYLE.values()],
               loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Mean faithfulness score by prompt language and variant\n"
        "Hypothesis: cf / fi faithfulness lowest in English"
        " — model trained on English data resists counterfactual prompts",
        fontsize=12,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    results = load_faithfulness_records(JUDGED_DIR)
    summary = (results
               .groupby(["model", "dataset", "variant", "language"])["faithfulness_pct"]
               .mean()
               .round(1))
    print(summary.to_string())
    plot_faithfulness(results, OUTPUT_PATH)
