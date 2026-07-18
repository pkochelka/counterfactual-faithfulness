import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from judged_io import iter_judged_records

_DEFAULT_JUDGED_DIR = Path(__file__).parent.parent / "data" / "judged"
JUDGED_DIR = Path(os.environ.get("JUDGED_DIR", _DEFAULT_JUDGED_DIR))
_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")
OUTPUT_PATH = Path(__file__).parent / f"faithfulness_by_language{_SUFFIX}.png"

DEFAULT_LANGUAGE_ORDER = ["en", "cs", "sk", "hsb"]
LANGUAGE_ORDER = [
    lang
    for lang in os.environ.get("LANGUAGES", " ".join(DEFAULT_LANGUAGE_ORDER)).replace(",", " ").split()
]

# x-axis groups: EN and HSB are language-agnostic (same for every dataset), while
# "Same"/"Other" resolve relative to each dataset's own target language (cs-qa's
# target is cs, sk-qa's is sk) so a cs-qa panel and an sk-qa panel are comparable.
GROUP_ORDER = ["EN", "Same", "Other", "HSB"]


def language_group(dataset: str, language: str) -> str:
    if language == "en":
        return "EN"
    if language == "hsb":
        return "HSB"
    target_language = dataset.split("-")[0]  # "cs-qa" -> "cs", "sk-qa" -> "sk"
    return "Same" if language == target_language else "Other"


VARIANT_ORDER = ["fa", "cf", "fi"]
VARIANT_STYLE = {
    "fa": {"color": "#2ca02c", "linestyle": "-",  "label": "fa  (factual — upper bound)"},
    "cf": {"color": "#ff7f0e", "linestyle": "--", "label": "cf  (counterfactual)"},
    "fi": {"color": "#d62728", "linestyle": ":",  "label": "fi  (fictional)"},
}

MAX_SCORE = 5

# CVD-validated categorical palette (worst adjacent-pair ΔE 13.3); models are
# assigned slots in fixed sorted order, paired with a distinct marker each so
# identity never rests on color alone. This is the shared source of truth —
# faithfulness_by_variant.py and faithfulness_by_predicted_label.py import it
# from here so a given model gets the same color/marker in every plot.
MODEL_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#0891b2", "#e34948",
                "#008300", "#4a3aa7", "#e87ba4", "#eb6834"]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<"]


def load_faithfulness_records(judged_dir: Path) -> pd.DataFrame:
    records = []
    for r in iter_judged_records(judged_dir):
        score = r.record.get("parsed", {}).get("faithfulness_score")
        if score is None:
            continue
        records.append(dict(model=r.model, dataset=r.dataset,
                            variant=r.variant, language=r.language,
                            language_group=language_group(r.dataset, r.language),
                            faithfulness_score=score,
                            faithfulness_pct=score / MAX_SCORE * 100))
    return pd.DataFrame(records)


def plot_faithfulness(results: pd.DataFrame, output_path: Path) -> None:
    models = sorted(results["model"].unique())
    if len(models) > len(MODEL_COLORS):
        raise ValueError(f"{len(models)} models but only {len(MODEL_COLORS)} palette slots; "
                         "extend MODEL_COLORS/MODEL_MARKERS with validated entries")

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Data-driven lower bound so lower-scoring groups (e.g. HSB) are not
    # clipped by a fixed near-5 window. Floor to the nearest 0.5 below the
    # minimum observed group mean, but never show less than down to 3.0.
    # language_group already resolves "Same"/"Other" per dataset, so cs-qa
    # and sk-qa rows are pooled together for a given model/group.
    means_by_group = (results
                      .groupby(["model", "language_group"])["faithfulness_score"]
                      .mean())
    lo = means_by_group.min() if not means_by_group.empty else 4.5
    y_lower = min(3.0, (lo * 2 // 1) / 2)

    for model, color, marker in zip(models, MODEL_COLORS, MODEL_MARKERS):
        means = (results[results["model"] == model]
                 .groupby("language_group")["faithfulness_score"]
                 .mean()
                 .reindex(GROUP_ORDER))
        ax.plot(GROUP_ORDER, means, marker=marker, color=color,
                linewidth=2, markersize=7, label=model)

    ax.set_ylim(y_lower, 5.15)
    ax.axhline(5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Prompt language (relative to dataset)")
    ax.set_ylabel("Mean faithfulness score (1-5)")

    handles = [Line2D([0], [0], color=color, marker=marker, linewidth=2, markersize=7)
               for color, marker in zip(MODEL_COLORS[:len(models)], MODEL_MARKERS[:len(models)])]
    fig.legend(handles, models, loc="lower center", ncol=min(5, len(models)),
               fontsize=9, bbox_to_anchor=(0.5, -0.1))

    fig.suptitle(
        "Mean faithfulness score by prompt language, aggregated across cs-qa/sk-qa and variants (fa/cf/fi)\n"
        "Same = prompt language matches the dataset's target language, Other = the other target language",
        fontsize=12,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    results = load_faithfulness_records(JUDGED_DIR)
    summary = (results
               .groupby(["model", "language_group"])["faithfulness_score"]
               .mean()
               .round(2))
    print(summary.to_string())
    plot_faithfulness(results, OUTPUT_PATH)
