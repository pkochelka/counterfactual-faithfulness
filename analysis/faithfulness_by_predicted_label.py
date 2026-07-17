import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from faithfulness_by_classification_split import load_split_rows
from faithfulness_by_variant import MODEL_COLORS, MODEL_MARKERS, stacked_model_legend

_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")
OUTPUT_PATH = Path(__file__).parent / f"faithfulness_by_predicted_label{_SUFFIX}.png"

# X-axis is the label the model itself assigned, not the source variant.
PREDICTED_ORDER = ["FA", "FI", "CFA"]
PREDICTED_TICK_LABELS = {"FA": "fa", "FI": "fi", "CFA": "cfa"}


def plot_faithfulness_by_predicted_label(rows: pd.DataFrame, output_path: Path) -> None:
    models = sorted(rows["model"].unique())
    if len(models) > len(MODEL_COLORS):
        raise ValueError(f"{len(models)} models but only {len(MODEL_COLORS)} palette slots; "
                         "extend MODEL_COLORS/MODEL_MARKERS with validated entries")

    _, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(PREDICTED_ORDER))

    means_by_model = {}
    for model in models:
        means_by_model[model] = (rows[rows["model"] == model]
                                 .groupby("predicted_label")["faithfulness_pct"]
                                 .mean()
                                 .reindex(PREDICTED_ORDER)
                                 / 100)
    for model, color, marker in zip(models, MODEL_COLORS, MODEL_MARKERS):
        ax.plot(positions, means_by_model[model], color=color, marker=marker,
                linewidth=2, markersize=7, label=model)

    ax.set_xticks(positions)
    ax.set_xticklabels([PREDICTED_TICK_LABELS[p] for p in PREDICTED_ORDER])
    y_min = min(means.min() for means in means_by_model.values())
    ax.set_ylim(min(0.67, y_min - 0.02), 1.0)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Mean faithfulness")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    stacked_model_legend(ax, models, MODEL_COLORS, MODEL_MARKERS)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    rows = load_split_rows()
    # Drop rows whose classifier output did not parse to a valid label.
    rows = rows[rows["predicted_label"].isin(PREDICTED_ORDER)]
    summary = (rows
               .groupby(["model", "predicted_label"])
               .agg(mean_faithfulness=("faithfulness_pct", lambda s: s.mean() / 100),
                    n=("faithfulness_pct", "size"))
               .round(3))
    print(summary.to_string())
    plot_faithfulness_by_predicted_label(rows, OUTPUT_PATH)
