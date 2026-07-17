import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path

from faithfulness_by_language import (JUDGED_DIR, MODEL_COLORS, MODEL_MARKERS,
                                       load_faithfulness_records)

_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")
OUTPUT_PATH = Path(__file__).parent / f"faithfulness_by_variant{_SUFFIX}.png"

# Data files use "cf" for the counterfactual variant; the plot shows it as "cfa"
# to match the classifier labels (FA / FI / CFA).
VARIANT_ORDER = ["fa", "fi", "cf"]
VARIANT_TICK_LABELS = {"fa": "fa", "fi": "fi", "cf": "cfa"}


def stacked_model_legend(ax, names, colors, markers, *, fontsize=12,
                         ncol=5, y_offset=-0.15):
    """Compact legend below the axes: each model name centered under its symbol.

    Matplotlib fills legend columns top-to-bottom, so interleaving each symbol
    handle with an invisible handle carrying the name stacks them vertically.
    """
    handles, labels = [], []
    for name, color, marker in zip(names, colors, markers):
        handles.append(Line2D([0], [0], color=color, marker=marker,
                              linewidth=2, markersize=7))
        labels.append("")
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(name)
    # Pad to a multiple of ncol so column splitting never separates a
    # symbol/name pair (matplotlib fills columns as evenly as possible).
    models_per_col = -(-len(names) // ncol)
    for _ in range(ncol * models_per_col - len(names)):
        handles.extend([Line2D([], [], linestyle="none")] * 2)
        labels.extend(["", ""])
    legend = ax.legend(handles, labels, ncol=ncol, loc="upper center",
                       bbox_to_anchor=(0.5, y_offset), frameon=False,
                       fontsize=fontsize, handletextpad=0, columnspacing=1.4,
                       labelspacing=0.4, borderpad=0)
    # Zero out the invisible handle's box on name rows and center-align each
    # column so the name sits directly under the symbol.
    for column in legend._legend_handle_box.get_children():
        column.align = "center"
        for name_row in column.get_children()[1::2]:
            name_row.get_children()[0].set_width(0)
    return legend


def plot_faithfulness_by_variant(results: pd.DataFrame, output_path: Path) -> None:
    models = sorted(results["model"].unique())
    if len(models) > len(MODEL_COLORS):
        raise ValueError(f"{len(models)} models but only {len(MODEL_COLORS)} palette slots; "
                         "extend MODEL_COLORS/MODEL_MARKERS with validated entries")

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = range(len(VARIANT_ORDER))

    for model, color, marker in zip(models, MODEL_COLORS, MODEL_MARKERS):
        means = (results[results["model"] == model]
                 .groupby("variant")["faithfulness_pct"]
                 .mean()
                 .reindex(VARIANT_ORDER)
                 / 100)
        ax.plot(positions, means, color=color, marker=marker,
                linewidth=2, markersize=7, label=model)

    ax.set_xticks(positions)
    ax.set_xticklabels([VARIANT_TICK_LABELS[v] for v in VARIANT_ORDER])
    ax.set_ylim(0.67, 1.0)
    ax.set_xlabel("Variant")
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
    results = load_faithfulness_records(JUDGED_DIR)
    summary = (results
               .groupby(["model", "variant"])["faithfulness_pct"]
               .mean()
               .div(100)
               .round(3))
    print(summary.to_string())
    plot_faithfulness_by_variant(results, OUTPUT_PATH)
