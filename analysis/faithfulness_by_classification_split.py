import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from classification_accuracy import HARD_LABEL, LABEL_ORDER, LANGUAGE_ORDER
from faithfulness_by_language import MAX_SCORE, VARIANT_STYLE

CLASSIFIED_DIR = Path(__file__).parent.parent / "data" / "classified"
JUDGED_DIR = Path(__file__).parent.parent / "data" / "judged"

SOURCE_VARIANTS = ["fa", "cf"]
# Predicted labels reuse the colour/linestyle of the variant they correspond to.
PREDICTED_TO_VARIANT = {label: variant for variant, label in HARD_LABEL.items()}


def judged_model_dir(model: str) -> Path:
    return JUDGED_DIR / model.replace(".", "_")


def read_predicted_labels(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)[["eid", "sentence"]].rename(columns={"sentence": "predicted_label"})


def read_faithfulness_scores(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rows.append(dict(eid=record["eid"],
                             faithfulness_pct=record["parsed"]["faithfulness_score"] / MAX_SCORE * 100))
    return pd.DataFrame(rows)


def load_split_rows() -> pd.DataFrame:
    rows = []
    for model_dir in sorted(CLASSIFIED_DIR.iterdir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            dataset, variant, language = csv_path.stem.split("_")
            if variant not in SOURCE_VARIANTS:
                continue
            jsonl_path = next(judged_model_dir(model_dir.name)
                              .glob(f"judge_{dataset}_{variant}_{language}_*.jsonl"))
            merged = read_predicted_labels(csv_path).merge(
                read_faithfulness_scores(jsonl_path), on="eid")
            merged["model"] = model_dir.name
            merged["dataset"] = dataset
            merged["variant"] = variant
            merged["language"] = language
            rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def plot_split(rows: pd.DataFrame, output_path: Path) -> None:
    models = sorted(rows["model"].unique())
    datasets = sorted(rows["dataset"].unique())

    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(5 * len(datasets), 4 * len(models)),
        sharey=True, sharex=True, squeeze=False,
    )

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            panel = rows[(rows["model"] == model) & (rows["dataset"] == dataset)]
            legend_lines = []

            for label in LABEL_ORDER:
                split = panel[panel["predicted_label"] == label]
                if split.empty:
                    continue
                means = (split.groupby("language")["faithfulness_pct"]
                         .mean().reindex(LANGUAGE_ORDER))
                variant = PREDICTED_TO_VARIANT[label]
                line, = ax.plot(LANGUAGE_ORDER, means, linewidth=2, marker="o",
                                color=VARIANT_STYLE[variant]["color"],
                                linestyle=VARIANT_STYLE[variant]["linestyle"])
                legend_lines.append((line, f"predicted {label} (n={len(split)})"))

            ax.axhline(100, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_title(f"{model} / {dataset}", fontsize=10)
            ax.set_xlabel("Prompt language" if row == len(models) - 1 else "")
            ax.set_ylabel("Mean faithfulness (%)" if col == 0 else "")
            ax.legend([line for line, _ in legend_lines],
                      [label for _, label in legend_lines], fontsize=7, loc="lower left")

    fig.suptitle(
        "Faithfulness by prompt language, split by the model's predicted label\n"
        "Each line groups factual (fa) and counterfactual (cf) rows by what the model classified them as",
        fontsize=12,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    rows = load_split_rows()
    summary = (rows
               .groupby(["model", "dataset", "predicted_label", "language"])
               .agg(mean_faithfulness=("faithfulness_pct", "mean"),
                    n=("faithfulness_pct", "size"))
               .round(1))
    print(summary.to_string())

    output_path = Path(__file__).parent / "faithfulness_by_classification_split.png"
    plot_split(rows, output_path)
