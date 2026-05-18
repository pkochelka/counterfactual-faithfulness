import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLASSIFIED_DIR = Path(__file__).parent.parent / "data" / "classified"
OUTPUT_PATH = Path(__file__).parent / "classification_accuracy.png"
CONFUSION_OUTPUT_PATH = Path(__file__).parent / "classification_confusion_matrix.png"

LABEL_ORDER = ["CFA", "FA", "FI"]

HARD_LABEL = {"cf": "CFA", "fa": "FA", "fi": "FI"}
SOFT_LABELS = {"cf": {"CFA", "FI"}, "fa": {"FA"}, "fi": {"FI", "CFA"}}


def load_accuracy_records(classified_dir: Path) -> pd.DataFrame:
    records = []
    for model_dir in sorted(classified_dir.iterdir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            dataset, variant, language = csv_path.stem.split("_")
            predictions = pd.read_csv(csv_path)["sentence"]
            hard_accuracy = (predictions == HARD_LABEL[variant]).mean() * 100
            soft_accuracy = predictions.isin(SOFT_LABELS[variant]).mean() * 100
            records.append(
                dict(model=model_dir.name, dataset=dataset, variant=variant,
                     language=language, hard_accuracy=hard_accuracy, soft_accuracy=soft_accuracy)
            )
    return pd.DataFrame(records)


def load_prediction_records(classified_dir: Path) -> pd.DataFrame:
    records = []
    for model_dir in sorted(classified_dir.iterdir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            dataset, variant, _ = csv_path.stem.split("_")
            true_label = HARD_LABEL[variant]
            for predicted in pd.read_csv(csv_path)["sentence"]:
                records.append(dict(model=model_dir.name, dataset=dataset,
                                    true_label=true_label, predicted_label=predicted))
    return pd.DataFrame(records)


def plot_confusion_matrices(predictions: pd.DataFrame, output_path: Path) -> None:
    models = sorted(predictions["model"].unique())
    datasets = sorted(predictions["dataset"].unique())
    fig, axes = plt.subplots(len(models), len(datasets),
                             figsize=(5 * len(datasets), 5 * len(models)))

    for row, model in enumerate(models):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            subset = predictions[(predictions["model"] == model) &
                                 (predictions["dataset"] == dataset)]
            cm = (
                pd.crosstab(subset["true_label"], subset["predicted_label"],
                            rownames=["True"], colnames=["Predicted"])
                .reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
            )
            cm_pct = cm.div(cm.sum(axis=1), axis=0) * 100
            annot = cm.astype(str) + "\n(" + cm_pct.round(1).astype(str) + "%)"
            sns.heatmap(cm_pct, ax=ax, annot=annot, fmt="", cmap="Blues",
                        vmin=0, vmax=100, linewidths=0.5,
                        cbar_kws={"label": "Row %"})
            ax.set_title(f"{model} / {dataset}", fontsize=11)
            ax.set_xlabel("Predicted" if row == len(models) - 1 else "")
            ax.set_ylabel("True" if col == 0 else "")

    fig.suptitle(
        "Confusion matrix (hard accuracy) — row-normalised\nAnnotations: count  (row %)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def pivot_for(results: pd.DataFrame, model: str, metric: str) -> pd.DataFrame:
    return (
        results[results["model"] == model]
        .pivot_table(index=["dataset", "variant"], columns="language", values=metric)
    )


def plot_accuracy(results: pd.DataFrame, output_path: Path) -> None:
    models = sorted(results["model"].unique())
    n_models = len(models)
    metrics = [("hard_accuracy", "Hard accuracy"), ("soft_accuracy", "Soft accuracy")]

    fig, axes = plt.subplots(2, n_models, figsize=(7 * n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for row, (metric, metric_label) in enumerate(metrics):
        for col, model in enumerate(models):
            ax = axes[row, col]
            sns.heatmap(
                pivot_for(results, model, metric),
                ax=ax, annot=True, fmt=".1f",
                vmin=0, vmax=100, cmap="RdYlGn", linewidths=0.5,
                cbar_kws={"label": "Accuracy (%)"},
            )
            ax.set_title(f"{model}  —  {metric_label}", fontsize=12)
            ax.set_xlabel("Prompt language")
            ax.set_ylabel("Dataset / true variant" if col == 0 else "")

    fig.suptitle(
        "Classification accuracy by model, dataset, true variant, and prompt language\n"
        "Soft accuracy: cf and fi are interchangeable (both CFA / FI are accepted)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    results = load_accuracy_records(CLASSIFIED_DIR)
    print(results.sort_values(["model", "dataset", "variant", "language"]).to_string(index=False))
    plot_accuracy(results, OUTPUT_PATH)

    predictions = load_prediction_records(CLASSIFIED_DIR)
    plot_confusion_matrices(predictions, CONFUSION_OUTPUT_PATH)
