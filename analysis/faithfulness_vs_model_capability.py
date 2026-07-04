"""Plot LLM-judge faithfulness and predicted-label rates against model capability
(parameter count and the Artificial Analysis Intelligence Index), one panel per
(dataset, prompt language). Capability values come from artificialanalysis.ai
(Intelligence Index v4.1) and the models' published parameter counts.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from faithfulness_by_language import load_faithfulness_records, LANGUAGE_ORDER, VARIANT_STYLE
from classification_accuracy import CLASSIFIED_DIR, LABEL_ORDER, HARD_LABEL

JUDGED_DIR = Path(__file__).parent.parent / "data" / "judged"

LABEL_COLOR = {label: VARIANT_STYLE[variant]["color"] for variant, label in HARD_LABEL.items()}

MODEL_INFO = {
    "gpt-oss-120b":                   {"label": "gpt-oss-120b", "size_b": 117.0, "aaii": 24},
    "qwen3_5-122b":                   {"label": "Qwen3.5-122B", "size_b": 125.0, "aaii": 32},
    "qwen3_5-9b":                     {"label": "Qwen3.5-9B",   "size_b": 9.7,   "aaii": 21},
    "qwen3-1_7b":                     {"label": "Qwen3-1.7B",   "size_b": 1.7,   "aaii": 7},
    "llama-4-scout-17b-16e-instruct": {"label": "Llama4-Scout", "size_b": 17.0,  "aaii": 10},
    "gemma4-31B":                     {"label": "Gemma4-31B",   "size_b": 30.7,  "aaii": 29},
    "gemma4-e4b":                     {"label": "Gemma4-E4B",   "size_b": 8.0,   "aaii": 12},
    "gemma4-e2b":                     {"label": "Gemma4-E2B",   "size_b": 5.1,   "aaii": 9},
    "tiny-aya-global":                {"label": "Tiny-Aya",     "size_b": 3.35,  "aaii": 5},
}

CAPABILITY_AXES = {
    "size_b": {"xlabel": "Model size (total parameters, B)", "logx": True,
               "fname": "faithfulness_vs_model_size.png",
               "title": "Mean faithfulness vs model size (total parameters)"},
    "aaii":   {"xlabel": "Artificial Analysis Intelligence Index", "logx": False,
               "fname": "faithfulness_vs_intelligence_index.png",
               "title": "Mean faithfulness vs Artificial Analysis Intelligence Index"},
}


def panel_grid(df):
    datasets = sorted(df["dataset"].unique())
    languages = [l for l in LANGUAGE_ORDER if l in df["language"].unique()]
    fig, axes = plt.subplots(
        len(datasets), len(languages),
        figsize=(4.8 * len(languages), 4.0 * len(datasets)),
        sharey=True, squeeze=False,
    )
    return datasets, languages, fig, axes


def save(output_path):
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def load_prediction_records():
    records = []
    for model_dir in sorted(CLASSIFIED_DIR.iterdir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            dataset, variant, language = csv_path.stem.split("_")
            for predicted in pd.read_csv(csv_path)["sentence"]:
                records.append(dict(model=model_dir.name, dataset=dataset, variant=variant,
                                    language=language, predicted_label=predicted))
    return pd.DataFrame(records)


def prediction_rates(preds):
    counts = (preds.groupby(["dataset", "language", "model", "predicted_label"])
              .size().unstack("predicted_label", fill_value=0)
              .reindex(columns=LABEL_ORDER, fill_value=0))
    return counts.div(counts.sum(axis=1), axis=0) * 100


def plot_faithfulness_vs_capability(results, metric, output_path):
    means = results.groupby(["dataset", "language", "model"])["faithfulness_pct"].mean()
    datasets, languages, fig, axes = panel_grid(results)
    cfg = CAPABILITY_AXES[metric]

    for row, dataset in enumerate(datasets):
        for col, language in enumerate(languages):
            ax = axes[row, col]
            points = sorted(
                (info[metric], means.loc[dataset, language, model], info["label"])
                for model, info in MODEL_INFO.items()
                if (dataset, language, model) in means.index)
            xs = [x for x, _, _ in points]
            ys = [y for _, y, _ in points]
            ax.plot(xs, ys, color="#bbbbbb", linewidth=1, zorder=1)
            ax.scatter(xs, ys, color="#1f77b4", zorder=2)
            for x, y, label in points:
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

            if cfg["logx"]:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{dataset} / prompt={language}", fontsize=10)
            ax.set_xlabel(cfg["xlabel"] if row == len(datasets) - 1 else "")
            ax.set_ylabel("Mean faithfulness (%)" if col == 0 else "")

    fig.suptitle(
        cfg["title"] + "\nMean LLM-judge faithfulness over fa+cf variants, "
        "per dataset and prompt language",
        fontsize=12,
    )
    save(output_path)


def plot_prediction_rates_vs_size(preds, output_path, source_label):
    rates = prediction_rates(preds)
    datasets, languages, fig, axes = panel_grid(preds)

    for row, dataset in enumerate(datasets):
        for col, language in enumerate(languages):
            ax = axes[row, col]
            models = sorted(
                (m for m in MODEL_INFO if (dataset, language, m) in rates.index),
                key=lambda m: MODEL_INFO[m]["size_b"])
            sizes = [MODEL_INFO[m]["size_b"] for m in models]
            shares = [[rates.loc[(dataset, language, m), label] for m in models]
                      for label in LABEL_ORDER]
            ax.stackplot(sizes, *shares,
                         colors=[LABEL_COLOR[label] for label in LABEL_ORDER],
                         labels=[f"predicted {label}" for label in LABEL_ORDER])

            ax.set_xscale("log")
            ax.set_xlim(min(sizes), max(sizes))
            ax.set_ylim(0, 100)
            ax.set_title(f"{dataset} / prompt={language}", fontsize=10)
            ax.set_xlabel("Model size (total parameters, B)" if row == len(datasets) - 1 else "")
            ax.set_ylabel("Prediction rate (%)" if col == 0 else "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(LABEL_ORDER),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Predicted-label rate vs model size (total parameters) — {source_label} source sentences\n"
        "Share of sentences classified as each label, per dataset and prompt language",
        fontsize=12,
    )
    save(output_path)


if __name__ == "__main__":
    faithfulness = load_faithfulness_records(JUDGED_DIR)
    print(faithfulness.groupby(["dataset", "language", "model"])["faithfulness_pct"]
          .mean().round(2).to_string())
    for metric in CAPABILITY_AXES:
        plot_faithfulness_vs_capability(
            faithfulness, metric, Path(__file__).parent / CAPABILITY_AXES[metric]["fname"])

    predictions = load_prediction_records()
    for variant, source_label in [("fa", "fa (factual)"), ("cf", "cf (counterfactual)")]:
        plot_prediction_rates_vs_size(
            predictions[predictions["variant"] == variant],
            Path(__file__).parent / f"prediction_rates_vs_model_size_{variant}.png",
            source_label)
