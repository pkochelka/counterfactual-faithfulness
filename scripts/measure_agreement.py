import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sample_for_annotation import FLUENCY_DIR, load_judgments

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = ROOT / "data" / "annotation"

SCORE_VALUES = [1, 2, 3, 4, 5]
DIMENSIONS = {
    "faithfulness": ("human_faithfulness_score", "judge_faithfulness_score"),
    "fluency": ("human_fluency_score", "judge_fluency_score"),
}
SERIES_COLORS = {"human": "#2a78d6", "LLM judge": "#1baf7a"}


def quadratic_weighted_kappa(human_scores, judge_scores):
    observed = (
        pd.crosstab(human_scores, judge_scores)
        .reindex(index=SCORE_VALUES, columns=SCORE_VALUES, fill_value=0)
        .to_numpy()
    )
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.array([[(a - b) ** 2 for b in SCORE_VALUES] for a in SCORE_VALUES])
    return 1 - (weights * observed).sum() / (weights * expected).sum()


def parse_scores(column):
    return pd.to_numeric(column.astype(str).str.rstrip("?"), errors="coerce")


def fetch_judge_fluency_scores(key, fluency_dir):
    """Look up judge_fluency_score in data/judged_fluency for keys missing that column."""
    judgments = load_judgments(fluency_dir)

    def lookup(row):
        record = judgments.get(
            (row["model"], row["dataset"], row["variant"], row["prompt_language"], str(row["eid"]))
        )
        return None if record is None else record.get("parsed", {}).get("fluency_score")

    return key.apply(lookup, axis=1)


def agreement_metrics(human_scores, judge_scores):
    absolute_difference = (human_scores - judge_scores).abs()
    return pd.Series({
        "n": len(human_scores),
        "exact_agreement": (absolute_difference == 0).mean(),
        "within_1_agreement": (absolute_difference <= 1).mean(),
        "mean_absolute_error": absolute_difference.mean(),
        "mean_human": human_scores.mean(),
        "mean_judge": judge_scores.mean(),
        "spearman_correlation": human_scores.rank().corr(judge_scores.rank()),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(human_scores, judge_scores),
    })


def print_markdown_table(report):
    header = ["annotator"] + list(report.columns)
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for annotator, row in report.iterrows():
        cells = [annotator] + [f"{value:g}" for value in row]
        print("| " + " | ".join(cells) + " |")


def plot_score_distributions(scored, dimension, human_column, judge_column, output_path):
    annotators = sorted(scored["Annotator"].unique())
    fig, axes = plt.subplots(
        1, len(annotators), figsize=(3.2 * len(annotators), 3.4), sharey=True
    )
    for axis, annotator in zip(np.atleast_1d(axes), annotators):
        group = scored[scored["Annotator"] == annotator]
        series = {"human": group[human_column], "LLM judge": group[judge_column]}
        for offset, (label, scores) in zip((-0.21, 0.21), series.items()):
            shares = scores.value_counts(normalize=True).reindex(SCORE_VALUES, fill_value=0)
            axis.bar(
                [score + offset for score in SCORE_VALUES], shares,
                width=0.38, color=SERIES_COLORS[label], label=label,
            )
        axis.set_title(f"{annotator} (n={len(group)})", fontsize=10)
        axis.set_xticks(SCORE_VALUES)
        axis.set_xlabel("score")
        axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#c3c2b7")
        axis.tick_params(colors="#52514e")
    first_axis = np.atleast_1d(axes)[0]
    first_axis.set_ylabel("share of items")
    first_axis.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{dimension}: score distributions, human vs LLM judge")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {output_path}")


def write_faithfulness_disagreements(merged, output_path):
    columns = [
        "uid", "Annotator", "prompt_language", "triples", "sentence",
        "human_faithfulness_score", "judge_faithfulness_score", "score_difference",
        "human_faithfulness_reason", "judge_incorrect_information",
    ]
    disagreements = merged.dropna(subset=["human_faithfulness_score"]).copy()
    disagreements["human_faithfulness_score"] = disagreements["human_faithfulness_score"].astype(int)
    disagreements["score_difference"] = (
        disagreements["human_faithfulness_score"] - disagreements["judge_faithfulness_score"]
    )
    disagreements = disagreements.sort_values("score_difference", key=abs, ascending=False)
    disagreements[columns].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(disagreements)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Measure human vs. LLM-judge agreement.")
    parser.add_argument("--sample", default="annotation_sample_we_used.csv",
                        help="Blind sample CSV (under data/annotation) with human_* scores.")
    parser.add_argument("--key", default="annotation_key_we_used.csv",
                        help="Key CSV (under data/annotation) with judge_* scores.")
    parser.add_argument("--fluency-dir", type=Path, default=FLUENCY_DIR,
                        help="Fallback source for judge_fluency_score when the key lacks that column.")
    args = parser.parse_args()

    human = pd.read_csv(ANNOTATION_DIR / args.sample, encoding="utf-8-sig")
    judge = pd.read_csv(ANNOTATION_DIR / args.key, encoding="utf-8-sig")
    human["human_faithfulness_score"] = parse_scores(human["human_faithfulness_score"])
    human["human_fluency_score"] = parse_scores(human["human_fluency_score"])
    if "judge_fluency_score" not in judge.columns:
        judge["judge_fluency_score"] = fetch_judge_fluency_scores(judge, args.fluency_dir)
        missing = judge["judge_fluency_score"].isna().sum()
        if missing:
            print(f"judge_fluency_score not in {args.key}; fetched from {args.fluency_dir} "
                  f"({missing}/{len(judge)} rows had no matching record, e.g. other variants)")
    merged = human.merge(
        judge[[
            "uid", "judge_faithfulness_score", "judge_incorrect_information",
            "judge_fluency_score",
        ]],
        on="uid",
        how="inner",
        validate="one_to_one",
    )
    write_faithfulness_disagreements(merged, ANNOTATION_DIR / "faithfulness_disagreements.csv")

    for dimension, (human_column, judge_column) in DIMENSIONS.items():
        scored = merged.dropna(subset=[human_column, judge_column])
        report = {"ALL": agreement_metrics(scored[human_column], scored[judge_column])}
        for annotator, group in scored.groupby("Annotator"):
            report[annotator] = agreement_metrics(group[human_column], group[judge_column])
        print(f"\n### {dimension}: human vs judge\n")
        print_markdown_table(pd.DataFrame(report).T.round(3))
        plot_score_distributions(
            scored, dimension, human_column, judge_column,
            ANNOTATION_DIR / f"score_distributions_{dimension}.png",
        )


if __name__ == "__main__":
    main()
