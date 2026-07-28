"""Build the compact LaTeX tables used in the article.

The script reads the released study data directly and writes the LaTeX tables
and cluster-bootstrap summaries used in the article.

Example (from the repository root):

    python analysis/build_results_tables.py --bootstrap-samples 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
COMBO_RE = re.compile(r"(cs-qa|sk-qa)_(fa|cf|fi)_(en|cs|sk|hsb)")
REPEAT_RE = re.compile(r"^sentence_\d+$")
LABELS = ("CFA", "FA", "FI")
EXPECTED_LABEL = {"cf": "CFA", "fa": "FA", "fi": "FI"}
MODEL_ORDER = (
    "qwen3.5-122b",
    "gpt-oss-120b",
    "gemma4-31B",
    "llama4-scout",
    "qwen3_5-9b",
    "gemma4-e4b",
    "gemma4-e2b",
    "tiny-aya-global",
    "qwen3-1_7b",
)
MODEL_DISPLAY = {
    "gemma4-31B": "Gemma4-31B",
    "gpt-oss-120b": "GPT-OSS-120B",
    "qwen3.5-122b": "Qwen3.5-122B",
    "llama4-scout": "Llama4-Scout",
    "qwen3_5-9b": "Qwen3.5-9B",
    "gemma4-e4b": "Gemma4-E4B",
    "gemma4-e2b": "Gemma4-E2B",
    "tiny-aya-global": "Tiny-Aya",
    "qwen3-1_7b": "Qwen3-1.7B",
}
ARTICLE_MODEL_DISPLAY = {
    "qwen3.5-122b": "Qwen3.5 122B",
    "gpt-oss-120b": "GPT OSS 120B",
    "gemma4-31B": "Gemma4 31B",
    "llama4-scout": "Llama4 17B",
    "qwen3_5-9b": "Qwen3.5 9B",
    "gemma4-e4b": "Gemma4 E4B",
    "gemma4-e2b": "Gemma4 E2B",
    "tiny-aya-global": "Tiny Aya",
    "qwen3-1_7b": "Qwen3 1.7B",
}
MODEL_SHORT = {
    "gemma4-31B": "G4-31",
    "gpt-oss-120b": "OSS-120",
    "qwen3.5-122b": "Q-122",
    "llama4-scout": "L4-S",
    "qwen3_5-9b": "Q3.5-9",
    "gemma4-e4b": "G4-E4",
    "gemma4-e2b": "G4-E2",
    "tiny-aya-global": "T-Aya",
    "qwen3-1_7b": "Q3-1.7",
}
MODEL_ALIASES = {
    "qwen3_5-122b": "qwen3.5-122b",
    "llama4-scout-openrouter": "llama4-scout",
}
SIZE_GROUP_ENDS = {"gpt-oss-120b", "gemma4-e4b"}
SIZE_GROUPS = (
    ("Large", MODEL_ORDER[:2]),
    ("Medium", MODEL_ORDER[2:6]),
    ("Small", MODEL_ORDER[6:]),
)
ISSUE_COLUMNS = (
    ("language_or_style", "Lang"),
    ("unsupported_or_hallucinated", "Hall"),
    ("reverse_relation", "Rev"),
    ("missing_information", "Miss"),
    ("wrong_relation", "Rel"),
    ("wrong_entity", "Ent"),
)
ISSUE_CATEGORY_NAMES = tuple(name for name, _ in ISSUE_COLUMNS)
CLASSIFICATION_KEYS = ["model", "dataset", "variant", "language", "eid"]
BOOTSTRAP_CLUSTER_KEYS = ["dataset", "variant", "eid"]


def canonical_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def format_extreme(value: float, extreme: float, *, suffix: str = "") -> str:
    """Bold values tied at the displayed two-decimal precision."""
    text = f"{value:.2f}"
    if text == f"{extreme:.2f}":
        return rf"\textbf{{{text}{suffix}}}"
    return text + suffix


def parse_combo(value: str) -> tuple[str, str, str] | None:
    match = COMBO_RE.search(value)
    return match.groups() if match else None


def parse_label(value: Any) -> str:
    """Apply the same conservative recovery rules as report generation."""
    text = str(value or "").strip()
    if not text:
        return ""
    compact = text.strip("`*_ \t\r\n.:;-")
    if compact.upper() in LABELS:
        return compact.upper()
    found = re.findall(r"\b(CFA|FA|FI)\b", text, re.I)
    if len(found) == 1:
        return found[0].upper()
    if len(found) > 1:
        return ""
    for label, pattern in (("CFA", r"\bcounterfactual\b|\bkontrafaktu[aá]ln"),
                           ("FI", r"\bfictional\b|\bfiktivn|\bfiktívn"),
                           ("FA", r"\bfactual\b|\bfaktick")):
        if re.search(pattern, text, re.I):
            return label
    return ""


def load_generated(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_dir in sorted((data_dir / "generated").glob("*")):
        if not model_dir.is_dir():
            continue
        model = canonical_model(model_dir.name)
        for path in sorted(model_dir.glob("*.csv")):
            combo = parse_combo(path.stem)
            if combo is None:
                continue
            dataset, variant, language = combo
            frame = pd.read_csv(path, dtype={"eid": str}, keep_default_na=False)
            for record in frame[["eid", "size"]].to_dict("records"):
                rows.append({"model": model, "dataset": dataset, "variant": variant,
                             "language": language, "eid": str(record["eid"]),
                             "triple_count": int(record["size"])})
    return pd.DataFrame(rows)


def load_judgments(data_dir: Path, directory: str, score_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((data_dir / directory).glob("*/*.jsonl")):
        if path.name.endswith(".failures.jsonl"):
            continue
        fallback_model = canonical_model(path.parent.name)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                combo = parse_combo(str(record.get("source_label") or path.stem))
                if combo is None:
                    continue
                parsed = record.get("parsed") or {}
                score = parsed.get(score_key)
                if score not in (1, 2, 3, 4, 5):
                    continue
                source_id = str(record.get("source_id") or "")
                model = canonical_model(source_id.split("__", 1)[0]) if "__" in source_id else fallback_model
                dataset, variant, language = combo
                rows.append({"model": model, "dataset": dataset, "variant": variant,
                             "language": language, "eid": str(record.get("eid", "")),
                             "score": int(score), "timestamp": str(record.get("timestamp", ""))})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Reruns can append replacement records.  The latest valid judgment is the
    # one the table should use, not an accidental duplicate line.
    return (frame.sort_values("timestamp")
                 .drop_duplicates(["model", "dataset", "variant", "language", "eid"], keep="last"))


def load_classification(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_dir in sorted((data_dir / "classified").glob("*")):
        if not model_dir.is_dir():
            continue
        model = canonical_model(model_dir.name)
        for path in sorted(model_dir.glob("*.csv")):
            combo = parse_combo(path.stem)
            if combo is None:
                continue
            dataset, variant, language = combo
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            answer_columns = [column for column in frame.columns if REPEAT_RE.fullmatch(column)]
            answer_columns = answer_columns or ["sentence"]
            for record in frame.to_dict("records"):
                parsed_labels = [parse_label(record.get(column, "")) for column in answer_columns]
                labels = [label for label in parsed_labels if label]
                vote = Counter(labels).most_common(1)[0][0] if labels else ""
                label_counts = Counter(labels)
                row = {"model": model, "dataset": dataset, "variant": variant,
                       "language": language, "eid": str(record["eid"]),
                       "predicted_label": vote,
                       "correct": vote == EXPECTED_LABEL[variant],
                       "unanimous": len(labels) == len(answer_columns) and len(set(labels)) == 1,
                       "invalid_rate": 1 - len(labels) / len(answer_columns)}
                for label in LABELS:
                    row[f"vote_share_{label}"] = label_counts.get(label, 0) / len(labels) if labels else 0.0
                rows.append(row)
    return pd.DataFrame(rows)


def classification_figure(classification: pd.DataFrame) -> str:
    """Generate the Gemma4-31B pooled confusion matrix and its LaTeX block."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    predictions = classification[classification["model"] == "gemma4-31B"]
    if predictions.empty:
        raise ValueError("Classification data does not contain gemma4-31B.")

    cm = (
        pd.crosstab(predictions["variant"].map(EXPECTED_LABEL),
                    predictions["predicted_label"],
                    rownames=["True"], colnames=["Predicted"])
        .reindex(index=["CFA", "FA", "FI"], columns=LABELS, fill_value=0)
    )
    cm_pct = cm.div(cm.sum(axis=1), axis=0) * 100
    annot = cm.astype(str) + "\n(" + cm_pct.round(1).astype(str) + "%)"

    figure_path = REPO_ROOT.parent / "article" / "classification_confusion_matrix_v3_gemma4-31B.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(cm_pct, ax=ax, annot=annot, fmt="", cmap="Blues",
                vmin=0, vmax=100, linewidths=0.5,
                cbar_kws={"label": "Row %"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return "\n".join([
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{classification_confusion_matrix_v3_gemma4-31B.png}",
        r"\caption{Classification confusion matrix for Gemma4-31B, pooled over both datasets and all four prompt languages. Rows show the true class and columns the predicted class. Each cell gives the number of examples and the percentage within its true-class row.}",
        r"\label{fig:classification-confusion-gemma}",
        r"\end{figure}",
    ])


def model_table(
    faith: pd.DataFrame,
    fluency: pd.DataFrame,
    classification: pd.DataFrame,
) -> str:
    """Render the combined classification and generation table."""
    faith_means = faith.groupby("model")["score"].mean()
    fluency_means = fluency.groupby("model")["score"].mean()
    metrics = classification.groupby("model").agg(
        accuracy=("correct", "mean"),
        unanimity=("unanimous", "mean"),
    )
    distributions = (
        classification[classification["predicted_label"].isin(LABELS)]
        .groupby("model")["predicted_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=("FA", "FI", "CFA"), fill_value=0)
        * 100
    )
    accuracy_max = metrics["accuracy"].max()
    unanimity_max = metrics["unanimity"].max()
    faith_max = faith_means.max()
    fluency_max = fluency_means.max()

    def distribution_text(model: str) -> str:
        raw_values = distributions.loc[model, ["FA", "FI", "CFA"]]
        values = np.floor(raw_values).astype(int)
        remainder = 100 - int(values.sum())
        for label in (raw_values - values).nlargest(remainder).index:
            values[label] += 1
        return "/".join(rf"\pz{value}" if value < 10 else str(value) for value in values)

    def percentage(value: float, extreme: float) -> str:
        text = f"{value:.1f}"
        return rf"\textbf{{{text}}}" if text == f"{extreme:.1f}" else text

    lines = [
        r"\begin{table}[t]", r"\centering", r"\def\pz{\phantom{0}}",
        r"\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lccc>{\hspace{1mm}}cc@{}}",
        r"\toprule", "Model & Acc. & Unan. & FA/FI/CFA " + r"\% & Faith. & Flu. \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        lines.append(
            f"{ARTICLE_MODEL_DISPLAY[model]} & "
            f"{percentage(metrics.loc[model, 'accuracy'] * 100, accuracy_max * 100)} & "
            f"{percentage(metrics.loc[model, 'unanimity'] * 100, unanimity_max * 100)} & "
            f"{distribution_text(model)} & "
            f"{format_extreme(faith_means[model], faith_max)} & "
            f"{format_extreme(fluency_means[model], fluency_max)} " + r"\\"
        )
        if model in SIZE_GROUP_ENDS:
            lines.append(r"\hdashline[0.5pt/2pt]")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Classification and generation results. Acc. is exact majority-vote classification accuracy, Unan. is the \% of examples with unanimous classification in five runs. FA/FI/CFA~\% is the percentage distribution of majority predictions across classes. Faith. and Flu. are mean LLM judge scores for faithfulness and fluency on a 1--5 scale. Model size groups are split by dashed lines.}",
        r"\label{tab:model-results}", r"\end{table}",
    ])
    return "\n".join(lines)


def model_ci_table(faith: pd.DataFrame, fluency: pd.DataFrame, model_cis: pd.DataFrame) -> str:
    faith_means = faith.groupby("model")["score"].mean()
    fluency_means = fluency.groupby("model")["score"].mean()
    faith_max = faith_means.max()
    fluency_max = fluency_means.max()
    ci_lookup = model_cis.set_index(["metric", "model"])
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lcc}",
        r"\toprule", r"Model & Faith. & Flu. \\", r"\midrule",
    ]
    for model in MODEL_ORDER:
        faith_ci = ci_lookup.loc[("faithfulness", model)]
        fluency_ci = ci_lookup.loc[("fluency", model)]
        lines.append(
            rf"\texttt{{{MODEL_DISPLAY[model]}}} & {format_extreme(faith_means[model], faith_max)} "
            rf"{{\scriptsize [{faith_ci['ci_95_low']:.2f}, {faith_ci['ci_95_high']:.2f}]}} & "
            rf"{format_extreme(fluency_means[model], fluency_max)} "
            rf"{{\scriptsize [{fluency_ci['ci_95_low']:.2f}, {fluency_ci['ci_95_high']:.2f}]}} \\")
        if model in SIZE_GROUP_ENDS:
            lines.append(r"\hdashline[0.5pt/2pt]")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Mean LLM-judge scores on a 1--5 scale, with 95\% cluster-bootstrap confidence intervals in brackets. The bootstrap resamples source RDF items while retaining their language-specific observations.}",
        r"\label{tab:model-results-ci}", r"\end{table}",
    ])
    return "\n".join(lines)


def classification_table(classification: pd.DataFrame) -> str:
    metrics = classification.groupby("model").agg(accuracy=("correct", "mean"),
                                                    unanimity=("unanimous", "mean"),
                                                    invalid=("invalid_rate", "mean"))
    accuracy_max = metrics["accuracy"].max()
    unanimity_max = metrics["unanimity"].max()
    invalid_min = metrics["invalid"].min()
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lccc}",
        r"\toprule", r"Model & Accuracy & Unanim. & Invalid \\", r"\midrule",
    ]
    for model in MODEL_ORDER:
        lines.append(rf"\texttt{{{MODEL_DISPLAY[model]}}} & "
                     rf"{format_extreme(metrics.loc[model, 'accuracy'] * 100, accuracy_max * 100, suffix=r'\%')} & "
                     rf"{format_extreme(metrics.loc[model, 'unanimity'] * 100, unanimity_max * 100, suffix=r'\%')} & "
                     rf"{format_extreme(metrics.loc[model, 'invalid'] * 100, invalid_min * 100, suffix=r'\%')} \\")
        if model in SIZE_GROUP_ENDS:
            lines.append(r"\hdashline[0.5pt/2pt]")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Classification results. Accuracy is exact majority-vote accuracy. Unanim. is the percentage of examples with five parsed, identical labels. Invalid is the percentage of individual classification attempts that could not be conservatively parsed as FA, CFA, or FI.}",
        r"\label{tab:classification-results}", r"\end{table}",
    ])
    return "\n".join(lines)


def triple_table(faith: pd.DataFrame, generated: pd.DataFrame) -> str:
    rows = faith.merge(generated, on=["model", "dataset", "variant", "language", "eid"], how="left")
    rows["bucket"] = pd.cut(rows["triple_count"], [0, 1, 3, 9], labels=["1", "2--3", "4--9"])
    means = rows.groupby(["model", "bucket"], observed=True)["score"].mean().unstack()
    overall = rows.groupby("model")["score"].mean()
    pooled = rows.groupby("bucket", observed=True)["score"].mean()
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lcccc}",
        r"\toprule", r"Model & 1 & 2--3 & 4--9 & Overall \\", r"\midrule",
    ]
    maxima = {"1": means["1"].max(), "2--3": means["2--3"].max(), "4--9": means["4--9"].max(),
              "overall": overall.max()}
    for model in MODEL_ORDER:
        values = [(means.loc[model, "1"], maxima["1"]), (means.loc[model, "2--3"], maxima["2--3"]),
                  (means.loc[model, "4--9"], maxima["4--9"]), (overall[model], maxima["overall"])]
        formatted = [format_extreme(value, maximum) for value, maximum in values]
        lines.append(rf"\texttt{{{MODEL_DISPLAY[model]}}} & " + " & ".join(formatted) + r" \\")
        if model in SIZE_GROUP_ENDS:
            lines.append(r"\hdashline[0.5pt/2pt]")
    lines.extend([
        r"\midrule",
        rf"All models & {pooled['1']:.2f} & {pooled['2--3']:.2f} & {pooled['4--9']:.2f} & {rows['score'].mean():.2f} \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Average faithfulness (1--5, judged by DeepSeek-V4-Pro). Columns 1, 2--3, and 4--9 give the number of triples in the input set. Values pool both datasets, all four prompt languages, and the factual, counterfactual, and fictional variants.}",
        r"\label{tab:faithfulness-triples}", r"\end{table}",
    ])
    return "\n".join(lines)


def predicted_class_table(faith: pd.DataFrame, classification: pd.DataFrame) -> str:
    """Faithfulness grouped by each model's self-classification, soft-weighted by vote share.

    Each example contributes to every label's mean in proportion to the share of
    its classification repeats that voted for that label (rather than being
    hard-assigned to the single majority label), so low-agreement examples are
    split fractionally across labels instead of arbitrarily rounded to one.
    """
    keys = ["model", "dataset", "variant", "language", "eid"]
    weight_cols = [f"vote_share_{label}" for label in LABELS]
    rows = faith.merge(classification[keys + weight_cols], on=keys, how="inner")

    means = {}
    counts = {}
    for label in LABELS:
        weight = rows[f"vote_share_{label}"]
        grouped = rows.assign(_weighted_score=rows["score"] * weight, _weight=weight).groupby("model")
        weight_sum = grouped["_weight"].sum()
        means[label] = grouped["_weighted_score"].sum() / weight_sum
        counts[label] = weight_sum
    means = pd.DataFrame(means)
    counts = pd.DataFrame(counts)
    maxima = means.max(axis=0)

    def score_with_count(score: float, count: float, maximum: float) -> str:
        count_text = f"{round(count):,}".replace(",", "{,}")
        score_text = f"{score:.2f}"
        if score_text == f"{maximum:.2f}":
            score_text = r"\mathbf{" + score_text + "}"
        return "$" + score_text + "^{" + count_text + "}$"

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lccc}",
        r"\toprule", r"Model & Pred. FA & Pred. FI & Pred. CFA " + r"\\", r"\midrule",
    ]
    for model in MODEL_ORDER:
        cells = [score_with_count(means.loc[model, label], counts.loc[model, label], maxima[label])
                 if label in means.columns and pd.notna(means.loc[model, label])
                 else r"--"
                 for label in ("FA", "FI", "CFA")]
        lines.append(rf"\texttt{{{MODEL_DISPLAY[model]}}} & " + " & ".join(cells) + r" \\")
        if model in SIZE_GROUP_ENDS:
            lines.append(r"\hdashline[0.5pt/2pt]")

    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Mean faithfulness scores (1--5) grouped by the model's self-classification and weighted by the five classification votes. Pred. denotes the predicted class, and superscripts give the effective weighted sample size.}",
        r"\label{tab:faithfulness-predicted-class-detailed}", r"\end{table}",
    ])
    return "\n".join(lines)


def perceived_class_bootstrap(
    faith: pd.DataFrame,
    classification: pd.DataFrame,
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap perceived-class means and paired size-group contrasts.

    Source RDF items are resampled as clusters.  The same sampled clusters are
    used for all three perceived classes, so the FA-CFA and FI-CFA differences
    are paired.  The contrast intervals use a Bonferroni correction for the six
    planned comparisons (two contrasts in each of three model-size groups).
    """
    labels = ("FA", "FI", "CFA")
    weight_columns = [f"vote_share_{label}" for label in labels]
    rows = faith.merge(
        classification[CLASSIFICATION_KEYS + weight_columns],
        on=CLASSIFICATION_KEYS,
        how="inner",
        validate="one_to_one",
    )
    cell_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    # Six two-sided comparisons share a family-wise alpha of 0.05.
    adjusted_tail = 0.05 / (2 * 6)
    # Preserve the original table's deterministic pointwise-CI stream.  The
    # paired contrasts below use a separate shared resampling stream.
    cell_rng = np.random.default_rng(seed)

    for group_index, (group_name, models) in enumerate(SIZE_GROUPS):
        subset = rows[rows["model"].isin(models)].copy()
        clusters = pd.MultiIndex.from_frame(
            subset[BOOTSTRAP_CLUSTER_KEYS].drop_duplicates().sort_values(BOOTSTRAP_CLUSTER_KEYS)
        )
        numerators: list[np.ndarray] = []
        denominators: list[np.ndarray] = []
        pointwise_intervals: list[tuple[float, float]] = []
        for label in labels:
            weight_column = f"vote_share_{label}"
            subset["_weighted_score"] = subset["score"] * subset[weight_column]
            pointwise_grouped = (
                subset.groupby(BOOTSTRAP_CLUSTER_KEYS, sort=False)
                .agg(numerator=("_weighted_score", "sum"), denominator=(weight_column, "sum"))
            )
            pointwise_numerator = pointwise_grouped["numerator"].to_numpy()
            pointwise_denominator = pointwise_grouped["denominator"].to_numpy()
            pointwise_boot = np.empty(samples)
            position = 0
            while position < samples:
                batch_size = min(250, samples - position)
                sampled_indices = cell_rng.integers(
                    len(pointwise_grouped),
                    size=(batch_size, len(pointwise_grouped)),
                )
                pointwise_boot[position:position + batch_size] = (
                    pointwise_numerator[sampled_indices].sum(axis=1)
                    / pointwise_denominator[sampled_indices].sum(axis=1)
                )
                position += batch_size
            pointwise_intervals.append(
                tuple(float(value) for value in np.quantile(pointwise_boot, [0.025, 0.975]))
            )
            grouped = (
                pointwise_grouped
                .reindex(clusters, fill_value=0)
            )
            numerators.append(grouped["numerator"].to_numpy())
            denominators.append(grouped["denominator"].to_numpy())
        numerator_array = np.stack(numerators, axis=1)
        denominator_array = np.stack(denominators, axis=1)
        observed = numerator_array.sum(axis=0) / denominator_array.sum(axis=0)

        rng = np.random.default_rng(seed + group_index)
        boot = np.empty((samples, len(labels)))
        position = 0
        while position < samples:
            batch_size = min(200, samples - position)
            sampled_indices = rng.integers(
                len(clusters), size=(batch_size, len(clusters))
            )
            sampled_numerators = numerator_array[sampled_indices].sum(axis=1)
            sampled_denominators = denominator_array[sampled_indices].sum(axis=1)
            boot[position:position + batch_size] = sampled_numerators / sampled_denominators
            position += batch_size

        for label_index, label in enumerate(labels):
            lower, upper = pointwise_intervals[label_index]
            cell_rows.append({
                "model_size": group_name,
                "predicted_label": label,
                "mean_score": float(observed[label_index]),
                "ci_95_low": float(lower),
                "ci_95_high": float(upper),
                "effective_count": float(denominator_array[:, label_index].sum()),
                "source_item_clusters": len(clusters),
                "bootstrap_samples": samples,
            })

        cfa_index = labels.index("CFA")
        for comparison_label in ("FA", "FI"):
            comparison_index = labels.index(comparison_label)
            differences = boot[:, comparison_index] - boot[:, cfa_index]
            lower, upper = np.quantile(differences, [0.025, 0.975])
            adjusted_lower, adjusted_upper = np.quantile(
                differences, [adjusted_tail, 1 - adjusted_tail]
            )
            contrast_rows.append({
                "model_size": group_name,
                "comparison": f"Pred. {comparison_label} - Pred. CFA",
                "mean_difference": float(observed[comparison_index] - observed[cfa_index]),
                "ci_95_low": float(lower),
                "ci_95_high": float(upper),
                "bonferroni_familywise_95_low": float(adjusted_lower),
                "bonferroni_familywise_95_high": float(adjusted_upper),
                "bonferroni_comparisons": 6,
                "significant_familywise_05": bool(adjusted_lower > 0 or adjusted_upper < 0),
                "source_item_clusters": len(clusters),
                "bootstrap_samples": samples,
            })

    return pd.DataFrame(cell_rows), pd.DataFrame(contrast_rows)


def perceived_class_ci_table(cells: pd.DataFrame) -> str:
    """Render the model-size perceived-class table with bootstrap intervals."""
    lookup = cells.set_index(["model_size", "predicted_label"])
    bootstrap_samples = int(cells["bootstrap_samples"].iloc[0])
    maxima = {
        label: cells[cells["predicted_label"] == label]["mean_score"].max()
        for label in ("FA", "FI", "CFA")
    }
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}", r"\toprule",
        "Models & Pred. FA & Pred. FI & Pred. CFA " + r"\\", r"\midrule",
    ]
    for group_name, _ in SIZE_GROUPS:
        values = []
        for label in ("FA", "FI", "CFA"):
            row = lookup.loc[(group_name, label)]
            score = format_extreme(row["mean_score"], maxima[label])
            values.append(
                f"{score} " + r"{\scriptsize[" +
                f"{row['ci_95_low']:.2f}, {row['ci_95_high']:.2f}" + r"]}"
            )
        lines.append(f"{group_name} & " + " & ".join(values) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Mean faithfulness scores (1--5) by model-size group and the model's classification of the input RDF triples, soft-weighted by classification vote share. Pred.\ denotes the predicted class. Brackets show 95\% cluster-bootstrap confidence intervals ({bootstrap_samples:,} resamples).}}",
        r"\label{tab:faithfulness-predicted-class-CI}", r"\end{table}",
    ])
    return "\n".join(lines)


def true_perceived_class_bootstrap(
    faith: pd.DataFrame,
    classification: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap macro-model means for the true x perceived class cells."""
    labels = ("FA", "FI", "CFA")
    weight_columns = [f"vote_share_{label}" for label in labels]
    rows = faith.merge(
        classification[CLASSIFICATION_KEYS + weight_columns],
        on=CLASSIFICATION_KEYS,
        how="inner",
        validate="one_to_one",
    )
    rows["true_label"] = rows["variant"].map(EXPECTED_LABEL)
    output: list[dict[str, Any]] = []

    for true_index, true_label in enumerate(labels):
        subset = rows[rows["true_label"] == true_label].copy()
        clusters = pd.MultiIndex.from_frame(
            subset[BOOTSTRAP_CLUSTER_KEYS].drop_duplicates().sort_values(BOOTSTRAP_CLUSTER_KEYS)
        )
        numerators: list[np.ndarray] = []
        denominators: list[np.ndarray] = []
        for perceived_label in labels:
            weight_column = f"vote_share_{perceived_label}"
            subset["_weighted_score"] = subset["score"] * subset[weight_column]
            grouped = (
                subset.groupby(BOOTSTRAP_CLUSTER_KEYS + ["model"])
                .agg(numerator=("_weighted_score", "sum"), denominator=(weight_column, "sum"))
                .reset_index()
            )
            numerator = (
                grouped.pivot(index=BOOTSTRAP_CLUSTER_KEYS, columns="model", values="numerator")
                .reindex(index=clusters, columns=MODEL_ORDER, fill_value=0)
                .fillna(0)
                .to_numpy()
            )
            denominator = (
                grouped.pivot(index=BOOTSTRAP_CLUSTER_KEYS, columns="model", values="denominator")
                .reindex(index=clusters, columns=MODEL_ORDER, fill_value=0)
                .fillna(0)
                .to_numpy()
            )
            numerators.append(numerator)
            denominators.append(denominator)
        # cluster x model x perceived class
        numerator_array = np.stack(numerators, axis=2)
        denominator_array = np.stack(denominators, axis=2)
        observed = np.mean(
            numerator_array.sum(axis=0) / denominator_array.sum(axis=0), axis=0
        )

        rng = np.random.default_rng(seed + true_index)
        boot = np.empty((samples, len(labels)))
        position = 0
        while position < samples:
            batch_size = min(100, samples - position)
            sampled_indices = rng.integers(
                len(clusters), size=(batch_size, len(clusters))
            )
            sampled_numerators = numerator_array[sampled_indices].sum(axis=1)
            sampled_denominators = denominator_array[sampled_indices].sum(axis=1)
            boot[position:position + batch_size] = np.mean(
                sampled_numerators / sampled_denominators, axis=1
            )
            position += batch_size

        for perceived_index, perceived_label in enumerate(labels):
            lower, upper = np.quantile(boot[:, perceived_index], [0.025, 0.975])
            output.append({
                "true_label": true_label,
                "predicted_label": perceived_label,
                "mean_score": float(observed[perceived_index]),
                "ci_95_low": float(lower),
                "ci_95_high": float(upper),
                "effective_count": float(subset[f"vote_share_{perceived_label}"].sum()),
                "source_item_clusters": len(clusters),
                "bootstrap_samples": samples,
            })
    return pd.DataFrame(output)


def true_perceived_class_table(cells: pd.DataFrame) -> str:
    """Render the joint true x model-perceived appendix table."""
    lookup = cells.set_index(["true_label", "predicted_label"])
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccc}",
        r"\toprule", r"True class & Pred.\ FA & Pred.\ FI & Pred.\ CFA " + r"\\",
        r"\midrule",
    ]
    for row_index, true_label in enumerate(("FA", "FI", "CFA")):
        scores = []
        counts = []
        for perceived_label in ("FA", "FI", "CFA"):
            row = lookup.loc[(true_label, perceived_label)]
            scores.append(
                f"{row['mean_score']:.2f} " + r"{\scriptsize [" +
                f"{row['ci_95_low']:.2f}, {row['ci_95_high']:.2f}" + r"]}"
            )
            count = f"{round(row['effective_count']):,}".replace(",", "{,}")
            counts.append(r"{\scriptsize " + count + "}")
        lines.append(true_label + " & " + " & ".join(scores) + r" \\")
        lines.append("  & " + " & ".join(counts) + r" \\")
        if row_index < 2:
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Mean faithfulness by ground-truth and model-perceived input class. Scores are macro-averaged across models and soft-weighted by the five classification votes; brackets show 95\% cluster-bootstrap confidence intervals. The smaller numbers below the scores are effective counts, pooling the corresponding fractional vote weights.}",
        r"\label{tab:faithfulness-true-perceived}", r"\end{table}",
    ])
    return "\n".join(lines)


def prompt_language_conditions() -> dict[str, Any]:
    """Return the shared EN/Same/Other/HSB prompt-language predicates."""
    return {
        "EN": lambda frame: frame["language"] == "en",
        "Same": lambda frame: ((frame["dataset"] == "cs-qa") & (frame["language"] == "cs"))
                              | ((frame["dataset"] == "sk-qa") & (frame["language"] == "sk")),
        "Other": lambda frame: ((frame["dataset"] == "cs-qa") & (frame["language"] == "sk"))
                               | ((frame["dataset"] == "sk-qa") & (frame["language"] == "cs")),
        "HSB": lambda frame: frame["language"] == "hsb",
    }


def language_condition_table(faith: pd.DataFrame, fluency: pd.DataFrame) -> str:
    """Compare English, local-language, cross-local, and HSB prompts."""
    conditions = prompt_language_conditions()

    def means(frame: pd.DataFrame) -> pd.DataFrame:
        rows = {}
        for group_name, models in SIZE_GROUPS:
            subset = frame[frame["model"].isin(models)]
            rows[group_name] = {
                condition: subset.loc[mask(subset), "score"].mean()
                for condition, mask in conditions.items()
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    faith_means = means(faith)
    fluency_means = means(fluency)
    faith_maxima = faith_means.max(axis=0)
    fluency_maxima = fluency_means.max(axis=0)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule", r"Models & EN & Same & Other & HSB \\", r"\midrule",
    ]
    for group_name, _ in SIZE_GROUPS:
        cells = [f"{format_extreme(faith_means.loc[group_name, condition], faith_maxima[condition])} / "
                 f"{format_extreme(fluency_means.loc[group_name, condition], fluency_maxima[condition])}"
                 for condition in conditions]
        lines.append(group_name + " & " + " & ".join(cells) + r" \\")
    pooled = []
    for condition, mask in conditions.items():
        pooled.append(f"{faith.loc[mask(faith), 'score'].mean():.2f} / "
                      f"{fluency.loc[mask(fluency), 'score'].mean():.2f}")
    lines.extend([
        r"\midrule", "Overall & " + " & ".join(pooled) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Faithfulness and fluency by prompt language and model-size group. Each cell shows mean Faithfulness\ / Fluency LLM judge scores on a 1--5 scale. EN and HSB pool all data. ``Same'' pools Czech local data with Czech prompts and Slovak data with Slovak prompts; ``Other'' pools the opposite pairings.}",
        r"\label{tab:language-condition-results}", r"\end{table}",
    ])
    return "\n".join(lines)


def issue_language_conditions() -> tuple[tuple[str, Any], ...]:
    """Return the prompt-language groupings used by the language table."""
    return tuple(prompt_language_conditions().items())


def issue_category_values(subset: pd.DataFrame) -> list[str]:
    """Format the six non-exclusive issue percentages as separate values."""
    denominator = len(subset)
    if not denominator:
        return ["--"] * len(ISSUE_CATEGORY_NAMES)
    values = []
    for category in ISSUE_CATEGORY_NAMES:
        count = int(subset["issue_categories"].map(lambda labels: category in labels).sum())
        values.append(f"{100 * count / denominator:.1f}")
    return values


def attach_issue_categories(rows: pd.DataFrame, issue_examples: pd.DataFrame) -> pd.DataFrame:
    """Attach non-exclusive issue labels while retaining all judged outputs."""
    merged = rows.merge(
        issue_examples[CLASSIFICATION_KEYS + ["issue_categories"]],
        on=CLASSIFICATION_KEYS,
        how="left",
        validate="one_to_one",
    )
    merged["issue_categories"] = merged["issue_categories"].map(
        lambda labels: labels if isinstance(labels, list) else []
    )
    return merged


def issue_grouped_category_table(
    rows_frame: pd.DataFrame,
    groups: tuple[tuple[str, Any], ...],
    *,
    row_header: str,
    caption: str,
    label: str,
    include_overall: bool = True,
    include_n: bool = True,
) -> str:
    """Render non-exclusive issue incidence over all outputs in each group."""
    table_rows: list[str] = []
    for name, mask in groups:
        subset = rows_frame[mask(rows_frame)]
        values = issue_category_values(subset)
        if include_n:
            values.append(f"{len(subset):,}".replace(",", "{,}"))
        table_rows.append(name + " & " + " & ".join(values) + r" \\")

    if include_overall:
        overall_values = issue_category_values(rows_frame)
        if include_n:
            overall_values.append(f"{len(rows_frame):,}".replace(",", "{,}"))
        table_rows.extend([r"\midrule", "Overall & " + " & ".join(overall_values) + r" \\"])

    column_spec = "l" + "r" * len(ISSUE_COLUMNS) + ("r" if include_n else "")
    header = row_header + " & " + " & ".join(header for _, header in ISSUE_COLUMNS)
    if include_n:
        header += r" & $N$"
    header += r" \\"
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])


def issue_language_category_table(faith: pd.DataFrame, issue_examples: pd.DataFrame) -> str:
    rows = attach_issue_categories(faith, issue_examples)
    group_sizes = [len(rows[mask(rows)]) for _, mask in issue_language_conditions()]
    if len(set(group_sizes)) != 1:
        raise ValueError(f"Prompt-language conditions are not balanced: {group_sizes}")
    table_rows = []
    for name, mask in issue_language_conditions():
        table_rows.append(name + " & " + " & ".join(issue_category_values(rows[mask(rows)])) + r" \\")
    count_text = f"{group_sizes[0]:,}"
    return "\n".join([
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        "Condition & " + " & ".join(header for _, header in ISSUE_COLUMNS) + r" \\",
        r"\midrule", *table_rows, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Error-category incidence (\% of all judged outputs) by prompt-language condition; categories are non-exclusive. Each condition has {count_text} outputs.}}",
        r"\label{tab:faithfulness-error-types-language}", r"\end{table}",
    ])


def issue_variant_category_table(faith: pd.DataFrame, issue_examples: pd.DataFrame) -> str:
    rows = attach_issue_categories(faith, issue_examples)
    groups = tuple(
        (label, lambda frame, value=value: frame["variant"] == value)
        for label, value in (("FA", "fa"), ("FI", "fi"), ("CFA", "cf"))
    )
    return issue_grouped_category_table(
        rows,
        groups,
        row_header="Variant",
        caption=(
            "Error-category incidence by source variant. Values are percentages of all judged outputs; "
            "categories are non-exclusive. $N$ is the number of outputs."
        ),
        label="tab:faithfulness-error-types-variant",
    )


def issue_predicted_size_all_category_table(
    faith: pd.DataFrame,
    issue_examples: pd.DataFrame,
    classification: pd.DataFrame,
) -> str:
    """Render issue incidence over all outputs by size group and predicted class."""
    rows = faith.merge(
        classification[CLASSIFICATION_KEYS + ["predicted_label"]],
        on=CLASSIFICATION_KEYS,
        how="inner",
        validate="one_to_one",
    )
    rows = rows[rows["predicted_label"].isin(LABELS)].copy()
    rows = attach_issue_categories(rows, issue_examples)

    table_rows: list[str] = []
    for group_index, (group_name, models) in enumerate(SIZE_GROUPS):
        for label_index, predicted_label in enumerate(("FA", "FI", "CFA")):
            subset = rows[
                rows["model"].isin(models) & (rows["predicted_label"] == predicted_label)
            ]
            denominator = len(subset)
            values = []
            for category in ISSUE_CATEGORY_NAMES:
                count = int(subset["issue_categories"].map(lambda labels: category in labels).sum())
                values.append(f"{100 * count / denominator:.1f}" if denominator else "--")
            count_text = f"{denominator:,}".replace(",", "{,}")
            group_cell = rf"\multirow{{3}}{{*}}{{{group_name}}}" if label_index == 0 else ""
            table_rows.append(
                f"{group_cell} & {predicted_label} & "
                + " & ".join(values) + f" & {count_text} " + r"\\"
            )
        if group_index < len(SIZE_GROUPS) - 1:
            table_rows.append(r"\hdashline[0.5pt/2pt]")

    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        "Models & Class & " + " & ".join(header for _, header in ISSUE_COLUMNS) + r" & $N$ \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Error-category incidence (\% outputs) by predicted class and model size; categories are non-exclusive. $N$ is the number of instances (unequal values reflect model prediction distributions, see Table~\ref{tab:model-results}).}",
        r"\label{tab:faithfulness-error-types-predicted-size-all}",
        r"\end{table}",
    ])


def variant_table(faith: pd.DataFrame) -> str:
    group_means = {}
    group_overall = {}
    for group_name, models in SIZE_GROUPS:
        subset = faith[faith["model"].isin(models)]
        group_means[group_name] = subset.groupby("variant")["score"].mean()
        group_overall[group_name] = subset["score"].mean()
    means = pd.DataFrame(group_means).T
    overall = pd.Series(group_overall)
    pooled = faith.groupby("variant")["score"].mean()
    maxima = means.max(axis=0)
    overall_max = overall.max()
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{lcccc}",
        r"\toprule", r"Model size & FA & FI & CFA & Overall \\", r"\midrule",
    ]
    for group_name, _ in SIZE_GROUPS:
        lines.append(rf"{group_name} & "
                     rf"{format_extreme(means.loc[group_name, 'fa'], maxima['fa'])} & "
                     rf"{format_extreme(means.loc[group_name, 'fi'], maxima['fi'])} & "
                     rf"{format_extreme(means.loc[group_name, 'cf'], maxima['cf'])} & "
                     rf"{format_extreme(overall.loc[group_name], overall_max)} \\")
    lines.extend([
        r"\midrule",
        rf"Overall & {pooled['fa']:.2f} & {pooled['fi']:.2f} & {pooled['cf']:.2f} & {faith['score'].mean():.2f} \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Mean faithfulness scores (1--5) by true data variant and model-size group. FA denotes factual inputs, FI fictional inputs, and CFA counterfactual inputs. Scores pool both datasets and all four prompt languages.}",
        r"\label{tab:faithfulness-variants}", r"\end{table}",
    ])
    return "\n".join(lines)


def load_issue_examples(data_dir: Path) -> pd.DataFrame:
    """Load the current non-exclusive issue annotations from the report CSV."""
    path = data_dir / "reports" / "issue_examples" / "all_issue_examples.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Issue report not found: {path}. Regenerate reports with inspect_judged_results.py first."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"issue_categories", "faithfulness_score", "generator_model", "dataset", "variant", "language", "eid"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Issue report {path} is missing columns: {sorted(missing)}")
    frame["model"] = frame["generator_model"].map(canonical_model)
    frame["score"] = pd.to_numeric(frame["faithfulness_score"], errors="coerce")
    frame["issue_categories"] = frame["issue_categories"].str.split("; ")
    frame = frame[frame["score"].isin([1, 2, 3, 4])].copy()
    # A judged row is identified by its source item and prompt language.
    frame = frame.drop_duplicates(["model", "dataset", "variant", "language", "eid"])
    return frame


def issue_category_table(
    faith: pd.DataFrame,
    issue_examples: pd.DataFrame,
    display: str = "percentage",
) -> str:
    """Render issue incidence by model over all judged outputs."""
    if display not in {"percentage", "count"}:
        raise ValueError(f"Unsupported issue table format: {display}")

    all_rows = attach_issue_categories(faith, issue_examples)
    rows: list[str] = []
    for model in [*MODEL_ORDER, "__all__"]:
        subset = all_rows if model == "__all__" else all_rows[all_rows["model"] == model]
        denominator = len(subset)
        values: list[str] = []
        for category in ISSUE_CATEGORY_NAMES:
            count = int(subset["issue_categories"].map(lambda labels: category in labels).sum())
            if display == "percentage":
                values.append(f"{100 * count / denominator:.1f}" if denominator else "--")
            else:
                values.append(str(count))
        values.append(f"{denominator:,}".replace(",", "{,}"))
        if model == "__all__":
            rows.append(r"\midrule")
        label = "All models" if model == "__all__" else rf"\texttt{{{MODEL_SHORT[model]}}}"
        rows.append(label + " & " + " & ".join(values) + r" \\")
        if model != "__all__" and model in SIZE_GROUP_ENDS:
            rows.append(r"\hdashline[0.5pt/2pt]")

    format_note = (
        "Values are percentages of all judged outputs"
        if display == "percentage"
        else "Counts are non-exclusive: one example may contribute to multiple categories"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Model & " + " & ".join(header for _, header in ISSUE_COLUMNS) + r" & $N$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Error-category incidence by generator model. {format_note}; categories are non-exclusive. $N$ is the number of outputs.}}",
        r"\label{tab:faithfulness-error-types}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def model_cluster_bootstrap(scores: pd.DataFrame, metric: str, samples: int,
                            rng: np.random.Generator) -> list[dict[str, Any]]:
    """Estimate CIs for model means by resampling source RDF-item clusters.

    Each cluster is one dataset x variant x eid item. Its language-specific
    observations remain together, avoiding an independence assumption between
    several prompts derived from the same source RDF input.
    """
    output: list[dict[str, Any]] = []
    for model in [model for model in MODEL_ORDER if model in set(scores["model"])]:
        model_scores = scores[scores["model"] == model]
        if model_scores.empty:
            continue
        blocks = [group["score"].to_numpy()
                  for _, group in model_scores.groupby(["dataset", "variant", "eid"])]
        observed = float(np.concatenate(blocks).mean())
        boot = np.empty(samples)
        for sample_index in range(samples):
            sampled = [blocks[index] for index in rng.integers(len(blocks), size=len(blocks))]
            boot[sample_index] = np.concatenate(sampled).mean()
        lower = float(np.quantile(boot, 0.025))
        upper = float(np.quantile(boot, 0.975))
        output.append({"metric": metric, "model": model,
                       "source_item_clusters": len(blocks), "judged_rows": len(model_scores),
                       "mean_score": observed, "ci_95_low": lower, "ci_95_high": upper,
                       "ci_95_half_width": max(observed - lower, upper - observed)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <data-dir>/results_tables).")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    # The broader per-model/per-variant issue-table bundle is not used in the
    # current article, so its --issue-format option and generation calls are
    # intentionally disabled.
    # parser.add_argument("--issue-format", choices=("percentage", "count"), default="percentage")
    args = parser.parse_args()
    output = args.output or args.data_dir / "results_tables"
    output.mkdir(parents=True, exist_ok=True)

    generated = load_generated(args.data_dir)
    faith = load_judgments(args.data_dir, "judged", "faithfulness_score")
    fluency = load_judgments(args.data_dir, "judged_fluency", "fluency_score")
    classification = load_classification(args.data_dir)
    issue_examples = load_issue_examples(args.data_dir)
    if generated.empty or faith.empty or fluency.empty or classification.empty:
        raise SystemExit("Required v3 generated, judged, judged_fluency, or classified data is empty.")

    model_cis = model_cluster_bootstrap(faith, "faithfulness", args.bootstrap_samples,
                                        np.random.default_rng(args.seed))
    model_cis += model_cluster_bootstrap(fluency, "fluency", args.bootstrap_samples,
                                         np.random.default_rng(args.seed + 1))
    model_cis_frame = pd.DataFrame(model_cis)
    perceived_cells, perceived_contrasts = perceived_class_bootstrap(
        faith, classification, args.bootstrap_samples, args.seed + 2
    )
    true_perceived_cells = true_perceived_class_bootstrap(
        faith, classification, args.bootstrap_samples, args.seed + 5
    )
    perceived_ci_latex = perceived_class_ci_table(perceived_cells)
    true_perceived_latex = true_perceived_class_table(true_perceived_cells)
    model_latex = model_table(faith, fluency, classification)
    language_latex = language_condition_table(faith, fluency)
    issue_predicted_size_all_latex = issue_predicted_size_all_category_table(
        faith, issue_examples, classification
    )
    triple_latex = triple_table(faith, generated)
    model_ci_latex = model_ci_table(faith, fluency, model_cis_frame)
    variant_latex = variant_table(faith)
    predicted_class_latex = predicted_class_table(faith, classification)
    issue_language_latex = issue_language_category_table(faith, issue_examples)

    (output / "faithfulness_predicted_class_ci.tex").write_text(
        perceived_ci_latex + "\n", encoding="utf-8"
    )
    (output / "faithfulness_true_perceived.tex").write_text(
        true_perceived_latex + "\n", encoding="utf-8"
    )
    # These older outputs are intentionally not generated because neither the
    # standalone classification table/figure nor the broader issue tables occur
    # in article.tex.
    # figure_latex = classification_figure(classification)
    # issue_table = issue_category_table(faith, issue_examples)
    # issue_variant_latex = issue_variant_category_table(faith, issue_examples)

    # Keep the same order as the tables in article/article.tex.
    tables = "\n\n".join([
        model_latex,
        language_latex,
        perceived_ci_latex,
        issue_predicted_size_all_latex,
        triple_latex,
        model_ci_latex,
        variant_latex,
        true_perceived_latex,
        predicted_class_latex,
        issue_language_latex,
    ]) + "\n"
    (output / "results_tables.tex").write_text(tables, encoding="utf-8")
    (output / "results_coverage.csv").unlink(missing_ok=True)
    model_cis_frame.to_csv(output / "bootstrap_model_cis.csv", index=False, float_format="%.6f")
    perceived_cells.to_csv(
        output / "bootstrap_perceived_class_cells.csv", index=False, float_format="%.6f"
    )
    perceived_contrasts.to_csv(
        output / "bootstrap_perceived_class_contrasts.csv", index=False, float_format="%.6f"
    )
    true_perceived_cells.to_csv(
        output / "bootstrap_true_perceived_cells.csv", index=False, float_format="%.6f"
    )
    (output / "bootstrap_model_comparisons.csv").unlink(missing_ok=True)
    (output / "results_summary.md").write_text(
        "# Results-table inputs\n\n"
        "The tables are generated from the complete v3 dataset. "
        "`bootstrap_model_cis.csv` contains 95% cluster-bootstrap confidence intervals for each model's mean score. "
        "`bootstrap_perceived_class_contrasts.csv` contains the six paired perceived-class contrasts, "
        "including Bonferroni-adjusted 95% family-wise intervals. "
        "A cluster is one source RDF item (dataset x variant x eid), retaining its available language-specific observations.\n",
        encoding="utf-8",
    )
    print(f"Wrote {output / 'results_tables.tex'}")
    print(f"Wrote {output / 'bootstrap_model_cis.csv'}")
    print(f"Wrote {output / 'bootstrap_perceived_class_contrasts.csv'}")


if __name__ == "__main__":
    main()
