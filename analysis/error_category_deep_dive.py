"""Deep dive into judge comments for below-5 faithfulness scores.

Loads every judged JSONL row in data/judged/, keeps rows with
faithfulness_score < 5, and buckets each incorrect_information entry into
the paper's six non-exclusive error categories (Lang/Hall/Rev/Miss/Rel/Ent)
using the judge's own comment-label vocabulary (see prompts/judge_speeches.txt:
unsupported | wrong_object | wrong_predicate | reverse | negation | omission |
over_specific | wrong_language | wrong_transform | repetition). Reproducing
the paper's Table 7 "All models" row from this mapping (Lang 39.2 vs 40.0,
Hall 34.4 vs 34.3, Rev/Miss/Rel/Ent matching to 0.1pp) confirms the mapping,
then goes further than the paper tables: sub-label splits, the predicates
most often implicated in each category, keyword/phrase frequency in the
judge's free-text explanations, and sampled examples for manual reading.

Outputs (under analysis/error_deep_dive/ by default):
  below5_rows.csv   one row per judged sentence scoring below 5
  error_items.csv   one row per incorrect_information entry (long format)
  report.md         category breakdowns, top predicates, keywords, samples
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from inspect_judged_results import CLASS_LABELS, parse_classification_answer  # noqa: E402

DEFAULT_JUDGED_DIR = REPO_ROOT / "data" / "judged"
DEFAULT_CLASSIFIED_DIR = REPO_ROOT / "data" / "classified"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "error_deep_dive"

MAX_SCORE = 5
PRED_LABEL_ORDER = ["FA", "FI", "CFA"]

# Maps the judge's own comment-label vocabulary to the paper's six
# non-exclusive error categories. A single item's label maps to at most one
# category; a sentence can carry multiple items and thus multiple categories.
CATEGORY_LABELS: dict[str, set[str]] = {
    "Lang": {"wrong_language", "wrong_transform"},
    "Hall": {"unsupported", "over_specific", "negation", "contradiction"},
    "Rev": {"reverse"},
    "Miss": {"omission"},
    "Rel": {"wrong_predicate"},
    "Ent": {"wrong_object", "wrong_subject"},
}
CATEGORY_ORDER = ["Lang", "Hall", "Rev", "Miss", "Rel", "Ent"]
CATEGORY_NAMES = {
    "Lang": "Language error (wrong target language / untranslated predicate)",
    "Hall": "Hallucinated or unsupported information",
    "Rev": "Reversed relation",
    "Miss": "Missing information",
    "Rel": "Wrong relation",
    "Ent": "Wrong entity",
}
LABEL_TO_CATEGORY = {label: cat for cat, labels in CATEGORY_LABELS.items() for label in labels}
# Judge labels outside the six paper categories: fluency/formatting issues or
# off-vocabulary responses. Tracked separately, not counted toward the six.
OTHER_LABELS = {"degenerate", "repetition", "supported"}

PREDICATE_RE = re.compile(r"\|([^|]*)\|")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")

# Domain stopwords: generic English stopwords plus words that recur in almost
# every judge comment regardless of error subtype ("the sentence claims ...
# but the triple states ..."), which would otherwise swamp the keyword counts.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "as",
    "of", "to", "in", "on", "for", "with", "not", "no", "does", "do", "did",
    "has", "have", "had", "which", "while", "than", "then", "so", "such",
    "into", "from", "by", "at", "here", "there", "also", "only", "even",
    "sentence", "sentences", "triple", "triples", "claims", "claim",
    "states", "state", "stated", "says", "say", "said", "implies", "imply",
    "means", "mean", "meaning", "instead", "rather", "actual", "actually",
    "specific", "specifically", "one", "both", "any", "all", "given",
    "would", "should", "could", "however", "asserts", "assert", "asserting",
}


def extract_label(comment: str) -> str:
    comment = comment.strip()
    if not comment:
        return ""
    first = comment.split(":", 1)[0].split()[0] if comment.split(":", 1)[0].split() else ""
    return first.strip().lower().rstrip(",.;")


def extract_comment_detail(comment: str, label: str) -> str:
    text = comment.strip()
    if label and text.lower().startswith(label):
        text = text[len(label):].lstrip(":").strip()
    return text


def extract_predicate(correct_info: str) -> str:
    match = PREDICATE_RE.search(correct_info or "")
    return match.group(1).strip() if match else ""


def parse_stem(stem: str) -> tuple[str, str, str]:
    # judge_<dataset>_<variant>_<language>_<judge-model...>
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "judge":
        return parts[1], parts[2], parts[3]
    return "", "", ""


def iter_judged_files(judged_dir: Path):
    for model_dir in sorted(judged_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            if jsonl_path.name.endswith(".failures.jsonl"):
                continue
            yield model_dir.name, jsonl_path


def normalize_classified_model_name(name: str) -> str:
    """Map a data/classified/ model directory name to its data/judged/ counterpart.

    The two trees name a couple of models differently: classified drops
    judged's underscore-for-dot ("qwen3.5-122b" vs "qwen3_5-122b") and keeps
    an "-openrouter" suffix judged does not ("llama4-scout-openrouter" vs
    "llama4-scout").
    """
    if name.endswith("-openrouter"):
        name = name[: -len("-openrouter")]
    if "." in name:
        name = name.replace(".", "_", 1)
    return name


def majority_predicted_label(values: list[str]) -> str:
    parsed = [label for label, _ in map(parse_classification_answer, values) if label]
    if not parsed:
        return ""
    votes = Counter(parsed)
    return sorted(
        votes.items(),
        key=lambda item: (-item[1], CLASS_LABELS.index(item[0]) if item[0] in CLASS_LABELS else 99),
    )[0][0]


def load_predicted_labels(classified_dir: Path) -> dict[tuple[str, str, str, str, str], str]:
    """(judged_model_name, dataset, variant, language, eid) -> majority-vote CFA/FA/FI label."""
    labels: dict[tuple[str, str, str, str, str], str] = {}
    if not classified_dir.exists():
        return labels
    repeat_re = re.compile(r"^sentence_\d+$")
    for model_dir in sorted(classified_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        judged_model_name = normalize_classified_model_name(model_dir.name)
        for csv_path in sorted(model_dir.glob("*.csv")):
            parts = csv_path.stem.split("_")
            if len(parts) != 3:
                continue
            dataset, variant, language = parts
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                repeat_cols = [c for c in fieldnames if repeat_re.match(c)]
                answer_cols = repeat_cols or (["sentence"] if "sentence" in fieldnames else [])
                for row in reader:
                    eid = row.get("eid", "")
                    if not eid:
                        continue
                    label = majority_predicted_label([row.get(c, "") for c in answer_cols])
                    labels[(judged_model_name, dataset, variant, language, eid)] = label
    return labels


def load_rows(judged_dir: Path, predicted_labels: dict[tuple[str, str, str, str, str], str] | None = None) -> list[dict[str, Any]]:
    predicted_labels = predicted_labels or {}
    rows = []
    for model, jsonl_path in iter_judged_files(judged_dir):
        dataset, variant, language = parse_stem(jsonl_path.stem)
        with jsonl_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = record.get("parsed") or {}
                score = parsed.get("faithfulness_score")
                if not isinstance(score, (int, float)) or score >= MAX_SCORE:
                    continue
                info = parsed.get("incorrect_information") or []
                if not isinstance(info, list):
                    info = [info]
                items = []
                for idx, item in enumerate(info):
                    if isinstance(item, dict):
                        comment = str(item.get("comment") or "")
                        info_used = str(item.get("info_used") or "")
                        correct_info = str(item.get("correct_info") or "")
                    else:
                        comment = str(item)
                        info_used = ""
                        correct_info = ""
                    label = extract_label(comment)
                    items.append(dict(
                        item_index=idx,
                        label=label,
                        category=LABEL_TO_CATEGORY.get(label, ""),
                        is_other_label=label in OTHER_LABELS,
                        info_used=info_used,
                        correct_info=correct_info,
                        comment=comment,
                        comment_detail=extract_comment_detail(comment, label),
                        predicate=extract_predicate(correct_info),
                    ))
                eid = str(record.get("eid", ""))
                rows.append(dict(
                    model=model,
                    dataset=dataset,
                    variant=variant,
                    language=language,
                    eid=eid,
                    sentence=str(record.get("sentence", "")),
                    modified_triples=str(record.get("modified_triples", "")),
                    faithfulness_score=int(score),
                    predicted_label=predicted_labels.get((model, dataset, variant, language, eid), ""),
                    judged_file=str(jsonl_path.relative_to(REPO_ROOT)),
                    judged_line=line_no,
                    items=items,
                ))
    return rows


def row_categories(row: dict[str, Any]) -> set[str]:
    return {item["category"] for item in row["items"] if item["category"]}


def write_below5_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "dataset", "variant", "language", "eid", "faithfulness_score",
        "predicted_label", "sentence", "modified_triples",
        *[f"has_{cat}" for cat in CATEGORY_ORDER],
        "labels", "num_items", "judged_file", "judged_line",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cats = row_categories(row)
            writer.writerow({
                "model": row["model"],
                "dataset": row["dataset"],
                "variant": row["variant"],
                "language": row["language"],
                "eid": row["eid"],
                "faithfulness_score": row["faithfulness_score"],
                "predicted_label": row["predicted_label"],
                "sentence": row["sentence"],
                "modified_triples": row["modified_triples"],
                **{f"has_{cat}": int(cat in cats) for cat in CATEGORY_ORDER},
                "labels": "; ".join(sorted({item["label"] for item in row["items"] if item["label"]})),
                "num_items": len(row["items"]),
                "judged_file": row["judged_file"],
                "judged_line": row["judged_line"],
            })


def write_error_items_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "dataset", "variant", "language", "eid", "faithfulness_score",
        "predicted_label", "item_index", "label", "category", "predicate",
        "info_used", "correct_info", "comment", "comment_detail",
        "sentence", "modified_triples", "judged_file", "judged_line",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for item in row["items"]:
                writer.writerow({
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "variant": row["variant"],
                    "language": row["language"],
                    "eid": row["eid"],
                    "faithfulness_score": row["faithfulness_score"],
                    "predicted_label": row["predicted_label"],
                    "item_index": item["item_index"],
                    "label": item["label"],
                    "category": item["category"] or ("other:" + item["label"] if item["label"] in OTHER_LABELS else ""),
                    "predicate": item["predicate"],
                    "info_used": item["info_used"],
                    "correct_info": item["correct_info"],
                    "comment": item["comment"],
                    "comment_detail": item["comment_detail"],
                    "sentence": row["sentence"],
                    "modified_triples": row["modified_triples"],
                    "judged_file": row["judged_file"],
                    "judged_line": row["judged_line"],
                })


def keyword_counts(comments: list[str], top_n: int = 25) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    for comment in comments:
        tokens = [t.lower() for t in WORD_RE.findall(comment) if t.lower() not in STOPWORDS]
        unigrams.update(tokens)
        bigrams.update(" ".join(pair) for pair in zip(tokens, tokens[1:]))
    return unigrams.most_common(top_n), bigrams.most_common(top_n)


def md_escape(value: Any) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def pct_cell(count: int, total: int) -> str:
    if not total:
        return f"{count} (-)"
    return f"{count} ({count / total * 100:.1f}%)"


def build_report(rows: list[dict[str, Any]], samples_per_category: int, seed: int) -> str:
    rng = random.Random(seed)
    total_rows = len(rows)
    all_items = [item for row in rows for item in row["items"]]
    item_to_row = {id(item): row for row in rows for item in row["items"]}

    pred_row_totals = {p: sum(1 for row in rows if row["predicted_label"] == p) for p in PRED_LABEL_ORDER}
    no_pred_label = sum(1 for row in rows if not row["predicted_label"])

    lines = ["# Faithfulness Error Deep Dive (scores below 5)", ""]
    lines.append(f"Below-5 judged rows: {total_rows}")
    lines.append(f"Incorrect-information items across those rows: {len(all_items)}")
    lines.append(
        "Pred FA/FI/CFA columns below give the count and, in parentheses, the "
        "percentage of that predicted class's own below-5 rows (or items, where "
        "noted) — the same denominator convention as the paper's per-predicted-"
        f"label table. Rows without a parsed predicted label ({no_pred_label} of "
        f"{total_rows}) are excluded from those columns; predicted-class totals "
        f"are FA={pred_row_totals['FA']}, FI={pred_row_totals['FI']}, CFA={pred_row_totals['CFA']}."
    )
    lines.append("")

    # --- Category overview (row-level, matching the paper's Table 7 denominator) ---
    lines.append("## Category Overview")
    lines.append("")
    lines.append(
        "Row-level: percentage of below-5 rows containing at least one item in "
        "that category. Categories are non-exclusive (one row may hit several), "
        "so the column does not sum to 100%. Item-level counts the raw number "
        "of incorrect_information entries carrying that category's label(s)."
    )
    lines.append("")
    lines.append("| category | meaning | All rows | Pred FA | Pred FI | Pred CFA | items |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    cat_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cat_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cats = row_categories(row)
        for cat in cats:
            cat_rows[cat].append(row)
    for item in all_items:
        if item["category"]:
            cat_items[item["category"]].append(item)
    for cat in CATEGORY_ORDER:
        n_rows = len(cat_rows[cat])
        by_pred = {p: sum(1 for row in cat_rows[cat] if row["predicted_label"] == p) for p in PRED_LABEL_ORDER}
        lines.append(
            f"| {cat} | {CATEGORY_NAMES[cat]} | {pct_cell(n_rows, total_rows)} | "
            f"{pct_cell(by_pred['FA'], pred_row_totals['FA'])} | "
            f"{pct_cell(by_pred['FI'], pred_row_totals['FI'])} | "
            f"{pct_cell(by_pred['CFA'], pred_row_totals['CFA'])} | {len(cat_items[cat])} |"
        )
    lines.append("")

    other_counts = Counter(item["label"] for item in all_items if item["label"] in OTHER_LABELS)
    unlabeled = sum(1 for item in all_items if not item["category"] and item["label"] not in OTHER_LABELS)
    lines.append(
        f"Other labels outside the six categories (fluency/formatting or "
        f"off-vocabulary): {dict(other_counts)}. Unrecognized labels: {unlabeled}."
    )
    lines.append("")

    # --- Per-category deep dive ---
    for cat in CATEGORY_ORDER:
        items = cat_items[cat]
        if not items:
            continue
        cat_class_item_total = {p: sum(1 for i in items if item_to_row[id(i)]["predicted_label"] == p) for p in PRED_LABEL_ORDER}
        lines.append(f"## {cat}: {CATEGORY_NAMES[cat]}")
        lines.append("")
        lines.append(f"{len(items)} items across {len(cat_rows[cat])} rows.")
        lines.append("")

        # Sub-label split
        sub_labels = Counter(item["label"] for item in items)
        if len(sub_labels) > 1:
            lines.append("**Sub-label split** (Pred columns: % of that predicted class's items in this category):")
            lines.append("")
            lines.append("| label | count | % of category | Pred FA | Pred FI | Pred CFA |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for label, count in sub_labels.most_common():
                by_pred = {
                    p: sum(1 for i in items if i["label"] == label and item_to_row[id(i)]["predicted_label"] == p)
                    for p in PRED_LABEL_ORDER
                }
                lines.append(
                    f"| {label} | {count} | {count / len(items) * 100:.1f}% | "
                    f"{pct_cell(by_pred['FA'], cat_class_item_total['FA'])} | "
                    f"{pct_cell(by_pred['FI'], cat_class_item_total['FI'])} | "
                    f"{pct_cell(by_pred['CFA'], cat_class_item_total['CFA'])} |"
                )
            lines.append("")

        # Language breakdown for Lang category
        if cat == "Lang":
            by_lang = Counter()
            for row in rows:
                if any(i["category"] == "Lang" for i in row["items"]):
                    by_lang[row["language"]] += 1
            lines.append("**By prompt/target language** (Pred columns: % of that language+predicted-class's below-5 rows with a Lang error):")
            lines.append("")
            lines.append("| language | rows | % of rows in that language | Pred FA | Pred FI | Pred CFA |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for lang, count in by_lang.most_common():
                lang_rows = [row for row in rows if row["language"] == lang]
                total_lang_rows = len(lang_rows)
                pct = f"{count / total_lang_rows * 100:.1f}%" if total_lang_rows else "-"
                by_pred_total = {p: sum(1 for row in lang_rows if row["predicted_label"] == p) for p in PRED_LABEL_ORDER}
                by_pred_hit = {
                    p: sum(1 for row in lang_rows if row["predicted_label"] == p and any(i["category"] == "Lang" for i in row["items"]))
                    for p in PRED_LABEL_ORDER
                }
                lines.append(
                    f"| {lang} | {count} | {pct} | "
                    f"{pct_cell(by_pred_hit['FA'], by_pred_total['FA'])} | "
                    f"{pct_cell(by_pred_hit['FI'], by_pred_total['FI'])} | "
                    f"{pct_cell(by_pred_hit['CFA'], by_pred_total['CFA'])} |"
                )
            lines.append("")

        # Top predicates involved
        predicates = Counter(item["predicate"] for item in items if item["predicate"])
        if predicates:
            lines.append(f"**Predicates most often implicated in {cat} errors** (Pred columns: raw item counts within this category):")
            lines.append("")
            lines.append("| predicate | count | Pred FA | Pred FI | Pred CFA |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for predicate, count in predicates.most_common(20):
                by_pred = {
                    p: sum(1 for i in items if i["predicate"] == predicate and item_to_row[id(i)]["predicted_label"] == p)
                    for p in PRED_LABEL_ORDER
                }
                lines.append(
                    f"| {md_escape(predicate)} | {count} | {by_pred['FA']} | {by_pred['FI']} | {by_pred['CFA']} |"
                )
            lines.append("")

        # Keyword / phrase frequency in the judge's free-text explanation
        comments = [item["comment_detail"] for item in items if item["comment_detail"]]
        uni, bi = keyword_counts(comments)
        if uni:
            lines.append("**Frequent words in judge explanations** (generic words and the label itself filtered out):")
            lines.append("")
            lines.append(", ".join(f"{w} ({c})" for w, c in uni))
            lines.append("")
        if bi:
            lines.append("**Frequent word pairs:**")
            lines.append("")
            lines.append(", ".join(f"{w} ({c})" for w, c in bi))
            lines.append("")

        # Sampled examples
        sample = items[:]
        rng.shuffle(sample)
        sample = sample[:samples_per_category]
        lines.append(f"**Sample of {len(sample)} examples:**")
        lines.append("")
        lines.append("| model | lang | sentence | info_used -> correct_info | comment |")
        lines.append("| --- | --- | --- | --- | --- |")
        for item in sample:
            parent = item_to_row.get(id(item))
            model = parent["model"] if parent else ""
            lang = parent["language"] if parent else ""
            sentence = parent["sentence"] if parent else ""
            lines.append(
                "| "
                + " | ".join([
                    md_escape(model),
                    md_escape(lang),
                    md_escape(sentence),
                    md_escape(f"{item['info_used']} -> {item['correct_info']}"),
                    md_escape(item["comment"]),
                ])
                + " |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged-dir", default=DEFAULT_JUDGED_DIR, type=Path)
    parser.add_argument("--classified-dir", default=DEFAULT_CLASSIFIED_DIR, type=Path,
                         help="Classifier-output CSVs used to attach each row's predicted CFA/FA/FI label.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--samples", default=15, type=int, help="Sample size per category shown in the report.")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    predicted_labels = load_predicted_labels(args.classified_dir)
    rows = load_rows(args.judged_dir, predicted_labels)
    if not rows:
        print(f"No below-5 rows found under {args.judged_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_below5_rows_csv(args.output_dir / "below5_rows.csv", rows)
    write_error_items_csv(args.output_dir / "error_items.csv", rows)
    report = build_report(rows, args.samples, args.seed)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")

    total_rows = len(rows)
    print(f"Below-5 judged rows: {total_rows}")
    for cat in CATEGORY_ORDER:
        n = sum(1 for row in rows if cat in row_categories(row))
        print(f"  {cat}: {n} rows ({n / total_rows * 100:.1f}%)")
    print(f"Wrote {args.output_dir / 'below5_rows.csv'}")
    print(f"Wrote {args.output_dir / 'error_items.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
