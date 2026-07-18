"""Deep dive into judge comments for below-5 fluency scores.

The fluency analogue of error_category_deep_dive.py. Loads every judged
record under data/judged_fluency/, keeps rows with fluency_score < 5, and
buckets each fluency_comment into non-exclusive error categories.

Unlike the faithfulness judge (prompts/judge_speeches.txt), the fluency judge
(prompts/judge_fluency.txt) returns one freeform English comment per row with
no fixed label vocabulary — it only names the dimensions to judge (grammar,
inflection/agreement, word order, spelling/orthography, punctuation, word
choice, naturalness). So categories here are recovered with keyword/phrase
regexes tuned against a sample of real comments (~97% of below-5 comments hit
at least one category; see git history for the tuning pass), rather than
parsed from an explicit label the judge emits. Treat category boundaries as
approximate, not authoritative like the faithfulness label mapping.

Reuses the predicted CFA/FA/FI label loader from error_category_deep_dive.py
so every table also breaks down by the generator's predicted classification.

Outputs (under analysis/fluency_deep_dive/ by default):
  below5_rows.csv   one row per judged sentence scoring below 5 on fluency
  report.md         category breakdowns, keywords, samples
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from error_category_deep_dive import (  # noqa: E402
    DEFAULT_CLASSIFIED_DIR,
    PRED_LABEL_ORDER,
    load_predicted_labels,
    md_escape,
    pct_cell,
)
from judged_io import iter_judged_records  # noqa: E402

DEFAULT_JUDGED_DIR = REPO_ROOT / "data" / "judged_fluency"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "fluency_deep_dive"

MAX_SCORE = 5

# Keyword/phrase regexes recovering the fluency judge's implicit error
# dimensions from its freeform comment. Non-exclusive: one comment can match
# several (e.g. a case error described as also sounding awkward).
CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    "WrongLanguage": re.compile(
        r"\b(?:sentence|text)\b[^.]{0,20}\bis not\b[^.]{0,60}\b(?:upper sorbian|czech|slovak|english)\b"
        r"|\bnot written in\b"
        r"|\bno\b[^.]{0,15}\b(?:upper sorbian|czech|slovak|english)\b[^.]{0,15}\b(?:grammar|vocabulary|words)\b"
        r"|\bwritten in (?:czech|slovak|english|upper sorbian)[^.]{0,15}, not\b"
        r"|\bnot (?:a |real |natural )?upper sorbian\b"
        r"|\bmostly in (?:english|czech|slovak)\b, not\b",
        re.I,
    ),
    "NotASentence": re.compile(
        r"\bnot a (?:complete |proper |coherent |single )*(?:sentence|clause)\b"
        r"|pipe-separated|concatenation of|rdf-like|list of terms|meta-commentary"
        r"|editing notes|lacks a verb|\brun-on\b|\bfragment\b",
        re.I,
    ),
    "GrammarCaseAgreement": re.compile(
        r"\b(genitive|locative|instrumental|dative|accusative|nominative|vocative|case\b|"
        r"agree\w*|inflect\w*|conjugat\w*|declens\w*|declin\w*|verb form|preposition\w*|"
        r"tense\b|participle|infinitive|article\b)\b",
        re.I,
    ),
    "WordOrder": re.compile(r"\bword order\b", re.I),
    "Spelling": re.compile(r"\b(spelling|misspell\w*|orthograph\w*|typo)\b", re.I),
    "Punctuation": re.compile(r"\b(punctuation|comma\w*|hyphenat\w*)\b", re.I),
    "LexicalChoiceStyle": re.compile(
        r"\b(awkward\w*|unnatural\w*|idiomatic|word.choice|calque|not natural|"
        r"sounds?\s+(odd|strange|generic)|collocation|redundan\w*|not (?:the )?standard|"
        r"generic\b|lexical error)\b",
        re.I,
    ),
    "CrossLanguage": re.compile(
        r"\b(czech|slovak|english|polish)\s+(word|words|term|terms|form|forms|spelling|"
        r"adjective|noun|verb|conjunction|influence|name|loanword|borrowing)\b"
        r"|instead of (?:the |its )?(?:czech|slovak|english|upper sorbian) equivalent"
        r"|untranslated|not translated|left in (?:czech|slovak|english)"
        r"|\bloanword\b|\bborrowing\b|\banglicism\b"
        r"|\bis\s+(?:czech|slovak|english)\b[^.]{0,15}(?:, not|\.)"
        r"|the (?:czech|slovak|english) name for"
        r"|translated (?:to|into) (?:english|czech|slovak)",
        re.I,
    ),
    "DateNumberFormat": re.compile(
        r"\bdate (format|range)\b|\biso format\b|\bnumeric format\b|\bnumber format\b|\bnumeral\b", re.I
    ),
}
CATEGORY_ORDER = list(CATEGORY_PATTERNS.keys())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "as",
    "of", "to", "in", "on", "for", "with", "not", "no", "does", "do", "did",
    "has", "have", "had", "which", "while", "than", "then", "so", "such",
    "into", "from", "by", "at", "here", "there", "also", "only", "even",
    "sentence", "sentences", "text", "otherwise", "overall", "understandable",
    "correct", "correctly", "clear", "natural", "naturally", "mostly", "fully",
    "would", "should", "could", "however", "one", "both", "any", "all", "given",
    "rest", "minor", "noticeable", "slight", "slightly", "contains", "makes",
    "making", "instead", "more", "most", "like", "e.g",
}


def categorize(comment: str) -> set[str]:
    return {cat for cat, pattern in CATEGORY_PATTERNS.items() if pattern.search(comment)}


def load_rows(judged_dir: Path, predicted_labels: dict[tuple[str, str, str, str, str], str] | None = None) -> list[dict[str, Any]]:
    predicted_labels = predicted_labels or {}
    rows = []
    for r in iter_judged_records(judged_dir):
        parsed = r.record.get("parsed") or {}
        score = parsed.get("fluency_score")
        if not isinstance(score, int) or score >= MAX_SCORE:
            continue
        comment = str(parsed.get("fluency_comment") or "")
        language = str(r.record.get("language") or r.language)
        eid = str(r.record.get("eid", ""))
        rows.append(dict(
            model=r.model,
            dataset=r.dataset,
            variant=r.variant,
            language=language,
            eid=eid,
            sentence=str(r.record.get("sentence", "")),
            source_lexical_terms="; ".join(r.record.get("source_lexical_terms") or []),
            fluency_score=score,
            comment=comment,
            categories=categorize(comment),
            predicted_label=predicted_labels.get((r.model, r.dataset, r.variant, language, eid), ""),
            judged_file=str(r.path.relative_to(REPO_ROOT)),
            judged_line=r.line_no,
        ))
    return rows


def write_below5_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "dataset", "variant", "language", "eid", "fluency_score",
        "predicted_label", "sentence", "source_lexical_terms",
        *[f"has_{cat}" for cat in CATEGORY_ORDER],
        "comment", "judged_file", "judged_line",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row["model"],
                "dataset": row["dataset"],
                "variant": row["variant"],
                "language": row["language"],
                "eid": row["eid"],
                "fluency_score": row["fluency_score"],
                "predicted_label": row["predicted_label"],
                "sentence": row["sentence"],
                "source_lexical_terms": row["source_lexical_terms"],
                **{f"has_{cat}": int(cat in row["categories"]) for cat in CATEGORY_ORDER},
                "comment": row["comment"],
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


def build_report(rows: list[dict[str, Any]], samples_per_category: int, seed: int) -> str:
    rng = random.Random(seed)
    total_rows = len(rows)

    pred_row_totals = {p: sum(1 for row in rows if row["predicted_label"] == p) for p in PRED_LABEL_ORDER}
    no_pred_label = sum(1 for row in rows if not row["predicted_label"])

    lines = ["# Fluency Error Deep Dive (scores below 5)", ""]
    lines.append(f"Below-5 judged rows: {total_rows}")
    no_category = sum(1 for row in rows if not row["categories"])
    lines.append(
        f"Rows matching no recovered category: {no_category} "
        f"({no_category / total_rows * 100:.1f}%) — these still have a real fluency "
        "complaint in their comment, it just didn't match any of the keyword patterns below."
    )
    lines.append(
        "Pred FA/FI/CFA columns give the count and, in parentheses, the percentage of "
        "that predicted class's own below-5 rows — same convention as error_category_deep_dive.py. "
        f"Rows without a parsed predicted label ({no_pred_label} of {total_rows}) are excluded from "
        f"those columns; predicted-class totals are FA={pred_row_totals['FA']}, "
        f"FI={pred_row_totals['FI']}, CFA={pred_row_totals['CFA']}."
    )
    lines.append("")

    lines.append("## Category Overview")
    lines.append("")
    lines.append(
        "Categories are recovered from the judge's freeform comment via keyword/phrase "
        "regexes (see module docstring), not an explicit label, and are non-exclusive — "
        "one comment can hit several, so the column does not sum to 100%."
    )
    lines.append("")
    lines.append("| category | All rows | Pred FA | Pred FI | Pred CFA |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    cat_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for cat in row["categories"]:
            cat_rows[cat].append(row)
    for cat in CATEGORY_ORDER:
        n_rows = len(cat_rows[cat])
        by_pred = {p: sum(1 for row in cat_rows[cat] if row["predicted_label"] == p) for p in PRED_LABEL_ORDER}
        lines.append(
            f"| {cat} | {pct_cell(n_rows, total_rows)} | "
            f"{pct_cell(by_pred['FA'], pred_row_totals['FA'])} | "
            f"{pct_cell(by_pred['FI'], pred_row_totals['FI'])} | "
            f"{pct_cell(by_pred['CFA'], pred_row_totals['CFA'])} |"
        )
    lines.append("")

    for cat in CATEGORY_ORDER:
        cat_row_list = cat_rows[cat]
        if not cat_row_list:
            continue
        lines.append(f"## {cat}")
        lines.append("")
        lines.append(f"{len(cat_row_list)} rows.")
        lines.append("")

        by_lang = Counter(row["language"] for row in cat_row_list)
        lines.append("**By prompt/target language** (Pred columns: % of that language+predicted-class's below-5 rows in this category):")
        lines.append("")
        lines.append("| language | rows | % of rows in that language | Pred FA | Pred FI | Pred CFA |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for lang, count in by_lang.most_common():
            lang_rows = [row for row in rows if row["language"] == lang]
            total_lang_rows = len(lang_rows)
            pct = f"{count / total_lang_rows * 100:.1f}%" if total_lang_rows else "-"
            by_pred_total = {p: sum(1 for row in lang_rows if row["predicted_label"] == p) for p in PRED_LABEL_ORDER}
            by_pred_hit = {
                p: sum(1 for row in lang_rows if row["predicted_label"] == p and cat in row["categories"])
                for p in PRED_LABEL_ORDER
            }
            lines.append(
                f"| {lang} | {count} | {pct} | "
                f"{pct_cell(by_pred_hit['FA'], by_pred_total['FA'])} | "
                f"{pct_cell(by_pred_hit['FI'], by_pred_total['FI'])} | "
                f"{pct_cell(by_pred_hit['CFA'], by_pred_total['CFA'])} |"
            )
        lines.append("")

        comments = [row["comment"] for row in cat_row_list if row["comment"]]
        uni, bi = keyword_counts(comments)
        if uni:
            lines.append("**Frequent words in judge comments** (generic words filtered out):")
            lines.append("")
            lines.append(", ".join(f"{w} ({c})" for w, c in uni))
            lines.append("")
        if bi:
            lines.append("**Frequent word pairs:**")
            lines.append("")
            lines.append(", ".join(f"{w} ({c})" for w, c in bi))
            lines.append("")

        sample = cat_row_list[:]
        rng.shuffle(sample)
        sample = sample[:samples_per_category]
        lines.append(f"**Sample of {len(sample)} examples:**")
        lines.append("")
        lines.append("| model | lang | score | sentence | comment |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for row in sample:
            lines.append(
                "| "
                + " | ".join([
                    md_escape(row["model"]),
                    md_escape(row["language"]),
                    md_escape(row["fluency_score"]),
                    md_escape(row["sentence"]),
                    md_escape(row["comment"]),
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
        print(f"No below-5 fluency rows found under {args.judged_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_below5_rows_csv(args.output_dir / "below5_rows.csv", rows)
    report = build_report(rows, args.samples, args.seed)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")

    total_rows = len(rows)
    print(f"Below-5 fluency rows: {total_rows}")
    for cat in CATEGORY_ORDER:
        n = sum(1 for row in rows if cat in row["categories"])
        print(f"  {cat}: {n} rows ({n / total_rows * 100:.1f}%)")
    print(f"Wrote {args.output_dir / 'below5_rows.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
