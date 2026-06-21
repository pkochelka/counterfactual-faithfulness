# Data and Reports Package

This archive contains the active benchmark data, generated outputs, model
classification artifacts, LLM-judge annotations, and summary reports for the
counterfactual faithfulness experiments.

## Top-Level Structure

```text
data/
prompts/
reports/
```

`data/` contains source data and machine-produced artifacts. `prompts/` contains
the prompt templates used for generation, classification, and judging.
`reports/` contains the analyzed judge results and convenient summaries.

## Data

```text
data/
  GEM-v2-D2T-SharedTask/
  webnlg/
  cs-qa/
  sk-qa/
  generated/
  classified/
  judged/
```

- `GEM-v2-D2T-SharedTask/`: original WebNLG-style XML sources.
- `webnlg/`, `cs-qa/`, `sk-qa/`: flat source CSV files used for generation and
  judging. Variants are named `cf` for counterfactual, `fa` for factual, and
  `fi` for fictional where available.
- `generated/<model>/`: generated sentence CSVs. File names follow
  `<dataset>_<variant>_<language>.csv`, for example `webnlg_cf_en.csv`.
- `classified/<model>/`: classification outputs for generated sentences. A
  model was prompted to classify whether each generated sentence is factual
  (`fa`), counterfactual (`cf`), or fictional (`fi`) with respect to the source
  triples.
- `judged/<model>/`: LLM-judge JSONL annotations for generated outputs. These
  contain faithfulness scores and judge explanations.

## Prompts

```text
prompts/
  generate_speeches.json
  classify_speeches.json
  judge_speeches.txt
```

- `generate_speeches.json`: multilingual generation prompts for turning RDF
  triples into sentences.
- `classify_speeches.json`: multilingual classification prompts asking a model
  to decide whether triples are factual (`fa`), counterfactual (`cf`), or
  fictional (`fi`).
- `judge_speeches.txt`: LLM-judge faithfulness prompt template. Runtime fields
  such as `{category}`, `{sentence}`, and `{modified_triples}` are filled by the
  judging code.

## Reports

```text
reports/
  index.md
  overall_report.md
  overall_statistics.csv
  overall_statistics.json
  comparisons/
  score_cases/
  models/
```

- `index.md`: short entry point for the report tree.
- `overall_report.md`: human-readable global summary of all judged records.
- `overall_statistics.csv` / `.json`: machine-readable overall statistics.
- `comparisons/`: aggregate comparisons by dataset, variant, language, judge
  model, and model/dataset/variant/language combination.
- `score_cases/`: examples grouped by judge score, plus focused issue reports
  such as reverse relation, wrong entity, wrong relation, and unsupported or
  hallucinated information.
- Top-level issue reports such as `unsupported_or_hallucinated_issues.md`,
  `wrong_entity_issues.md`, and `wrong_relation_issues.md`: focused Markdown
  summaries of selected error categories for quick reading.
- `models/<model>/`: per-generator-model summaries, statistics, score cases,
  and dataset/variant breakdowns.


The active `reports/overall_report.md` in this package summarizes 43,778 judged
records. It reports a mean faithfulness score of 4.9461 and a score-1/2 failure
rate of 0.64%.
