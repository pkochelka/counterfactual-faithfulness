# Data and Reports

Layout and record schemas for the released study data tracked under `data/`
and the derived report artifacts produced from it.

## Data

```text
data/
  cs-qa/            cf.csv  fa.csv  fi.csv
  sk-qa/            cf.csv  fa.csv  fi.csv
  source_csv/       pristine copy of the same six flat tables
  generated/        <model>/<dataset>_<variant>_<language>.csv
  classified/       <model>/<dataset>_<variant>_<language>.csv
  judged/           <model>/judge_<dataset>_<variant>_<language>_<judge-model>.jsonl
  judged_fluency/   <model>/judge_<dataset>_<variant>_<language>_<judge-model>.jsonl
  annotation/       human annotation sample, key, and agreement outputs
```

- `cs-qa/`, `sk-qa/`: flat source triple tables used for both generation and
  judging, produced by `cus-qa-to-triples/`. Variants: `cf` counterfactual,
  `fa` factual, `fi` fictional.
- `generated/<model>/`: one sentence per `eid`, generated from the triples.
  Languages: `en`, `cs`, `sk`, `hsb`. Generator models range from
  qwen3-1_7b (1.7B) to qwen3_5-122b (125B).
- `classified/<model>/`: the generator's own classification of each triple set
  as factual/counterfactual/fictional (prompt: `prompts/classify_speeches.json`).
- `judged/<model>/`: faithfulness judgments (judge: deepseek-v4-pro;
  prompt: `prompts/judge_speeches.txt`). `parsed` holds `faithfulness_score`
  (1–5) and `incorrect_information` (list of `info_used` / `correct_info` /
  `comment` objects, empty at score 5).
- `judged_fluency/<model>/`: fluency judgments (prompt: `prompts/judge_fluency.txt`).
  `parsed` holds `fluency_score` (1–5) and a freeform English `fluency_comment`.
- `*.failures.jsonl` sidecars next to judged files hold failed judge calls
  (no parsed score); every loader skips them.
- `annotation/`: `annotation_sample_we_used.csv` (blind sample with human
  scores), `annotation_key_we_used.csv` (matching judge scores),
  and the outputs of `scripts/measure_agreement.py` (disagreement listing,
  score-distribution plots).

### Judged record fields

Each judged JSONL record carries `eid`, `sentence`, `parsed`, source
identification (`source_label`, `source_id`), judge metadata (`judge_model`,
`requested_judge_model`, `requested_reasoning`, `provider`, `request_cost`,
`timestamp`), and per-dimension context: faithfulness records include
`modified_triples` / `num_modified_triples`; fluency records include
`language`, `language_name`, and `source_lexical_terms`.

Note: the stored records in this tree were trimmed to save space — bulky
reproducible fields (`prompt`, `raw_response`, `usage`, `source_path`,
`modified_triples_json`, `requested_judge_api_url`) were removed from the
archived files. Fresh judge runs still write them.

## Prompts

```text
prompts/
  generate_speeches.json          multilingual generation prompts (triples -> sentence)
  classify_speeches.json          multilingual fa/cf/fi classification prompts
  judge_speeches.txt              faithfulness judge prompt (+ label vocabulary)
  judge_fluency.txt               fluency judge prompt
  prompt-tuning-iterations/       historical faithfulness-prompt iterations
```

The faithfulness prompt instructs the judge to treat the modified triples as
the complete source of truth, even when they contradict real-world facts.

## Reports

Reports are regenerated from the judged trees, not stored canonically:

- `analysis/*.py` write plots next to themselves (`analysis/*.png`) and report
  trees under `analysis/error_deep_dive/` and `analysis/fluency_deep_dive/`.
- `analysis/fluency_report.py` writes per-score case listings under
  `data/reports/fluency/`.
- `inspect_judged_results.py` writes the full faithfulness report tree
  (overall statistics, per-model summaries, score cases, issue reports).
- `analysis/build_results_tables.py` reads the released study outputs and the
  regenerated issue report, then writes the paper's LaTeX tables and bootstrap
  summaries under `data/results_tables/`.

## Current totals

102,492 scored records per dimension: 9 models × 4 prompt languages × 2,847
evaluated RDF inputs across the two datasets and three variants.
Mean faithfulness 4.32 with 14.6% of records scoring 1–2; mean fluency 3.74
with 23.9% scoring 1–2.
