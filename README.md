# counterfactual-faithfulness

Benchmarking LLM faithfulness to presented counterfactual data (in form of triplets) when asked to generate a sentence, across languages and model families.

Nine generator models (1.7B–125B) turn RDF-style triples from two datasets (`cs-qa`, `sk-qa`) into single sentences in four prompt languages (`en`, `cs`, `sk`, `hsb`), for three source variants: `fa` (factual), `cf` (counterfactual), and `fi` (fictional). An LLM judge then scores every sentence on two dimensions: **faithfulness** to the triples (treating them as the complete source of truth even when they contradict reality) and **fluency** in the target language. A stratified sample was human-annotated to validate the judge.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -r llm-judge/requirements.txt
```

API credentials go into `.env.local` at the repo root (never committed): `BASE_URL` plus one or more token variables for generation, and judge endpoint settings described in [llm-judge/README_judge.md](llm-judge/README_judge.md).

## Repository layout

- `generate_speeches.py`, `generate_all.py`, `api_caller.py` — sentence generation and classification through OpenAI-compatible APIs
- `llm-judge/` — faithfulness and fluency judging (CLI + batch runners) and a Streamlit browser; see [llm-judge/README_judge.md](llm-judge/README_judge.md)
- `analysis/` — plots and reports built from the judged records
- `scripts/` — human-annotation sampling and human-vs-judge agreement
- `cus-qa-to-triples/` — pipeline that converts the CUS-QA dataset into factual/counterfactual/fictional triples; see [cus-qa-to-triples/README.md](cus-qa-to-triples/README.md)
- `prompts/` — generation, classification, and judge prompt templates
- `inspect_judged_results.py` — writes a full Markdown/CSV report tree from judged records

## Data

`data/` is not tracked by git. See [DATA_AND_REPORTS_README.md](DATA_AND_REPORTS_README.md) for the full layout and record schemas. In short:

- `data/{cs-qa,sk-qa}/{cf,fa,fi}.csv` — flat source triple tables (produced by `cus-qa-to-triples/`)
- `data/generated/<model>/<dataset>_<variant>_<language>.csv` — generated sentences
- `data/classified/<model>/` — the generator's own fa/cf/fi classification of each item
- `data/judged/` and `data/judged_fluency/` — judge JSONL records per model

## Generation

```bash
python3 generate_speeches.py \
  --model "qwen3.5-122b" \
  --dataset "cs-qa" \
  --variant "cf" \
  --language "cs" \
  --kind original \
  --token-env-vars KEY1,KEY2 \
  --concurrency-per-key 4
```

`--task classified` runs the classification prompt instead (the model labels each triple set as fa/cf/fi); output then goes under `data/classified/`. `generate_all.py` runs the whole model × dataset × variant × language grid, skipping outputs that already exist.

## Judging

Faithfulness, one file (the source triples are inferred from the CSV filename — `cs-qa`/`sk-qa` outputs are judged against the same flat tables used for generation):

```bash
python llm-judge/judge_csv.py data/generated/qwen3_5-9b/cs-qa_cf_cs.csv \
  --sample-size all \
  --model deepseek-v4-pro \
  --judge-base-url https://your-openai-compatible-endpoint/v1 \
  --token-env-vars KEY1 \
  --concurrency-per-key 4 \
  --output-dir data/judged
```

Full batches with retry and logging, for both dimensions:

```bash
python llm-judge/run_judge_batch.py   --token-env-vars KEY1,KEY2 --sample-size all
python llm-judge/run_fluency_batch.py --token-env-vars KEY1,KEY2 --sample-size all
```

The fluency runner is the same batch driver with `--fluency` injected; fluency output defaults to `data/judged_fluency/`. The Streamlit browser (`streamlit run llm-judge/browser_app.py`) can browse entries, compare model outputs, and run judge batches interactively. See [llm-judge/README_judge.md](llm-judge/README_judge.md) for providers, concurrency, resume behavior, and the full CLI reference.

## Analysis

Each script reads `data/judged/` or `data/judged_fluency/` and writes its plot/report next to itself or under `data/reports/`:

```bash
python analysis/faithfulness_by_language.py        # mean score by language group, per model
python analysis/faithfulness_by_variant.py         # fa vs cf vs fi
python analysis/fluency_by_language.py             # fluency analogue
python analysis/classification_accuracy.py         # can models tell fa/cf/fi apart?
python analysis/faithfulness_by_classification_split.py
python analysis/error_category_deep_dive.py        # judge error labels -> categories
python analysis/fluency_error_deep_dive.py
python analysis/fluency_report.py                  # per-score case listings
python inspect_judged_results.py                   # full report tree
```

## Human annotation and judge validation

`scripts/sample_for_annotation.py` draws a stratified blind sample (`annotation_sample*.csv` for annotators, `annotation_key*.csv` with the judge's scores). `scripts/measure_agreement.py` computes human-vs-judge agreement (exact/within-1 agreement, Spearman, quadratic weighted kappa) per annotator and dimension, and plots score distributions; when the key lacks fluency scores it fetches them from `data/judged_fluency/`.

## References

### CUS-QA dataset

```text
CUS-QA: Local-Knowledge-Oriented Open-Ended Question Answering Dataset, Libovický et al, 2025
```
