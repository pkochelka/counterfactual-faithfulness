# llm-judge

Utilities for browsing generated WebNLG-style sentences and judging whether they are faithful to the modified triples used by this benchmark.

Run all commands below from the repository root:

```bash
cd /path/to/counterfactual-faithfulness
```

## What is here

- `browser_app.py`: Streamlit workspace for browsing entries, comparing model outputs, and running judge batches.
- `judge_csv.py`: command-line judge runner for sentence CSV files.
- `webnlg_utils.py`: shared XML/CSV loading, prompt building, API calls, annotation loading, and JSONL writing.
- `runtime_state.py`: in-memory Streamlit background-batch state.
- `outputs/`: local judge JSONL outputs. This directory is ignored by git.

## Expected files

The judge can work from either original GEM XML files or flat CSVs created from them.

Expected XML files:

```text
data/GEM-v2-D2T-SharedTask/D2T-1-CFA_WebNLG_CounterFactual.xml
data/GEM-v2-D2T-SharedTask/D2T-1-FA_WebNLG_Factual.xml
data/GEM-v2-D2T-SharedTask/D2T-1-FI_WebNLG_Fictional.xml
```

Optional flat CSVs from `xml2csv.py`:

```text
data/webnlg_cf.csv
data/webnlg_fa.csv
data/webnlg_fi.csv
```

Generated sentence CSVs must contain at least:

```text
eid,sentence
```

Typical generated files live at the repo root, for example:

```text
sentences_webnlg_cf_qwen3.5-122b.csv
sentences_webnlg_fa_qwen3.5-122b.csv
```

Generated sentence CSVs, text exports, `data/`, and `llm-judge/outputs/` are ignored by git.

## Environment

Create `.env.local` in the repository root:

```text
counterfactual-faithfulness/.env.local
```

For sentence generation with `generate_speeches.py`, set:

```text
BASE_URL=https://your-openai-compatible-endpoint
AUTH_TOKEN=your_token_if_required
```

For judging through OpenRouter, set:

```text
AUTH_TOKEN=your_openrouter_token
```

Do not commit `.env.local`. The Streamlit app also has a sidebar token field, which can be used instead of or in addition to the environment file.

## Install dependencies

Use the project environment you normally use for this repo. If creating a fresh one:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r llm-judge/requirements.txt
```

## Prepare data

Convert a GEM XML file to a flat CSV:

```bash
python xml2csv.py data/GEM-v2-D2T-SharedTask/D2T-1-CFA_WebNLG_CounterFactual.xml \
  --output data/webnlg_cf.csv
```

Generate model sentences:

```bash
python generate_speeches.py --model "qwen3.5-122b" --dataset "webnlg_cf"
```

The judge can also read existing sentence CSVs directly as long as they have `eid,sentence`.

## Run the web app

```bash
streamlit run llm-judge/browser_app.py
```

Optional custom port:

```bash
streamlit run llm-judge/browser_app.py --server.port 8502
```

In the sidebar:

- Choose an XML or CSV dataset source. The presets point to the expected files under `data/`.
- Load one or more model output CSVs. Use one path per line.
- Use optional labels with `Label :: path/to/file.csv`.
- Choose the judge model and token settings.
- Keep the default annotation directory unless you need a custom location:

```text
llm-judge/outputs
```

Batch judging uses the current filtered entries and currently visible outputs. Existing annotations are skipped by default for the selected judge model. `Batch row limit` means the number of non-skipped judge calls to run; already-judged skipped rows do not count against the limit. `0` means no limit.

## Run without the web app

Dry-run one prompt without calling the API:

```bash
python llm-judge/judge_csv.py sentences_webnlg_cf_qwen3.5-122b.csv \
  --sample-size 1 \
  --head \
  --dry-run \
  --output-dir /tmp/llm-judge-dry-run
```

Judge the first 20 rows:

```bash
python llm-judge/judge_csv.py sentences_webnlg_cf_qwen3.5-122b.csv \
  --sample-size 20 \
  --head \
  --model openai/gpt-5.2 \
  --output-dir llm-judge/outputs
```

Judge the whole CSV:

```bash
python llm-judge/judge_csv.py sentences_webnlg_cf_qwen3.5-122b.csv \
  --sample-size all \
  --model openai/gpt-5.2 \
  --output-dir llm-judge/outputs
```

Pass an explicit XML file when inference from the CSV filename is not enough:

```bash
python llm-judge/judge_csv.py custom_sentences.csv \
  --xml data/GEM-v2-D2T-SharedTask/D2T-1-CFA_WebNLG_CounterFactual.xml \
  --sample-size 20 \
  --output-dir llm-judge/outputs
```

Use `--force` to rewrite existing annotation rows for the same `eid`, source, and judge model in the output JSONL file.

Useful CLI arguments:

- `--xml path/to/source.xml`
- `--sample-size 20`
- `--sample-size all`
- `--head`
- `--seed 7`
- `--limit 50`
- `--model openai/gpt-5.2`
- `--max-tokens 5000`
- `--output-dir llm-judge/outputs`
- `--label custom_source_name`
- `--force`
- `--dry-run`

## Output format

Judge outputs are JSONL files written under:

```text
llm-judge/outputs/
```

Typical filename pattern:

```text
judge_<source-stem>_<judge-model>.jsonl
```

Each record includes metadata such as:

- `eid`
- `source_label`
- `source_path`
- `source_id`
- `judge_model`
- `requested_judge_model`
- `provider`
- `request_cost`
- `timestamp`
- `sentence`
- `raw_response`
- `parsed`

The current expected judge response schema is:

```json
{
  "faithfulness_score": 5,
  "incorrect_information": [
    {
      "info_used": "exact unsupported or wrong claim from the sentence",
      "correct_info": "triple-backed correction or missing constraint",
      "comment": "brief explanation based only on the triples"
    }
  ],
  "style_comment": "short comment about style and fluency"
}
```

The prompt tells the judge to treat the modified triples as the complete source of truth, even if they contradict real-world facts.

## Troubleshooting

If `AUTH_TOKEN` is missing, create `.env.local` at the repo root or paste the token into the Streamlit sidebar.

If OpenRouter returns an invalid model error, use a real OpenRouter model id, for example:

```text
openai/gpt-5.2
```

If the app cannot find data, check that the XML files are under `data/GEM-v2-D2T-SharedTask/` or pass explicit paths in the sidebar.
