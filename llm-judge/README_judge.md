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
- `data/judged/`: local judge JSONL outputs. This directory is ignored by git.

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
data/webnlg/cf.csv
data/webnlg/fa.csv
data/webnlg/fi.csv
```

Generated sentence CSVs must contain at least:

```text
eid,sentence
```

Typical generated files live under `data/generated/<model>/`, for example:

```text
data/generated/qwen3.5-122b/webnlg_cf_cs.csv
data/generated/qwen3.5-122b/webnlg_fa_cs.csv
```

Generated sentence CSVs, text exports, `data/`, and `data/judged/` are ignored by git.

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

For judging, OpenRouter is the default provider. Set:

```text
AUTH_TOKEN=your_openrouter_token
```

To use a different OpenAI-compatible provider, add either a base URL or a full chat completions endpoint:

```text
JUDGE_BASE_URL=https://api.openai.com/v1
AUTH_TOKEN=your_provider_token
```

or:

```text
JUDGE_API_URL=https://api.openai.com/v1/chat/completions
AUTH_TOKEN=your_provider_token
```

`JUDGE_BASE_URL` and `JUDGE_API_URL` are only for the judge. Sentence generation still uses `BASE_URL`.

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
  --output data/webnlg/cf.csv
```

Generate model sentences:

```bash
python generate_speeches.py --model "qwen3.5-122b" --dataset "webnlg" --variant "cf" --language "cs"
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
- Choose the judge model, provider endpoint, and token settings.
- Leave the judge endpoint at `https://openrouter.ai/api/v1/chat/completions` for OpenRouter, or paste another OpenAI-compatible base URL such as `https://api.openai.com/v1`.
- Keep the default annotation directory unless you need a custom location:

```text
data/judged
```

Batch judging uses the current filtered entries and currently visible outputs. Existing annotations are skipped by default for the selected judge model. `Batch row limit` means the number of non-skipped judge calls to run; already-judged skipped rows do not count against the limit. `0` means no limit.

## Run without the web app

Dry-run one prompt without calling the API:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_cs.csv \
  --sample-size 1 \
  --head \
  --dry-run \
  --output-dir /tmp/llm-judge-dry-run
```

Judge the first 20 rows:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_cs.csv \
  --sample-size 20 \
  --head \
  --model openai/gpt-5.2 \
  --output-dir data/judged
```

Judge through another OpenAI-compatible provider:

```bash
AUTH_TOKEN=your_provider_token \
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_cs.csv \
  --sample-size 20 \
  --head \
  --model gpt-4.1-mini \
  --judge-base-url https://api.openai.com/v1 \
  --output-dir data/judged
```

Local or self-hosted endpoints work the same way if they implement `/chat/completions`:

```bash
AUTH_TOKEN=unused \
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_cs.csv \
  --sample-size 20 \
  --head \
  --model local-model \
  --judge-base-url http://localhost:8000/v1 \
  --output-dir data/judged
```

Judge the whole CSV:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_cs.csv \
  --sample-size all \
  --model openai/gpt-5.2 \
  --concurrency 5 \
  --output-dir data/judged
```

Pass an explicit XML file when inference from the CSV filename is not enough:

```bash
python llm-judge/judge_csv.py custom_sentences.csv \
  --xml data/GEM-v2-D2T-SharedTask/D2T-1-CFA_WebNLG_CounterFactual.xml \
  --sample-size 20 \
  --output-dir data/judged
```

Use `--force` to rewrite existing annotation rows for the same `eid`, source, and judge model in the output JSONL file.
By default, the CLI appends each judged row to disk immediately, so stopping the process midway still preserves already completed rows. When you rerun without `--force`, rows that already exist for the same source and judge model are skipped automatically. With `--force`, the target JSONL file is cleared before the run starts.

Useful CLI arguments:

- `--xml path/to/source.xml`
- `--sample-size 20`
- `--sample-size all`
- `--head`
- `--seed 7`
- `--limit 50`
- `--concurrency 5`
- `--model openai/gpt-5.2`
- `--judge-base-url https://api.openai.com/v1`
- `--max-tokens 5000`
- `--output-dir data/judged`
- `--label custom_source_name`
- `--force`
- `--dry-run`

## Output format

Judge outputs are JSONL files written under:

```text
data/judged/
```

Typical layout:

```text
data/judged/<generator-model>/judge_<source-stem>_<judge-model>.jsonl
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
- `requested_judge_api_url`
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

If another provider returns a 404, check whether you pasted a base URL or a full chat completions URL. Both of these are accepted:

```text
https://api.openai.com/v1
https://api.openai.com/v1/chat/completions
```

If the app cannot find data, check that the XML files are under `data/GEM-v2-D2T-SharedTask/` or pass explicit paths in the sidebar.
