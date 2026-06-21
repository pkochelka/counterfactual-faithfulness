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

Judge multiple CSV files through a custom provider, with both file-level and row-level parallelism:

```bash
export AUTH_TOKEN="$OPENAI_API_KEY"

find data/generated/qwen3.5-122b -name 'webnlg_*.csv' -print0 \
  | xargs -0 -n 1 -P 3 python llm-judge/judge_csv.py \
      --sample-size all \
      --model gpt-4.1-mini \
      --judge-base-url https://api.openai.com/v1 \
      --concurrency 4 \
      --output-dir data/judged
```

Here `xargs -P 3` runs up to three CSV files at once, and `--concurrency 4` runs up to four judge requests in parallel within each CSV. The maximum number of in-flight requests is therefore roughly `3 * 4 = 12`. Tune both numbers to match the provider's rate limits.

If you have several equivalent provider keys in `.env.local`, pass their environment variable names explicitly and set per-key concurrency. For example, three keys with `--concurrency-per-key 4` allow up to twelve in-flight judge requests, with no more than four using any one key:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_en.csv \
  --sample-size all \
  --model glm-5 \
  --judge-base-url https://llm.ai.e-infra.cz/v1 \
  --token-env-vars EINFRA_JR,EINFRA_AP,EINFRA_PK \
  --concurrency-per-key 4 \
  --output-dir data/judged
```

`--retry-sleep` is the short pause between transient retries. `--long-retry-sleep` is one longer cooldown used once per request before continuing normal retry attempts. HTTP 429, HTTP 5xx, timeouts, and connection errors are retried; authentication errors are not treated as retryable.

For long runs where you want one file at a time plus automatic retry/fallback, use `run_judge_batch.py`. It runs each matching CSV sequentially, logs stdout/stderr to a log file, retries a failed file once at the original concurrency, then retries again with lower concurrency. `judge_csv.py` exits nonzero when any rows fail, so the batch runner can detect provider/network errors reliably.

Example e-INFRA run with multiple keys:

```bash
../.venv/bin/python llm-judge/run_judge_batch.py \
  --language en \
  --token-env-vars EINFRA_JR,EINFRA_AP,EINFRA_PK \
  --model glm-5 \
  --judge-base-url https://llm.ai.e-infra.cz/v1 \
  --concurrency-per-key 4 \
  --fallback-concurrency 3 \
  --output-dir data/judged
```

By default logs go to `logs/llm-judge-batch-<timestamp>.log`. You can set an explicit path with `--log-file logs/judge-en.log`.

Pass an explicit XML file when inference from the CSV filename is not enough:

```bash
python llm-judge/judge_csv.py custom_sentences.csv \
  --xml data/GEM-v2-D2T-SharedTask/D2T-1-CFA_WebNLG_CounterFactual.xml \
  --sample-size 20 \
  --output-dir data/judged
```

For `cs-qa` and `sk-qa`, automatic inference uses the flat CSV tables under `data/<dataset>/<variant>.csv`, because those are the tables used by `generate_speeches.py` and `generate_all.py`. This prevents an alignment bug where `cs-qa_cf` generated sentences were compared against `cus-qa-to-triples/data/CounterFactual-triples.xml`, whose counterfactual substitutions can differ for the same `eid`.

Examples:

```bash
python llm-judge/judge_csv.py data/generated/gpt-oss-120b/cs-qa_cf_en.csv \
  --xml data/cs-qa/cf.csv \
  --sample-size all \
  --output-dir data/judged

python llm-judge/judge_csv.py data/generated/gpt-oss-120b/sk-qa_cf_en.csv \
  --xml data/sk-qa/cf.csv \
  --sample-size all \
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
- `--timeout 150`
- `--output-dir data/judged`
- `--output-path data/judged/gpt-oss-120b/judge_cs-qa_cf_cs_glm-5.jsonl`
- `--skip-existing-eids`
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
