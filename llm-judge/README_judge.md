# llm-judge

Utilities for browsing generated sentences and judging them against the modified triples used by this benchmark, on two dimensions: faithfulness to the triples and linguistic fluency.

Run all commands below from the repository root:

```bash
cd /path/to/counterfactual-faithfulness
```

## What is here

- `browser_app.py`: Streamlit workspace for browsing entries, comparing model outputs, and running judge batches.
- `judge_csv.py`: command-line judge runner for sentence CSV files (faithfulness by default, fluency with `--fluency`).
- `run_judge_batch.py`: sequential batch driver with logging, retry, and a concurrency fallback.
- `run_fluency_batch.py`: thin wrapper around the batch driver that injects `--fluency` and the fluency default model.
- `webnlg_utils.py`: shared XML/CSV loading, prompt building, API calls, annotation loading, and JSONL writing.
- `runtime_state.py`: in-memory Streamlit background-batch state.

Judge JSONL outputs go to `data/judged/` (faithfulness) and `data/judged_fluency/` (fluency).

## Expected files

The judge scores generated sentences against source triples. For `cs-qa` and `sk-qa` the source triples are the flat tables also used for generation:

```text
data/cs-qa/cf.csv  data/cs-qa/fa.csv  data/cs-qa/fi.csv
data/sk-qa/cf.csv  data/sk-qa/fa.csv  data/sk-qa/fi.csv
```

Generated sentence CSVs must contain at least:

```text
eid,sentence
```

Typical generated files live under `data/generated/<model>/`, for example:

```text
data/generated/qwen3.5-122b/cs-qa_cf_cs.csv
data/generated/qwen3.5-122b/sk-qa_fa_en.csv
```

The whole `data/` tree is ignored by git.

## Environment

Create `.env.local` in the repository root:

```text
counterfactual-faithfulness/.env.local
```

For sentence generation with `generate_speeches.py`, set:

```text
BASE_URL=https://your-openai-compatible-endpoint
KEY1=your_token_if_required
```

For command-line judging, OpenRouter is the default provider. Set a named key and pass that name with `--token-env-vars`:

```text
OPENROUTER_API_KEY=your_openrouter_token
```

To use a different OpenAI-compatible provider, add either a base URL or a full chat completions endpoint:

```text
JUDGE_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_provider_token
```

or:

```text
JUDGE_API_URL=https://api.openai.com/v1/chat/completions
OPENAI_API_KEY=your_provider_token
```

`JUDGE_BASE_URL` and `JUDGE_API_URL` are only for the judge. Sentence generation still uses `BASE_URL`.

Do not commit `.env.local`. The Streamlit app also has a sidebar token field and can still read `AUTH_TOKEN` as an app default.

## Install dependencies

Use the project environment you normally use for this repo. If creating a fresh one:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r llm-judge/requirements.txt
```

## Prepare data

The `cs-qa`/`sk-qa` flat tables come from the `cus-qa-to-triples/` pipeline and live under `data/{cs-qa,sk-qa}/`. Generate model sentences from them:

```bash
python generate_speeches.py --model "qwen3.5-122b" --dataset "cs-qa" --variant "cf" --language "cs" --kind original
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
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/cs-qa_cf_cs.csv \
  --sample-size 1 \
  --head \
  --dry-run \
  --output-dir /tmp/llm-judge-dry-run
```

Judge the first 20 rows:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/cs-qa_cf_cs.csv \
  --sample-size 20 \
  --head \
  --model openai/gpt-5.2 \
  --token-env-vars OPENROUTER_API_KEY \
  --output-dir data/judged
```

Judge through another OpenAI-compatible provider:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/cs-qa_cf_cs.csv \
  --sample-size 20 \
  --head \
  --model gpt-4.1-mini \
  --judge-base-url https://api.openai.com/v1 \
  --token-env-vars OPENAI_API_KEY \
  --output-dir data/judged
```

Local or self-hosted endpoints work the same way if they implement `/chat/completions`:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/cs-qa_cf_cs.csv \
  --sample-size 20 \
  --head \
  --model local-model \
  --judge-base-url http://localhost:8000/v1 \
  --token-env-vars LOCAL_API_KEY \
  --output-dir data/judged
```

Judge the whole CSV:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/cs-qa_cf_cs.csv \
  --sample-size all \
  --model openai/gpt-5.2 \
  --token-env-vars OPENROUTER_API_KEY \
  --concurrency-per-key 5 \
  --output-dir data/judged
```

Judge multiple CSV files through a custom provider, with both file-level and per-key row-level parallelism:

```bash
find data/generated/qwen3.5-122b -name 'cs-qa_*.csv' -print0 \
  | xargs -0 -n 1 -P 3 python llm-judge/judge_csv.py \
      --sample-size all \
      --model gpt-4.1-mini \
      --judge-base-url https://api.openai.com/v1 \
      --token-env-vars OPENAI_API_KEY \
      --concurrency-per-key 4 \
      --output-dir data/judged
```

Here `xargs -P 3` runs up to three CSV files at once, and `--concurrency-per-key 4` runs up to four judge requests per listed key within each CSV. With one key, the maximum number of in-flight requests is roughly `3 * 4 = 12`. Tune both numbers to match the provider's rate limits.

If you have several equivalent provider keys in `.env.local`, pass their environment variable names explicitly and set per-key concurrency. For example, three keys with `--concurrency-per-key 4` allow up to twelve in-flight judge requests, with no more than four using any one key:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/cs-qa_cf_en.csv \
  --sample-size all \
  --model deepseek-v4-pro \
  --judge-base-url https://your-openai-compatible-endpoint/v1 \
  --token-env-vars KEY1,KEY2,KEY3 \
  --concurrency-per-key 4 \
  --output-dir data/judged
```

`--retry-sleep` is the short pause between transient retries. `--long-retry-sleep` is one longer cooldown used once per request before continuing normal retry attempts. HTTP 429, HTTP 5xx, timeouts, and connection errors are retried; authentication errors are not treated as retryable.

For long runs where you want one file at a time plus automatic retry/fallback, use `run_judge_batch.py`. It runs each matching CSV sequentially, logs stdout/stderr to a log file, retries a failed file once at the original per-key concurrency, then retries again with lower per-key concurrency. `judge_csv.py` exits nonzero when any rows fail, so the batch runner can detect provider/network errors reliably.

Example batch run with multiple keys:

```bash
python llm-judge/run_judge_batch.py \
  --language en \
  --token-env-vars KEY1,KEY2,KEY3 \
  --model deepseek-v4-pro \
  --judge-base-url https://your-openai-compatible-endpoint/v1 \
  --concurrency-per-key 4 \
  --fallback-concurrency-per-key 3 \
  --output-dir data/judged
```

By default logs go to `logs/llm-judge-batch-<timestamp>.log`. You can set an explicit path with `--log-file logs/judge-en.log`.

## Fluency judging

`--fluency` switches `judge_csv.py` to the second dimension: how well the sentence is written in its target language, independent of faithfulness (prompt: `prompts/judge_fluency.txt`). The matching source triples are used only to provide a lexical term set, and output defaults to `data/judged_fluency/` instead of `data/judged/`:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/cs-qa_cf_cs.csv \
  --fluency \
  --sample-size all \
  --model deepseek-v4-pro \
  --judge-base-url https://your-openai-compatible-endpoint/v1 \
  --token-env-vars KEY1 \
  --output-dir data/judged_fluency
```

For full batches, `run_fluency_batch.py` forwards everything to `run_judge_batch.py` with `--fluency` and the fluency default model injected:

```bash
python llm-judge/run_fluency_batch.py --token-env-vars KEY1,KEY2 --sample-size all
```

Pass an explicit source file when inference from the CSV filename is not enough:

```bash
python llm-judge/judge_csv.py custom_sentences.csv \
  --xml data/cs-qa/cf.csv \
  --sample-size 20 \
  --token-env-vars OPENROUTER_API_KEY \
  --output-dir data/judged
```

For `cs-qa` and `sk-qa`, automatic inference uses the flat CSV tables under `data/<dataset>/<variant>.csv`, because those are the tables used by `generate_speeches.py` and `generate_all.py`. This prevents an alignment bug where `cs-qa_cf` generated sentences were compared against `cus-qa-to-triples/data/CounterFactual-triples.xml`, whose counterfactual substitutions can differ for the same `eid`.

Examples:

```bash
python llm-judge/judge_csv.py data/generated/gpt-oss-120b/cs-qa_cf_en.csv \
  --xml data/cs-qa/cf.csv \
  --sample-size all \
  --token-env-vars OPENROUTER_API_KEY \
  --output-dir data/judged

python llm-judge/judge_csv.py data/generated/gpt-oss-120b/sk-qa_cf_en.csv \
  --xml data/sk-qa/cf.csv \
  --sample-size all \
  --token-env-vars OPENROUTER_API_KEY \
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
- `--token-env-vars OPENAI_API_KEY`
- `--concurrency-per-key 5`
- `--model openai/gpt-5.2`
- `--judge-base-url https://api.openai.com/v1`
- `--max-tokens 5000`
- `--timeout 150`
- `--output-dir data/judged`
- `--output-path data/judged/gpt-oss-120b/judge_cs-qa_cf_cs_deepseek-v4-pro.jsonl`
- `--skip-existing-eids`
- `--label custom_source_name`
- `--force`
- `--dry-run`

## Output format

Judge outputs are JSONL files written under `data/judged/` (faithfulness) or `data/judged_fluency/` (fluency), one file per source CSV:

```text
data/judged/<generator-model>/judge_<source-stem>_<judge-model>.jsonl
```

Failed judge calls go to a `judge_<source-stem>_<judge-model>.failures.jsonl` sidecar; all downstream loaders skip those.

Each record includes `eid`, `sentence`, and `parsed` (the judge's verdict), source identification (`source_label`, `source_id`), and judge metadata (`judge_model`, `requested_judge_model`, `requested_judge_api_url`, `provider`, `request_cost`, `timestamp`), plus the full `prompt`, `raw_response`, and token `usage`. Note that the archived runs stored in this repo's data tree were trimmed of the bulky reproducible fields (`prompt`, `raw_response`, `usage`, `source_path`, `modified_triples_json`, `requested_judge_api_url`) to save space; fresh runs still write them.

The faithfulness judge response schema (`prompts/judge_speeches.txt`) is:

```json
{
  "incorrect_information": [
    {
      "info_used": "exact unsupported or wrong claim from the sentence",
      "correct_info": "triple-backed correction or missing constraint",
      "comment": "label plus brief explanation based only on the triples"
    }
  ],
  "faithfulness_score": 5
}
```

An empty `incorrect_information` list means the score must be 5. The fluency judge (`prompts/judge_fluency.txt`) returns:

```json
{
  "fluency_score": 5,
  "fluency_comment": "short English comment on grammar, word order, and naturalness"
}
```

The faithfulness prompt tells the judge to treat the modified triples as the complete source of truth, even if they contradict real-world facts.

## Troubleshooting

If the command-line judge says token variables are missing, create `.env.local` at the repo root and pass the variable names with `--token-env-vars`. In the Streamlit app, you can also paste the token into the sidebar.

If OpenRouter returns an invalid model error, use a real OpenRouter model id, for example:

```text
openai/gpt-5.2
```

If another provider returns a 404, check whether you pasted a base URL or a full chat completions URL. Both of these are accepted:

```text
https://api.openai.com/v1
https://api.openai.com/v1/chat/completions
```

If the app cannot find data, check that the flat source tables are under `data/{cs-qa,sk-qa}/`, or pass explicit paths in the sidebar.
