# counterfactual-faithfulness

Benchmarking LLM faithfulness to presented counterfactual data (in form of triplets) when asked to generate a sentence, across languages and model families.

## Setup

### Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Deactivate when done

```bash
deactivate
```

## Usage

The repo expects structured data under `data/`:

- `data/webnlg/{cf,fa,fi}.csv`
- `data/{cs-qa,sk-qa}/{cf,fa}.csv`
- `data/GEM-v2-D2T-SharedTask/*.xml`

Use `xml2csv.py` to convert the GEM XML files into the `data/webnlg/*.csv` layout when needed.

### Speech generation

Generate fluent sentences based on the triples in your dataset using parallel LLM API calls. Set your `AUTH_TOKEN` and OpenAI-compatible API `BASE_URL` in `.env.local`.

```bash
python3 generate_speeches.py --model "qwen3.5-122b" --dataset "webnlg" --variant "cf" --language "cs"
```

To use multiple equivalent provider keys from `.env.local`, pass their environment variable names and the per-key concurrency:

```bash
python3 generate_speeches.py \
  --model "qwen3.5-122b" \
  --dataset "webnlg" \
  --variant "cf" \
  --language "cs" \
  --token-env-vars EINFRA_JR,EINFRA_AP,EINFRA_PK \
  --concurrency-per-key 4
```

### Judging and browsing outputs

The `llm-judge/` tools can browse generated sentence CSVs, compare outputs, and run an LLM judge against the modified triples. OpenRouter is the default judge endpoint, but you can select any OpenAI-compatible endpoint with `JUDGE_BASE_URL`, `JUDGE_API_URL`, the Streamlit sidebar, or the CLI `--judge-base-url` option.

Start the Streamlit workspace from the repository root:

```bash
streamlit run llm-judge/browser_app.py
```

Run the judge without the web app:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_cs.csv --sample-size 20 --head --output-dir data/judged
```

For `cs-qa` and `sk-qa`, generated CSVs are judged against the same flat source tables used for generation:

- `data/cs-qa/{cf,fa}.csv`
- `data/sk-qa/{cf,fa}.csv`

This is intentional. Do not judge `cs-qa_cf` outputs against `cus-qa-to-triples/data/CounterFactual-triples.xml` unless that XML is known to be the exact source used to generate the sentence CSV. If the source cannot be inferred, pass the matching file explicitly with `--xml data/cs-qa/cf.csv` or `--xml data/sk-qa/cf.csv`.

Run command-line judging through a custom provider with parallelism across files and within each file:

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

`xargs -P 3` runs three files at once; `--concurrency 4` runs four examples at once inside each file, for about twelve in-flight judge calls.
The per-request HTTP timeout defaults to 150 seconds and can be changed with `--timeout`.

For long full evaluations with one file at a time, retry, fallback from concurrency 4 to 3, and log files, use:

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

See `llm-judge/README_judge.md` for expected XML/CSV files, `.env.local` setup, batch judging behavior, and full CLI examples.

## References

#### CUS-QA dataset

```
CUS-QA: Local-Knowledge-Oriented Open-Ended Question Answering Dataset, Libovický et al, 2025
```

#### web_nlg

```
The 2024 GEM Shared Task on Multilingual Data-to-Text Generation and Summarization; https://gem-benchmark.com/shared_task
```
