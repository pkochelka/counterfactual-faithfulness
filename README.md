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
- `data/cs-qa/{cf,fa}.csv`
- `data/GEM-v2-D2T-SharedTask/*.xml`

Use `xml2csv.py` to convert the GEM XML files into the `data/webnlg/*.csv` layout when needed.

### Speech generation

Generate fluent English sentences based on the triples in your dataset using parallel LLM API calls. Set your `AUTH_TOKEN` and API `BASE_URL` in `.env.local`.

```bash
python3 generate_speeches.py --model "qwen3.5-122b" --dataset "webnlg" --variant "cf" --language "cs"
```

### Judging and browsing outputs

The `llm-judge/` tools can browse generated sentence CSVs, compare outputs, and run an LLM judge against the modified triples.

Start the Streamlit workspace from the repository root:

```bash
streamlit run llm-judge/browser_app.py
```

Run the judge without the web app:

```bash
python llm-judge/judge_csv.py data/generated/qwen3.5-122b/webnlg_cf_cs.csv --sample-size 20 --head --output-dir data/judged
```

See `llm-judge/README_judge.md` for expected XML/CSV files, `.env.local` setup, batch judging behavior, and full CLI examples.
