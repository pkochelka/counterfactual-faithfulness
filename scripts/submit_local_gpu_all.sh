#!/usr/bin/env bash
# Submit one local-GPU vLLM job per model. Each lands on its own GPU node, so the
# two models run in parallel. Run from the project root:
#   scripts/submit_local_gpu_all.sh
#
# Extra job variables go through EXTRA_VARS (folded into the SAME qsub -v), NOT a
# second -v: this PBS replaces an earlier -v with a later one, so a separate
# `-v SKIP_EXISTING=0` would silently drop MODEL_ID/MODEL_NAME/MAX_MODEL_LEN and
# every job would fall back to the gemma defaults. Examples:
#   EXTRA_VARS=SKIP_EXISTING=0 scripts/submit_local_gpu_all.sh
#   EXTRA_VARS="LIMIT=2,DATASETS=cs-qa,LANGUAGES=en" scripts/submit_local_gpu_all.sh
# Any real qsub flags (e.g. -l, -q) can still be passed as args; just not -v.
set -euo pipefail

JOB="scripts/metacentrum_local_gpu.pbs"
EXTRA_VARS="${EXTRA_VARS:-}"   # comma-separated KEY=VALUE pairs, folded into -v

# Each entry: MODEL_ID|MODEL_NAME|MAX_MODEL_LEN
MODELS=(
  "google/gemma-4-E4B-it|gemma4-e4b|1536"
  "CohereLabs/tiny-aya-global|tiny-aya-global|1536"
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r model_id model_name max_len <<< "$spec"
  vars="MODEL_ID=${model_id},MODEL_NAME=${model_name},MAX_MODEL_LEN=${max_len}"
  [ -n "$EXTRA_VARS" ] && vars="${vars},${EXTRA_VARS}"
  echo "Submitting $model_name ($model_id) -v $vars"
  qsub -v "$vars" "$@" "$JOB"
done
