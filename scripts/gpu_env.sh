# Source this inside a Metacentrum GPU job:  source scripts/gpu_env.sh
# Pins the HuggingFace cache to /storage, activates the GPU venv (vLLM), and
# remaps the PBS GPU UUID(s) to the integer indices vLLM expects.
#
# Assumes the current working directory is the project root (the PBS job does
# `cd "$PROJECT_DIR"` before sourcing this).

# Keep every HF cache under the project on /storage so weights download once and
# are shared across jobs instead of landing on the tiny, ephemeral scratch.
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE"

# Gated repos (gemma, tiny-aya) need an HF token. HF_TOKEN is exported in the
# user's ~/.bashrc, which interactive shells source but PBS job shells do not, so
# the job would otherwise hit 401 on first download. If it is not already in the
# environment (e.g. via `qsub -v HF_TOKEN`), pull the value straight from .bashrc
# without sourcing the whole file.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.bashrc" ]]; then
    _hf_tok=$(sed -n 's/^[[:space:]]*export[[:space:]]\{1,\}HF_TOKEN=//p' "$HOME/.bashrc" | tr -d "\"' " | head -n1)
    [[ -n "$_hf_tok" ]] && export HF_TOKEN="$_hf_tok"
fi

# Select the GPU Python env (vLLM lives here, separate from the CPU .venv used by
# the e-infra API pipeline). Override with GPU_VENV if you named it differently.
# We export PYTHON and prepend bin/ to PATH instead of relying on `activate`,
# because the reused MIST env is non-standard and ships no activate script.
GPU_VENV="${GPU_VENV:-.venv-gpu}"
if [[ -f "$GPU_VENV/bin/activate" ]]; then
    source "$GPU_VENV/bin/activate"
    export PYTHON="${PYTHON:-$GPU_VENV/bin/python}"
elif [[ -x "$GPU_VENV/bin/python" ]]; then
    export PATH="$PWD/$GPU_VENV/bin:$PATH"
    export PYTHON="${PYTHON:-$PWD/$GPU_VENV/bin/python}"
else
    echo "WARN: no GPU Python at $GPU_VENV/bin/python; create/symlink it first (see METACENTRUM_RUNBOOK.md)" >&2
fi

# PBS exposes allocated GPU(s) as UUIDs in CUDA_VISIBLE_DEVICES, but vLLM needs
# integer indices. Remap UUIDs -> indices using nvidia-smi's cgroup-restricted
# view (which only sees this job's GPUs).
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] && command -v nvidia-smi >/dev/null; then
    _map=$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader)
    _ids=""
    IFS=',' read -ra _uuids <<< "$CUDA_VISIBLE_DEVICES"
    for _u in "${_uuids[@]}"; do
        _u="${_u// /}"
        _idx=$(echo "$_map" | awk -F', ' -v u="$_u" '$1==u {print $2}')
        _ids="${_ids:+$_ids,}${_idx}"
    done
    export CUDA_VISIBLE_DEVICES="$_ids"
fi

echo "PROJECT=$PWD"
echo "python: $(command -v python)"
echo "HF_HOME=$HF_HOME"
echo "HF_TOKEN=${HF_TOKEN:+<set>}${HF_TOKEN:-<UNSET>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
