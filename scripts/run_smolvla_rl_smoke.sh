#!/usr/bin/env bash
# Run inside tmux. No credentials are embedded, persisted, or echoed.
set -euo pipefail

: "${SMOLVLA_SFT_PATH:?Set the actual LIBERO-10 SFT checkpoint path}"
: "${SMOLVLA_TOKENIZER_PATH:?Set the matching tokenizer directory}"
: "${SMOLVLA_STATS_PATH:?Set the matching Flux dataset statistics JSON}"
: "${WANDB_API_KEY:?Export WANDB_API_KEY in this tmux shell}"

for artifact in "$SMOLVLA_SFT_PATH" "$SMOLVLA_TOKENIZER_PATH" "$SMOLVLA_STATS_PATH"; do
    if [[ ! -e "$artifact" ]]; then
        printf 'Required artifact is missing: %s\n' "$artifact" >&2
        exit 1
    fi
done

export FLUXVLA_ROOT
FLUXVLA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$FLUXVLA_ROOT${RLINF_ROOT:+:$RLINF_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export FLUX_RL_EXTERNAL_DISTRIBUTED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE=online
export WANDB_PROJECT=fluxvla-rl-smolvla
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/flux-smolvla-ray}
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=17179869184

smoke_artifacts=${SMOLVLA_ARTIFACT_ROOT:-$FLUXVLA_ROOT/work_dirs/rlinf-smolvla}
smoke_root=${SMOLVLA_RESULTS_ROOT:-$smoke_artifacts/results}
smoke_name=${SMOLVLA_RUN_NAME:-smolvla_libero10_smoke_$(date +%Y%m%d_%H%M%S)}
smoke_run="$smoke_root/$smoke_name"
smoke_python=${FLUX_RL_PYTHON:-python}
smoke_config=${SMOLVLA_CONFIG_NAME:-libero_10_ppo_fluxvla_smolvla_smoke}
smoke_entry=${SMOLVLA_ENTRY:-train}
if [[ "$smoke_entry" != train && "$smoke_entry" != eval ]]; then
    printf 'SMOLVLA_ENTRY must be train or eval\n' >&2
    exit 1
fi
mkdir -p "$smoke_root"
mkdir "$smoke_run"  # Refuse to overwrite a prior run.
export WANDB_DIR="$smoke_run"
cd "$FLUXVLA_ROOT"
printf 'Run directory: %s\n' "$smoke_run"

"$smoke_python" -u -m "fluxvla.rl.$smoke_entry" \
    "--config-name=$smoke_config" \
    "actor.model.model_path='$SMOLVLA_SFT_PATH'" \
    "actor.model.fluxvla.tokenizer_path='$SMOLVLA_TOKENIZER_PATH'" \
    "actor.model.fluxvla.norm_stats_path='$SMOLVLA_STATS_PATH'" \
    "runner.logger.log_path='$smoke_root'" \
    "runner.logger.experiment_name='$smoke_name'" \
    "$@" 2>&1 | tee "$smoke_run/driver.log"
