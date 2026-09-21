#!/usr/bin/env bash
# Start in tmux. First establish the recipe's matching SFT reference.
set -euo pipefail
smol_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smol_artifacts=${SMOLVLA_ARTIFACT_ROOT:-$smol_script_dir/../work_dirs/rlinf-smolvla}
smol_bundle=${SMOLVLA_BUNDLE_DIR:-$smol_artifacts/weights/FluxVLA-SmolVLA-LIBERO10}
export SMOLVLA_SFT_PATH=${SMOLVLA_SFT_PATH:-$smol_bundle/checkpoints/step-057096-epoch-36-loss=0.2340.safetensors}
export SMOLVLA_TOKENIZER_PATH=${SMOLVLA_TOKENIZER_PATH:-$smol_bundle/tokenizer}
export SMOLVLA_STATS_PATH=${SMOLVLA_STATS_PATH:-$smol_bundle/dataset_statistics.json}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SMOLVLA_CONFIG_NAME=${SMOLVLA_CONFIG_NAME:-libero_10_ppo_fluxvla_smolvla_8gpu_200}
smol_run_name=${SMOLVLA_RUN_NAME:-smolvla_libero10_8gpu_200_$(date +%Y%m%d_%H%M%S)}
SMOLVLA_RUN_NAME="${smol_run_name}_sft_reference" SMOLVLA_ENTRY=eval \
    bash "$smol_script_dir/run_smolvla_rl_smoke.sh" runner.only_eval=true
smol_results_root=${SMOLVLA_RESULTS_ROOT:-$smol_artifacts/results}
if ! grep -qx 'FLUXVLA_EVALUATION_COMPLETED' \
    "$smol_results_root/${smol_run_name}_sft_reference/driver.log"; then
    printf 'SFT reference did not complete; refusing to start PPO.\n' >&2
    exit 1
fi
SMOLVLA_RUN_NAME="$smol_run_name" SMOLVLA_ENTRY=train \
    bash "$smol_script_dir/run_smolvla_rl_smoke.sh" "$@"
