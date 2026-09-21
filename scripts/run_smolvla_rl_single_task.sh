#!/usr/bin/env bash
# A bounded task-6 experiment: SFT reference, 50 PPO rounds, restored eval.
set -euo pipefail
smol_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
smol_artifacts=${SMOLVLA_ARTIFACT_ROOT:-$smol_script_dir/../work_dirs/rlinf-smolvla}
smol_bundle=${SMOLVLA_BUNDLE_DIR:-$smol_artifacts/weights/FluxVLA-SmolVLA-LIBERO10}
export SMOLVLA_SFT_PATH=${SMOLVLA_SFT_PATH:-$smol_bundle/checkpoints/step-057096-epoch-36-loss=0.2340.safetensors}
export SMOLVLA_TOKENIZER_PATH=${SMOLVLA_TOKENIZER_PATH:-$smol_bundle/tokenizer}
export SMOLVLA_STATS_PATH=${SMOLVLA_STATS_PATH:-$smol_bundle/dataset_statistics.json}
export SMOLVLA_CONFIG_NAME=libero_10_ppo_fluxvla_smolvla_task6
export SMOLVLA_RUN_NAME=${SMOLVLA_RUN_NAME:-smolvla_libero10_task6_50_$(date +%Y%m%d_%H%M%S)}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash "$smol_script_dir/run_smolvla_rl_200.sh"

smol_results_root=${SMOLVLA_RESULTS_ROOT:-$smol_artifacts/results}
smol_checkpoint="$smol_results_root/$SMOLVLA_RUN_NAME/checkpoints/global_step_50"
if [[ ! -f "$smol_checkpoint/actor/model_state_dict/full_weights.pt" ]]; then
    printf 'Final checkpoint is absent; refusing restored evaluation.\n' >&2
    exit 1
fi
SMOLVLA_RUN_NAME="${SMOLVLA_RUN_NAME}_restored_eval50" SMOLVLA_ENTRY=eval \
    bash "$smol_script_dir/run_smolvla_rl_smoke.sh" \
    runner.only_eval=true "runner.ckpt_path='$smol_checkpoint'"
if ! grep -qx 'FLUXVLA_EVALUATION_COMPLETED' \
    "$smol_results_root/${SMOLVLA_RUN_NAME}_restored_eval50/driver.log"; then
    printf 'Restored final evaluation did not complete.\n' >&2
    exit 1
fi
