#!/usr/bin/env bash
# Start in tmux: online auth, matching SFT reference, PPO, restored monitor.
set -euo pipefail
if [[ $# != 0 ]]; then
    printf 'This fixed 1000-round protocol accepts no CLI overrides.\n' >&2
    exit 1
fi
smol_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SMOLVLA_CONFIG_NAME=libero_10_ppo_fluxvla_smolvla_8gpu_1000
export SMOLVLA_RUN_NAME=${SMOLVLA_RUN_NAME:-smolvla_libero10_8gpu_1000_$(date +%Y%m%d_%H%M%S)}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
export USE_TF=0
export FLUX_RL_EXTERNAL_DISTRIBUTED=1
export PYTHONPATH="$smol_script_dir/..${RLINF_ROOT:+:$RLINF_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
smol_python=${FLUX_RL_PYTHON:-python}
"$smol_python" -c 'from fluxvla.rl.utils.logging import require_wandb; require_wandb(); print("Online W&B authentication passed", flush=True)'

bash "$smol_script_dir/run_smolvla_rl_200.sh"

smol_artifacts=${SMOLVLA_ARTIFACT_ROOT:-$smol_script_dir/../work_dirs/rlinf-smolvla}
smol_bundle=${SMOLVLA_BUNDLE_DIR:-$smol_artifacts/weights/FluxVLA-SmolVLA-LIBERO10}
export SMOLVLA_SFT_PATH=${SMOLVLA_SFT_PATH:-$smol_bundle/checkpoints/step-057096-epoch-36-loss=0.2340.safetensors}
export SMOLVLA_TOKENIZER_PATH=${SMOLVLA_TOKENIZER_PATH:-$smol_bundle/tokenizer}
export SMOLVLA_STATS_PATH=${SMOLVLA_STATS_PATH:-$smol_bundle/dataset_statistics.json}
smol_results_root=${SMOLVLA_RESULTS_ROOT:-$smol_artifacts/results}
smol_checkpoint="$smol_results_root/$SMOLVLA_RUN_NAME/checkpoints/global_step_1000"
if [[ ! -f "$smol_checkpoint/actor/model_state_dict/full_weights.pt" ]]; then
    printf 'Step-1000 checkpoint is absent; training is not complete.\n' >&2
    exit 1
fi
SMOLVLA_RUN_NAME="${SMOLVLA_RUN_NAME}_restored_eval1000" SMOLVLA_ENTRY=eval \
    bash "$smol_script_dir/run_smolvla_rl_smoke.sh" \
    runner.only_eval=true "runner.ckpt_path='$smol_checkpoint'"
if ! grep -qx 'FLUXVLA_EVALUATION_COMPLETED' \
    "$smol_results_root/${SMOLVLA_RUN_NAME}_restored_eval1000/driver.log"; then
    printf 'Restored step-1000 evaluation did not complete.\n' >&2
    exit 1
fi
printf 'FLUXVLA_MIXED_1000_COMPLETED\n'
