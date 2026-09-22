#!/usr/bin/env bash
# Run INSIDE tmux, after the explicit real-worker gates pass. No watchdog.
set -euo pipefail
if [[ -z ${TMUX:-} ]]; then
    printf 'Start this long experiment in tmux.\n' >&2
    exit 1
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/robotwin_rl_env.sh
RL_PY="$FLUX_ROBOTWIN_ROOT/envs/flux-robotwin/bin/python"
"$RL_PY" - <<'PY'
import json, os
from pathlib import Path
from fluxvla.rl.utils.logging import require_wandb
root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
for name in ('model_parity', 'environment', 'fsdp_rank0', 'fsdp_rank1', 'ray_gate'):
    if not json.loads((root / 'preflight' / f'{name}.json').read_text())['passed']:
        raise SystemExit(f'Gate failed: {name}')
require_wandb()
PY
"$RL_PY" scripts/robotwin_gpu_budget.py --phase baseline -- \
    "$RL_PY" -m fluxvla.rl.eval \
    --config-name=robotwin_adjust_bottle_eval_fluxvla_pi05 \
    > "$FLUX_ROBOTWIN_ROOT/logs/sft-baseline.log" 2>&1
"$RL_PY" - <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from robotwin_paired_report import load_episodes, wilson
root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
seeds = json.loads((root / 'protocol/test_seeds.json').read_text())['adjust_bottle']['success_seeds']
values = load_episodes(root / 'results/adjust_bottle_sft_test/episodes/test', seeds)
result = {'n': len(values), 'successes': int(values.sum()), 'rate': float(values.mean()),
          'wilson_95ci': wilson(values.sum(), len(values)), 'all_150_seeds_once': True}
(root / 'results/sft_baseline.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
PY
"$RL_PY" scripts/robotwin_gpu_budget.py --phase train -- \
    "$RL_PY" -m fluxvla.rl.train \
    --config-name=robotwin_adjust_bottle_ppo_fluxvla_pi05 \
    > "$FLUX_ROBOTWIN_ROOT/logs/ppo-train.log" 2>&1
# Validation-only selection; read fixed paths rather than selecting on test.
readarray -t RL_CHECKPOINTS < <("$RL_PY" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['FLUX_ROBOTWIN_ROOT']) / 'results/adjust_bottle_pi05_seed1234'
selection = json.loads((root / 'checkpoint_selection.json').read_text())['best']
assert selection['split'] == 'validation'
print(root / f"checkpoints/global_step_{selection['step']}")
print(json.loads((root / 'last_checkpoint.json').read_text())['path'])
PY
)
[[ ${#RL_CHECKPOINTS[@]} == 2 ]] || { printf 'No valid best/last checkpoints.\n' >&2; exit 1; }
for RL_INDEX in 0 1; do
    if [[ $RL_INDEX == 0 ]]; then RL_LABEL=best; else RL_LABEL=last; fi
    "$RL_PY" scripts/robotwin_gpu_budget.py --phase final -- \
        "$RL_PY" -m fluxvla.rl.eval \
        --config-name=robotwin_adjust_bottle_eval_fluxvla_pi05 \
        "runner.ckpt_path=${RL_CHECKPOINTS[$RL_INDEX]}" \
        "runner.logger.experiment_name=adjust_bottle_rl_${RL_LABEL}_test" \
        > "$FLUX_ROBOTWIN_ROOT/logs/rl-${RL_LABEL}-test.log" 2>&1
done
"$RL_PY" scripts/robotwin_paired_report.py \
    --seeds "$FLUX_ROBOTWIN_ROOT/protocol/test_seeds.json" \
    --sft "$FLUX_ROBOTWIN_ROOT/results/adjust_bottle_sft_test/episodes/test" \
    --best "$FLUX_ROBOTWIN_ROOT/results/adjust_bottle_rl_best_test/episodes/test" \
    --last "$FLUX_ROBOTWIN_ROOT/results/adjust_bottle_rl_last_test/episodes/test" \
    --output "$FLUX_ROBOTWIN_ROOT/results/paired_report.json"
