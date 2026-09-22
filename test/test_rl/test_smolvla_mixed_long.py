"""The long mixed-task run changes budget, not PPO or sampling semantics."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.train import prepare_environment, validate_frontend_cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('evaluation', [False, True])
def test_long_recipe_preserves_mixed_control(tiny_assets, evaluation):
    prepare_environment()
    overrides = [
        f'actor.model.model_path={tiny_assets[2]}',
        f'actor.model.fluxvla.tokenizer_path={tiny_assets[3]}',
        'actor.model.fluxvla.norm_stats_path=/not-read.json',
        f'runner.only_eval={str(evaluation).lower()}'
    ]
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        short = compose(
            config_name='libero_10_ppo_fluxvla_smolvla_8gpu_200',
            overrides=overrides)
        long = compose(
            config_name='libero_10_ppo_fluxvla_smolvla_8gpu_1000',
            overrides=overrides)
    validate_frontend_cfg(long, evaluation=evaluation)
    assert long.runner.max_steps == long.runner.max_epochs == 1000
    assert long.runner.save_interval == long.runner.val_check_interval == 25
    assert long.runner.resume_dir is None
    assert long.env.train.get('task_id_filter') is None
    assert long.env.eval.total_num_envs * long.env.eval.rollout_epoch == 100
    # Undo the deliberately changed budget/log identity and require all
    # interpolated settings to match, including scheduler and video paths.
    for field in ('max_steps', 'max_epochs', 'save_interval',
                  'val_check_interval'):
        long.runner[field] = short.runner[field]
    long.runner.logger.experiment_name = short.runner.logger.experiment_name
    assert OmegaConf.to_container(long, resolve=True) == \
        OmegaConf.to_container(short, resolve=True)


@pytest.mark.parametrize('checkpoint,completed', [(False, False),
                                                  (True, False), (True, True)])
def test_long_launcher_requires_restored_eval(tmp_path, checkpoint, completed):
    stub = tmp_path / 'bash'
    stub.write_text('''#!/bin/bash
set -eu
if [[ "$1" == */run_smolvla_rl_200.sh ]]; then
    if [[ "$TEST_CHECKPOINT" == 1 ]]; then
        ckpt="$SMOLVLA_RESULTS_ROOT/$SMOLVLA_RUN_NAME/checkpoints"
        mkdir -p "$ckpt/global_step_1000/actor/model_state_dict"
        touch "$ckpt/global_step_1000/actor/model_state_dict/full_weights.pt"
    fi
else
    mkdir -p "$SMOLVLA_RESULTS_ROOT/$SMOLVLA_RUN_NAME"
    printf '%s\\n' "$TEST_EVAL_OUTPUT" > \
        "$SMOLVLA_RESULTS_ROOT/$SMOLVLA_RUN_NAME/driver.log"
fi
''')
    stub.chmod(0o755)
    result = subprocess.run(
        ['/bin/bash',
         str(ROOT / 'scripts/run_smolvla_rl_mixed_long.sh')],
        env=dict(
            os.environ,
            PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
            FLUX_RL_PYTHON='/bin/true',
            SMOLVLA_RESULTS_ROOT=str(tmp_path),
            SMOLVLA_RUN_NAME='test',
            TEST_CHECKPOINT=str(int(checkpoint)),
            TEST_EVAL_OUTPUT='FLUXVLA_EVALUATION_COMPLETED'
            if completed else 'Traceback: restored eval failed'),
        capture_output=True,
        text=True,
        timeout=15)
    assert (result.returncode == 0) == (checkpoint and completed)
    assert ('FLUXVLA_MIXED_1000_COMPLETED' in result.stdout) == (
        checkpoint and completed)
