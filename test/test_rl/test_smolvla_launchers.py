"""Exercise portable shell entrypoints without Ray, models or credentials."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def launcher_env(tmp_path):
    bundle = tmp_path / 'bundle with spaces'
    bundle.mkdir()
    checkpoint = bundle / 'loss=0.2340.safetensors'
    checkpoint.touch()
    tokenizer = bundle / 'tokenizer'
    tokenizer.mkdir()
    stats = bundle / 'statistics.json'
    stats.write_text('{}')
    return dict(
        os.environ,
        SMOLVLA_SFT_PATH=str(checkpoint),
        SMOLVLA_TOKENIZER_PATH=str(tokenizer),
        SMOLVLA_STATS_PATH=str(stats),
        SMOLVLA_RESULTS_ROOT=str(tmp_path / 'results with spaces'),
        SMOLVLA_RUN_NAME='test',
        SMOLVLA_ENTRY='eval',
        SMOLVLA_CONFIG_NAME='libero_10_eval_fluxvla_smolvla',
        FLUX_RL_PYTHON='/bin/echo',
        WANDB_API_KEY='unit-test-placeholder')


def run_smoke(env):
    return subprocess.run(
        ['bash', str(ROOT / 'scripts/run_smolvla_rl_smoke.sh')],
        env=env,
        capture_output=True,
        text=True,
        timeout=15)


def test_portable_paths_and_refuse_overwrite(tmp_path):
    env = launcher_env(tmp_path)
    result = run_smoke(env)
    assert result.returncode == 0, result.stderr
    log = Path(env['SMOLVLA_RESULTS_ROOT']) / 'test/driver.log'
    original = log.read_text()
    assert '-m fluxvla.rl.eval' in original
    assert "actor.model.model_path='" + env[
        'SMOLVLA_SFT_PATH'] + "'" in original
    assert env['WANDB_API_KEY'] not in original + result.stdout + result.stderr
    assert run_smoke(env).returncode != 0
    assert log.read_text() == original


def test_launcher_requires_online_credentials(tmp_path):
    env = launcher_env(tmp_path)
    env.pop('WANDB_API_KEY')
    result = run_smoke(env)
    assert result.returncode != 0
    assert 'WANDB_API_KEY' in result.stderr
    assert not Path(env['SMOLVLA_RESULTS_ROOT']).exists()


@pytest.mark.parametrize('checkpoint,completed', [(False, False),
                                                  (True, False), (True, True)])
def test_single_task_restored_eval_gate(tmp_path, checkpoint, completed):
    stub = tmp_path / 'bash'
    stub.write_text('''#!/bin/bash
set -eu
if [[ "$1" == */run_smolvla_rl_200.sh ]]; then
    if [[ "$TEST_CHECKPOINT" == 1 ]]; then
        run_dir="$SMOLVLA_RESULTS_ROOT/$SMOLVLA_RUN_NAME"
        ckpt="$run_dir/checkpoints/global_step_50/actor/model_state_dict"
        mkdir -p "$ckpt"
        touch "$ckpt/full_weights.pt"
    fi
else
    mkdir -p "$SMOLVLA_RESULTS_ROOT/$SMOLVLA_RUN_NAME"
    run_dir="$SMOLVLA_RESULTS_ROOT/$SMOLVLA_RUN_NAME"
    printf '%s\\n' "$TEST_EVAL_OUTPUT" > "$run_dir/driver.log"
fi
exit 0
''')
    stub.chmod(0o755)
    result = subprocess.run(
        ['/bin/bash',
         str(ROOT / 'scripts/run_smolvla_rl_single_task.sh')],
        env=dict(
            os.environ,
            PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
            SMOLVLA_RESULTS_ROOT=str(tmp_path),
            SMOLVLA_RUN_NAME='test',
            TEST_CHECKPOINT=str(int(checkpoint)),
            TEST_EVAL_OUTPUT='FLUXVLA_EVALUATION_COMPLETED'
            if completed else 'Traceback: restored evaluation failed'),
        capture_output=True,
        text=True,
        timeout=15)
    assert (result.returncode == 0) == (checkpoint and completed)
    restored_log = tmp_path / 'test_restored_eval50/driver.log'
    assert restored_log.exists() == checkpoint


@pytest.mark.parametrize('script,option',
                         [('prepare_smolvla_rl.py', '--output'),
                          ('smolvla_libero_preflight.py', '--bundle')])
def test_artifact_tools_expose_portable_cli(script, option):
    result = subprocess.run(
        [sys.executable,
         str(ROOT / 'scripts' / script), '--help'],
        env=dict(
            os.environ,
            PYTHONPATH=str(ROOT) + os.pathsep +
            os.environ.get('PYTHONPATH', '')),
        capture_output=True,
        text=True,
        timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert option in result.stdout
