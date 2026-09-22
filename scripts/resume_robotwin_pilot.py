#!/usr/bin/env python3
"""Resume the interrupted pilot, preserving its budget and held-out test."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fluxvla.rl.benchmarks.robotwin.protocol import choose_best
from fluxvla.rl.utils.logging import require_wandb


def selection_records(run_dir):
    """Resolve each validation score to its own run's durable checkpoint."""
    data = json.loads((run_dir / 'checkpoint_selection.json').read_text())
    records = []
    for row in data['records']:
        checkpoint = run_dir / f"checkpoints/global_step_{row['step']}"
        if not (checkpoint /
                'actor/model_state_dict/full_weights.pt').is_file():
            raise FileNotFoundError(checkpoint)
        records.append(dict(row, checkpoint=str(checkpoint)))
    return records


def main():
    if not os.environ.get('TMUX'):
        raise RuntimeError('Launch long jobs inside tmux')
    root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
    budget = json.loads((root / 'gpu_budget.json').read_text())
    if time.time() >= budget['training_deadline'] - 600:
        raise RuntimeError(
            'Insufficient original training budget; do not extend')
    require_wandb()
    original = root / 'results/adjust_bottle_pi05_seed1234'
    name = 'adjust_bottle_pi05_seed1234_resume60'
    resumed = root / 'results' / name
    checkpoint = original / 'checkpoints/global_step_60'
    if not (checkpoint / 'actor/dcp_checkpoint/.metadata').is_file():
        raise FileNotFoundError(checkpoint)
    # Never overwrite a completed/partial scored test or a previous resume.
    for label in ('best', 'last'):
        if (root / f'results/adjust_bottle_rl_{label}_test').exists():
            raise FileExistsError(f'Test output already exists: {label}')
    resumed.mkdir(exist_ok=False)
    (resumed / 'resume_protocol.json').write_text(
        json.dumps(
            {
                'parent_run':
                str(original),
                'parent_wandb_run':
                'vcr4joqv',
                'resume_checkpoint':
                str(checkpoint),
                'reason':
                'Host RAM OOM during round-70 validation; no GPU OOM',
                'train_envs_before':
                32,
                'train_envs_after':
                16,
                'global_batch_size':
                64,
                'micro_batch_size':
                2,
                'algorithm_and_learning_rates':
                'unchanged',
                'evaluation_protocol':
                'unchanged; validation 8x4, test 10x15',
                'budget':
                budget,
                'resume_boundary':
                ('Actor/optimizer restored by RLinf; simulator and rollout '
                 'RNG restart. Not a bitwise continuation.'),
                'memory_status':
                ('Reduced parallelism mitigates OOM; underlying simulator '
                 'memory growth is not proven fixed.'),
            },
            indent=2) + '\n')

    def stage(phase, log_name, module, overrides):
        command = [
            sys.executable, 'scripts/robotwin_gpu_budget.py', '--phase', phase,
            '--', sys.executable, '-u', '-m', module, *overrides
        ]
        log_path = root / 'logs' / log_name
        print(
            json.dumps({
                'phase': phase,
                'log': str(log_path),
                'command': command
            }),
            flush=True)
        with log_path.open('x') as stream:
            subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    stage('train', 'ppo-resume60.log', 'fluxvla.rl.train', [
        '--config-name=robotwin_adjust_bottle_ppo_fluxvla_pi05',
        f'runner.resume_dir={checkpoint}',
        f'runner.logger.experiment_name={name}',
        'env.train.total_num_envs=16',
    ])
    # Merge ONLY validation records. A new log directory must not discard the
    # original round-20 best or resolve it under the resumed checkpoint tree.
    records = selection_records(original) + selection_records(resumed)
    best = choose_best(records)
    last = json.loads((resumed / 'last_checkpoint.json').read_text())
    if int(last['step']) <= 60:
        raise RuntimeError('No durable post-resume checkpoint')
    selection = {'records': records, 'best': best, 'last': last}
    (root / 'results/final_checkpoint_selection.json'
     ).write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection), flush=True)
    for label, path in (('best', best['checkpoint']), ('last', last['path'])):
        stage('final', f'rl-{label}-test.log', 'fluxvla.rl.eval', [
            '--config-name=robotwin_adjust_bottle_eval_fluxvla_pi05',
            f'runner.ckpt_path={path}',
            f'runner.logger.experiment_name=adjust_bottle_rl_{label}_test',
        ])
    subprocess.run([
        sys.executable,
        'scripts/robotwin_paired_report.py',
        '--seeds',
        str(root / 'protocol/test_seeds.json'),
        '--sft',
        str(root / 'results/adjust_bottle_sft_test/episodes/test'),
        '--best',
        str(root / 'results/adjust_bottle_rl_best_test/episodes/test'),
        '--last',
        str(root / 'results/adjust_bottle_rl_last_test/episodes/test'),
        '--output',
        str(root / 'results/paired_report.json'),
    ],
                   check=True)


if __name__ == '__main__':
    main()
