#!/usr/bin/env python3
"""Evaluate validation-selected best/last checkpoints without reselection."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from robotwin_paired_report import compare, load_episodes
from run_robotwin_8gpu import verify_reused_sft_protocol

from fluxvla.rl.experiment import choose_best, require_wandb


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('Run evaluation inside tmux')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        raise ValueError('Tag must be a simple unique experiment name')
    if os.environ.get('FLUX_RL_BUDGET_FILE'):
        raise RuntimeError(
            'Unset the expired training budget for this requested evaluation')
    root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
    source = Path(__file__).resolve().parents[1]
    training = args.training_dir.resolve()
    completion = json.loads((training / 'runner_completion.json').read_text())
    records = json.loads(
        (training / 'checkpoint_selection.json').read_text())['records']
    best = choose_best(records)
    last = json.loads((training / 'last_checkpoint.json').read_text())
    if completion['step'] != last['step']:
        raise RuntimeError('Completion and last checkpoint disagree')
    checkpoints = {
        'best': training / f"checkpoints/global_step_{best['step']}",
        'last': Path(last['path']),
    }
    for checkpoint in checkpoints.values():
        if not (checkpoint /
                'actor/model_state_dict/full_weights.pt').is_file():
            raise FileNotFoundError(checkpoint)
    output = root / 'results' / args.tag
    names = {
        label: f'{args.tag}_{label}_step{step}_test'
        for label, step in (('best', best['step']), ('last', last['step']))
    }
    for path in [
            output, *(root / 'results' / name for name in names.values())
    ]:
        if path.exists():
            raise FileExistsError(f'Refusing to append or overwrite: {path}')

    reference = json.loads(
        (root / 'results/eight_gpu_protocol.json').read_text())
    signature = verify_reused_sft_protocol()
    if signature != reference['evaluation_signature']:
        raise RuntimeError(
            'Evaluation settings differ from audited SFT protocol')
    critical = ('fluxvla/rl/bridge/', 'fluxvla/rl/rollout_worker.py',
                'fluxvla/rl/env_worker.py', 'fluxvla/rl/robotwin_env.py',
                'fluxvla/rl/robotwin_runtime.py', 'fluxvla/rl/eval.py')
    for name, expected in reference['source_sha256'].items():
        if name.startswith(critical) and digest(source / name) != expected:
            raise RuntimeError(
                f'Evaluation source changed since SFT audit: {name}')
    seeds_path = root / 'protocol/test_seeds.json'
    seeds = json.loads(
        seeds_path.read_text())['adjust_bottle']['success_seeds']
    if len(seeds) != 150 or len(set(seeds)) != 150:
        raise ValueError('Expected exactly 150 unique official test seeds')
    baseline_dir = root / 'results/adjust_bottle_sft_test/episodes/test'
    baseline = load_episodes(baseline_dir, seeds)
    require_wandb()
    output.mkdir()
    protocol = {
        'started_at': time.time(),
        'training_dir': str(training),
        'completion': completion,
        'best': best,
        'last': last,
        'names': names,
        'selection':
        'Validation only; ties choose earliest. No test-set reselection.',
        'baseline_dir': str(baseline_dir),
        'baseline_successes': int(baseline.sum()),
        'baseline_note': reference['baseline_note'],
        'evaluation_signature': signature,
        'test_seeds_sha256': digest(seeds_path),
        'checkpoints':
        {label: str(path)
         for label, path in checkpoints.items()},
        'source_sha256': {
            str(path.relative_to(source)): digest(path)
            for path in [
                *sorted((source / 'fluxvla/rl').rglob('*.py')), *sorted((
                    source / 'configs/rl').rglob('*.yaml')), source /
                'configs/rl/model/pi05_robotwin.py',
                Path(__file__).resolve(), source /
                'scripts/robotwin_paired_report.py'
            ]
        },
    }
    save_new(output / 'protocol.json', protocol)
    print(
        json.dumps({
            'phase': 'protocol_locked',
            'output': str(output),
            'best_step': best['step'],
            'last_step': last['step'],
            'sft_successes': int(baseline.sum())
        }),
        flush=True)
    reports = {}
    for label, checkpoint in checkpoints.items():
        weights = checkpoint / 'actor/model_state_dict/full_weights.pt'
        command = [
            sys.executable, '-u', '-m', 'fluxvla.rl.eval',
            '--config-name=robotwin_adjust_bottle_eval_fluxvla_pi05_8gpu',
            f'runner.ckpt_path={checkpoint}',
            f'runner.logger.experiment_name={names[label]}',
            'runner.logger.logger_backends=[wandb,tensorboard]'
        ]
        log_path = output / f'{label}.log'
        save_new(
            output / f'{label}_launch.json', {
                'command': command,
                'started_at': time.time(),
                'weights_sha256': digest(weights),
                'weights_bytes': weights.stat().st_size,
            })
        print(
            json.dumps({
                'phase': label,
                'log': str(log_path),
                'experiment': names[label]
            }),
            flush=True)
        started = time.time()
        with log_path.open('x') as stream:
            subprocess.run(
                command,
                cwd=source,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True)
        episodes = root / 'results' / names[label] / 'episodes/test'
        reports[label] = compare(baseline, load_episodes(episodes, seeds))
        reports[label]['wall_seconds_including_initialization'] = time.time(
        ) - started
        save_new(output / f'{label}_report.json', reports[label])
        print(
            json.dumps({
                'phase': f'{label}_complete',
                **reports[label]
            }),
            flush=True)
    save_new(output / 'paired_report.json', reports)
    save_new(output / 'completion.json', {
        'completed_at': time.time(),
        'status': 'complete'
    })
    print(
        json.dumps({
            'phase': 'complete',
            'report': str(output / 'paired_report.json')
        }),
        flush=True)


if __name__ == '__main__':
    main()
