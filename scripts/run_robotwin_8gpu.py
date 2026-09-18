#!/usr/bin/env python3
"""Budget-preserving SFT -> eight-GPU PPO -> paired held-out evaluation."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from robotwin_paired_report import load_episodes, wilson

from fluxvla.rl.experiment import choose_best, require_wandb


def verify_reused_sft_protocol():
    """Reuse the baseline only if evaluation semantics are unchanged."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from fluxvla.rl.train import prepare_environment
    source = prepare_environment()
    signatures = []
    with initialize_config_dir(
            version_base='1.3', config_dir=str(source / 'configs/rl')):
        for name in ('robotwin_adjust_bottle_eval_fluxvla_pi05',
                     'robotwin_adjust_bottle_eval_fluxvla_pi05_8gpu'):
            cfg = compose(config_name=name)
            model = OmegaConf.to_container(cfg.rollout.model, resolve=True)
            # This option is used only by train-mode sampling/bootstrap, not
            # paired ODE evaluation (which always uses five envs per worker).
            model['fluxvla'].pop('rollout_micro_batch_size', None)
            env = cfg.env.eval
            signatures.append({
                'model': model,
                'noise_seed': cfg.actor.seed,
                'env': {
                    key: OmegaConf.to_container(env[key], resolve=True)
                    if OmegaConf.is_config(env[key]) else env[key]
                    for key in ('env_type', 'total_num_envs', 'rollout_epoch',
                                'seed', 'auto_reset', 'ignore_terminations',
                                'max_episode_steps',
                                'max_steps_per_rollout_epoch', 'center_crop',
                                'assets_path', 'seeds_path', 'task_config')
                },
                'rollout_workers': 2,
                'env_workers': 2,
            })
    if signatures[0] != signatures[1]:
        raise RuntimeError(
            'Evaluation semantics changed; cannot reuse original SFT baseline')
    return signatures[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-original-sft', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('All long jobs must run inside tmux')
    root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
    source = Path(__file__).resolve().parents[1]
    gate_path = root / 'preflight/eight_gpu_v1/ray_gate.json'
    gate = json.loads(gate_path.read_text())
    if not gate['passed'] or gate['actor_world_size'] != 4 or gate[
            'rollout_world_size'] != 2:
        raise RuntimeError('Eight-GPU gate did not pass')
    if any(row['train_live_subenvs']
           for row in gate['env_memory_after_rollout']):
        raise RuntimeError(
            'Simulator objects were not released at rollout boundary')
    probability_gate_path = (
        root / 'preflight/eight_gpu_probability_v1/ray_gate.json')
    probability_gate = json.loads(probability_gate_path.read_text())
    if not probability_gate['passed']:
        raise RuntimeError(
            'Full shuffled training-probability gate did not pass')
    loss_gate_path = root / 'preflight/eight_gpu_loss_sync_v1/ray_gate.json'
    if not json.loads(loss_gate_path.read_text())['passed']:
        raise RuntimeError(
            'Actual first-update PPO loss input audit did not pass')
    budget = json.loads((root / 'gpu_budget.json').read_text())
    if time.time() >= budget['training_deadline'] - 600:
        raise RuntimeError('Original training budget insufficient; '
                           'automatic extension forbidden')
    require_wandb()
    names = {
        'sft': 'adjust_bottle_sft_test'
        if args.reuse_original_sft else 'adjust_bottle_8gpu_sft_test',
        'train': 'adjust_bottle_pi05_8gpu_seed1234',
        'best': 'adjust_bottle_8gpu_rl_best_test',
        'last': 'adjust_bottle_8gpu_rl_last_test',
    }
    for label, name in names.items():
        if label == 'sft' and args.reuse_original_sft:
            continue
        if (root / 'results' / name).exists():
            raise FileExistsError(
                f'Refusing to append into existing experiment: {name}')
    protocol = {
        'started_at':
        time.time(),
        'original_budget':
        budget,
        'names':
        names,
        'initialization':
        'Original SFT; fresh optimizer and value head, not resume60',
        'reused_original_sft':
        args.reuse_original_sft,
        'evaluation_signature':
        verify_reused_sft_protocol() if args.reuse_original_sft else None,
        'baseline_note':
        ('If reused, the prior 150-seed baseline was not rerun on the new '
         'physical GPU placement; model, simulator, worker counts, seeds '
         'and ODE settings are unchanged.'),
        'resources':
        subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'
        ],
                                text=True),
        'gate':
        str(gate_path),
        'full_probability_gate':
        str(probability_gate_path),
        'loss_input_gate':
        str(loss_gate_path),
        'train_envs':
        32,
        'rollout_epoch':
        4,
        'micro_batch_size':
        4,
        'global_batch_size':
        512,
        'update_epoch':
        2,
        'actor_lr':
        1e-6,
        'value_lr':
        1e-4,
        'gradient_sync':
        'Every microbatch; optimizer still accumulates global batch 512',
        'memory_control':
        ('Explicit train/eval environment offload; per-round RSS logging. '
         'Not proof that all native leaks are fixed.'),
        'selection':
        ('Validation only, ties choose earlier; 32 fixed seeds. '
         'Held-out 150 seeds scored exactly once per SFT/best/last.'),
        'source_sha256': {
            str(path.relative_to(source)):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                *sorted((source / 'fluxvla/rl').rglob('*.py')), *sorted((
                    source / 'configs/rl').glob('*.yaml')), *sorted((
                        source / 'scripts').glob('robotwin_*.py')),
                Path(__file__).resolve()
            ]
        },
    }
    with (root / 'results/eight_gpu_protocol.json').open('x') as stream:
        json.dump(protocol, stream, indent=2)

    def stage(phase, log_name, module, args):
        log_path = root / 'logs' / log_name
        print(json.dumps({'phase': phase, 'log': str(log_path)}), flush=True)
        with log_path.open('x') as stream:
            subprocess.run([
                sys.executable,
                'scripts/robotwin_gpu_budget.py',
                '--phase',
                phase,
                '--',
                sys.executable,
                '-u',
                '-m',
                module,
                *args,
            ],
                           stdout=stream,
                           stderr=subprocess.STDOUT,
                           check=True)

    eval_config = '--config-name=robotwin_adjust_bottle_eval_fluxvla_pi05_8gpu'
    if not args.reuse_original_sft:
        stage('baseline', 'sft-8gpu.log', 'fluxvla.rl.eval', [eval_config])
    seeds = json.loads((root / 'protocol/test_seeds.json'
                        ).read_text())['adjust_bottle']['success_seeds']
    baseline = load_episodes(root / 'results' / names['sft'] / 'episodes/test',
                             seeds)
    baseline_report = {
        'n': len(baseline),
        'successes': int(baseline.sum()),
        'rate': float(baseline.mean()),
        'wilson_95ci': wilson(baseline.sum(), len(baseline))
    }
    (root / 'results/sft_8gpu_baseline.json'
     ).write_text(json.dumps(baseline_report, indent=2) + '\n')
    print(json.dumps({'baseline': baseline_report}), flush=True)
    if time.time() >= budget['training_deadline'] - 600:
        raise RuntimeError(
            'Insufficient time for a complete PPO round and validation/save')
    stage('train', 'ppo-8gpu.log', 'fluxvla.rl.train',
          ['--config-name=robotwin_adjust_bottle_ppo_fluxvla_pi05_8gpu'])
    training = root / 'results' / names['train']
    completion = json.loads((training / 'runner_completion.json').read_text())
    records = json.loads(
        (training / 'checkpoint_selection.json').read_text())['records']
    best = choose_best(records)
    last = json.loads((training / 'last_checkpoint.json').read_text())
    if completion['step'] != last['step']:
        raise RuntimeError('Final model was not durably checkpointed')
    paths = {
        'best': str(training / f"checkpoints/global_step_{best['step']}"),
        'last': last['path']
    }
    (root / 'results/eight_gpu_checkpoint_selection.json').write_text(
        json.dumps(
            {
                'records': records,
                'best': best,
                'last': last,
                'paths': paths,
                'completion': completion
            },
            indent=2) + '\n')
    for label, path in paths.items():
        stage('final', f'rl-8gpu-{label}-test.log', 'fluxvla.rl.eval', [
            eval_config, f'runner.ckpt_path={path}',
            f'runner.logger.experiment_name={names[label]}'
        ])
    subprocess.run([
        sys.executable,
        'scripts/robotwin_paired_report.py',
        '--seeds',
        str(root / 'protocol/test_seeds.json'),
        '--sft',
        str(root / 'results' / names['sft'] / 'episodes/test'),
        '--best',
        str(root / 'results' / names['best'] / 'episodes/test'),
        '--last',
        str(root / 'results' / names['last'] / 'episodes/test'),
        '--output',
        str(root / 'results/eight_gpu_paired_report.json'),
    ],
                   check=True)


if __name__ == '__main__':
    main()
