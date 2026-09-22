#!/usr/bin/env python3
"""Audit restored rollout samples under shuffled and grad-enabled forwards."""

import argparse
import json
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from fluxvla.rl.models.builder import build_pi05_policy
from fluxvla.rl.train import prepare_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample')
    parser.add_argument('--sft', action='store_true')
    parser.add_argument(
        '--output', default='preflight/eight_gpu_v1/shuffled_probability.json')
    args = parser.parse_args()
    root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
    source = prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(source / 'configs/rl')):
        cfg = compose(config_name='robotwin_adjust_bottle_preflight_8gpu')
    model = build_pi05_policy(cfg.actor.model).cuda()
    checkpoint = (
        root / 'results/adjust_bottle_integration_gate_8gpu' /
        'checkpoints/global_step_1/actor/model_state_dict/'
        'full_weights.pt')
    if not args.sft:
        model.load_state_dict(
            torch.load(checkpoint, map_location='cpu', weights_only=True),
            strict=True)
    sample = torch.load(
        args.sample or root / 'preflight/eight_gpu_v1/ray_sample_0.pt',
        weights_only=True)
    inputs = {
        key: value.cuda()
        for key, value in sample['forward_inputs'].items()
    }
    old = sample['prev_logprobs'].cuda()
    size = len(old)
    torch.manual_seed(1234)
    permutations = {
        'ordered': torch.arange(size),
        'shuffled': torch.randperm(size)
    }
    report = {
        'denoise_indices': inputs['denoise_inds'][:, 0].tolist(),
        'checks': []
    }
    for order, permutation in permutations.items():
        for grad in (False, True):
            model.train(grad)
            for start in range(0, size, 4):
                indices = permutation[start:start + 4]
                with torch.set_grad_enabled(grad):
                    outputs = model(
                        forward_inputs={
                            key: value[indices].contiguous()
                            for key, value in inputs.items()
                        })
                diff = outputs['logprobs'].detach() - old[indices]
                report['checks'].append({
                    'order':
                    order,
                    'grad_enabled':
                    grad,
                    'start':
                    start,
                    'element_max_abs_drift':
                    diff.abs().max().item(),
                    'chunk_logratio':
                    diff.sum(dim=(1, 2)).tolist(),
                })
                del outputs
    path = root / args.output
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
