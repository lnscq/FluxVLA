#!/usr/bin/env python3
"""Two real GPU ranks: FSDP2/PPO/backward and backend DCP save/restore.

This does not stand in for the separate Ray actor-to-rollout communication
gate. Input must be an actual RoboTwin reset observation, not a mock env.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def fingerprints(model, frozen_only=False):
    result = {}
    for name, parameter in model.named_parameters():
        if frozen_only and parameter.requires_grad:
            continue
        data = local(parameter.detach()).cpu().contiguous().numpy().tobytes()
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def main():
    import rlinf.algorithms  # noqa: F401
    from rlinf.algorithms.registry import policy_loss
    from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy

    from fluxvla.rl.bridge.builder import build_pi05_policy
    from fluxvla.rl.train import prepare_environment

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl'))
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    # Flux's distributed logging may initialize the group during import when
    # torchrun supplies RANK/WORLD_SIZE. Reuse that actual NCCL group.
    if not dist.is_initialized():
        dist.init_process_group('nccl')
    if dist.get_backend() != 'nccl':
        raise RuntimeError('The GPU gate requires the NCCL backend')
    if dist.get_world_size() != 2:
        raise ValueError('This gate requires exactly two actual GPU ranks')
    root = prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(config_name='robotwin_adjust_bottle_ppo_fluxvla_pi05')
    OmegaConf.resolve(cfg)
    raw = torch.load(
        args.root / 'preflight/real_observation.pt', weights_only=True)
    # Two observations per rank, as in the actual actor microbatch.
    raw = {
        key: value * 2 if isinstance(value, list) else value.repeat(
            2, *([1] * (value.ndim - 1)))
        for key, value in raw.items()
    }
    start = time.monotonic()
    model = build_pi05_policy(cfg.actor.model).cuda()
    _, old = model.predict_action_batch(raw, mode='train')
    strategy = FSDP2Strategy(
        cfg.actor, world_size=2, dp_group=dist.group.WORLD)
    model = strategy.wrap_model(model, init_device_mesh('cuda', (2, )))
    # Match RLinf's lifecycle: optimizer owns the sharded parameters BEFORE
    # first forward, not temporary unsharded views visible during a forward.
    optimizer = torch.optim.AdamW(
        [{
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith('value_head.')
            ],
            'lr':
            5e-6
        }, {
            'params': list(model.value_head.parameters()),
            'lr': 1e-4
        }],
        betas=(.9, .95))
    from rlinf.utils.utils import warmup_optimizer_state
    warmup_optimizer_state(optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    frozen = fingerprints(model, frozen_only=True)
    before = fingerprints(model)
    model.train()
    output = model(forward_inputs=old['forward_inputs'], compute_entropy=True)
    max_drift = float(
        (output['logprobs'] - old['prev_logprobs']).detach().abs().max())
    torch.testing.assert_close(
        output['logprobs'], old['prev_logprobs'], rtol=1e-5, atol=1e-5)
    advantages = torch.tensor([[1.], [-0.5]], device='cuda')
    loss, _ = policy_loss(
        loss_type='actor_critic',
        task_type='embodied',
        reward_type='chunk_level',
        logprob_type='chunk_level',
        single_action_dim=14,
        logprobs=output['logprobs'],
        old_logprobs=old['prev_logprobs'],
        values=output['values'],
        prev_values=old['prev_values'],
        advantages=advantages,
        returns=old['prev_values'] + advantages,
        loss_mask=torch.ones(2, 1, dtype=torch.bool, device='cuda'),
        clip_ratio_low=.2,
        clip_ratio_high=.2,
        value_clip=.2,
        huber_delta=10.)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite PPO loss')
    loss.backward()
    gradients = [
        local(p.grad) for p in model.parameters() if p.grad is not None
    ]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Non-finite or absent gradient')
    grad_norm = float(strategy.clip_grad_norm_(model))
    rank_norms = [None, None]
    dist.all_gather_object(rank_norms, grad_norm)
    assert rank_norms[0] == rank_norms[
        1], 'Global gradient clipping must agree across ranks'
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    after = fingerprints(model)
    assert frozen == fingerprints(model, frozen_only=True)
    changed = [name for name in before if before[name] != after[name]]
    assert any(name.startswith('llm_expert.') for name in changed)
    assert any(name.startswith('value_head.') for name in changed)
    checkpoint = args.root / 'preflight/fsdp_checkpoint'
    strategy.save_checkpoint(model, optimizer, scheduler, str(checkpoint))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    strategy.load_checkpoint(model, optimizer, scheduler, str(checkpoint))
    assert fingerprints(model) == after
    report = {
        'passed': True,
        'rank': rank,
        'world_size': 2,
        'old_logprob_max_abs_drift': max_drift,
        'loss': float(loss.detach()),
        'gradient_norm': grad_norm,
        'frozen_unchanged': True,
        'expert_and_value_updated': True,
        'dcp_restored': True,
        'elapsed_seconds': time.monotonic() - start,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated()
    }
    (args.root / f'preflight/fsdp_rank{rank}.json'
     ).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
