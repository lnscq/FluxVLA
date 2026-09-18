#!/usr/bin/env python3
"""Replay one failed microbatch through repeated backward, with no updates."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from fluxvla.rl.bridge.builder import build_pi05_policy
from fluxvla.rl.train import prepare_environment


def main():
    import rlinf.algorithms  # noqa: F401
    from rlinf.algorithms.registry import policy_loss
    from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy

    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    torch.manual_seed(1234)
    if world > 1:
        dist.init_process_group('nccl')
    root = Path(os.environ['FLUX_ROBOTWIN_ROOT'])
    source = prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(source / 'configs/rl')):
        cfg = compose(config_name='robotwin_adjust_bottle_preflight_8gpu')
    # Explicitly exercise the unsafe old mode unless the comparison arm asks
    # for per-microbatch sync; production defaults to the workaround.
    cfg.actor.fsdp_config.enable_gradient_accumulation = os.environ.get(
        'FLUX_ACCUM_SYNC_EACH') != '1'
    model = build_pi05_policy(cfg.actor.model).cuda()
    strategy = None
    if world > 1:
        strategy = FSDP2Strategy(
            cfg.actor, world_size=world, dp_group=dist.group.WORLD)
        model = strategy.wrap_model(
            model,
            init_device_mesh('cuda', (world, ), mesh_dim_names=['fsdp']))
    sample = torch.load(
        root / 'preflight/eight_gpu_loss_v1/loss_failed_sample_rank_0.pt',
        weights_only=True)
    inputs = {
        key: value.cuda()
        for key, value in sample['forward_inputs'].items()
    }
    old = sample['prev_logprobs'].cuda()

    def local(tensor):
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    churn = os.environ.get('FLUX_ACCUM_CHURN') == '1'
    if churn:
        from rlinf.utils.utils import warmup_optimizer_state
        optimizer = torch.optim.AdamW([
            parameter
            for parameter in model.parameters() if parameter.requires_grad
        ],
                                      lr=1e-6)
        warmup_optimizer_state(optimizer)
        optimizer.zero_grad()
    before = {
        name: local(parameter.detach()).clone()
        for name, parameter in model.named_parameters()
    } if not churn else {}
    model.train()
    report = []
    for step in range(5):
        if churn:
            sample = torch.load(
                root / ('preflight/eight_gpu_loss_v1/'
                        f'loss_failed_sample_rank_{step % 4}.pt'),
                weights_only=True)
            inputs = {
                key: value.cuda().contiguous()
                for key, value in sample['forward_inputs'].items()
            }
            old = sample['prev_logprobs'].cuda()
        if strategy:
            strategy.before_micro_batch(model, is_last_micro_batch=step == 4)
        output = model(forward_inputs=inputs)
        difference = output['logprobs'].detach() - old
        loss, metrics = policy_loss(
            loss_type='actor_critic',
            task_type='embodied',
            reward_type='chunk_level',
            logprob_type='chunk_level',
            single_action_dim=14,
            logprobs=output['logprobs'],
            old_logprobs=old,
            values=output['values'],
            prev_values=torch.zeros(4, 1, device='cuda'),
            advantages=torch.tensor([[1.], [-.5], [.3], [-.2]], device='cuda'),
            returns=torch.ones(4, 1, device='cuda'),
            clip_ratio_low=.2,
            clip_ratio_high=.2,
            value_clip=.2,
            huber_delta=10.)
        row = {
            'step': step,
            'drift': difference.abs().max().item(),
            'chunk_logratio': difference.sum(dim=(1, 2)).tolist(),
            'ratio': metrics['actor/ratio']
        }
        loss.backward()
        row['changed_weights'] = [
            name for name, parameter in model.named_parameters()
            if name in before
            and not torch.equal(local(parameter.detach()), before[name])
        ]
        row['weights_checked'] = bool(before)
        report.append(row)
        print(json.dumps(row), flush=True)
        del loss, output
    name = os.environ.get('FLUX_ACCUM_LABEL', f'world{world}')
    path = (
        root / 'preflight/eight_gpu_loss_v1' /
        f'accumulation_{name}_rank{rank}.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
