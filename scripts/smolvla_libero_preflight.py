"""Real LIBERO reset/step gate, optionally with the published SmolVLA SFT."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir


def main():
    from rlinf.envs.libero.libero_env import LiberoEnv

    from fluxvla.rl.train import prepare_environment

    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', action='store_true')
    artifacts = Path(
        os.environ.get(
            'SMOLVLA_ARTIFACT_ROOT',
            Path(__file__).resolve().parents[1] / 'work_dirs/rlinf-smolvla'))
    parser.add_argument(
        '--bundle',
        type=Path,
        default=artifacts / 'weights/FluxVLA-SmolVLA-LIBERO10')
    parser.add_argument(
        '--output', type=Path, default=artifacts / 'preflight/smolvla')
    args = parser.parse_args()
    root = prepare_environment()
    bundle = args.bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(config_name='libero_10_ppo_fluxvla_smolvla_smoke')
    cfg.env.train.total_num_envs = 1
    env = LiberoEnv(
        cfg.env.train,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None)
    try:
        obs, info = env.reset()
        torch.save(obs, output / 'real_observation.pt')
        print(
            'Real LIBERO reset:', {
                key: list(value.shape)
                for key, value in obs.items()
                if isinstance(value, torch.Tensor)
            },
            flush=True)
        report = dict(reset=True, synthetic=False)
        if args.policy:
            from fluxvla.rl.models.builder import build_smolvla_policy
            cfg.actor.model.model_path = str(
                bundle / 'checkpoints' /
                'step-057096-epoch-36-loss=0.2340.safetensors')
            cfg.actor.model.fluxvla.tokenizer_path = str(bundle / 'tokenizer')
            cfg.actor.model.fluxvla.norm_stats_path = str(
                bundle / 'dataset_statistics.json')
            model = build_smolvla_policy(cfg.actor.model).cuda().eval()
            noise = torch.randn(1, 50, 32, device='cuda')
            actions, _ = model.predict_action_batch(
                obs, mode='eval', noise=noise)
            assert torch.isfinite(actions).all()
            _, old = model.predict_action_batch(obs, mode='train')
            model.train()
            replay = model(forward_inputs=old['forward_inputs'])
            drift = float((replay['logprobs'] -
                           old['prev_logprobs']).detach().abs().max())
            assert drift < 1e-4, drift
            report.update(
                sft_loaded=True,
                logprob_max_abs_drift=drift,
                actions_shape=list(actions.shape))
            actions = actions.cpu().numpy()
        else:
            actions = np.zeros((1, 10, 7), dtype=np.float32)
            actions[..., -1] = -1
        result = env.chunk_step(actions)
        report['chunk_step'] = True
        print('Real LIBERO chunk step OK', flush=True)
        report_path = output / ('policy.json' if args.policy else 'env.json')
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        assert result is not None
    finally:
        env.env.close()


if __name__ == '__main__':
    main()
