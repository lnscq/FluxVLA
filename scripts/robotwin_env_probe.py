#!/usr/bin/env python3
"""Real reset/step witness, saves a real observation for the FSDP gate."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.robotwin_env import AuditedRoboTwinEnv
from fluxvla.rl.train import prepare_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl'))
    args = parser.parse_args()
    root = prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(config_name='robotwin_adjust_bottle_ppo_fluxvla_pi05')
    OmegaConf.resolve(cfg)
    env_cfg = cfg.env.train
    start = time.monotonic()
    env = AuditedRoboTwinEnv(env_cfg, 1, 0, 1, SimpleNamespace())
    dest = args.root / 'preflight'
    dest.mkdir(parents=True, exist_ok=True)
    try:
        obs, _ = env.reset()
        obs['task_descriptions'] = [
            str(item) for item in obs['task_descriptions']
        ]
        assert obs['states'].shape == (1, 14)
        assert obs['wrist_images'].shape[1] == 2
        assert obs['main_images'].dtype == torch.uint8
        torch.save(obs, dest / 'real_observation.pt')
        # Hold the measured joint/gripper pose for one simulated control step.
        result = env.chunk_step(obs['states'][:, None].float())
        assert torch.isfinite(result[1]).all()
        import imageio.v2 as imageio
        frames = [
            obs['main_images'][0].numpy(),
            result[0][-1]['main_images'][0].numpy()
        ]
        imageio.mimsave(dest / 'reset_step.mp4', frames, fps=2)
        report = {
            'passed': True,
            'seed': int(env.reset_state_ids[0]),
            'state_shape': list(obs['states'].shape),
            'wrist_shape': list(obs['wrist_images'].shape),
            'device': torch.cuda.get_device_name(),
            'elapsed_seconds': time.monotonic() - start
        }
        (dest /
         'environment.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        env.offload()


if __name__ == '__main__':
    main()
