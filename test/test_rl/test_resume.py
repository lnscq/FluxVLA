"""OOM continuation keeps batch contracts and the original validation best."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.benchmarks.robotwin.protocol import choose_best
from fluxvla.rl.rlinf_registry import register
from fluxvla.rl.train import prepare_environment, validate_frontend_cfg

ROOT = Path(__file__).resolve().parents[2]


def resume_module():
    spec = importlib.util.spec_from_file_location(
        'resume_robotwin_pilot', ROOT / 'scripts/resume_robotwin_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_record(folder, step, score):
    checkpoint = folder / f'checkpoints/global_step_{step}'
    weights = checkpoint / 'actor/model_state_dict/full_weights.pt'
    weights.parent.mkdir(parents=True)
    weights.touch()
    (folder / 'checkpoint_selection.json').write_text(
        json.dumps({
            'records': [{
                'step': step,
                'success_rate': score,
                'split': 'validation'
            }]
        }))
    return checkpoint


def test_resume_preserves_original_best_path(tmp_path):
    old, new = tmp_path / 'original', tmp_path / 'resumed'
    checkpoint = make_record(old, 20, 31 / 32)
    make_record(new, 70, 31 / 32)
    loader = resume_module().selection_records
    best = choose_best(loader(old) + loader(new))
    assert best['step'] == 20
    assert best['checkpoint'] == str(checkpoint)


def test_resume_rejects_missing_checkpoint(tmp_path):
    (tmp_path / 'checkpoint_selection.json').write_text(
        json.dumps({
            'records': [{
                'step': 70,
                'success_rate': 1,
                'split': 'validation'
            }]
        }))
    with pytest.raises(FileNotFoundError):
        resume_module().selection_records(tmp_path)


def test_resume_16_env_global_batch_contract(tiny_assets, monkeypatch):
    import rlinf.config as backend
    prepare_environment()
    register()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(
            config_name='robotwin_adjust_bottle_ppo_fluxvla_pi05',
            overrides=[
                f'actor.model.model_path={tiny_assets[2]}',
                'env.train.total_num_envs=16'
            ])
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg)
    monkeypatch.setattr(backend, 'Cluster', lambda *a, **k: None)
    monkeypatch.setattr(
        backend, 'HybridComponentPlacement',
        lambda *a, **k: SimpleNamespace(get_world_size=lambda name: 2))
    backend.validate_embodied_cfg(cfg)
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.micro_batch_size == 2
    assert cfg.env.train.total_num_envs * (200 // 50) == 64
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 32
    assert cfg.actor.optim.lr == 5e-6
    assert cfg.algorithm.update_epoch == 5
