"""Nested Hydra recipes retain root namespaces and backend semantics."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from mmengine.config import Config
from omegaconf import OmegaConf

from fluxvla.rl.train import prepare_environment

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs/rl'
RECIPES = sorted(
    str(path.relative_to(CONFIGS).with_suffix(''))
    for path in (CONFIGS / 'benchmarks').rglob('*.yaml')
    if not path.stem.startswith('base'))


def test_config_groups_are_explicit():
    assert not list(CONFIGS.glob('*.yaml'))
    assert not list(CONFIGS.glob('*.json'))
    assert len(RECIPES) == 8
    assert not list((CONFIGS / 'experiments').rglob('*.yaml'))
    assert not list((CONFIGS / 'diagnostics').rglob('*.yaml'))
    assert not list((CONFIGS / 'model').glob('*'))
    assert (CONFIGS / 'runtime/robotwin/nvidia_icd.json').is_file()
    common = OmegaConf.load(CONFIGS / 'base/ppo.yaml')
    assert 'model' not in common.actor
    assert 'wrap_policy' not in common.actor.fsdp_config
    assert 'max_episode_steps' not in common.env.train
    for benchmark in ('libero', 'robotwin'):
        other = 'robotwin' if benchmark == 'libero' else 'libero'
        for path in (CONFIGS / 'benchmarks' / benchmark).rglob('*.yaml'):
            assert f'/benchmarks/{other}/' not in path.read_text()
            assert '/experiments/' not in path.read_text()
            assert '/diagnostics/' not in path.read_text()


@pytest.mark.parametrize('name', RECIPES)
def test_all_nested_recipes_compose_at_root(name):
    prepare_environment()
    with initialize_config_dir(version_base='1.3', config_dir=str(CONFIGS)):
        cfg = compose(config_name=name)
    assert not {'benchmarks', 'experiments', 'diagnostics', 'base'} & set(cfg)
    assert cfg.algorithm.loss_type == 'actor_critic'
    assert cfg.weight_syncer.type == 'bucket'
    assert cfg.actor.fsdp_config.strategy == 'fsdp2'
    assert cfg.env.eval.is_eval
    assert cfg.env.eval.auto_reset and cfg.env.eval.ignore_terminations
    assert not cfg.env.train.video_cfg.save_video
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert raw['env']['train']['reward_coef'] == '${algorithm.reward_coef}'
    assert cfg.actor.fsdp_config.mixed_precision.cast_forward_inputs is False
    smolvla = 'smolvla' in name
    expected = ['SmolVLMEncoderLayer'
                ] if smolvla else ['GemmaDecoderLayer', 'SiglipEncoderLayer']
    assert list(cfg.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap
                ) == expected
    assert cfg.actor.model.model_type == ('fluxvla_smolvla'
                                          if smolvla else 'fluxvla_pi05')
    assert cfg.env.train.env_type == ('robotwin'
                                      if 'robotwin' in name else 'libero')
    assert cfg.runner.only_eval == (Path(name).stem in ('eval', 'eval_8gpu'))


def test_robotwin_mmengine_base_resolves_after_move():
    cfg = Config.fromfile(str(CONFIGS / 'models/pi05/robotwin.py'))
    assert cfg.model.n_action_steps == 50
    assert cfg.model.num_steps == 5
    assert cfg.model.openpi_fp32_flow
    assert cfg.model.vision_backbone.openpi_stem_fp32
