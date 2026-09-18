#!/usr/bin/env python3
"""Cross-process OpenPI/Flux parity witness; run each side in its own venv."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def trace_vision(model, output):
    """Capture the first camera to localize cross-version precision drift."""

    def hook(name):

        def capture(module, inputs, result):
            key = f'vision_trace/{name}'
            if key not in output:
                if isinstance(result, tuple):
                    result = result[0]
                output[key] = result.detach().cpu().float()

        return capture

    for name in ('embeddings', 'encoder.layers.0.layer_norm1',
                 'encoder.layers.0.self_attn', 'encoder.layers.0.layer_norm2',
                 'encoder.layers.0.mlp', 'encoder.layers.0',
                 'encoder.layers.26', 'post_layernorm'):
        model.get_submodule(name).register_forward_hook(hook(name))


def trace_language(model, output):
    output['language_trace/inv_freq'] = model.rotary_emb.inv_freq.detach().cpu(
    ).float()

    def hook(name):

        def capture(module, inputs, result):
            key = f'language_trace/{name}'
            if key not in output:
                if isinstance(result, tuple):
                    result = result[0]
                output[key] = result.detach().cpu().float()

        return capture

    for name in ('rotary_emb', 'layers.0.input_layernorm',
                 'layers.0.self_attn.q_proj', 'layers.0.self_attn.k_proj',
                 'layers.0.self_attn.v_proj', 'layers.0.self_attn',
                 'layers.0.post_attention_layernorm', 'layers.0.mlp',
                 'layers.0', 'layers.1', 'layers.17', 'norm'):
        model.get_submodule(name).register_forward_hook(hook(name))


def example():
    rng = np.random.default_rng(20260916)
    return {
        'main_images':
        torch.from_numpy(
            rng.integers(256, size=(1, 240, 320, 3), dtype=np.uint8)),
        'wrist_images':
        torch.from_numpy(
            rng.integers(256, size=(1, 2, 240, 320, 3), dtype=np.uint8)),
        'states':
        torch.from_numpy(rng.uniform(0.1, 0.8, (1, 14)).astype(np.float32)),
        'task_descriptions': ['Adjust_the bottle\nwith the left arm'],
    }


def reference(args, raw, noise):
    # Read the unmodified reference transform without importing RLinf/Ray.
    import importlib.util

    import sentencepiece
    from openpi import transforms
    from openpi.models.model import Observation
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared.normalize import NormStats
    spec = importlib.util.spec_from_file_location(
        'reference_aloha',
        str(
            Path(
                os.environ.get('RLINF_ROOT',
                               Path(__file__).resolve().parents[2] / 'RLinf'))
            / 'rlinf/models/embodiment/openpi/policies/aloha_policy.py'))
    aloha = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = aloha
    spec.loader.exec_module(aloha)
    AlohaInputs, AlohaOutputs = aloha.AlohaInputs, aloha.AlohaOutputs

    stats = json.loads(
        (args.weights / 'physical-intelligence/robotwin/norm_stats.json'
         ).read_text())['norm_stats']
    stats = {
        key: NormStats(**{k: np.asarray(v)
                          for k, v in row.items()})
        for key, row in stats.items()
    }
    tokenizer = object.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 200
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(args.weights / 'paligemma_tokenizer.model'))
    row = AlohaInputs()({
        'state': raw['states'][0].numpy(),
        'prompt': raw['task_descriptions'][0],
        'images': {
            'cam_high': raw['main_images'][0].numpy(),
            'cam_left_wrist': raw['wrist_images'][0, 0].numpy(),
            'cam_right_wrist': raw['wrist_images'][0, 1].numpy()
        }
    })
    for transform in (transforms.Normalize(stats, use_quantiles=True),
                      transforms.ResizeImages(224, 224),
                      transforms.TokenizePrompt(
                          tokenizer, discrete_state_input=True),
                      transforms.PadStatesAndActions(32)):
        row = transform(row)
    from torch.utils._pytree import tree_map
    data = tree_map(lambda x: torch.as_tensor(np.asarray(x)).unsqueeze(0), row)
    observation = Observation.from_dict(data)
    output = {
        'images': torch.cat(list(observation.images.values()), dim=1),
        'img_masks':
        torch.stack(list(observation.image_masks.values()), dim=1),
        'states': observation.state.float(),
        'lang_tokens': observation.tokenized_prompt,
        'lang_masks': observation.tokenized_prompt_mask
    }
    # Verify action restoration independently of model parity as well.
    restored = transforms.Unnormalize(
        stats, use_quantiles=True)({
            'state': row['state'],
            'actions': noise[0].numpy()
        })
    restored = transforms.AbsoluteActions([True] * 6 + [False] + [True] * 6 +
                                          [False])(
                                              restored)
    output['restored_actions'] = torch.from_numpy(
        AlohaOutputs()(restored)['actions']).float().unsqueeze(0)
    if not args.preprocess_only:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import (PI0Pytorch,
                                                       make_att_2d_masks)
        from safetensors.torch import load_file
        config = Pi0Config(pi05=True, action_horizon=50, action_dim=32)
        model = PI0Pytorch(config)
        state = {}
        index = json.loads(
            (args.weights /
             'model.safetensors.index.json').read_text())['weight_map']
        for file in sorted(set(index.values())):
            state.update(load_file(str(args.weights / file)))
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or any(not key.startswith('value_head.')
                                      for key in result.unexpected_keys):
            raise ValueError(f'Reference checkpoint mismatch: {result}')
        del state
        model.eval().to('cuda')
        trace_vision(
            model.paligemma_with_expert.paligemma.model.vision_tower.
            vision_model, output)
        trace_language(model.paligemma_with_expert.paligemma.language_model,
                       output)
        observation = Observation.from_dict(
            tree_map(lambda x: x.to('cuda'), data))
        torch.set_float32_matmul_precision('highest')
        # Explicit class method bypasses the constructor's compiled instance
        # method: compile/CUDA graphs are disabled for this experiment.
        with torch.no_grad():
            observation_parts = model._preprocess_observation(
                observation, train=False)
            images, masks, tokens, token_mask, state = observation_parts
            prefix, mask, attention = model.embed_prefix(
                images, masks, tokens, token_mask)
            output['prefix'] = prefix.cpu().float()
            llm = model.paligemma_with_expert.paligemma.language_model
            llm.config._attn_implementation = 'eager'
            hidden, cache = model.paligemma_with_expert.forward(
                inputs_embeds=[prefix, None],
                attention_mask=model._prepare_attention_masks_4d(
                    make_att_2d_masks(mask, attention)),
                position_ids=mask.cumsum(1) - 1,
                past_key_values=None,
                use_cache=True)
            output['hidden'] = hidden[0].cpu().float()
            output['velocity'] = model.denoise_step(
                state, mask, cache, noise.cuda(),
                torch.ones(1, device='cuda')).cpu()
            output['actions'] = PI0Pytorch.sample_actions(
                model,
                torch.device('cuda'),
                observation,
                noise=noise.cuda(),
                num_steps=5).cpu()
    return output


def flux(args, raw, noise):
    from omegaconf import OmegaConf

    from fluxvla.rl.bridge.builder import build_pi05_policy
    from fluxvla.rl.bridge.robotwin_observation import \
        RoboTwinObservationAdapter
    stats = json.loads(
        (args.weights /
         'physical-intelligence/robotwin/norm_stats.json').read_text())
    adapter = RoboTwinObservationAdapter(
        norm_stats=stats,
        tokenizer_path=args.weights / 'paligemma_tokenizer.model')
    output = adapter(raw)
    output['restored_actions'] = adapter.env_actions(
        noise, 50, 14, env_obs=raw)
    if not args.preprocess_only:
        cfg = OmegaConf.create({
            'model_path': str(args.weights),
            'num_steps': 5,
            'num_action_chunks': 50,
            'action_dim': 14,
            'fluxvla': {
                'config_path':
                str(
                    Path(__file__).resolve().parents[1] /
                    'configs/rl/model/pi05_robotwin.py'),
                'action_horizon':
                50,
                'observation_adapter':
                'robotwin',
                'compute_dtype':
                'bf16',
                'noise_level':
                0.3,
                'tokenizer_path':
                str(args.weights / 'paligemma_tokenizer.model'),
                'norm_stats_path':
                str(args.weights /
                    'physical-intelligence/robotwin/norm_stats.json')
            }
        })
        model = build_pi05_policy(cfg).eval().cuda()
        trace_vision(model.vision_backbone.vision.vision_model, output)
        trace_language(model.llm_backbone, output)
        obs = model._prepare_obs(raw)
        with torch.no_grad(), model._network_context():
            prefix, _, _ = model.embed_prefix(obs['images'],
                                              obs['lang_tokens'],
                                              obs['img_masks'],
                                              obs['lang_masks'])
            output['prefix'] = model._cast_gemma_input(prefix).cpu().float()
            hidden, mask, cache = model._prefix(obs)
            output['hidden'] = hidden.cpu().float()
            output['velocity'] = model._velocity(obs, mask, cache,
                                                 noise.cuda(),
                                                 torch.ones(
                                                     1, device='cuda')).cpu()
        _, result = model.predict_action_batch(
            raw, mode='eval', noise=noise.cuda())
        output['actions'] = result['forward_inputs']['model_action'].reshape(
            1, 50, 32).cpu()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('side', choices=['reference', 'flux', 'compare'])
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl'))
    parser.add_argument('--preprocess-only', action='store_true')
    args = parser.parse_args()
    args.weights = args.root / 'weights/RLinf-Pi05-RoboTwin-SFT-adjust_bottle'
    dest = args.root / 'preflight'
    dest.mkdir(parents=True, exist_ok=True)
    suffix = 'preprocess' if args.preprocess_only else 'model'
    if args.side == 'compare':
        outputs = [
            torch.load(dest / f'{side}_{suffix}.pt', weights_only=True)
            for side in ('reference', 'flux')
        ]
        report, passed = {}, True
        for key, expected in outputs[0].items():
            actual = outputs[1][key]
            difference = (actual.float() - expected.float()).abs()
            exact = key in ('images', 'img_masks', 'lang_tokens', 'lang_masks',
                            'states')
            okay = torch.equal(actual, expected) if exact else torch.allclose(
                actual, expected, atol=1e-3, rtol=1e-3)
            report[key] = {
                'max_abs': float(difference.max()),
                'mean_abs': float(difference.mean()),
                'passed': okay
            }
            passed &= okay
        report['passed'] = passed
        (dest / f'{suffix}_parity.json'
         ).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise SystemExit(1)
        return
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    noise = torch.randn(
        1, 50, 32, generator=torch.Generator().manual_seed(1234))
    output = globals()[args.side](args, example(), noise)
    torch.save(output, dest / f'{args.side}_{suffix}.pt')
    print(f'{args.side.upper()}_{suffix.upper()}_EXPORTED', flush=True)


if __name__ == '__main__':
    sys.path.insert(
        0,
        os.environ.get('RLINF_ROOT',
                       str(Path(__file__).resolve().parents[2] / 'RLinf')))
    main()
