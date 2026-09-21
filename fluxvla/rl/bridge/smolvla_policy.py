"""SmolVLA frontend with the shared RLinf flow-SDE PPO contract."""

import torch
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead

from fluxvla.engines.utils.model_utils import make_att_2d_masks
from fluxvla.models.vlas.smolvla_flowmatching import SmolVLAFlowMatching
from .flow_policy import FlowPPOPolicyMixin
from .sampler import timesteps


class FluxSmolVLARLPolicy(FlowPPOPolicyMixin, SmolVLAFlowMatching, BasePolicy):
    """Preserve SFT keys; freeze VLM and its state projection, add a critic.

    Text/expert decoder forwards are bypassed by SmolVLA's interleaving.
    Never FSDP-wrap LlamaDecoderLayer: its unshard hooks would not execute.
    The recipe leaves these parameters under the root FSDP2 forward.
    """

    _no_split_modules = ('SmolVLMEncoderLayer', )

    @property
    def n_action_steps(self):
        return self.chunk_size

    def configure_rl(self,
                     observation_adapter,
                     *,
                     action_chunk=10,
                     action_dim=7,
                     noise_level=0.5,
                     compute_dtype=torch.float32,
                     rollout_micro_batch_size=None,
                     value_hidden_sizes=(512, 128)):
        if not 0 < action_chunk <= self.chunk_size:
            raise ValueError('action_chunk must be within the model horizon')
        if not 0 < action_dim <= self.max_action_dim:
            raise ValueError('action_dim must be within the model dimension')
        if noise_level <= 0 or not torch.isfinite(torch.tensor(noise_level)):
            raise ValueError('PPO requires a finite, positive noise_level')
        if compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('compute_dtype must be FP32 or BF16')
        if (rollout_micro_batch_size is not None
                and rollout_micro_batch_size < 1):
            raise ValueError('rollout_micro_batch_size must be positive')
        timesteps(self.num_steps, 'cpu')
        self.observation_adapter = observation_adapter
        self.action_chunk = int(action_chunk)
        self.action_env_dim = int(action_dim)
        self.noise_level = float(noise_level)
        self.compute_dtype = compute_dtype
        self.rollout_micro_batch_size = rollout_micro_batch_size
        self.value_head = ValueHead(
            self.vlm_backbone.hidden_size,
            hidden_sizes=value_hidden_sizes,
            activation='relu')
        self.requires_grad_(False)
        for name in ('llm_expert', 'action_in_proj', 'action_out_proj',
                     'action_time_mlp_in', 'action_time_mlp_out',
                     'value_head'):
            getattr(self, name).requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for name in ('vlm_backbone', 'state_proj'):
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
        return self

    def _prefix(self, obs):
        # Same helpers and ordering as native forward_model's prefix-only
        # path, retaining its final hidden states for the value head as well.
        # No cache/hidden states are persisted in the rollout trajectory.
        with torch.no_grad(), self._network_context():
            hidden, mask, attention = self.embed_prefix(
                obs['images'], obs['img_masks'], obs['lang_tokens'],
                obs['lang_masks'], obs['states'])
            attention = make_att_2d_masks(mask, attention)
            positions = mask.cumsum(dim=1) - 1
            layers, self_attention = self._get_model_layers()
            cache = [None] * self.num_vlm_layers
            for index, layer in enumerate(layers[0]):
                forward = (
                    self._forward_attn_layer if self_attention[index] else
                    self._forward_cross_attn_layer)
                output, _ = forward(
                    layer,
                    layers[1][index],
                    hidden,
                    None,
                    positions,
                    attention,
                    use_cache=True,
                    past_key_values=cache,
                    layer_idx=index)
                hidden = self._apply_residual_ffn(layer, output, hidden)
            hidden = self.vlm_backbone.norm(hidden)
        return hidden, mask, cache

    def _velocity(self, obs, mask, cache, x, time):
        with self._network_context():
            return self._denoise_step(x, mask, cache, time).float()
