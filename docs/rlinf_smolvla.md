# SmolVLA reinforcement learning with RLinf

This optional frontend registers `model_type=fluxvla_smolvla` through the same
`RLINF_EXT_MODULE=fluxvla.rl.rlinf_registry` entry as PI0.5. RLinf still owns
PPO/GAE, environments, FSDP2, weight synchronization and checkpoints. Neither
the RLinf checkout nor the base SmolVLA model implementation is patched.

## Scope and architecture

The initial recipe is LIBERO-10, two RGB cameras, raw 8D proprio, 32D padded
model actions, 7D environment actions, horizon 50, executed chunk 10, and 10
denoising steps. This is an integration recipe, not tuned PPO hyperparameters.
RoboTwin SmolVLA is explicitly rejected until matching dual-arm SFT weights
and a SmolVLA-specific preprocessing contract are validated.

- `bridge/flow_policy.py`: shared PI0.5/SmolVLA rollout, immutable FP32 chains,
  one-random-step flow-SDE (noise level 0.5), Gaussian log-probability replay,
  entropy, bootstrap values and batch-first tensor payloads.
- `bridge/smolvla_policy.py`: native SmolVLA prefix/cache and velocity helpers,
  preserving SFT parameter names. Evaluation calls native `predict_action`.
- `bridge/builder.py`: native PT/safetensors/indexed shards and the existing
  LeRobot name mapping; all expected SFT tensors must be present and match
  shape. The value head is initialized separately, after loading the SFT model.
- `bridge/observation.py`: Flux's SmolVLA-configured image letterboxing,
  tokenizer/prompt, mean/std proprio normalization and action denormalization.
  RLinf images are already rotated. The adapter does not rotate again.

The VLM, including the vision connector, and `state_proj` remain frozen.
`state_proj` is deliberately frozen because it belongs to the cached prefix;
training it would require a differentiable prefix recomputation contract.
The action expert, action input/output projections, action-time MLP and new
value head are trainable. The value head pools final valid prefix tokens
(images, language and state) with a masked mean.

Actor and rollout both have FP32 weights and local BF16 network autocast.
Probability arithmetic and saved chains stay FP32. Training rollout compute
microbatch matches actor microbatch to reduce batch-shape-dependent BF16
drift. ODE evaluation retains its per-worker environment batch size.

### FSDP caveat

SmolVLA interleaves VLM/expert layers by calling their constituent operations,
not `LlamaDecoderLayer.forward`. Wrapping those decoder modules would bypass
their FSDP unshard hooks. This recipe wraps normal vision encoder forwards,
`LinearProjector` and `ValueHead`; text/expert parameters remain owned by the
root FSDP2 module. This is correctness-first and gathers more root parameters
at once than decoder-level sharding. Profile real-model memory/throughput
before scaling. Do not copy PI0.5's Gemma wrapping configuration.

## Dependencies and initial artifacts

Use the same isolated RL environment as the PI0.5 bridge: Python 3.10,
Torch 2.8, Transformers 5.3.0, NumPy 1.26.4, Hydra 1.3.2 and RLinf core.
See [the integration guide](rlinf.md) for installation limitations.
Install RLinf core separately, without its `embodied` extra's Transformers
4.x constraints. LIBERO and its simulator/assets remain required for real
environment runs; the missing offline demonstration dataset directory does
not block online PPO.

Supply all three matching artifacts explicitly:

1. SmolVLA LIBERO-10 SFT checkpoint (not PI0.5 or just a generic base model).
2. The tokenizer directory used by that checkpoint.
3. Flux dataset statistics JSON, keyed by `libero_10_no_noops`, containing
   `proprio` and `action` mean/std statistics. Do not substitute PI0.5 statistics.

The model Python config is
`configs/smolvla/smolvla_libero_10_finetune.py`. Override
`actor.model.fluxvla.config_path` if the SFT architecture differs.
Historical experimental solver/LVFD patches in another working tree are not
part of this implementation: the reference is the committed native SmolVLA.

## Commands

From this checkout, using the RL Python environment:

```bash
export RLINF_ROOT=/absolute/path/RLinf
export PYTHONPATH="$PWD:$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FLUX_RL_EXTERNAL_DISTRIBUTED=1
export USE_TF=0

# Configuration only; starts neither Ray nor training.
python -m fluxvla.rl.train --config-name=libero_10_ppo_fluxvla_smolvla \
  actor.model.model_path=/absolute/path/smolvla_sft.safetensors \
  actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer \
  actor.model.fluxvla.norm_stats_path=/absolute/path/dataset_statistics.json \
  --cfg job --resolve
```

The launchers use the active environment's `python` by default; set
`FLUX_RL_PYTHON=/absolute/path/to/python` to choose another interpreter.
Set `RLINF_ROOT` for a source checkout, or install RLinf into that interpreter.
No machine-specific Python, repository or storage paths are embedded.

To prepare the published SFT bundle, in tmux:

```bash
export SMOLVLA_ARTIFACT_ROOT=/absolute/path/to/rlinf-smolvla
python scripts/prepare_smolvla_rl.py
# If direct Hugging Face access is unavailable, explicitly choose a mirror:
# python scripts/prepare_smolvla_rl.py --endpoint https://hf-mirror.com
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python scripts/smolvla_libero_preflight.py --policy
```

Without `SMOLVLA_ARTIFACT_ROOT`, artifacts default to
`work_dirs/rlinf-smolvla` under this checkout. `--output` overrides the
download destination; preflight accepts `--bundle` and `--output`.
The 200-round and single-task launchers derive checkpoint/tokenizer/statistics
paths from this artifact root, or from `SMOLVLA_BUNDLE_DIR`. Individual
`SMOLVLA_SFT_PATH`, `SMOLVLA_TOKENIZER_PATH`, and `SMOLVLA_STATS_PATH` overrides
take precedence. `SMOLVLA_RESULTS_ROOT` overrides the results directory.
Run only one experiment per Ray instance; these scripts do not implement
concurrent-job resource isolation.

For a real run, open `tmux new -s smolvla-rl`, activate the environment and
set these variables there. Remove `--cfg job --resolve` and add a unique
`runner.logger.experiment_name`. Formal training must set
`runner.logger.logger_backends=[wandb]`; `WANDB_API_KEY` is read only from the
environment and authentication failure stops training. No key belongs in a
config or command history. The default recipe uses TensorBoard for local checks.

Use `python -m fluxvla.rl.eval --config-name=libero_10_eval_fluxvla_smolvla`
with the same three artifact overrides for native-ODE SFT evaluation. To
evaluate PPO weights, additionally set `runner.ckpt_path` to a saved RLinf
checkpoint. For training recovery use `runner.resume_dir` instead; the initial
SFT path remains distinct. The default evaluation layout is not a paired
benchmark protocol; lock seeds, episode counts and noise before reporting gains.

## Tests and evidence boundary

### Real LIBERO smoke run

`scripts/prepare_smolvla_rl.py` downloads the published FluxVLA LIBERO-10 SFT
bundle (revision and hashes pinned in the script) into the selected artifact
directory. The checkpoint, tokenizer and statistics are verified together.
The measured runs used CPFS and an explicitly selected mirror as a transport
fallback; the checkpoint SHA256 matches the previously recorded SFT baseline.

The full SFT model passed `scripts/smolvla_libero_preflight.py --policy` on a
real LIBERO observation: reset, native ODE action generation, ten environment
steps, and stochastic transition replay with zero maximum probability drift.
This is not a success-rate evaluation.

The `libero_10_ppo_fluxvla_smolvla_smoke` recipe runs five PPO rounds on four
GPUs (actor 0-1, environment/rollout 2-3), eight training environments, and
per-round checkpoint/evaluation. It enables the existing actor probability
and finite-gradient audits, and requires online W&B. Run inside tmux:

```bash
export SMOLVLA_SFT_PATH=/absolute/path/sft.safetensors
export SMOLVLA_TOKENIZER_PATH=/absolute/path/tokenizer
export SMOLVLA_STATS_PATH=/absolute/path/dataset_statistics.json
# WANDB_API_KEY must already be present in this shell's environment.
bash scripts/run_smolvla_rl_smoke.sh
```

The launcher quotes Hydra path values, including SFT filenames containing
`=`, and refuses to overwrite a prior run directory. It creates no watchdog.

### Eight-GPU 200-round run

Inside a tmux shell with `WANDB_API_KEY` already exported:

```bash
bash scripts/run_smolvla_rl_200.sh
```

The launcher uses the prepared SFT bundle and a unique run name.
It first evaluates the original SFT model, then starts fresh PPO (no smoke-run
resume). The recipe assigns actor ranks to GPUs 0-3, rollout to 4-5 and
environments to 6-7. It uses 32 training environments, global batch 128,
microbatch 2, two PPO update epochs, actor LR `1e-6` and value LR `1e-4`.
The limit is 200 runner rounds, not 200 individual optimizer updates.
Checkpoints and monitoring evaluations run every ten rounds.

The SFT reference and periodic monitor each use ten parallel environments and
ten evaluation epochs: ten tasks with initial-state trials 0-9 per task,
100 episodes total. Evaluation noise is seeded with `20260920` plus worker
rank and isolated from the training RNG. This fixed monitoring set is not
held out from training; improvement here is not an independent benchmark gain.

Standalone evaluation needs the complete `rollout.model`, so the SmolVLA
base aliases it to `actor.model`. RLinf's Ray shutdown hook can mask a failed
driver exit status; the sequential launcher therefore also requires the
`FLUXVLA_EVALUATION_COMPLETED` signal emitted only after evaluation and metric
logging return successfully. A failed reference must not start PPO.

### Single-task isolation experiment

Run `bash scripts/run_smolvla_rl_single_task.sh` inside tmux with W&B
credentials already in the environment. This bounded experiment selects
LIBERO-10 task 6 (white mug on plate and chocolate pudding to its right),
using the backend's `task_id_filter: [6]` for both training and evaluation.
It starts from the original multi-task SFT weights, not the previous PPO
checkpoint, and does not perform additional SFT.

The actor/rollout/environment placement, 32 training environments, global
batch 128, microbatch 2, two PPO update epochs, learning rates, noise level
0.5 and original normalization remain unchanged from the eight-GPU recipe.
The limit is 50 rounds with checkpoint/evaluation every five rounds.
Evaluation uses ten environments and five epochs: task 6 trials 0-49 exactly
once. These states overlap the training pool; this is a diagnostic monitor,
not a held-out generalization claim. A fresh 50-episode SFT reference is
required; the earlier ten-episode task score is not a matching baseline.

The launcher requires successful SFT evaluation before PPO. After training,
it requires the step-50 full checkpoint and runs a separate restored-weight
evaluation with the identical protocol. A missing checkpoint or absent
evaluation completion marker fails the launcher. It creates no watchdog.
Compare the restored result to both the fresh SFT reference and the in-run
step-50 evaluation; do not hide reproducibility differences or substitute
the best intermediate monitoring score for the final result.

`test_smolvla_single_task.py` validates train/eval configuration, actual
backend random sampling restricted to task 6, and disjoint worker pools
covering the 50 requested evaluation trials without duplicates.

### Measured outcomes (2026-09-20 to 2026-09-21)

These are integration experiments, not tuned or official benchmark results.
All rows use the matching SFT/reference layout and fixed monitoring states.

| Experiment                                      | SFT success_once | Final in-run eval | Restored final eval |
| ----------------------------------------------- | ---------------- | ----------------- | ------------------- |
| LIBERO-10 mixed tasks, 200 rounds, 100 episodes | 67/100           | 56/100            | 53/100              |
| Task 6 only, 50 rounds, 50 episodes             | 23/50            | 28/50             | 28/50               |

The mixed-task recipe degraded; the three-episode discrepancy between in-run
and restored evaluation remains unresolved. The single-task restored run
matched the in-run result episode by episode, with 12 rescues and seven
regressions against SFT. Its success_at_end improved from 19/50 to 26/50.
Single-task periodic success varied from 42% to 64%, so the final 56% is not
evidence of stable convergence. These monitoring states overlap training;
neither run establishes held-out gains, and different per-task data budgets
and run lengths prevent attributing the difference solely to task mixing.

The mixed-task run completed 200 rounds and the single-task run completed
50 rounds on eight GPUs with actual environment sampling, FSDP2 updates and
actor-to-rollout synchronization. The latter training loop took about
2 h 10 min. Probability/finite-gradient checks passed. The mixed-task final
checkpoint audit found the VLM/state projection unchanged from SFT and the
action expert/projections updated, with no missing SFT keys.
The longer ten-task experiment proposal was not executed and is not part of
this delivery. Logs, checkpoints, videos, credentials and downloaded assets
are deliberately excluded from source control.

### Regression and synthetic distributed tests

```bash
python -m pytest test/test_rl -q

# Optional: two actual GPUs, tiny real model, synthetic observations only.
# Run in tmux; explicitly select two free GPUs.
CUDA_VISIBLE_DEVICES=0,1 RUN_SMOLVLA_GPU_PROBE=1 \
  python -m pytest test/test_rl/test_smolvla.py -k gpu_fsdp -q
```

The tests cover native preprocessing/cache/velocity/ODE parity, PPO replay,
real RLinf loss/backward, frozen parameters, bootstrap values, trajectory
alignment, strict native/mapped loading, state-dict copying, CLI composition
and worker registration. The opt-in probe checks two NCCL ranks, actual RLinf
FSDP2, two optimizer updates and distributed checkpoint save/restore.

The tested environment is Python 3.10.21, Torch 2.8.0+cu128,
Transformers 5.3.0, NumPy 1.26.4 and Hydra 1.3.2. The RLinf checkout is
`88988ad5999619953b18d10b15bea1d3a6e1cfce`; this frontend is based on FluxVLA
PI0.5 integration commit `527cd8708c8a130be5af3d8ff59ff4c9a3eb3f1b`.
Other measured package versions: Ray 2.58.0, W&B 0.21.0,
Safetensors 0.8.0, MMEngine 0.10.7 and pytest 8.4.2.

Historical regression on 2026-09-20, with `RUN_SMOLVLA_GPU_PROBE=1`:
**82 passed, 21 warnings in 95.33 seconds** (the existing 68 tests plus 14
SmolVLA cases).

Publication regression on 2026-09-21, with the same GPU probe enabled:
**97 passed, 22 warnings in 120.16 seconds**, including the later launcher
and evaluation-isolation tests. All applicable repository pre-commit hooks
passed. Warnings concern dependency deprecations/metadata and CUDA RNG
device initialization; they are not failed checks.

A separate full-SFT, real-LIBERO preflight also passed: environment reset,
strict SFT loading, action shape `[1, 10, 7]`, maximum pre-update log-probability
drift **0.0**, and execution of one action chunk. This is a short interface
gate, not an episode-success evaluation.

The actual two-GPU synthetic probe passed: both ranks had maximum pre-update
log-probability drift **0.0**, completed two PPO updates, changed expert/value
parameters, preserved frozen parameters, and restored the distributed
checkpoint. Peak allocated model/test tensor memory was approximately 67 MiB
per rank; this is a tiny-model measurement, not a full SmolVLA sizing estimate.

The interface/synthetic checks alone do not establish real-environment
performance; the measured runs above provide that separate evidence.
They do not establish convergence or reliable gains across tasks/seeds.
The PI0.5 RoboTwin results cannot be attributed to SmolVLA.
