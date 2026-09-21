# PI0.5 reinforcement learning with RLinf

SmolVLA also uses this backend through `fluxvla_smolvla`; see the
[SmolVLA integration guide](rlinf_smolvla.md) for its LIBERO-10 recipe,
model-specific FSDP constraints and validation boundaries.

FluxVLA supplies the PI0.5 policy, observation/action adapters, flow-SDE sampling,
and launch configuration. RLinf supplies environment orchestration, PPO/GAE,
FSDP2, weight synchronization, and checkpoint storage. Actor and rollout workers
construct the same policy in their own processes; this is not an HTTP model server.

The integration is optional: ordinary FluxVLA imports do not require RLinf.
The external model name is `fluxvla_pi05`, registered in both the driver and Ray
workers through `RLINF_EXT_MODULE=fluxvla.rl.rlinf_registry`.

## Code map

| Module                                                        | Responsibility                                                                                             |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `fluxvla/rl/bridge/`                                          | Strict SFT loading, policy, shared Gaussian probability functions, precision, LIBERO and RoboTwin adapters |
| `fluxvla/rl/train.py`, `eval.py`, `rlinf_registry.py`         | Hydra entry points and external registration                                                               |
| `rollout_worker.py`, `actor_worker.py`, `env_worker.py`       | Small RLinf worker extensions, bootstrap values, actual-loss probability checks                            |
| `robotwin_env.py`, `robotwin_runtime.py`                      | Deterministic scene seeds, episode auditing, process-local simulator compatibility                         |
| `experiment.py`, `pilot_runner.py`, `preflight.py`            | Validation-only selection, bounded pilot orchestration, real-worker validation                             |
| `demo_video.py`                                               | Opt-in 25 FPS physics-time camera recording without splitting action chunks                                |
| `configs/rl/`                                                 | LIBERO-10 and RoboTwin adjust_bottle recipes, four/eight-GPU placements                                    |
| `scripts/*robotwin*`, `scripts/setup_openpi_reference_env.sh` | Resource preparation, environment setup, diagnostics, paired reports and demos                             |
| `test/test_rl/`                                               | Small real PI0.5 models, actual RLinf losses and interface regression tests                                |

## Dependencies and paths

Use a separate Python 3.10 environment with Torch 2.8, Transformers 5.3.0,
NumPy 1.26.4 and Hydra 1.3.2. Install `requirements-rlinf.txt` and RLinf core;
do not install `RLinf[embodied]`, whose Transformers 4.x constraints conflict
with this FluxVLA recipe. Simulator dependencies and assets are separate.

```bash
python -m pip install -r requirements-rlinf.txt -e /absolute/path/RLinf
export RLINF_ROOT=/absolute/path/RLinf
export FLUX_ROBOTWIN_ROOT=/absolute/path/robotwin_experiment
source scripts/robotwin_rl_env.sh
```

The environment script locates the current FluxVLA checkout automatically.
`RLINF_ROOT` defaults to a sibling RLinf checkout. `FLUX_ROBOTWIN_ROOT` contains
weights, simulator checkout/assets, seeds, environments, logs and results.
Historical defaults and case-study documents retain the original experiment's
CPFS paths; set the variables explicitly on another machine. Setup scripts are
overlay recipes based on the documented pre-existing Flux environment, not
standalone installers for an arbitrary machine.

## Entry points

```bash
# Parse a configuration without training.
python -m fluxvla.rl.train --config-name=libero_10_ppo_fluxvla_pi05 \
  actor.model.model_path=/absolute/path/sft.safetensors --cfg job --resolve

# Run long jobs inside tmux. W&B credentials come only from the environment.
python -m fluxvla.rl.train \
  --config-name=robotwin_adjust_bottle_ppo_fluxvla_pi05_8gpu \
  runner.logger.experiment_name=my_unique_training_run

# Evaluate an existing RL checkpoint; do not use resume_dir here.
python -m fluxvla.rl.eval \
  --config-name=robotwin_adjust_bottle_eval_fluxvla_pi05_8gpu \
  runner.ckpt_path=/absolute/path/checkpoints/global_step_30 \
  runner.logger.experiment_name=my_unique_test_run

# Optional dense demos: replay only the first fixed batch, separately from tests.
python -m fluxvla.rl.eval \
  --config-name=robotwin_adjust_bottle_eval_fluxvla_pi05_8gpu \
  runner.ckpt_path=/absolute/path/checkpoints/global_step_30 \
  runner.logger.experiment_name=my_unique_demo_run \
  env.eval.rollout_epoch=1 env.eval.video_cfg.save_video=false \
  +env.eval.full_video_dir=/absolute/path/new_demo_directory
```

Keep initial SFT weights (`actor.model.model_path`) separate from training
resume checkpoints (`runner.resume_dir`) and evaluation checkpoints
(`runner.ckpt_path`). Never reuse an episode output directory for a retry.

`run_robotwin_8gpu.py`, `run_robotwin_final_eval.py` and `run_robotwin_demos.py`
are deliberately audited case-study launchers: they require the recorded gates,
source hashes, baselines and run directories. They are not drop-in launchers for
new experiments. Formatting or source changes correctly invalidate old strict
source-hash checks; do not bypass them or relabel a replay as the original run.
The module entry points above are the reusable interface.

## Validation and evidence boundaries

```bash
python -m pytest test/test_rl -q
python -m pre_commit run --files fluxvla/rl/*.py fluxvla/rl/bridge/*.py test/test_rl/*.py
```

Run pre-commit from Python 3.10, matching the repository CI and its pinned hooks.
The suite uses generated tokenizers, small real models and synthetic observations;
it does not download pretrained models or perform LIBERO training. Importing the
complete Flux model catalog may initialize CUDA through an unrelated backbone;
the recorded environment used a visible GPU even for CPU tensor tests.

Real RoboTwin experiments additionally exercised multi-GPU backward, frozen
parameters, unchanged-weight log-probabilities, actor-to-rollout synchronization,
and distributed checkpoint restoration. The Torch 2.8/FSDP2 configuration uses
per-microbatch gradient synchronization to avoid an observed probability drift;
optimizer accumulation/global batch remain unchanged.

Historical numerical results were measured on the original experiment checkout,
not rerun merely by publishing this branch on newer upstream main. See the dated
records below for exact scope; interface tests do not replace GPU validation.

Publication check on 2026-09-18: rebased onto fork main
`ce4b90ec47f6a614616ef2c472890e4a2a0c005a`; all 68 RL interface tests passed
(52.54 seconds) and repository pre-commit checks passed under Python 3.10.
The test worktree reused three local compiled CUDA extensions whose sources
were unchanged between the original experiment and the new base. No compiled
extensions, weights, videos, credentials or experiment output are committed.

- [LIBERO interface and data flow](rlinf_pi05.md)
- [Initial dependency and interface validation](rlinf_pi05_validation.md)
- [RoboTwin setup, GPU investigations and pilot history](rlinf_robotwin_pilot.md)
- [200-round continuation and 150-seed final tests](rlinf_robotwin_final200_results.md)
- [Full-motion demo recording](rlinf_robotwin_full_demos.md)

The held-out results were SFT 129/150 (86%), validation-selected step 30 135/150
(90%), and final step 200 136/150 (90.67%). This is one task and one training
seed, not a claim of broad convergence or replication of RLinf's published
96.09%. Test results did not determine checkpoint selection.
