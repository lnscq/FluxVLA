# RL configuration layout

Run commands from the repository root. Hydra's config root remains
`configs/rl`; select a recipe by its relative path without `.yaml`.

| Directory              | Responsibility                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `base/`                | Shared PPO, worker, optimizer and precision defaults; no benchmark or model selection |
| `models/`              | Policy construction, native model config references and model overrides               |
| `backends/fsdp/`       | Model-specific FSDP wrapping rules                                                    |
| `benchmarks/libero/`   | LIBERO environment defaults and PI0.5/SmolVLA train/eval recipes                      |
| `benchmarks/robotwin/` | RoboTwin task protocol and PI0.5 train/eval recipes                                   |
| `runtime/`             | Simulator runtime files, including the NVIDIA Vulkan ICD                              |

## Standard recipes

```bash
python -m fluxvla.rl.train --config-name=benchmarks/libero/pi05/ppo \
  actor.model.model_path=/absolute/path/model.safetensors

python -m fluxvla.rl.train --config-name=benchmarks/libero/smolvla/ppo \
  actor.model.model_path=/absolute/path/model.safetensors \
  actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer \
  actor.model.fluxvla.norm_stats_path=/absolute/path/norm_stats.json

python -m fluxvla.rl.eval --config-name=benchmarks/robotwin/pi05/eval \
  actor.model.model_path=/absolute/path/robotwin-sft
```

SmolVLA standalone evaluation uses `benchmarks/libero/smolvla/eval`.
RoboTwin training uses `benchmarks/robotwin/pi05/ppo`.
See [the integration guide](../../docs/rlinf.md) for dependencies,
benchmark assets, checkpoint requirements and distributed launch details.

## Representative multi-GPU recipes

Use `benchmarks/libero/smolvla/ppo_8gpu` or
`benchmarks/robotwin/pi05/ppo_8gpu` for eight-GPU training. RoboTwin also
provides `benchmarks/robotwin/pi05/eval_8gpu` for matching evaluation.
These settings are runnable examples, not optimized convergence targets.

No smoke, preflight, fixed-duration or single-task YAML variants are kept.
Use ordinary Hydra overrides on a representative recipe instead:

```bash
# A longer run: same model, optimizer and sampling protocol.
python -m fluxvla.rl.train --config-name=benchmarks/libero/smolvla/ppo_8gpu \
  actor.model.model_path=/absolute/path/model.safetensors \
  actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer \
  actor.model.fluxvla.norm_stats_path=/absolute/path/norm_stats.json \
  runner.max_epochs=1000 runner.max_steps=1000 \
  runner.save_interval=25 runner.val_check_interval=25
```

For task isolation, override both `++env.train.task_id_filter=[6]` and
`++env.eval.task_id_filter=[6]`; set the evaluation episode count explicitly.
Use the shared `scripts/rl/run.sh` launcher or the module CLI. Tests compose
these same representative configs with in-memory overrides; internal
experiment and diagnostic launch scripts are not part of the user package.

## Composition and migration

- Recipe YAML files use `# @package _global_` and absolute defaults paths,
  so directory names do not become new keys in RLinf's configuration.
- Model and FSDP fragments are placed explicitly at `actor.model` and
  `actor.fsdp_config`. Do not add `_global_` to these fragments.
- The selected primary recipe declares the RLinf Hydra search path.
  Include shared bases, not other primary recipes with a search path.
- Benchmark configs must not depend on other benchmarks. Add a new YAML
  only for a reusable model, backend or benchmark recipe, not a test run.
- The old flat config names have been replaced by the paths above, without
  compatibility aliases. Repository launchers, tests and docs use the new
  names. Update external `--config-name` or `SMOLVLA_CONFIG_NAME` arguments.
- Retained recipes preserve hyperparameters and run identities. The RoboTwin
  MMEngine config and Vulkan ICD have moved; external path overrides need
  updating. Set `FLUX_ROBOTWIN_ROOT` to your assets/results directory; its
  default is now `work_dirs/robotwin` under this checkout, not a user path.
- This source reorganization does not migrate running jobs or checkpoints.
