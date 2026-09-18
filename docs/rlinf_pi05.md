# FluxVLA PI0.5 → RLinf：LIBERO-10 PPO

## 范围与仓库边界

本实现把 FluxVLA 的 PI0.5 接入当前 RLinf，同步 PPO + GAE；不复制 RLinf
训练循环、不修改 RLinf 或 dexbotic。全部新增源码、配置和测试在 FluxVLA。
这不是“启动 HTTP 推理服务后让 RLinf 请求动作”：actor 和 rollout 各自在进程内
构建相同的 Flux policy，只有 actor 反向传播，RLinf 负责同步权重。

| 层        | FluxVLA 提供                                            | RLinf 提供                                                            |
| --------- | ------------------------------------------------------- | --------------------------------------------------------------------- |
| 启动/注册 | `fluxvla/rl/train.py`、`rlinf_registry.py`              | Hydra 校验、Ray Cluster、组件 placement                               |
| 模型/数据 | `bridge/builder.py`、`pi05_policy.py`、`observation.py` | `get_model`、`BasePolicy`、`PolicyOutput`                             |
| 采样/训练 | `bridge/sampler.py`、薄 `FluxRolloutWorker`             | LIBERO 环境、rollout 调度、PPO/GAE、FSDP、checkpoint、bucket 权重同步 |

沿用 dexbotic 的外部注册思路：driver 在配置校验前调用 `register()`；
Ray worker 通过 `RLINF_EXT_MODULE=fluxvla.rl.rlinf_registry` 调用同一个函数。
模型名为 `fluxvla_pi05`。使用当前后端类 `EmbodiedFSDPActor`、`EnvWorker`、
`EmbodiedRunner`；不依赖 dexbotic 的旧入口或旧 actor 类名。
普通 `import fluxvla` 不导入 RLinf；只有 RL 入口/适配模块依赖它。

## 独立环境

推荐从已验证能导入 Flux PI0.5 的 Python 3.10 环境创建 overlay，避免重复安装
大型 CUDA 包，也不修改原环境：

```bash
/root/miniconda3/envs/fluxvla/bin/python -m venv --system-site-packages /root/.venvs/fluxvla-rlinf
source /root/.venvs/fluxvla-rlinf/bin/activate
cd /root/workspace/FluxVLA
python -m pip install -r requirements-rlinf.txt -e /root/workspace/RLinf
export PYTHONPATH=/root/workspace/FluxVLA${PYTHONPATH:+:$PYTHONPATH}
```

overlay 的依赖继承基环境，不是可脱离基环境搬运的独立发行包；继承的可选包
仍可能存在版本冲突，实际 `pip check` 结果和 TensorBoard/旧 TensorFlow 注意事项见
[验证记录](rlinf_pi05_validation.md)。
全新环境先安装合适的 Torch 2.8 CUDA wheel 和 Flux `requirements-base.txt`，
再安装上述 RL 依赖。不要安装 `RLinf[embodied]`：它的 Transformers 4.x
约束与 Flux 的 Transformers 5.3.0 不兼容。安装的是 RLinf 核心包；
LIBERO/robosuite/MuJoCo/EGL 和任务资产仍需在真实训练前准备，接口测试不需要它们。

本配方依赖 RLinf 源码 checkout 中的 `examples/embodiment/config`；入口自动定位，
非标准安装可显式设置 `RLINF_CONFIG_DIR=/absolute/path/to/embodiment/config`。
训练的所有 Ray 节点必须使用相同环境、可见的 Flux 源码、模型/分词器/统计文件；
如果连接已有 Ray 集群，需要在启动集群前准备这些路径和环境变量。

## 启动与配置

```bash
cd /root/workspace/FluxVLA
python -m fluxvla.rl.train --config-name=libero_10_ppo_fluxvla_pi05 \
  actor.model.model_path=/absolute/path/model.safetensors
```

只查看解析后的配置，不创建 Ray 集群、不加载大模型：

```bash
python -m fluxvla.rl.train --config-name=libero_10_ppo_fluxvla_pi05 \
  actor.model.model_path=/absolute/path/model.safetensors --cfg job --resolve
```

`configs/rl/libero_10_ppo_fluxvla_pi05.yaml` 通过 Hydra searchpath 复用 RLinf
`env/libero_10`、`hybrid_engines/fsdp`、`weight_syncer/bucket_syncer`。
模型结构/图像处理/tokenizer/归一化默认读取原 Flux 配方
`configs/pi05/pi05_paligemma_libero_10_full_finetune.py`，并不转换成 OpenPI 模型。

可覆盖：

```bash
actor.model.fluxvla.config_path=/absolute/path/flux_pi05_config.py
actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer.model
actor.model.fluxvla.norm_stats_path=/absolute/path/norm_stats.json
```

JSON 统计应与 Flux dataset_statistics 同结构，顶层包含配置中 `statistic_name`
（默认 `libero_10_no_noops`），其下有 `proprio`/`action` 等原始统计字段。
不是 OpenPI 的任意 `assets/.../norm_stats.json` 都能直接替代。
不覆盖 tokenizer 时沿用 Flux 配方的缓存/下载逻辑；离线运行应显式给出本地模型。

默认 horizon=10、执行 chunk=10、模型动作维度=32、环境动作维度=7、去噪=10 步。
冻结视觉/语言主干、多模态 projector；训练 action expert、动作/时间投影和 value head。
expert 未使用的词嵌入保持冻结。无 LoRA、compile、CUDA graph、混合 SFT。
`joint_logprob=false`、`noise_level=0.5`、`entropy_bonus=0`。

### SFT 初始化与 RL 恢复

`actor.model.model_path` 始终是初始 **SFT** 权重文件：支持原生 safetensors、PT/PTH
state dict、`{"model": state_dict}` 和 MMEngine 配方里的 `name_mapping`。
先逐参数严格覆盖原 PI0.5，再初始化 `value_head`。缺失、映射歧义、shape 不符均报错；
checkpoint 多余字段不用于主干，原生同名 key 优先于映射 key。
不允许找不到权重后退回随机主干。

RLinf 保存和恢复模型/优化器等训练状态；续训使用：

```bash
python -m fluxvla.rl.train \
  actor.model.model_path=/absolute/path/original_sft.safetensors \
  runner.resume_dir=/absolute/path/checkpoints/global_step_50
```

`runner.resume_dir` 不是 SFT 文件；不要把完整 RL checkpoint 当成 SFT 初始化。
续训仍需相同初始文件和模型配方供 actor/rollout 构造，再由 RLinf 覆盖训练状态。

## 接口与数据流

```text
RLinf EnvWorker → canonical env_obs → Flux observation adapter
  → PI0.5 rollout（ODE / 单步 SDE）→ 已反归一化环境动作 → EnvWorker
  → PolicyOutput / Trajectory → GAE → actor.forward(保存的 transition)
  → RLinf PPO loss / FSDP optimizer → bucket 权重同步 → rollout
```

观测包括 `main_images/wrist_images: uint8[B,H,W,3]`、`states: float[B,8]`、
`task_descriptions: list[str]`。RLinf 已旋转图像并把位置、axis-angle、夹爪状态拼好，
adapter 不再旋转或做四元数转换。复用 Flux 图像处理、prompt/tokenizer 和统计，
state 直接归一化并补零到 32 维。PI0.5 沿用原模型的 proprio 使用方式，
不会新添 state projector。

`predict_action_batch(env_obs, mode)` 返回 `(actions, result)`：

| 字段                   | 默认 shape / dtype            | 含义                                               |
| ---------------------- | ----------------------------- | -------------------------------------------------- |
| actions                | `[B,10,7]` / FP32             | Flux 反归一化及夹爪转换一次后的环境动作            |
| prev_logprobs          | `[B,10,7]` / FP32             | 选中 SDE transition 的逐元素 log-prob，未预先聚合  |
| prev_values            | `[B,1]` / FP32                | 有效 prefix token masked mean 经 critic 的状态价值 |
| images                 | `[B,6,224,224]` / FP32        | 两个 CHW 相机沿 channel 拼接                       |
| img_masks / lang_masks | `[B,2]` / `[B,L]` bool        | 图像和语言有效 token mask                          |
| lang_tokens / states   | `[B,L]` int64 / `[B,32]` FP32 | 模型输入                                           |
| chains                 | `[B,11,10,32]` / FP32         | train 的完整 latent chain 快照，始终在归一化坐标   |
| denoise_inds           | `[B,10]` int64                | 所选 transition 索引，每行重复相同索引             |
| action / model_action  | `[B,70]` / `[B,320]` FP32     | 环境动作 / 最终归一化模型动作                      |

图像、mask、tokens、state、chain 等放在 `result.forward_inputs`，是 **扁平 Tensor 字典**，
全部以 batch 为首维；不放 cache、Python 对象或嵌套 observation。
RLinf 的拆批、轨迹拼接和 actor 微批处理不需要特判本模型。

Eval 直接调用 Flux 原始 ODE `predict_action`；测试使用相同输入和初始 noise 比较。
Train 每批随机选一个去噪步使用 RLinf OpenPI 同公式的 `flow_sde`，其他步 Euler ODE；
保存克隆后的 chain 和已采样 transition 的概率。Actor 从 chain 取同一对 latent，
只重算均值/标准差，不重新采样。这里的概率是 **去噪 transition 的 Gaussian 概率**，
不是最终环境动作的精确边缘似然；按当前 OpenPI 配方只交付执行 chunk 的前 7 维，
再由 RLinf 处理 chunk-level 聚合。Gaussian entropy 可计算，默认不加入 loss。

`forward(forward_inputs=..., compute_logprobs=True, compute_values=True, compute_entropy=False)` 返回 `logprobs/values/entropy`。
bootstrap 通过 `get_values(final_obs)` 计算 `[B,1]`，不推进动作采样 RNG；
薄 rollout 子类只补 train/eval 参数与这一 bootstrap 调用，其余复用后端。

### 精度与 FSDP

actor/rollout 均持有 FP32 主参数，网络内部 autocast BF16，Gaussian 计算和 chain
保持 FP32。使用 FSDP2、`cast_forward_inputs=false`，避免 backend 把保存的 chain
降为 BF16。`param_dtype=fp32` 保持 gathered 参数与 rollout 相同：如果只把 actor
参数降成 BF16，embedding/norm 会与 rollout 的 FP32 参数 + autocast 路径不同，
更新前 log-prob 也可能漂移。BF16 矩阵/卷积计算由 policy 的局部 autocast 提供。
这比全 BF16 gathered 参数更耗显存，但保证两端采用相同精度策略。

显式 wrapping：`GemmaDecoderLayer`、`SiglipEncoderLayer`、`LinearProjector`、`ValueHead`。
配置保留 `use_orig_params=true`；FSDP2 本身按原参数管理，无 FSDP1 的 flatten 开关。
这里的分布式运行支持基于后端接口接入；尚未完成实际 GPU/FSDP 运行验证。

## 接口测试

```bash
cd /root/workspace/FluxVLA
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/.venvs/fluxvla-rlinf/bin/python -m pytest test/test_rl -q
```

使用本地生成的 SentencePiece tokenizer、小尺寸真实 SigLIP + Gemma PI0.5、
合成观测与临时 SFT checkpoint，不下载大模型、不启动 LIBERO。
覆盖配置/注册/入口、原 Flux 预处理与 ODE 一致性、概率复算、真实 RLinf PPO
梯度与冻结参数、PolicyOutput 拆合及轨迹、bootstrap、严格权重加载、状态复制，
以及真实 RLinf bucket 同步器的 CPU 本地传输。

实际运行结果和依赖版本见同目录的 `rlinf_pi05_validation.md`。

## 下一步：真实 GPU 短跑（尚未验证）

准备匹配上述 Flux 配方的 SFT checkpoint、tokenizer、统计和 RLinf LIBERO 资产，
确认 CUDA/EGL 可用及显存足够后，可尝试单 GPU 共置的两轮 PPO：

```bash
source /root/.venvs/fluxvla-rlinf/bin/activate
cd /root/workspace/FluxVLA
export PYTHONPATH=/root/workspace/FluxVLA${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python -m fluxvla.rl.train \
  --config-name=libero_10_ppo_fluxvla_pi05 \
  actor.model.model_path=/absolute/path/model.safetensors \
  actor.model.fluxvla.tokenizer_path=/absolute/path/tokenizer.model \
  runner.max_epochs=2 runner.max_steps=2 runner.save_interval=1 \
  runner.val_check_interval=-1 algorithm.update_epoch=1 \
  env.train.total_num_envs=2 env.eval.total_num_envs=2 \
  env.train.max_steps_per_rollout_epoch=20 \
  actor.micro_batch_size=1 actor.global_batch_size=4
```

这只是接口接通后的探针，不是成功率实验；20 步 rollout 可能远不足以完成任务。
先检查更新前 ratio/log-prob、loss/gradient 有限、checkpoint 保存恢复和权重同步，
再恢复 480 步 rollout 并开启定期评测。
**真实 LIBERO 训练、GPU/多机 FSDP、分布式权重传输/checkpoint 恢复、成功率提升均尚未验证。**
