# FluxVLA 接入 RLinf

FluxVLA 提供模型、预处理、policy 和启动入口；RLinf 提供环境采样、PPO/GAE、
FSDP2、权重同步及 checkpoint。actor 和 rollout 在各自进程内构建相同 Flux policy，
不通过 HTTP 推理服务调用模型，不修改 RLinf 源码。普通 FluxVLA 导入不强制依赖 RLinf。

## 阅读入口

- [PI0.5 接入](rlinf_pi05.md)：LIBERO-10、RoboTwin 模型/数据适配及启动。
- [SmolVLA 接入](rlinf_smolvla.md)：LIBERO-10 适配、FSDP 差异及启动。
- [扩展接口约定](rlinf_extension.md)：公共组件、变量命名及新增 VLA 的最小改动。

## 公共组件与数据流

| 文件                                             | 职责                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `fluxvla/rl/policy_specs.py`                     | 声明模型、builder、horizon 字段、支持的 adapter 和 FSDP 约束 |
| `fluxvla/rl/rlinf_registry.py`                   | driver/Ray worker 共用的外部注册和构建分发                   |
| `fluxvla/rl/bridge/builder.py`                   | 读取 MMEngine 配置，严格加载初始 SFT，再添加 critic          |
| `fluxvla/rl/bridge/adapters.py`                  | 构建环境观测/动作适配器                                      |
| `fluxvla/rl/bridge/flow_policy.py`、`sampler.py` | 共享初始化、冻结、flow-SDE 采样、概率复算与 bootstrap        |
| `fluxvla/rl/train.py`、`eval.py`                 | Hydra 配置与 RLinf 训练/独立评测入口                         |
| `fluxvla/rl/rollout_worker.py`                   | 显式传递 train/eval 模式，提供 bootstrap value               |

Driver 在配置校验前调用 `register()`；Ray worker 通过
`RLINF_EXT_MODULE=fluxvla.rl.rlinf_registry` 调用同一个函数。
注册名为 `fluxvla_pi05` 和 `fluxvla_smolvla`，新增兼容模型通过声明接入，
不向 train/eval/worker 添加逐模型判断。

```text
env_obs → adapter → rollout policy → 环境动作
                       ↓
PolicyOutput → Trajectory → GAE → actor.forward → PPO/FSDP2 更新
                                                       ↓
                             rollout ← RLinf 权重同步
```

两种 policy 共用三个接口：

- `predict_action_batch(env_obs, mode)` 返回动作及 `prev_logprobs`、
  `prev_values`、`forward_inputs`。
- `forward(forward_inputs=..., compute_logprobs=..., compute_values=..., compute_entropy=...)` 返回 `logprobs`、`values`、`entropy`。
- `get_values(env_obs)` 计算 bootstrap value，不推进动作采样 RNG。

Eval 调用原生 Flux ODE。Train 随机选一个去噪步采用 flow-SDE，其余步用 ODE，
保存不可变 FP32 chain。Actor 使用同一对 latent 重算 Gaussian transition 概率，
不重新采样。这不是最终环境动作的精确边缘似然，不能直接用于非 flow 模型。
`forward_inputs` 是 batch-first 扁平 Tensor 字典，不保存模型 cache；
动作反归一化和夹爪转换仅由 adapter 执行一次。

## 依赖与环境

在已能导入对应 Flux 模型的独立 Python 3.10 环境中，从仓库根目录执行：

```bash
python -m pip install -r requirements-rlinf.txt -e /absolute/path/RLinf
export RLINF_ROOT=/absolute/path/RLinf
export PYTHONPATH="$PWD:$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export FLUX_RL_EXTERNAL_DISTRIBUTED=1
```

不要安装 `RLinf[embodied]`，其 Transformers 4.x 限制与本配方冲突。
模拟器、渲染依赖和任务资产须单独准备。入口自动定位 RLinf 的
`examples/embodiment/config`；非标准安装设置 `RLINF_CONFIG_DIR`。
所有 Ray 节点须能访问同一源码、环境和模型文件。

已测试版本：Python 3.10.21、Torch 2.8.0+cu128、Transformers 5.3.0、NumPy 1.26.4、
Hydra 1.3.2、Ray 2.58.0、W&B 0.21.0、Safetensors 0.8.0、MMEngine 0.10.7、
pytest 8.4.2。RLinf commit：`88988ad5999619953b18d10b15bea1d3a6e1cfce`。
这不是完整 lockfile；历史 overlay 继承过其他模型/TensorFlow 的依赖冲突，
不能据此宣称整个多模型环境的 `pip check` 全部通过。

## 运行约定

长任务在 tmux 内执行。正式训练设置 `runner.logger.logger_backends=[wandb]`；
凭据仅从已有 `WANDB_API_KEY` 环境变量读取，认证失败不得静默转离线。
不要把密钥写入配置或脚本。每次使用独立 `runner.logger.experiment_name`，
一个 Ray 实例只运行一个实验。

| 配置字段                 | 用途                                           |
| ------------------------ | ---------------------------------------------- |
| `actor.model.model_path` | 初始 SFT，必须匹配模型配置、tokenizer 和统计   |
| `runner.resume_dir`      | 续训用 RLinf checkpoint，恢复模型和训练状态    |
| `runner.ckpt_path`       | 独立评测用 RLinf checkpoint 目录或完整权重文件 |

续训/评测仍需初始 SFT 配置完成模型构造。支持 PT/PTH、safetensors 及分片 index；
原生 key 优先于 `name_mapping`。缺参数、歧义或 shape 不符立即失败，
不会用随机主干继续训练；新增 value head 单独初始化。

当前使用同步 PPO + GAE、`joint_logprob=false`、`entropy_bonus=0`、FP32 主参数、
局部 BF16 autocast 和 FP32 chain；FSDP2 设置 `param_dtype=fp32`、
`cast_forward_inputs=false`。禁用 LoRA、compile、CUDA graph、gradient checkpointing
和混合 SFT。模型专属 FSDP wrapping 不能相互照搬。

## 验证

```bash
python -m pytest test/test_rl -q
# 在 tmux 中选择两张空闲 GPU，加入真实双卡更新/恢复测试。
CUDA_VISIBLE_DEVICES=0,1 RUN_SMOLVLA_GPU_PROBE=1 \
  python -m pytest test/test_rl -q
```

2026-09-22 重构后 107 项测试通过，含真实双卡 SmolVLA FSDP2 更新和恢复；
完整 SFT 的真实 LIBERO reset/action-chunk 短跑通过，概率复算误差为 0。
PI0.5 的 RoboTwin 多卡训练、通信与恢复此前已实际运行，不等同于 PI0.5 LIBERO
或其他模型/环境组合均已实测；未执行多机验收。

接口正确不代表收敛或涨点。现有超参是接入配方，不是最优方案；成功率对照必须
固定初始权重、任务、episode/noise seeds 和评测布局，不得用测试集挑 checkpoint。
