# FluxVLA PI0.5 / RLinf RoboTwin 四卡与八卡试验

本试验为 `adjust_bottle`、一个训练 seed 的探索性 PPO 试验。只有完整的本机 SFT/RL 配对评测才能判断涨点。RLinf 公布的 SFT 85.94%、PPO 96.09% 仅作外部参考，**不是 FluxVLA 官方成绩**。

## 2026-09-18：200 轮续训终评完成

用户手动启动的 200 轮续训正常结束；本次经用户请求，单独评测验证最佳 step 30 与最后
step 200。固定 150 个测试 seeds：SFT 129/150（复用），step 30 为 135/150（90.00%，
+4.00 个百分点），step 200 为 136/150（90.67%，+4.67 个百分点）。最佳仍按验证集选择
step 30，不用测试结果重新挑选。详见 [完整终评记录](rlinf_robotwin_final200_results.md)。
以下章节保留原八小时试验的历史预算与启动记录，不适用于这次已获授权的续训及终评。

## 2026-09-17 八卡重启方案

机器扩容后确认 8 × RTX PRO 5000 Blackwell 72GB、503 GiB 主机内存（标称 512GB）、64 CPU；旧进程随扩容重启结束。新会话为 `tmux attach -t flux-robotwin-8gpu`，不创建看门狗，不重置 `gpu_budget.json`。原训练截止仍为 03:11:56 +08，总截止 05:11:56；30 轮仅为配置上限，不代表剩余预算足以跑满。

新 run 从原始 SFT 开始，重置优化器及 value head，不接续发生回落的第 60 轮。四卡旧 best（第 20 轮）与所有原始结果保留；本次 best 只从本次验证记录选择。

| 项目            | 八卡设置                                                                           |
| --------------- | ---------------------------------------------------------------------------------- |
| GPU 分工        | actor 0–3（FSDP2）、rollout 4–5、env 6–7                                           |
| 每轮采样        | 32 环境 × 4 rollout epochs × 4 chunks = 512 chunks                                 |
| 优化器          | global batch 512、micro batch 4、update epoch 2，即每外层轮 2 次优化器更新         |
| 学习率          | actor 1e-6、value 1e-4                                                             |
| 其他 PPO/模型项 | gamma .99、lambda .95、noise .3、clip .2、entropy 0；chunk 50、denoise 5，冻结主干 |
| 验证 / 保存     | 每 5 轮；固定 32 validation seeds；预算到时额外验证并保存 last                     |
| 测试            | SFT / best / last 各固定 150 seeds，仍为 2 workers、10 env × 15 epochs             |

旧 run 小批量重复更新后验证分数大幅波动，因此扩大批量、降低 actor LR 和重复更新次数；这是一组保守试验参数，不是已验证的最优参数。新增主机内存不是提高学习率的理由。公共 RLinf 配方每轮 4096 chunks、global batch 2048、5 epochs，本方案没有完整复刻其采样规模。

当前 backend 实际读取 `env.train.enable_offload` / `env.eval.enable_offload`，本次显式打开这两个字段；仅顶层 `env.enable_offload` 不够。每轮采样结束后释放 SubEnv 引用，并记录每个环境 worker 的 RSS；释放 Python 引用不等于已证明修复原生库内存增长。

配置：`_robotwin_8gpu_base.yaml` 共享设置；PPO、preflight、eval 三个独立根配置各自声明 Hydra searchpath，避免嵌套主配置覆盖 searchpath。八卡门禁要求实际 4 actor / 2 rollout 同步、概率复算、PPO backward、冻结参数、保存恢复及环境对象释放全部通过。原四卡门禁不能替代八卡实测。

入口 `scripts/run_robotwin_8gpu.py` 在新八卡门禁通过、在线 W&B 认证成功且剩余训练预算足够后，依次运行 SFT → PPO → best/last 终评。源文件 SHA256 和新协议写入 `results/eight_gpu_protocol.json`，不覆盖原四卡 manifest。日志为 `logs/eight-gpu-launcher.log`、`sft-8gpu.log`、`ppo-8gpu.log`、`rl-8gpu-{best,last}-test.log`；选择与配对报告为 `results/eight_gpu_checkpoint_selection.json`、`eight_gpu_paired_report.json`。

Hydra 继承问题修复后，64 项回归测试通过（59.65 秒，`logs/tests-8gpu-v3.log`），W&B 在线认证成功。八卡门禁实测每批采样约 75–80 秒，因此八卡 runner 使用 600 秒预算余量在轮次边界验证、保存并退出，覆盖下一轮采样/更新与验证保存的预计耗时；不是看门狗。门禁完成状态以 `preflight/eight_gpu_v1/ray_gate.json` 为准。

### 八卡预检发现的训练路径概率漂移

02:38 +08 基础八卡门禁通过（462 秒），但只有 1 次 optimizer update 的首轮出现非零 PPO clipping，不能用原来的抽查通过来放行。追加检查显示：原始顺序 / 打乱 / grad-enabled 独立 forward 都为 0 漂移；真实 loss 路径第三个 microbatch 在 optimizer_steps=0 时却出现最大 13.6739 的单元素漂移。失败样本与逐批记录保存在 `preflight/eight_gpu_loss_v1/`，未开始正式训练。

最小复现进一步表明：同一个输入反复 backward 不复现；换批、重分配输入并预建优化器状态时，延后梯度同步的 FSDP2 复现 0.96384 漂移（四 rank 均如此），改为每 microbatch 同步后四 rank 全部为 0。脚本 `scripts/robotwin_accumulation_probe.py`，用 `FLUX_ACCUM_CHURN=1` 启用更接近 worker 的分配过程，比较有无 `FLUX_ACCUM_SYNC_EACH=1`；结果为 `accumulation_churn*_rank*.json`。这是当前 Torch 2.8 / FSDP2 / BF16 组合中的实测规避办法，尚未定位到上游具体缺陷。

八卡设置 `actor.fsdp_config.enable_gradient_accumulation: false`：该 RLinf 字段控制是否延后通信，不取消 optimizer 的梯度累积，global batch 512、micro batch 4、每外层轮 2 次更新均保留。新门禁 `preflight/eight_gpu_loss_sync_v1/ray_gate.json` 必须通过；旧基础门禁不足以证明训练安全。正式 actor 还会检查首个 optimizer update 的每一次实际 forward（包括梯度累积期间），指标 `actor/loss_input_max_abs_drift`，而不是只在训练前抽查一次。

预算内可用 `scripts/run_robotwin_8gpu.py --reuse-original-sft` 显式复用原始 129/150 基线；脚本核对两个评测配置的模型、预处理、ODE、seed、任务和两 worker 布局一致，并重新审计所有 150 个 seed。没有在新的物理 GPU 分工上重测 SFT，必须在结果中保留这一说明。每 phase 的环境 offload 只在整段评测结束后释放环境，不改变该段 episode 逻辑。默认不传此选项仍然先重测 SFT；不论哪条路径，训练前剩余不足 600 秒均拒绝启动。

03:00 +08：修正后的真实门禁通过（221.95 秒，128 chunks 的 1 次更新），四 rank 初始及恢复后概率差异为 0，213 个参数 tensor 更新、冻结参数不变、跨组权重 SHA 和 checkpoint 恢复一致，实际首轮 ratio=1、clip_fraction=0、approx_kl=0、梯度范数 49.964。66 项接口测试通过（55.53 秒，`logs/tests-8gpu-sync.log`）；预算退出前先写完最后一轮指标的补充回归 6 项通过（2.38 秒，`logs/tests-8gpu-final-log.log`），Ruff 通过。

03:00:51 +08 已在 `flux-robotwin-8gpu` 的 `pilot` 窗口启动 `scripts/run_robotwin_8gpu.py --reuse-original-sft`，`train-log` 跟随 `logs/ppo-8gpu.log`；03:00:59 进入 train 阶段，剩余训练预算约 656 秒。预计只够约 1 个外层 PPO 轮次，然后验证、保存、自动进入 best/last 终评。不能将这种短跑描述为收敛试验完成，更不能将门禁的正确概率比当作任务成功率提升。

03:03 +08 已完成实际 workers 初始化并进入首轮 32 env × 4 批采样。在线 W&B：<https://wandb.ai/3224392370-xi-an-jiaotong-university-/fluxvla-rl-robotwin/runs/l6e8u4cy>，ID 也写入 `results/adjust_bottle_pi05_8gpu_seed1234/wandb_run.json`。此时正式 global-batch-512 更新尚未结束；上面的 128-chunk 门禁结果不能冒充这一正式轮次的训练结果。后续实际状态以 `ppo-8gpu.log`、`runner_completion.json` 和配对报告为准。

## 状态与证据

所有运行文件位于 `/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl`（下文 `$RL_ROOT`）。

- 模型前向对齐已通过：隔离原生 OpenPI 与 Flux 进程，真实分片权重、固定观测和 noise；预处理、各层抽查、velocity、ODE actions 的最大差异均为 0，见 `preflight/model_parity.json`。
- 原有 LIBERO 接口测试保留，新增 RoboTwin 后 57 项通过，最新见 `logs/tests-eval-config.log`（48.83 秒）。包含真实 rollout 可选 `None` 观测字段拆批与训练/独立评测的 RLinf 配置校验测试。
- 单环境 reset/step 已通过 OIDN 2.3.3 重测，未再出现 OIDN CUDA 错误；`preflight/reset_oidn233.png` 已人工检查，见 `logs/env-probe-oidn.log`。
- 双 GPU FSDP2 已完成 backward、expert/value 更新、冻结参数核验、DCP 保存与恢复；显式 DP 组复核通过，两 rank 全局梯度范数一致（173.5832），概率漂移 0，见 `logs/fsdp-probe-v4.log`。
- 四卡真实 Ray 门禁通过：32 env、实际 PPO 更新、213 个参数 tensor 更新、冻结参数不变、完整 state SHA 跨组一致、actor checkpoint 保存/扰动/恢复一致；两个 rank 更新前及恢复后的概率漂移均为 0。耗时 225 秒，见 `preflight/ray_gate.json`、`logs/ray-gate-v5.log`。门禁仅 1 个 update epoch，不计为正式训练结果。
- SFT 基线已完成：129/150，86.0%，Wilson 95% CI 79.54%–90.66%，严格审计所有 150 个官方测试 seeds 恰好一次。见 `results/sft_baseline.json`、`results/adjust_bottle_sft_test/episodes/test/`；后端评测耗时 343 秒（不含 worker 初始化）。每 worker 5 个环境，显存观察值约 22.7 GB；不是精确 peak 测量。
- 2026-09-16 22:37 +08 正式 PPO 启动，首轮 5 个 update epochs 已完成并进入第二轮：总耗时 134.839 秒，actor 更新 57.257 秒，梯度范数有限（60.163）。在线 W&B run：`https://wandb.ai/3224392370-xi-an-jiaotong-university-/fluxvla-rl-robotwin/runs/vcr4joqv`。尚无 RL 150-seed 终评成绩，不宣称提升。
- 训练状态见 tmux 的 `train-log` 窗口、`logs/ppo-train.log` 和 `results/adjust_bottle_pi05_seed1234/wandb_run.json`。W&B 捕获的逐轮指标表位于 `results/wandb/wandb/run-20260916_223710-vcr4joqv/files/output.log`，该窗口实时显示此文件。

## 目录与接口

### 2026-09-17 OOM 中断与续跑

原 run `vcr4joqv` 在第 70 轮验证时因主机 RAM 达到 Ray 95% 阈值而终止；两个环境进程分别约 118.6 GB、93.1 GB，并非 GPU 显存 OOM。已完整保存至第 60 轮。固定 32 个验证 seeds 的成功数为：10/20/30/40/50/60 轮分别 27/31/16/26/25/20；逐 seed 审计均无遗漏或重复。策略出现明显回落，不能称为稳定收敛，第 20 轮最佳验证成绩也不是 150-seed 测试成绩。

01:58 +08 由 `scripts/resume_robotwin_pilot.py` 从第 60 轮恢复模型和优化器；新目录 `results/adjust_bottle_pi05_seed1234_resume60`、tmux `flux-robotwin-resume`、日志 `logs/ppo-resume60.log`、W&B run `lshwhvsn`。训练环境 32 → 16（OOM 允许的资源调整），micro batch 2/global batch 64、5 update epochs、学习率、验证与测试布局均不变。这会使每轮采样量从 128 降至 64 个 action chunks；模拟器及 rollout RNG 重新启动，不是逐位连续恢复。内存增长根因尚未证明修复，不提高或关闭 Ray 内存保护阈值。

续跑沿用原预算，不启动看门狗。终评前合并原 run 与续跑的验证记录，并保留每个 checkpoint 的所属目录；同分取更早者，原第 20 轮 best 不会丢失。终评测试集尚未用于续跑决策。完成后选取记录写入 `results/final_checkpoint_selection.json`，配对结果写入 `results/paired_report.json`。

续跑回归测试：`logs/tests-resume60.log`，60 项通过（53.12 秒）；新增覆盖 16-env/global-batch-64 配置、跨 run 保留原 best checkpoint 路径及缺失权重拒绝。

| Flux 文件                                | 职责                                                                                    |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| `fluxvla/rl/train.py`、`eval.py`         | 注册扩展、Hydra、RLinf worker/runner 启动                                               |
| `bridge/builder.py`                      | MMEngine 模型配置、严格分片/映射权重加载；主干加载完成后创建新 value head               |
| `bridge/robotwin_observation.py`         | 三相机、14D Aloha 状态、quantile、离散 state prompt、delta/夹爪还原                     |
| `bridge/pi05_policy.py`、`sampler.py`    | 原始 Flux ODE eval、单随机步 flow-SDE train、FP32 概率复算                              |
| `bridge/precision.py`                    | 与原生 OpenPI 一致的计算精度，保留 FP32 主参数                                          |
| `rollout_worker.py`                      | train/eval 显式传递、固定评测 RNG、不采样的 bootstrap value                             |
| `actor_worker.py`                        | 真实轨迹更新前概率抽查、非有限梯度立即失败；PPO/FSDP 仍由 RLinf 实现                    |
| `robotwin_env.py`、`robotwin_runtime.py` | 全量 seeds 分片、禁止 seed+1 偷换、prompt 与实际场景匹配、GPU 渲染路由、逐 episode 审计 |
| `pilot_runner.py`                        | 仅验证集选 best、同分早者、last checkpoint、W&B URL、预算边界保存                       |
| `preflight.py`                           | 实际四卡通信/训练/保存恢复门禁，不计入正式结果                                          |

数据流：RLinf env → Flux 预处理 → policy 归一化动作/chain → Flux 上下文相关动作反变换 → env；`forward_inputs` 是 batch-first 扁平 Tensor 字典，包含图像/masks/tokens/state、不可变 FP32 `chains`、`denoise_inds`。Actor 用同一对保存的 latent 复算 Gaussian transition 概率，不重新采样。RLinf 处理轨迹拼接、GAE、PPO 聚合、FSDP、checkpoint 和权重同步。

## 版本和精度

- RoboTwin `RLinf_support` commit：`0008ae6800df9f75fc8de7098bacb01735fd8fd2`。
- SFT：`RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` revision `fa8df6ed103db0f5549c122f3a17c00ba6426c98`，分片校验在 `weights/RLinf-Pi05-RoboTwin-SFT-adjust_bottle/manifest.json`。
- 资产：`TianxingChen/RoboTwin2.0` revision `981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042`，不用失效 OSS 链接。
- 主环境：Python 3.10.21、Torch 2.8.0+cu128、Transformers 5.3.0、NumPy 1.26.4、Hydra 1.3.2、SAPIEN 3.0.0b1、mplib 0.2.1。
- 参考环境：`rlinf-openpi==0.1.1`、`rlinf-transformer-openpi==4.53.2`；JAX 在 CPU，不安装 OpenPI 的 CUDA extra，也不替换主环境 Transformers。
- OIDN 2.3.3 官方 changelog 首次加入 Blackwell 支持；仅本实验进程通过 `LD_PRELOAD` 加载，不修改系统/flex-pi 的库。渲染版本差异必须作为外部成绩可比性限制记录；本机 SFT/RL 使用同一渲染配置。
- tokenizer 的 GCS URL 在本机遇 TLS 主机名错误，未关闭 TLS。使用公开 HF 字节镜像并锁定 SHA256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`，见 `tokenizer_source.json`。
- 创建模型后**先转 FP32 再加载**，避免 Gemma 构造出的 BF16 norm 参数吞掉 SFT 精度。网络局部 BF16；视觉 stem、action/time 投影、adaptive RMS dense、chain、概率保持所需 FP32。视觉 LayerNorm/图像 memory layout、非持久 RoPE buffers 与原生 OpenPI 对齐。
- 两个环境是独立 overlay，复用既有 Torch 安装但不写入原环境。重建还依赖本机 `/root/miniconda3/envs/fluxvla` 和 `/root/.venvs/fluxvla-rlinf`；不是可直接搬到任意机器的完整容器。
- 完整依赖版本见 `logs/{flux-robotwin,openpi-reference}-freeze.txt`；源码 commit、集成文件 SHA256、GPU/驱动和预算记录在 `preflight/runtime_manifest.json`，可用 `scripts/robotwin_record_versions.py` 刷新。GPU 2 曾同时运行非本任务的 openwam 作业（约 24.6 GB），未停止该作业；吞吐不能标注为独占四卡性能。
- RL 入口设置 `FLUX_RL_EXTERNAL_DISTRIBUTED=1`，避免 Flux 日志导入时让 Accelerate 抢建进程组；仅 RL 模式改变，普通 Flux 行为保持不变。TensorBoard 在 `USE_TF=0` 时使用自带兼容层，不加载旧 TensorFlow。

## 固定协议

GPU 0–1 actor FSDP2，GPU 2–3 各一个 rollout/env worker。32 train env、micro batch 2、global batch 64、update epoch 5、actor LR 5e-6、value LR 1e-4、gamma .99、lambda .95、noise .3、entropy bonus 0、joint logprob false。

三相机、14D 双臂、模型动作 32D，horizon/chunk 50、denoise 5；checkpoint 元数据 horizon 10 与公开评测代码 50 的差异已显式选择后者。关闭 DR、center crop、compile、CUDA graph、LoRA、混合 SFT。episode 上限 200。

训练 seed 1234；官方 1000 train seeds 固定划出 32 validation，剩余 968 全部参与训练。验证 8 env × 4 轮；最终测试固定官方 150 seeds，10 env × 15 轮，每个 seed 恰好一次。每 10 轮保存并验证，best 只按 validation 选择，同分早者。测试集不用于选择模型。

训练 rollout 的模型计算按 actor micro batch 2 分块；32 个环境仍保持并行，global batch 不变。真实 GPU 门禁发现 BF16 下 batch 16 直接采样、batch 2 复算时存在约 0.08 的 log-prob 漂移，因此统一模型计算分块并严格重测。Eval 不使用此训练分块，SFT/best/last 均保持每 worker 5 个环境的布局。评测视频记录 chunk 边界预览（每 50 个动作一个观测，1 fps），不是逐仿真步录像。

首轮 GPU 墙钟预算起点为 2026-09-16 21:11:56 +08:00，6 小时训练/基线截止 2026-09-17 03:11:56，8 小时总截止 05:11:56，最终以 `gpu_budget.json` 为准。没有自动强杀看门狗，不自动延长；launcher 检查阶段准入，runner 在 checkpoint 边界留出保存时间。

## 启动

所有长任务均在 tmux。准备会话：`tmux attach -t flux-robotwin-prep`。

2026-09-16 22:30 +08 启动正式顺序管线：`tmux attach -t flux-robotwin-pilot`。22:27 的首次入口发现 `only_eval` 需要完整 `rollout.model`，在消费测试 seeds 前退出；已在 eval YAML 中复用 `${actor.model}` 并通过后端校验回归。首次失败日志保存在 `logs/*-v1-config-failure.log`。当前阶段以 `logs/pilot-launcher.log`、`logs/sft-baseline.log`、`logs/ppo-train.log` 为准，启动管线不代表 PPO 已开始。

```bash
cd /root/workspace/FluxVLA
source scripts/robotwin_rl_env.sh
RL_PY="$FLUX_ROBOTWIN_ROOT/envs/flux-robotwin/bin/python"

# 重建隔离环境/资产时使用（已经完成的下载会校验复用）。
bash scripts/setup_robotwin_rl_env.sh
bash scripts/setup_openpi_reference_env.sh
bash scripts/setup_robotwin_oidn.sh
source scripts/robotwin_rl_env.sh

# 原生模型前向对齐：分别使用两个解释器；其后 compare 必须通过。
CUDA_VISIBLE_DEVICES=0 "$FLUX_ROBOTWIN_ROOT/envs/openpi-reference/bin/python" scripts/robotwin_parity.py reference
CUDA_VISIBLE_DEVICES=1 "$RL_PY" scripts/robotwin_parity.py flux
"$RL_PY" scripts/robotwin_parity.py compare

CUDA_VISIBLE_DEVICES=3 "$RL_PY" scripts/robotwin_env_probe.py
CUDA_VISIBLE_DEVICES=0,1 "$RL_PY" -m torch.distributed.run --standalone --nproc_per_node=2 scripts/robotwin_fsdp_probe.py
"$RL_PY" -m fluxvla.rl.preflight

# 上述门禁通过后，在新的 tmux 会话运行顺序管线：
# tmux new-session -s flux-robotwin-pilot
bash scripts/run_robotwin_pilot.sh
```

W&B 密钥仅从环境变量读取，正式 PPO 验证在线认证，缺失/离线均失败；不要把密钥写进命令、YAML、日志或 Git。项目 `fluxvla-rl-robotwin`，run 链接由 runner 写到 `wandb_run.json`。

初始 SFT 使用 `actor.model.model_path`。继续 PPO 使用 `runner.resume_dir=/.../checkpoints/global_step_N`；独立评测使用 `runner.ckpt_path` 指向同一目录或其 `actor/model_state_dict/full_weights.pt`，不能把 resume 当初始权重。首版不下载完整示范数据。

## 结果审计

每个 worker 将完成 episode 写到 `results/<experiment>/episodes/<split>/worker_N.jsonl`，记录 seed、checkpoint step、成败和回报。`scripts/robotwin_paired_report.py` 对重复/缺失测试 seeds 直接失败，报告 SFT/RL 成功数、成功率、百分点差、rescue/regression、各自 Wilson 95% CI 和配对 bootstrap 差值 CI。若失败/超预算，不补选容易 seeds，也不将门禁训练当作正式结果。

`run_robotwin_pilot.sh` 是顺序阶段 launcher，不是看门狗；只有四卡门禁成功才进入 SFT → PPO → best/last 终评。下载、测试、FSDP、Ray、SFT、训练、终评均保留独立日志。若重启失败的评测，应使用新的 experiment_name 保留原始逐 episode 文件，禁止在同一目录追加后挑选有效条目。
