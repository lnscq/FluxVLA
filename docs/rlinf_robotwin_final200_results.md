# RoboTwin adjust_bottle：200 轮续训后的固定测试集评测

2026-09-18 完成。结论：本机固定 150-seed 测试集上观察到相对 SFT 的提升；
不等价于多训练 seed 的稳定收益，也未复现外部公布的 96.09%。

## 结果

| 模型                      | 成功数 / 150 | 成功率 | 相对 SFT（百分点） | 新增成功 / 退步 |
| ------------------------- | ------------ | ------ | ------------------ | --------------- |
| 原 SFT 基线（复用）       | 129          | 86.00% | —                  | —               |
| 验证集选出的最佳：step 30 | 135          | 90.00% | +4.00              | 6 / 0           |
| 最后：step 200            | 136          | 90.67% | +4.67              | 9 / 2           |

配对差值的 percentile bootstrap 95% 区间分别为 +1.33～+7.33、+0.67～+9.33
个百分点（按 episode 重采样 20,000 次，统计 RNG seed 1234）。区间描述固定模型下
测试 episode 的变异，不包含训练 seed 变异；不据此宣称跨任务或多次训练的可靠提升。
成功率 Wilson 95% 区间：SFT 79.54%～90.66%，step 30 84.16%～93.85%，
step 200 84.94%～94.36%。

最佳和最后在测试前已锁定。验证集最高 30/32、同分取更早，故最佳保持 step 30；
不因 step 200 测试分高一个 episode 而改变选择规则。此前验证曲线有明显波动，
本次结果支持“小幅实际涨点”，不支持“随训练轮数持续改善或已充分收敛”。

## 协议与执行

- 训练来源：`adjust_bottle_pi05_8gpu_resume1_20260917_120908`，已正常到达 step 200。
- 两组均为全部 150 个官方测试 seeds，每 seed 恰好一次；严格拒绝缺失、重复或错误 seeds。
- 两个 rollout worker、两个环境 worker；每 worker 5 个环境，共 15 批。
- GPU 4–5 rollout、6–7 环境；无 actor 训练。episode 200 步，horizon/chunk 50，
  去噪 5 步，ODE、固定 episode/noise seeds、三相机及预处理与原协议一致。
- 复用已完成的 SFT 129/150 基线，不声称本次重新评测了 SFT。与原归档配置签名比较一致，
  同时核对关键适配/采样/环境源码 SHA256；原 SFT 使用不同的物理 GPU 分工，
  worker 数、批量和评测语义不变。原协议详见 `results/eight_gpu_protocol.json`。
- 所有长任务在 tmux `flux-robotwin-test-200` 中顺序执行，无看门狗、无新训练、无自动重试。
  W&B 在线，两组均正常退出。含 worker 初始化的耗时分别为 420.81、411.83 秒。
- 两份 checkpoint 的完整权重 SHA256、命令、源码校验、预先锁定的协议及独立日志已保存。

## 文件与链接

以下路径相对于 `/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl`：

- 完整配对报告：`results/adjust_bottle_final200_20260918_1453/paired_report.json`
- 协议及日志：同目录 `protocol.json`、`best_launch.json`、`last_launch.json`、
  `best.log`、`last.log`、`completion.json`。
- 总日志：`logs/adjust_bottle_final200_20260918_1453_launcher.log`。
- 最佳逐 episode / 视频：`results/adjust_bottle_final200_20260918_1453_best_step30_test/episodes/test/`
  及该 experiment 的 `videos/`。
- 最后逐 episode / 视频：`results/adjust_bottle_final200_20260918_1453_last_step200_test/episodes/test/`
  及该 experiment 的 `videos/`。
- [step 30 W&B](https://wandb.ai/3224392370-xi-an-jiaotong-university-/fluxvla-rl-robotwin/runs/ikf67njx)
- [step 200 W&B](https://wandb.ai/3224392370-xi-an-jiaotong-university-/fluxvla-rl-robotwin/runs/eo4ilx93)

专用执行器为 `scripts/run_robotwin_final_eval.py`，复用现有评测入口和配对统计，
要求唯一 tag、已有完整训练 checkpoint、tmux 和在线 W&B；不会训练或重新选择模型。
本次没有修改 policy 或 RLinf 后端。
