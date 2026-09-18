# RoboTwin 完整动作 demo（2026-09-18）

录制完成：step 30、step 200 各 10 个固定场景，每场景包含 `head.mp4`
和 `three_cameras.mp4`，共 40 个文件。全部 25 FPS，124–201 帧（4.96–8.04 秒）。
不是插帧或调整旧视频速度，而是在动作执行期间取得新的真实仿真画面。

## 数据位置

根路径为 `/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl/results/`：

- step 30：`adjust_bottle_full_demo_20260918_1517_best_step30/full_videos/`
- step 200：`adjust_bottle_full_demo_20260918_1517_last_step200/full_videos/`
- 完整索引：`adjust_bottle_full_demo_20260918_1517/demo_index.json`
- 子目录为 `worker_<rank>/seed_<实际场景seed>/`，包含两种视频和 `recording.json`。
  三相机横向顺序：主视角、左腕、右腕。

可直接查看同一场景 `worker_0/seed_100100088/head.mp4`，或对应的三相机版。
索引列出了所有场景及成败，不只提供成功示例。

## 录制与验证

- 保留原模型、噪声 seed、两个 rollout worker / 两个环境 worker、每 worker 5 个环境，
  只重放原测试的第一批 10 个场景；未根据成败选择场景。
- 保留 50 步 action chunk 及原 TOPP/物理控制逻辑，不拆分动作，不改成功判定。
- 250 Hz 物理仿真中每 10 个物理步采帧，以 25 FPS 播放，末帧补齐终止姿态。
  成功后原环境停止物理动作，视频到该终止姿态结束；失败则录到 episode 上限。
- 读取相机 RGB，不调用额外的 policy `get_obs()`，不覆盖 `now_obs`。
- 20 个场景成败均与原正式测试对应场景一致；视频帧数/FPS 与录制元数据一致。
- 全部 68 项接口测试通过（50.56 秒）。最初隐藏全部 GPU 的回归启动因已有 Eagle
  模块导入时要求 CUDA 而失败，随后仅开放未参与录制的 GPU 0 完成回归。
- 新逻辑通过 `+env.eval.full_video_dir=...` 显式启用；默认训练/评测不录制密集视频。
  接入位于 FluxVLA，未修改 RLinf 或 RoboTwin 源码。

这批是演示重放，不是新基准评测。其 10 个场景的局部成功率不能替代已完成的
150-seed 正式成绩，原正式视频、统计和 checkpoint 均未覆盖。

tmux：`flux-robotwin-full-demo`。执行器：`scripts/run_robotwin_demos.py`。
对应 W&B：step 30 `qidayr3p`、step 200 `wqkv37pt`。
总日志、回归及解码日志在根路径上一级的 `logs/full_demo_20260918_1517_*.log`。
