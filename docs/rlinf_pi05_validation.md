# PI0.5 / RLinf 接口验证记录

## 结果

2026-09-16，在 `/root/.venvs/fluxvla-rlinf` 运行：

```bash
cd /root/workspace/FluxVLA
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/.venvs/fluxvla-rlinf/bin/python -m pytest test/test_rl -q --tb=short \
  --junitxml=/tmp/fluxvla-rlinf-interface-tests.xml
```

结果：**35 passed, 15 warnings in 22.14s**，无跳过用例。
`ruff check fluxvla/rl test/test_rl` 通过；新增 Python 文件已用 Flux 的 YAPF 配置格式化。
15 条 pytest warning 来自已有 Pydantic 字段元数据和 Flux Gemma 的 Transformers
`input_embeds` 弃用提示，不是测试失败。

测试没有下载大模型或 tokenizer：生成本地 SentencePiece tokenizer 和小尺寸真实
SigLIP/Gemma PI0.5，使用合成 checkpoint/观测。视觉 hidden=16、1 层，语言 hidden=32、
expert hidden=16、各 2 层，horizon/denoise=3，执行 chunk=2；真实启动配方仍为 10/10/10。

## 验证层次

| 项目        | 实际验证内容与边界                                                                                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hydra/入口  | 真实 compose、RLinf embodied schema validator、CLI `--cfg job --resolve`；资源发现被替换，不创建 Ray 集群                                                                                                           |
| 注册        | driver `get_model` 构造真实 policy；独立 Python 进程调用真实 RLinf extension hook；不是 Ray 调度测试                                                                                                                |
| 可选依赖    | 普通 Flux 导入时阻断所有 RLinf import，仍导入成功                                                                                                                                                                   |
| 后端兼容    | Transformers 5.3.0 下真实 actor、env worker、FSDP 工具导入成功；RLinf/Ray 直接依赖约束检查通过                                                                                                                      |
| 预处理      | 原 Flux raw LIBERO pipeline 与 canonical RLinf 观测处理一致；不重复图像旋转或夹爪转换                                                                                                                               |
| Eval        | 独立的原始 PI0.5 与 adapter 加载相同参数，固定 noise、BF16 autocast 的模型动作逐值一致，传入 noise 未被修改                                                                                                         |
| Train/复算  | FP32、BF16 网络两种模式下，rollout 旧概率与更新前 actor 复算一致，chain 不被修改，固定 RNG 可重放                                                                                                                   |
| PPO         | 调用真实 RLinf `actor_critic` loss；两种精度下梯度有限，expert/critic 更新，冻结参数逐值不变                                                                                                                        |
| Value/GAE   | prefix masked mean 不受无效 token 干扰；bootstrap 不推进 RNG，与 rollout value 一致；真实 GAE 的终止 mask 和 bootstrap 计算符合手算                                                                                 |
| 数据拼接    | 真实 `PolicyOutput` 拆分/合并、Trajectory 合批；拆批后的概率复算仍对齐                                                                                                                                              |
| 初始化/恢复 | native PT、嵌套 PT、safetensors、原配方 OpenPI name_mapping；缺参数/shape 错误会失败；policy state dict 保存恢复一致                                                                                                |
| 权重复制    | 真实 RLinf `BucketWeightSyncer`，CPU 多 bucket、本地异步发送/接收回调、version=7、选择 trainable 参数与 persistent buffers；复制后参数和概率/value 一致；使用 `load_instant=False`，未验证默认 CUDA stream/通信路径 |
| FSDP        | 真实 wrapping 目标解析、FSDP2 遍历/选择逻辑；`fully_shard` 被记录函数替换，未做实际 shard 或分布式 backward                                                                                                         |
| 配置边界    | 对 joint logprob、LoRA、compile、非零 entropy bonus、actor 单独降参数精度、downcast chain 的错误配置提前报错                                                                                                        |

## 实际依赖版本

这是 overlay 环境的关键版本记录，不是整个 Flux 多模型环境的完整 lockfile。
RLinf 安装的是核心包，没有安装其 `embodied` extra。

```text
Python 3.10.21
torch==2.8.0+cu128
torchvision==0.23.0+cu128
transformers==5.3.0
numpy==1.26.4
hydra-core==1.3.2
omegaconf==2.3.0
mmengine==0.10.7
ray==2.58.0
rlinf==0.4.0
pytest==8.4.2
peft==0.19.1
accelerate==0.33.0
safetensors==0.8.0
sentencepiece==0.1.99
torchdata==0.11.0
gym==0.25.2
gymnasium==1.3.0
imageio==2.37.4
tensorboard==2.15.2
protobuf==6.33.6
swanlab==0.10.0
nvitop==1.7.1
```

验证时 RLinf commit：`88988ad5999619953b18d10b15bea1d3a6e1cfce`，其 tracked worktree 无改动。
FluxVLA base commit：`0cefaacd0a739a9b0d52116fc65822c0cbe6f46a`；原有用户修改保留，
本实现仅新增 RL 目录、配置、测试、依赖清单及文档。

### 依赖限制须知

overlay 使用 `--system-site-packages` 复用原 Flux 环境，新增安装仅落在 overlay，
没有卸载或覆盖原 conda 环境的包。**它不是依赖完全无冲突的干净环境**：
全量 `pip check` 仍因继承的 GR00T/robocasa 缺包或固定版本不匹配、以及
TensorFlow 2.15 与 overlay 的 protobuf/wrapt 不匹配而非零退出。
RLinf 核心及 Ray default 的直接依赖均满足；本页所列 PI0.5/后端接口测试全部通过。

额外实测了 `torch.utils.tensorboard.SummaryWriter` 创建、写入 scalar 和关闭，退出成功。
但 TensorBoard 自动发现继承的旧 TensorFlow 时，会输出
`MessageFactory.GetPrototype` 的 AttributeError 诊断及 CUDA 插件重复注册日志。
不把该环境标为通用 GR00T/robocasa/TensorFlow 环境；生产部署建议使用仅含本配方
所需包的干净 Python 3.10 环境，避免带入这些可选包。

## 尚未验证

未进行真实 LIBERO 环境创建/训练、大尺寸 PI0.5 GPU 推理或训练、Ray worker 集群执行、
GPU/多机 FSDP sharding/backward、实际 NCCL/bucket 通信、RLinf 分布式 checkpoint 的
保存恢复，也未测成功率或成功率提升。policy state dict 测试不等价于分布式恢复验证。
后续短跑命令和检查顺序见 [启动与接口说明](rlinf_pi05.md)。
