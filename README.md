# SAGIN-MARL

本仓库是面向空天地一体化网络（Space-Air-Ground Integrated Network,
SAGIN）的协同任务调度研究代码库。当前主线围绕 STARS
（Structured Topology-Aware Resource Scheduler）展开，目标是在地面用户、
无人机中继和 LEO 卫星组成的动态边缘计算场景中，联合学习 UAV 运动控制、
卫星选择和接入带宽分配策略。

项目同时保存三类内容：

- 可运行的仿真、训练、评估和诊断代码；
- 论文实验所需的轻量证据表、出图脚本和投稿图片源文件；
- 历史实验、调试记录和分支收紧过程中的 run 产物索引。

原始 `runs/`、checkpoint 和大体量日志默认不进入 Git。论文中的数值和图表应优先追溯
`docs/paper/evidence_tables/`、`docs/paper/reproduction/` 和
`docs/run_artifact_roots_20260909.md`，再回到运行机器上的原始 run 目录。

## 当前主线

当前正式代码路径是：

```text
native CUDA structured batch environment
+ StructuredActor / RelationalCritic
+ joint MC-GAE training
+ staged accel / satellite / bandwidth actor updates
+ access bandwidth macro decision interval K=5
```

默认配置：

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

默认训练入口：

```text
scripts/train_joint_mcgae.py
```

默认评估入口：

```text
scripts/evaluate_structured_mixed_heads_native.py
```

当前主配置对应 `3 UAV / 20 GU / 250 steps` 场景。正式实验通常使用
`64` 个并行环境和 `250` 个 environment steps，因此每次训练更新包含
`16000` 条环境转移。主实验路径使用学习式卫星选择，即
`fixed_satellite_strategy: false` 且 `exec_sat_source: policy`。

## 目录地图

```text
sagin_marl/
  sagin_marl/              Python package 源码
    env/                   SAGIN 环境、队列、链路、native CUDA runtime
    rl/                    STARS actor/critic、MAPPO/MC-GAE、baseline
    utils/                 checkpoint、日志、归一化和运行辅助
    viz/                   可视化辅助
  configs/                 当前配置、消融配置、smoke/perf 配置和历史归档
  scripts/                 训练、评估、诊断、实验启动和分析脚本
  tests/                   单元测试和 smoke tests
  docs/                    方法说明、实验记录、论文证据和维护文档
  runs/                    本地运行产物，默认被 Git 忽略
```

更细的目录说明见：

- `PROJECT_STRUCTURE.md`：当前源码、配置和脚本布局；
- `configs/README.md`：配置文件目录和使用边界；
- `scripts/README.md`：脚本入口、诊断脚本和 legacy 脚本说明；
- `docs/README.md`：研究记录、方法文档和历史实验导航。

## 方法概览

STARS 针对 UAV-LEO 协同调度的混合动作空间和动态拓扑结构做了结构化拆分：

- 每个 UAV 基于局部观测进行分布式执行；
- 观测包含自身状态、候选 GU 集合、可见 LEO 集合、邻近 UAV 集合以及相关队列和可行性信息；
- actor 将单时隙动作拆成运动控制、卫星子集选择和带宽分配三个阶段；
- UAV 运动后会刷新接入拓扑、卫星可见性和后续阶段的动作掩码；
- 运动动作经过安全修正层，安全修正量可作为训练中的辅助监督信号；
- centralized critic 在训练时编码 GU-UAV、UAV-LEO 和 UAV-UAV 关系，用于提供 stage-wise value estimates；
- 评估和部署时只需要分布式 actor，不需要 centralized critic 参与环境交互。

核心实现位置：

```text
sagin_marl/rl/structured_actor.py
sagin_marl/rl/structured_critic.py
sagin_marl/rl/stage_mcgae.py
sagin_marl/rl/native_actor_cuda.py
sagin_marl/env/structured_batch_env_core.py
sagin_marl/env/structured_gpu_rollout_runtime.py
sagin_marl/env/native_cuda/
```

## 环境准备

本机原生 Python 环境可能缺少常用依赖。
环境做代码检查和轻量 smoke；正式 native CUDA 训练和评估应在带 NVIDIA GPU 的机器上运行。

通用安装方式：

```bash
python -m pip install -r requirements.txt
```

如果使用 Windows/PowerShell 或 `torch.compile`，需要显式设置 ASCII 缓存路径，避免 Triton
和临时文件写入中文路径或系统盘。详细说明见：

```text
docs/guides/torch_compile_cache_setup_20260510.md
```

## 最小训练命令

正式训练建议从当前主配置派生，并显式传入 BW 和 SAT 的决策间隔：

```bash
python scripts/train_joint_mcgae.py \
  --config configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml \
  --run_dir runs/experiments/<run_name> \
  --updates 300 \
  --rollout_env_steps 250 \
  --num_envs 64 \
  --device cuda \
  --torch_threads 1 \
  --access_bw_decision_interval 5 \
  --sat_decision_interval 1 \
  --save_every 50
```

恢复训练时，`--updates` 表示目标总 update 数，不是额外追加的 update 数：

```bash
python scripts/train_joint_mcgae.py \
  --config configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml \
  --run_dir runs/experiments/<run_name> \
  --resume runs/experiments/<run_name>/checkpoint_update0150.pt \
  --updates 300 \
  --rollout_env_steps 250 \
  --num_envs 64 \
  --device cuda \
  --torch_threads 1 \
  --access_bw_decision_interval 5 \
  --sat_decision_interval 1 \
  --save_every 50
```

典型输出：

```text
runs/experiments/<run_name>/
  metrics.csv
  phase_trace.jsonl
  final.pt
  checkpoint_update*.pt
  best_stage_heads/
  diagnostics/
```

## 最小评估命令

评估学习式 checkpoint：

```bash
python scripts/evaluate_structured_mixed_heads_native.py \
  --config configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml \
  --base_checkpoint runs/experiments/<run_name>/final.pt \
  --episodes 64 \
  --num_envs 64 \
  --episode_seed_base 900000 \
  --policy_mode deterministic \
  --device cuda \
  --exec_accel_source policy \
  --exec_sat_source policy \
  --exec_bw_source policy \
  --access_bw_decision_interval 5 \
  --sat_decision_interval 1 \
  --out_dir runs/experiments/<run_name>/native_eval_final \
  --label final_policy
```

评估规则 baseline：

```bash
python scripts/evaluate_structured_mixed_heads_native.py \
  --config configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml \
  --baseline_policy cluster_center_queue_aware \
  --episodes 64 \
  --num_envs 64 \
  --episode_seed_base 900000 \
  --device cuda \
  --access_bw_decision_interval 5 \
  --sat_decision_interval 1 \
  --out_dir runs/experiments/<run_name>/native_eval_rule \
  --label qccs
```


## 论文资产与复现

论文相关内容集中在：

```text
docs/paper/
  manuscript_overleaf/     投稿图、PPT 图源和 Overleaf 辅助文件
  evidence_tables/         从正式 run 导出的 CSV/JSON 证据表
  reproduction/            聚合、绘图和证据复核脚本
```

当前约定是：

- 系统图和方法图的可编辑 PPT 源保存在 `docs/paper/manuscript_overleaf/figures/`；
- Section 5 性能图由 `docs/paper/reproduction/*.py` 从 `docs/paper/evidence_tables/` 生成。

复现和证据说明见：

```text
docs/paper/README.md
docs/paper/reproduction/README.md
docs/paper/evidence_tables/README.md
docs/run_artifact_roots_20260909.md
docs/run_registry.csv
```

## 检查与维护

轻量代码检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/sagin_marl_pycache \
python -m py_compile \
  sagin_marl/env/config.py \
  scripts/train_joint_mcgae.py \
  scripts/train_stage_mcgae.py
```

测试入口：

```bash
python -m pytest tests
```

论文出图脚本的最小语法检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/sagin_marl_pycache \
python -m py_compile docs/paper/reproduction/*.py
```