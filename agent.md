# Agent Guide

本文件给后续在本仓库工作的编码 agent 使用。重点是避免被旧脚本、旧配置和大量历史诊断文件带偏。

## 当前项目语境

项目：`SAGIN-MARL`

当前主线：

```text
native CUDA structured batch environment
+ joint MC-GAE critic training
+ accel / SAT / BW 三个 actor head
+ BW macro decision interval K=5
```

当前正式训练入口：

```text
scripts/train_joint_mcgae.py
```

当前正式评估入口：

```text
scripts/evaluate_structured_mixed_heads_native.py
```

当前主配置：

```text
configs/tmp/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

不要默认使用：

```text
scripts/train_structured.py
scripts/evaluate_structured.py
scripts/train_sat_mcgae.py
scripts/train_stage_mcgae.py
scripts/train.py
scripts/evaluate.py
```

这些脚本是历史 PPO、单阶段诊断、legacy 或普通 structured eval 入口。除非用户明确要复现旧实验，否则当前论文主线不用它们。

## 当前训练口径

正式 K=5 joint 训练命令：

```powershell
$env:TMP='D:\sagin_cache\tmp'
$env:TEMP='D:\sagin_cache\tmp'
$env:TORCHINDUCTOR_CACHE_DIR='D:\sagin_cache\torchinductor'
$env:TRITON_CACHE_DIR='D:\sagin_cache\triton'

.\.venv\Scripts\python.exe scripts\train_joint_mcgae.py `
  --config configs\tmp\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
  --run_dir runs\diagnostics\<run_name> `
  --updates 300 `
  --rollout_env_steps 250 `
  --num_envs 64 `
  --device cuda `
  --torch_threads 1 `
  --access_bw_decision_interval 5 `
  --sat_decision_interval 1 `
  --save_every 50
```

关 danger imitation 消融才加：

```powershell
--disable_danger_imitation
```

关键点：

- `access_bw_decision_interval=5` 是当前 BW macro K=5。
- `sat_decision_interval=1` 当前保持不做 SAT macro。
- `train_joint_mcgae.py` 会强制 joint training：accel/sat/bw 都训练，exec source 都是 policy。
- 当前 reward 默认 `positive_weighted_workload_level`。
- 当前 safety 默认使用 native shield，并保留 danger imitation。
- LR grow 当前关闭，KL early stop 和 LR decay 保留。

## 当前评估口径

正式 native eval：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_structured_mixed_heads_native.py `
  --config configs\tmp\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
  --base_checkpoint runs\diagnostics\<run_name>\final.pt `
  --episodes 64 `
  --num_envs 64 `
  --episode_seed_base 900000 `
  --policy_mode deterministic `
  --device cuda `
  --exec_accel_source policy `
  --exec_sat_source policy `
  --exec_bw_source policy `
  --access_bw_decision_interval 5 `
  --sat_decision_interval 1 `
  --out_dir runs\diagnostics\<run_name>\native_eval `
  --label final_ppp
```

Stage-best 替换：

```powershell
--accel_checkpoint runs\diagnostics\<run_name>\best_stage_heads\best_accel.pt
--sat_checkpoint   runs\diagnostics\<run_name>\best_stage_heads\best_sat.pt
--bw_checkpoint    runs\diagnostics\<run_name>\best_stage_heads\best_bw.pt
```

规则 baseline：

```powershell
--baseline_policy cluster_center_queue_aware
```

该 baseline 在 native eval 中表示：

```text
accel = cluster_center_queue_aware
sat   = queue_aware
bw    = queue_aware
```

必须显式传：

```text
--access_bw_decision_interval 5
--sat_decision_interval 1
```

否则评估可能不严格复现训练时的 macro 行为。

## 当前关键结果目录

K=5 主结果：

```text
runs/diagnostics/joint_mcgae_macro_k5_nogrow_u300_rerun_20260514
```

K=10 对照：

```text
runs/diagnostics/joint_mcgae_macro_k10_nogrow_u300_20260514
```

关 danger imitation：

```text
runs/diagnostics/joint_mcgae_macro_k5_nodanger_u300_20260515
```

主要看：

```text
metrics.csv
final.pt
checkpoint_update*.pt
best_stage_heads/best_*.pt
native_eval_stage_grid_8x_final_best/*_summary.json
```

## 当前关键文档

```text
docs/joint_mcgae_macro_k5_nogrow_u300_rerun_eval_20260514.md
```

K=5 主结果。

```text
docs/joint_mcgae_macro_k10_nogrow_u300_eval_20260514.md
```

K=10 对照。

```text
docs/joint_mcgae_k5_danger_imitation_ablation_20260515.md
```

danger imitation 消融。

```text
docs/joint_mcgae_training_summary_for_slides_20260514.md
```

导师汇报简版。

```text
docs/bw_access_macro_decision_interval_design_20260513.md
```

BW K=5 macro decision 设计。

```text
docs/torch_compile_cache_setup_20260510.md
```

torch.compile / Triton cache 路径。

## 核心代码位置

```text
sagin_marl/env/config.py
```

配置 dataclass 与 YAML 加载。

```text
sagin_marl/env/structured_batch_env_core.py
```

structured batch env、stage state、native runtime。

```text
sagin_marl/env/native_cuda/
```

native CUDA kernels / bindings。改动作执行、macro、safety、信道、队列时必须检查。

```text
sagin_marl/rl/structured_actor.py
```

accel / sat / bw actor。

```text
sagin_marl/rl/structured_critic.py
```

relational critic。

```text
sagin_marl/rl/distributions.py
```

PyTorch action distribution 与 log-prob 口径。

```text
sagin_marl/rl/native_actor_cuda.py
```

actor 权重导出到 native CUDA ABI。

```text
sagin_marl/rl/structured_mappo.py
```

MAPPO 工具、actor/critic eval、native binding 同步。

```text
sagin_marl/rl/structured_factory.py
```

按配置构建 actor/critic。

## 修改联动规则

改动作分布或 log-prob：

- 查 `sagin_marl/rl/distributions.py`
- 查 `sagin_marl/rl/structured_actor.py`
- 查 `sagin_marl/rl/native_actor_cuda.py`
- 查 `sagin_marl/env/native_cuda/`
- 做 PyTorch/native parity，不要只改一边。

改 BW macro：

- 查 `docs/bw_access_macro_decision_interval_design_20260513.md`
- 查 `scripts/train_joint_mcgae.py`
- 查 native rollout / BW action history / BW logprob history
- 确认 K=1 与新路径等价，K=5 只在 macro-start 更新 BW actor。

改 critic 训练：

- 当前 critic target 是 finite-horizon MC return。
- actor advantage 用训练后 critic 重新算 `A_gae(V)`。
- 不要无意切回普通 train-time GAE target。

改 safety：

- 当前正式训练使用 native shield。
- danger imitation 是训练辅助，不等于唯一安全来源。
- 当前日志有 `rollout_intervention_rate`，但没有 native solver 内部迭代次数。

改训练/评估命令：

- README、agent.md、PROJECT_STRUCTURE.md 要同步。
- 评估命令必须保留 `--access_bw_decision_interval 5`。

## 设计原则

不要因为“改动最小”就选一个方向。先判断问题类别：

```text
1. head 内部分布/优化问题
2. joint-action 结构问题
3. credit assignment 问题
4. 环境/执行语义问题
```

当前历史经验：

- BW 曾经有明确的分布和 score-gradient 钝化问题。
- SAT/BW 曾经被普通 PPO advantage 噪声淹没。
- Accel 会改变系统工作区间，不能只看单步局部收益。
- 旧的 per-head value / per-head surrogate 开关不是等价的真正 local credit，不要悄悄复活成“新方案”。

提出新方案前要说清：

- 它解决的是结构、credit、分布优化，还是环境语义。
- 为什么已有失败路径不适用。
- 什么现象可以证伪这个方案。

实现时不要隐式混多个 redesign：

- 不要悄悄改 reward。
- 不要悄悄改执行源。
- 不要悄悄改 PPO / critic target。
- 如果多件事一起改，要明确标成 architecture redesign，不要当成单机制对照。

## 常用检查

语法检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\train_joint_mcgae.py scripts\evaluate_structured_mixed_heads_native.py
```

测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --import-mode=importlib
```

改 native 或 actor distribution 后不要直接长训，先跑小规模 shapecheck / parity。

## 工作习惯

- 不要覆盖已有重要 run；新增实验用新 `run_dir`。
- 不要把 `runs/`、`.tmp`、profiler、cache 当成源码依据。
- 读历史结果时优先看对应 docs，再看 `metrics.csv` 和 `*_summary.json`。
- 如果训练卡住，先看 `phase_trace.jsonl` 定位阶段。
- 如果 stdout 中 cache 路径落到 C 盘或中文路径，先修环境变量再跑。
