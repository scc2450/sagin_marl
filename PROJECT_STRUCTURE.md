# PROJECT_STRUCTURE

本文件只保留当前仍然有用的主线结构。仓库里历史诊断脚本、旧 run、临时 profiler 文件很多，默认不要把它们当成当前训练入口。

## 当前主线

当前实验主线是：

```text
native CUDA structured batch env
+ StructuredActor/RelationalCritic
+ joint MC-GAE training
+ BW macro decision interval K=5
```

正式训练入口：

```text
scripts/train_joint_mcgae.py
```

正式评估入口：

```text
scripts/evaluate_structured_mixed_heads_native.py
```

当前主配置：

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

## 顶层目录

```text
sagin_marl/
  configs/                 当前和历史 YAML 配置
  docs/                    设计、诊断、实验记录
  sagin_marl/              Python package 源码
  scripts/                 训练、评估、诊断、出图脚本
  tests/                   单元测试和 smoke tests
  runs/                    本地训练/评估产物，默认不进 Git
  tools/                   辅助工具
  README.md                项目使用入口
  agent.md                 给编码 agent 的当前工作说明
  PROJECT_STRUCTURE.md     当前文件
```

## 关键配置

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

当前 joint MC-GAE / 3UAV-20GU / positive reward / relational critic 主配置。注意：

- `access_bw_decision_interval` 通常通过命令行传入，当前正式 K=5。
- `sat_decision_interval` 当前保持 1。
- `train_joint_mcgae.py` 会强制设置 joint 训练语义，包括 `train_accel/train_sat/train_bw=True` 和 `exec_*_source=policy`。

历史配置仍在 `configs/` 和 `configs/tmp/` 中保留，但复现实验前要确认是否属于当前主线。

## 核心源码

### 环境与 native CUDA

```text
sagin_marl/env/config.py
```

`SaginConfig`、默认值和 YAML 加载入口。新增配置项通常先改这里。

```text
sagin_marl/env/structured_batch_env_core.py
```

structured batch 环境核心，包含 runtime state、stage state、native rollout 对接、history/cache 写入等。

```text
sagin_marl/env/structured_sync_group.py
```

`GpuStructuredEnvGroup` / `GpuStructuredDriverGroup` 封装。

```text
sagin_marl/env/native_cuda/
```

CUDA kernel、actor kernel、bindings。改动作语义、环境转移、native rollout、macro decision、safety shield 时一定要同步检查这里。

### Actor / Critic / PPO

```text
sagin_marl/rl/structured_actor.py
```

当前 `StructuredActor`，包含：

- accel actor
- SAT subset actor
- BW Dirichlet actor

```text
sagin_marl/rl/structured_critic.py
```

当前 relational critic / world state value 网络。

```text
sagin_marl/rl/distributions.py
```

动作分布与 log-prob 口径。Accel 径向 squash、BW Dirichlet、SAT categorical 的 PyTorch 训练口径都在这里。

```text
sagin_marl/rl/native_actor_cuda.py
```

把 PyTorch actor 权重导出到 native CUDA rollout 使用的 ABI。

```text
sagin_marl/rl/structured_mappo.py
```

StructuredMAPPO 组件、actor/critic eval、native binding 同步、部分 PPO 更新工具函数。

```text
sagin_marl/rl/structured_factory.py
```

按配置构建 actor/critic。

## 当前训练脚本

```text
scripts/train_joint_mcgae.py
```

当前正式训练入口。关键行为：

- native rollout 收集 64 env x 250 step。
- critic 用 finite-horizon MC return 训练。
- 用训练后的 critic 重新计算 `A_gae(V)`。
- accel / sat / bw 三个 actor 独立 PPO 更新。
- BW 当前正式使用 macro decision interval `K=5`。
- 保存 `final.pt`、`checkpoint_update*.pt`、`best_stage_heads/best_*.pt`、`metrics.csv`、`phase_trace.jsonl`。

常用命令见 `README.md`。

## 当前评估脚本

```text
scripts/evaluate_structured_mixed_heads_native.py
```

当前正式 native deterministic eval 入口。用于：

- final checkpoint 评估
- stage-best head 替换评估
- 8 组合评估
- rule baseline 评估

必须显式传入：

```text
--access_bw_decision_interval 5
--sat_decision_interval 1
```

否则容易和训练时 macro 行为不一致。

## 仍有用的辅助脚本

```text
scripts/evaluation/evaluate_thesis_native_methods.py
```

论文方法对照 / baseline 汇总入口，适合批量评估 proposed、baseline、Lyapunov 等。

```text
scripts/analysis/export/generate_joint_mcgae_training_ppt.py
```

根据当前 joint MC-GAE 结果生成简版 PPT。

```text
scripts/analysis/export/export_tb_scalars.py
```

导出 TensorBoard scalar 曲线。

```text
scripts/analysis/analyze_thesis_fairness.py
```

论文 fairness / 指标分析辅助。

## 历史或调试入口

这些脚本仍可能有用，但不是当前 K=5 joint MC-GAE 主线：

```text
scripts/train_structured.py
scripts/evaluate_structured.py
scripts/train_sat_mcgae.py
scripts/train_stage_mcgae.py
scripts/train.py
scripts/evaluate.py
scripts/render_episode.py
```

使用它们前要先确认目标是不是历史 PPO、单阶段诊断、legacy PettingZoo 或普通 structured eval。

## 重要文档

```text
docs/joint_mcgae_training_summary_for_slides_20260514.md
```

给导师汇报用的 joint MC-GAE 简版材料。

```text
docs/joint_mcgae_macro_k5_nogrow_u300_rerun_eval_20260514.md
```

K=5 当前主结果记录。

```text
docs/joint_mcgae_macro_k10_nogrow_u300_eval_20260514.md
```

K=10 对照结果记录。

```text
docs/joint_mcgae_k5_danger_imitation_ablation_20260515.md
```

danger imitation 开关消融。

```text
docs/bw_access_macro_decision_interval_design_20260513.md
```

BW macro decision interval 设计与实现说明。

```text
docs/torch_compile_cache_setup_20260510.md
```

Windows + torch.compile / Triton 缓存路径说明。

## 当前 run 产物结构

典型训练目录：

```text
runs/diagnostics/<run_name>/
  metrics.csv
  phase_trace.jsonl
  final.pt
  checkpoint_update0050.pt
  checkpoint_update0100.pt
  ...
  checkpoint_update0300.pt
  best_stage_heads/
    best_accel.pt
    best_sat.pt
    best_bw.pt
  diagnostics/
  native_eval_stage_grid_8x_final_best/
```

最重要的数据：

- `metrics.csv`：训练曲线、EV、KL、episode reward、collision、耗时。
- `final.pt`：最终 joint checkpoint。
- `best_stage_heads/best_*.pt`：每个 stage 按训练期指标保存的 best head。
- `native_eval_stage_grid_8x_final_best/*_summary.json`：8 组合 deterministic eval 汇总。
- `native_eval_stage_grid_8x_final_best/*_episodes.csv`：逐 episode 评估数据。

## 测试入口

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --import-mode=importlib
```

改 native CUDA / actor distribution / macro decision 后，至少做：

- 相关 parity 或 smoke 脚本
- `py_compile` 对改动脚本
- 小规模 1 update shapecheck

完整训练前先确认 stdout 第一行显示 cache 路径和训练语义符合预期。
