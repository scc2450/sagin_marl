# SAGIN-MARL

面向空天地一体化网络（SAGIN）的多智能体强化学习仿真、训练与评估代码。当前主线已经从早期 PPO / PettingZoo 路径切到：

```text
native CUDA structured batch environment
+ joint MC-GAE critic training
+ accel / SAT / BW 三阶段 actor 更新
+ BW macro decision interval K=5
```

旧脚本仍保留用于历史复现和调试，但当前论文结果、K=5 训练、8 组合评估不要再默认使用旧入口。

## 当前主线入口

训练：

```text
scripts/train_joint_mcgae.py
```

评估：

```text
scripts/evaluate_structured_mixed_heads_native.py
```

当前主配置：

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

当前推荐设置：

```text
num_envs = 64
rollout_env_steps = 250
reward_mode = positive_weighted_workload_level
access_bw_decision_interval = 5
sat_decision_interval = 1
native safety shield = enabled
danger imitation = enabled
stage_actor_lr_grow_factor = 1.0
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## torch.compile / Triton 缓存路径

训练前建议在当前 PowerShell 进程设置 ASCII 缓存与临时目录，避免写入 C 盘或中文路径：

```powershell
$env:TMP='D:\sagin_cache\tmp'
$env:TEMP='D:\sagin_cache\tmp'
$env:TORCHINDUCTOR_CACHE_DIR='D:\sagin_cache\torchinductor'
$env:TRITON_CACHE_DIR='D:\sagin_cache\triton'
New-Item -ItemType Directory -Force $env:TMP,$env:TORCHINDUCTOR_CACHE_DIR,$env:TRITON_CACHE_DIR | Out-Null
```

这里用 `$env:` 只影响当前终端和子进程。Triton 在 Windows 上生成 `__triton_launcher*.lib/.exp` 时会使用 `TEMP/TMP`，所以这两个也要设。

详细说明见：

```text
docs/torch_compile_cache_setup_20260510.md
```

## 当前正式训练命令

K=5 joint MC-GAE：

```powershell
.\.venv\Scripts\python.exe scripts\train_joint_mcgae.py `
  --config configs\current\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
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

关掉 danger imitation 做消融时才加：

```powershell
--disable_danger_imitation
```

恢复训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_joint_mcgae.py `
  --config configs\current\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
  --run_dir runs\diagnostics\<run_name> `
  --resume runs\diagnostics\<run_name>\checkpoint_update0150.pt `
  --updates 300 `
  --rollout_env_steps 250 `
  --num_envs 64 `
  --device cuda `
  --torch_threads 1 `
  --access_bw_decision_interval 5 `
  --sat_decision_interval 1 `
  --save_every 50
```

说明：

- `--updates` 在 resume 时表示目标总 update 数，不是额外 update 数。
- `--access_bw_decision_interval 5` 是当前 BW macro K=5 的关键参数。
- `train_joint_mcgae.py` 会强制 joint 训练语义：`train_accel/train_sat/train_bw=True`，执行源为 policy。

## 当前正式评估命令

评估 final 全 policy：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_structured_mixed_heads_native.py `
  --config configs\current\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
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
  --out_dir runs\diagnostics\<run_name>\native_eval_final `
  --label final_ppp
```

评估规则 baseline：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_structured_mixed_heads_native.py `
  --config configs\current\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
  --baseline_policy cluster_center_queue_aware `
  --episodes 64 `
  --num_envs 64 `
  --episode_seed_base 900000 `
  --device cuda `
  --access_bw_decision_interval 5 `
  --sat_decision_interval 1 `
  --out_dir runs\diagnostics\<run_name>\native_eval_rule `
  --label rule_rrr
```

`cluster_center_queue_aware` 在 native eval 中对应：

```text
accel = cluster_center_queue_aware
sat   = queue_aware
bw    = queue_aware
```

评估 stage-best 全组合：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_structured_mixed_heads_native.py `
  --config configs\current\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
  --base_checkpoint runs\diagnostics\<run_name>\final.pt `
  --accel_checkpoint runs\diagnostics\<run_name>\best_stage_heads\best_accel.pt `
  --sat_checkpoint runs\diagnostics\<run_name>\best_stage_heads\best_sat.pt `
  --bw_checkpoint runs\diagnostics\<run_name>\best_stage_heads\best_bw.pt `
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
  --out_dir runs\diagnostics\<run_name>\native_eval_stagebest `
  --label best_ppp
```

8 组合评估使用同一个脚本，通过 `exec_*_source` 和可选 stage checkpoint 替换实现。已有结果通常放在：

```text
runs/diagnostics/<run_name>/native_eval_stage_grid_8x_final_best/
```

## 训练输出

典型 run 目录：

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
```

重点文件：

- `metrics.csv`：训练曲线，包含 MC return、EV、KL、episode reward、collision、episode length、intervention、耗时。
- `phase_trace.jsonl`：中断或卡住时定位阶段。
- `final.pt`：最终 checkpoint。
- `best_stage_heads/best_*.pt`：每个 stage 的 best head。

## 当前结果文档

```text
docs/joint_mcgae_macro_k5_nogrow_u300_rerun_eval_20260514.md
```

当前 K=5 主结果。

```text
docs/joint_mcgae_macro_k10_nogrow_u300_eval_20260514.md
```

K=10 对照。

```text
docs/joint_mcgae_k5_danger_imitation_ablation_20260515.md
```

danger imitation 开关消融。

```text
docs/joint_mcgae_training_summary_for_slides_20260514.md
```

给导师汇报用的简版材料。

```text
docs/bw_access_macro_decision_interval_design_20260513.md
```

BW macro K 的设计说明。

## 当前指标重点

训练期重点看：

- `rollout_episode_reward_mean`
- `rollout_completed_episode_reward_mean`
- `*_mc_return_mean`
- `explained_variance` / `ev[a,s,b]`
- `approx_kl_*`
- `rollout_collision_rate`
- `rollout_episode_length_mean`
- `rollout_intervention_rate`
- `iteration_sec / collect_sec / critic_sec / actor_sec`

评估期重点看：

- `reward_sum`
- `processed_ratio_eval`
- `drop_ratio_eval`
- `pre_backlog_steps_eval`
- `sat_overlap_eval`
- `collision_episode_fraction`
- `episode_length`

## 旧入口说明

以下脚本仍保留，但不是当前 K=5 joint MC-GAE 主线：

```text
scripts/train_structured.py
scripts/evaluate_structured.py
scripts/train_sat_mcgae.py
scripts/train_stage_mcgae.py
scripts/train.py
scripts/evaluate.py
scripts/render_episode.py
```

使用它们前先确认目标是历史 PPO、单阶段诊断、legacy PettingZoo，还是普通 structured eval。当前论文主结果和 8 组合评估优先使用 `train_joint_mcgae.py` 与 `evaluate_structured_mixed_heads_native.py`。

## 源码导航

```text
sagin_marl/env/config.py
```

配置入口。

```text
sagin_marl/env/structured_batch_env_core.py
```

structured batch 环境与 native runtime。

```text
sagin_marl/env/native_cuda/
```

CUDA kernel 和 bindings。

```text
sagin_marl/rl/structured_actor.py
```

三头 actor。

```text
sagin_marl/rl/structured_critic.py
```

relational critic。

```text
sagin_marl/rl/distributions.py
```

动作分布和 log-prob 口径。

```text
sagin_marl/rl/native_actor_cuda.py
```

PyTorch actor 到 native CUDA ABI 的同步。

```text
sagin_marl/rl/structured_mappo.py
```

actor/critic eval 与 MAPPO 工具函数。

更简洁的文件导航见 `PROJECT_STRUCTURE.md`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --import-mode=importlib
```

改 native CUDA、动作分布、BW macro、critic/actor 结构后，至少先做：

- `py_compile` 对改动脚本
- 小规模 1 update shapecheck
- 相关 parity / smoke 诊断

不要直接长训验证语义改动。
