# Baseline 算法说明

本文档说明当前分支中可运行的非学习式 baseline。历史上的拓扑感知 DPP 原型已经归档到 `docs/archive/legacy/`；当前主线已经保留 native structured evaluator 中的规则/MaxWeight 类 baseline，并补入一个 structured Python 版 `topology_dpp` 作为更强的非学习对照。

## 当前推荐命名

- `cluster_center_queue_aware`：当前工程规则基线。运动头跟踪 GU cluster center，SAT/BW 头使用 queue-aware heuristic。
- `maxweight_lyapunov`：当前推荐的 Lyapunov/MaxWeight 名称。它是 stage-wise queue-pressure controller，兼容旧别名 `lyapunov`。
- `dpp_no_mobility`：轻量消融。关闭移动，SAT/BW 仍使用 Lyapunov/MaxWeight source。
- `dpp_equal_bw`：轻量消融。Accel/SAT 使用 Lyapunov/MaxWeight，BW 改为 uniform。
- `dpp_greedy_sat`：轻量消融。Accel/BW 使用 Lyapunov/MaxWeight，SAT 改为 queue-aware heuristic。
- `topology_dpp`：拓扑感知 one-step DPP baseline。枚举候选 UAV 加速度，预测移动后的接入拓扑，再联合打分 access/backhaul/BW/SAT 决策；当前是 structured Python fallback，不是 native CUDA 快路径。
- `dpp_resource_hybrid`：资源分配优先 hybrid。Accel 沿用稳定的 cluster-center/queue-aware 移动，BW/SAT 由 topology DPP 在移动后的 stage observation 上决策。

## 当前 `maxweight_lyapunov` 的算法含义

当前实现不是完整枚举 joint acceleration 后重算拓扑的 one-step DPP optimizer，而是一个可在 native CUDA 路径中高效执行的 MaxWeight/Lyapunov 风格在线控制器。

直觉上，它根据当前队列压力和链路质量做三类动作：

- Accel：向高压力 GU 的加权方向移动，同时考虑邻居责任划分、避碰和能量项。
- BW：按 GU queue pressure 与 access service gain 分配带宽。
- SAT：按 UAV queue 与 SAT queue 的差压，以及 relay support 选择卫星子集。

保留旧名 `lyapunov` 是为了兼容已有脚本和历史结果；新实验和论文表格建议使用 `maxweight_lyapunov`。

## 与完整 DPP 的区别

完整的 topology-aware one-step DPP 应该在每步枚举候选 UAV 动作，预测移动后的 GU/UAV/SAT 拓扑，再对 access、backhaul、BW、SAT coupling 做一步式优化。当前分支已经迁入一个可运行的 Python/structured 版本，用作论文 strong non-learning benchmark；它还没有 native CUDA kernel 实现，所以评估速度会慢于 `maxweight_lyapunov`。

因此现阶段比较建议分两层：

- 当前强规则基线：`cluster_center_queue_aware` 与 `maxweight_lyapunov`。
- 主 non-learning benchmark 候选：`topology_dpp`，用于检验 learning policy 是否超过更强的拓扑感知 DPP 控制器。

## Native 评估命令

```bash
python scripts/evaluate_structured_mixed_heads_native.py \
  --config configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml \
  --baseline_policy maxweight_lyapunov \
  --episodes 64 \
  --num_envs 64 \
  --episode_seed_base 900000 \
  --device cuda \
  --access_bw_decision_interval 5 \
  --sat_decision_interval 1 \
  --out_dir runs/diagnostics/<run_name>/native_eval_maxweight_lyapunov \
  --label maxweight_lyapunov
```

轻量消融只需替换 `--baseline_policy`：

```text
cluster_center_queue_aware
queue_aware
maxweight_lyapunov
dpp_no_mobility
dpp_equal_bw
dpp_greedy_sat
topology_dpp
dpp_resource_hybrid
```

`topology_dpp` 和 `dpp_resource_hybrid` 可以使用同一个入口，但内部会自动走 structured Python baseline fallback，而不是 `_FIXED_POLICY_EXEC_SOURCE_MAP` 的 native source triple。

## 当前生效的 Lyapunov/MaxWeight 参数

当前 native kernel 明确使用的参数包括：

- `baseline_accel_gain`
- `baseline_repulse_gain`
- `baseline_repulse_radius_factor`
- `baseline_cluster_cruise_speed`
- `baseline_cluster_slow_radius`
- `baseline_cluster_stop_radius`
- `baseline_cluster_speed_tol`
- `baseline_cluster_vel_gain`
- `baseline_lyapunov_v`
- `baseline_lyapunov_urgency_alpha`
- `baseline_lyapunov_bw_service_scale`
- `baseline_lyapunov_sat_drift_weight`

配置中仍保留了一些历史参数，例如 `baseline_lyapunov_drift_weight`、`baseline_lyapunov_action_cost`、`baseline_lyapunov_ema_beta`、`baseline_lyapunov_bw_temp`、`baseline_lyapunov_bw_floor`、`baseline_lyapunov_sat_switch_bias`、`baseline_lyapunov_sat_abs_se_weight`、`baseline_lyapunov_sat_doppler_penalty`。这些参数目前会进入 native ABI 参数表，但当前 kernel 主路径没有实际使用它们；写论文或调参表时不要把它们解释为当前机制贡献。
