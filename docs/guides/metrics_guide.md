# Metrics Guide

本文档说明当前代码版本实际会稳定写出的三类结果：

- 训练日志：`runs/<RUN_DIR>/metrics.csv` 与训练 TensorBoard
- 评估日志：`eval_*.csv` 与评估 TensorBoard
- checkpoint 评估：`runs/<RUN_DIR>/checkpoint_eval.csv`

本文档对应当前正式三阶段配置：

- `configs/phase1_actions_curriculum_stage1_accel.yaml`
- `configs/phase1_actions_curriculum_stage2_bw.yaml`
- `configs/phase1_actions_curriculum_stage3_sat.yaml`

如果你打开的是更早版本的 run，可能会看到更多旧字段；那些旧字段不属于本文档讨论重点，但文中会标出仍保留的兼容字段。

## 1. 先看横坐标

当前项目里，不同指标不该共用同一种横坐标。

### 1.1 横坐标类别

- `update`
  - 指 PPO 参数更新次数。
  - 适合优化器和 PPO 诊断类指标。
- `total_env_steps`
  - 指累计环境交互步数，已经把并行环境都加总进去了。
  - 适合吞吐、队列、安全等“策略在环境里实际跑出来的表现”。
- `eval_episode`
  - 指评估脚本里的 episode 序号。
  - 适合单次评估输出。
- `checkpoint_update`
  - 指 checkpoint 对应的训练 update 编号。
  - 适合 `checkpoint_eval.csv`。

### 1.2 一个很重要的细节

训练 CSV 里的首列 `step` 始终还是 `update`。

也就是说：

- `metrics.csv` 里的行索引是 `update`
- 但训练 TensorBoard 里，环境表现类指标通常更适合按 `total_env_steps` 看
- PPO 诊断类指标仍更适合按 `update` 看

## 2. 统一归一化口径

当前正式奖励与主评估指标统一按

```text
A_ref = 所有 GU 的理论平均总到达比特数/步
```

做归一化。

当前正式配置里：

- `arrival_ref_mode: expected_arrival`
- `A_ref` 在环境初始化时固定
- 主指标不再以 `queue_max_*` 作为主归一化分母

这样做的目的是让训练奖励、checkpoint 排序和最终评估能共用同一物理口径。

## 3. 聚合口径总览

为了避免后面反复写长句，先约定几个聚合口径名称：

- `最近完整 episode 滚动均值`
  - 在训练中，对最近 `train_episode_stat_window` 个已完成 episode 做滚动平均。
- `当前 rollout step 均值`
  - 当前 update 这一整块采样数据里，对所有 env-step 求平均。
- `当前 update 优化统计`
  - 当前 update 里，跨 PPO epoch 和 minibatch 聚合出来的诊断量。
- `单个 eval episode`
  - 一条评估 episode 的结果。
- `checkpoint 多 episode 均值`
  - 一个 checkpoint 在多条 eval episode 上的平均结果。

## 4. 当前正式训练奖励

正式配置推荐使用 `reward_mode: controllable_flow`。

单步核心量：

```text
x_acc = gu_outflow_sum / A_ref
x_rel = uav_to_sat_inflow_sum / A_ref
g_pre = (B_pre_{t+1} - B_pre_t) / A_ref
d_pre = (gu_drop + uav_drop) / A_ref
```

其中：

- `B_pre = Q_gu_sum + Q_uav_sum`
- `g_pre` 只惩罚前两层 backlog 的正增长

单步训练奖励：

```text
r_train
= reward_w_access * x_acc
+ reward_w_relay * x_rel
- reward_w_pre_drop * d_pre
- reward_w_pre_growth * relu(g_pre)
```

默认正式权重：

- `reward_w_access = 0.5`
- `reward_w_relay = 0.5`
- `reward_w_pre_drop = 1.0`
- `reward_w_pre_growth = 0.2`

## 5. 训练日志 `metrics.csv`

### 5.1 当前最该看的训练指标

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `episode_reward` | CSV + TB | 最近完整 episode 滚动均值 | CSV:`update` / TB:`total_env_steps` | 完整 episode return 的滚动均值 |
| `rollout_reward_per_step` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均每步 reward |
| `x_acc` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均接入吞吐比 |
| `x_rel` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均回传吞吐比 |
| `g_pre` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均前两层 backlog 增长率 |
| `d_pre` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均前两层丢包率 |
| `processed_ratio_eval` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均系统最终处理能力 |
| `drop_ratio_eval` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均全链路丢包率 |
| `pre_backlog_steps_eval` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均前两层 backlog 等价步数 |
| `D_sys_report` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均系统 backlog / processed 比值 |
| `collision_rate` | CSV + TB | 最近完整 episode 滚动均值 | CSV:`update` / TB:`total_env_steps` | 最近完成 episode 的碰撞比例 |

### 5.2 PPO 诊断

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `policy_loss` | CSV + TB | 当前 update 优化统计 | `update` | PPO actor loss |
| `value_loss` | CSV + TB | 当前 update 优化统计 | `update` | critic MSE loss |
| `explained_variance` | CSV + TB | 当前 update 优化统计 | `update` | critic 对 return 的解释度 |
| `approx_kl` | CSV + TB | 当前 update 优化统计 | `update` | 近似 KL |
| `clip_frac` | CSV + TB | 当前 update 优化统计 | `update` | PPO clip 命中比例 |
| `entropy` | CSV + TB | 当前 update 优化统计 | `update` | 总熵 |
| `entropy_accel` | CSV + TB | 当前 update 优化统计 | `update` | accel 头熵 |
| `entropy_bw` | CSV + TB | 当前 update 优化统计 | `update` | bw 头熵 |
| `entropy_sat` | CSV + TB | 当前 update 优化统计 | `update` | sat 头熵 |

### 5.3 运行统计

下列字段仍然有用，建议保留：

- `completed_episode_count`
- `episode_length_mean`
- `actor_lr`
- `critic_lr`
- `update_time_sec`
- `rollout_time_sec`
- `optim_time_sec`
- `env_steps_per_sec`
- `update_steps_per_sec`
- `total_env_steps`
- `total_time_sec`

### 5.4 兼容保留字段

下面这些字段仍会保留，便于和旧实验对照，但正式分析不建议再把它们当主口径：

- `throughput_access_norm`
- `throughput_backhaul_norm`
- `episode_term_throughput_access`
- `episode_term_throughput_backhaul`

## 6. 评估日志 `eval_*.csv`

`evaluate.py` 每个 episode 写一行。

### 6.1 当前主评估指标

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `processed_ratio_eval` | CSV + TB | 单个 eval episode | `eval_episode` | 系统最终处理量 / `A_ref` |
| `drop_ratio_eval` | CSV + TB | 单个 eval episode | `eval_episode` | `(gu_drop + uav_drop + sat_drop) / A_ref` |
| `pre_backlog_steps_eval` | CSV + TB | 单个 eval episode | `eval_episode` | `(Q_gu + Q_uav) / A_ref` |
| `D_sys_report` | CSV + TB | 单个 eval episode | `eval_episode` | `(Q_gu + Q_uav + Q_sat) / max(sat_processed_bits, eps)` |

其中：

- `processed_ratio_eval` 是主排序指标 1
- `drop_ratio_eval` 是主排序指标 2
- `pre_backlog_steps_eval` 是主排序指标 3
- `D_sys_report` 只做报告，不进主排序

### 6.2 训练诊断辅助指标

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `x_acc_mean` | CSV + TB | 单个 eval episode | `eval_episode` | 整个 episode 的平均接入吞吐比 |
| `x_rel_mean` | CSV + TB | 单个 eval episode | `eval_episode` | 整个 episode 的平均回传吞吐比 |
| `g_pre_mean` | CSV + TB | 单个 eval episode | `eval_episode` | 整个 episode 的平均前两层 backlog 增长率 |
| `d_pre_mean` | CSV + TB | 单个 eval episode | `eval_episode` | 整个 episode 的平均前两层丢包率 |

### 6.3 仍然推荐保留的辅助字段

这些字段虽然不是新的主排序口径，但仍然有分析价值：

- `reward_sum`
- `steps`
- `terminated_early`
- `collision`
- `gu_queue_mean`
- `uav_queue_mean`
- `gu_queue_arrival_steps_p95`
- `uav_queue_arrival_steps_p95`

## 7. Checkpoint 评估 `checkpoint_eval.csv`

checkpoint 评估是“按固定 update 间隔，对当前 checkpoint 做多 episode 确定性评估”的汇总表。

### 7.1 核心结果字段

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `update` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 当前行对应的训练 update |
| `reward_sum` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 当前 checkpoint 的平均 return |
| `processed_ratio_eval` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 当前 checkpoint 的平均系统最终处理能力 |
| `drop_ratio_eval` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 当前 checkpoint 的平均全链路丢包率 |
| `pre_backlog_steps_eval` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 当前 checkpoint 的平均前两层 backlog 等价步数 |
| `D_sys_report` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 当前 checkpoint 的平均系统 backlog / processed 比值 |
| `collision_episode_fraction` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 发生碰撞的 episode 比例 |

### 7.2 固定参考策略字段

以 `fixed_` 开头的字段表示固定参考策略的同口径结果，例如：

- `fixed_reward_sum`
- `fixed_processed_ratio_eval`
- `fixed_drop_ratio_eval`
- `fixed_pre_backlog_steps_eval`
- `fixed_D_sys_report`
- `fixed_collision_episode_fraction`

当前正式配置常用的固定参考策略：

- Stage 1：`queue_aware`
- Stage 2：`queue_aware`
- Stage 3：`stage2_exec_fixed_sat`

### 7.3 排序与状态字段

| 字段 | 含义 |
| --- | --- |
| `processed_improved` | 是否刷新过 `processed_ratio_eval` 最优值 |
| `drop_improved` | 是否刷新过 `drop_ratio_eval` 最优值 |
| `pre_backlog_improved` | 是否刷新过 `pre_backlog_steps_eval` 最优值 |
| `model_improved` | 按主排序键是否成为新的最优 checkpoint |
| `quality_worsened` | 是否相对上次 checkpoint 同时变差 |
| `quality_worse_streak` | 连续质量恶化次数 |
| `reward_improved` | `reward_sum` 是否达到最小相对改进阈值 |
| `reward_plateau_streak` | 连续 reward 未达改进阈值次数 |
| `early_stop_triggered` | 是否触发 checkpoint-eval early stop |

## 8. 当前 checkpoint 排序与 early stop

### 8.1 当前 checkpoint 排序

当前最优 checkpoint 选择顺序是：

1. `processed_ratio_eval` 高
2. `drop_ratio_eval` 低
3. `pre_backlog_steps_eval` 低

这和训练 reward 不是一回事。

### 8.2 当前正式 early stop

当前正式 YAML 里生效的是 reward-plateau early stop：

- 从 `checkpoint_eval_start_update` 开始
- 每 `checkpoint_eval_interval_updates` 做一次 checkpoint eval
- 若 `reward_sum` 连续 `checkpoint_eval_reward_patience` 次没有达到最小相对改进
- 且 `collision_episode_fraction <= checkpoint_eval_reward_collision_threshold`
- 则提前停止

当前正式配置为：

- `checkpoint_eval_interval_updates = 50`
- `checkpoint_eval_start_update = 200`
- `checkpoint_eval_reward_patience = 5`
- `checkpoint_eval_reward_min_delta_rel = 0.005`
- `checkpoint_eval_reward_collision_threshold = 0.01`

`checkpoint_eval_sat_drop_early_stop_enabled` 在当前正式配置中为 `false`，因此不会因为质量恶化分支直接停训。

## 9. TensorBoard 页面对应关系

### 9.1 训练页

- `Training/Main`
  - `episode_reward`
  - `rollout_reward_per_step`
  - `x_acc`
  - `x_rel`
  - `g_pre`
  - `d_pre`
  - `processed_ratio_eval`
  - `drop_ratio_eval`
  - `pre_backlog_steps_eval`
  - `D_sys_report`
  - `collision_rate`
- `Training/PPO`
  - `policy_loss`
  - `value_loss`
  - `entropy`
  - `entropy_accel`
  - `entropy_bw`
  - `entropy_sat`
  - `approx_kl`
  - `clip_frac`
  - `explained_variance`

### 9.2 评估页

- `Eval/Main`
  - `reward_sum`
  - `processed_ratio_eval`
  - `drop_ratio_eval`
  - `pre_backlog_steps_eval`
  - `D_sys_report`
  - `x_acc_mean`
  - `x_rel_mean`
  - `g_pre_mean`
  - `d_pre_mean`
  - `collision`
  - `terminated_early`

## 10. 阶段化对比建议

不同阶段，baseline 也应按“动作集合一致”来选。

### 10.1 Stage 1

主对比建议：

- `queue_aware`
- `cluster_center`
- `centroid`

这是 `accel` 阶段，主要看 UAV 机动。

### 10.2 Stage 2

主对比建议：

- `queue_aware`
- `stage1 actor + queue_aware_bw`

这里更应该比较“同时有 `accel + bw` 的方法”，而不只是单动作基线。

### 10.3 Stage 3

主对比建议：

- `queue_aware`
- `stage2 actor + queue_aware_sat`

这里更应该比较“同时有 `accel + bw + sat` 的方法”。

## 11. 怎么看这些图最省力

如果只想快速判断一次训练值不值得继续看，建议按这个顺序：

1. `metrics.csv`
   - `x_acc`
   - `x_rel`
   - `g_pre`
   - `d_pre`
   - `collision_rate`
2. `checkpoint_eval.csv`
   - `processed_ratio_eval`
   - `drop_ratio_eval`
   - `pre_backlog_steps_eval`
   - `reward_plateau_streak`
   - `early_stop_triggered`
3. `eval_trained_final.csv` 与对应 baseline CSV
   - 看最终 trained vs baseline 的同口径对比
