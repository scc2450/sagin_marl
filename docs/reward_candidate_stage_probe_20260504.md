# Stage-Specific Reward Candidate Probe

目的：不训练 actor，只检查不同 reward 在当前 policy samples 附近，能不能给 PPO actor 提供可学习的一阶动作 credit。

这版诊断以真实 autograd 为准：

```text
A_oracle = G(s, a_sample) - mean_a G(s, a)
g = mean[A_oracle * grad log pi(a_sample | s)]
```

脚本会调用当前 stage actor 自己的 `evaluate_*().logprob`，再对真实 actor 参数做 `torch.autograd.grad`。旧 score proxy 仍会输出，但只作为辅助，不再作为结论依据。

## Invalid Slot 问题

之前 BW 的 proxy 用过近似 Dirichlet score，其中 `log(action)` 会被 invalid / 极小 action 放大。这是诊断 proxy 的问题，不是 BW actor 本身的问题。

当前 BW actor 的真实口径是：

```text
invalid score -> masked
invalid action slot -> log_prob 不贡献
```

`accel` 和 `sat` 没有同类 invalid-slot log 放大问题：

```text
accel: tanh Gaussian，没有 invalid slot。
sat: masked categorical / subset 选择，invalid 候选在 logits 层被 mask。
```

不过为了避免 proxy 和真实 PPO 梯度不一致，三类 stage 现在都统一看 true autograd gradient SNR。

## Candidate Reward

已有口径：

```text
weighted_workload_level
positive_weighted_workload_level
weighted_workload_delta
relative_weighted_workload_delta
controllable_flow
access_raw
```

新增诊断候选：

```text
accel_access_safe  = x_acc - d_pre - 0.05 * log1p(pre_backlog_steps) - 0.2 * close_risk - 2.0 * collision
accel_delta_safe   = weighted_workload_delta - 10.0 * close_risk - 100.0 * collision
accel_growth_safe  = x_acc - relu(g_pre) - d_pre - 0.5 * close_risk - 2.0 * collision
accel_pressure_safe = x_acc - d_pre - 0.25 * overflow_risk - 0.25 * downstream_pressure - 0.5 * close_risk - 2.0 * collision
accel_queue_relief = -g_pre - d_pre - 0.5 * close_risk - 2.0 * collision

sat_relay_processed = 0.5 * x_rel + 0.5 * processed_ratio_eval - drop_ratio_eval - 0.05 * sat_overlap_eval
sat_backhaul_drop   = x_rel - d_pre - 0.05 * sat_overlap_eval

bw_access_drop     = x_acc - d_pre - 0.2 * service_gap_risk_mean
bw_relative_access = x_acc / (1 + pre_backlog_steps) - 0.5 * d_pre
bw_growth_drop     = x_acc - relu(g_pre) - 2.0 * d_pre - 0.5 * service_gap_risk_mean
bw_queue_relief    = -g_pre - d_pre - 0.5 * service_gap_risk_mean
bw_access_pressure = x_acc - 2.0 * d_pre - 0.5 * overflow_risk - 0.5 * service_gap_risk_mean
```

其中 `sat_relay_processed` / `sat_backhaul_drop` 已接入训练 reward mode；其他新候选暂时只用于诊断。

## 诊断设置

```text
config: configs/tmp/structured_accel_sanity_3uav_20gu_t250_currentenv_ppo_positive_nowarmup.yaml
num_envs: 8
rollout_env_steps: 250
sample_rows: 8
policy_action_samples: 4
autograd_chunk_count: 4
accel std: 0.2
```

判断口径：

```text
true_grad_snr < 1
  当前 PPO 一阶梯度大概率被采样噪声淹没。

1 <= true_grad_snr < 3
  有信号但偏弱，训练可能慢、波动大。

true_grad_snr >= 3
  当前 reward + policy 分布下，一阶 PPO 信号比较可用。
```

## Accel 结果

文件：

```text
runs/diagnostics/reward_action_sensitivity/stage_accel_rows8_k4_truegrad_candidates_v2.json
```

Top true-autograd SNR：

```text
relative_weighted_workload_delta  1.395
bw_growth_drop                    1.352
accel_growth_safe                 1.339
bw_access_pressure                1.336
bw_relative_access                1.328
controllable_flow                 1.310
bw_access_drop                    1.305
accel_access_safe                 1.297
positive_weighted_workload_level  1.168
weighted_workload_delta           0.709
```

判断：新设计里“队列增长/相对变化”比单纯 level 好，但仍不到 3。也就是说 reward 方向有改善，但当前 PPO samples 附近仍只是弱信号。

## SAT 结果

文件：

```text
runs/diagnostics/reward_action_sensitivity/stage_sat_rows8_k4_truegrad_candidates.json
```

Top true-autograd SNR：

```text
sat_relay_processed               3.427
sat_backhaul_drop                 3.427
relative_weighted_workload_delta  2.148
bw_relative_access                2.022
controllable_flow                 1.989
positive_weighted_workload_level  1.617
weighted_workload_level           1.085
access_raw                        1.059
```

判断：SAT 的旧 proxy 结论不可靠。用真实 autograd 后，`sat_relay_processed` / `sat_backhaul_drop` 在这个 seed 上是明显最强候选。

## BW 结果

文件：

```text
runs/diagnostics/reward_action_sensitivity/stage_bw_rows8_k4_truegrad_candidates_v2.json
```

Top true-autograd SNR：

```text
accel_growth_safe       1.609
sat_relay_processed     1.454
sat_backhaul_drop       1.454
weighted_workload_delta 1.453
bw_relative_access      1.420
bw_growth_drop          1.371
controllable_flow       1.369
bw_access_drop          1.305
positive_level          1.093
weighted_level          0.927
```

判断：BW 有弱信号，但新候选也没有把 SNR 拉到好训区域。比较重要的是，BW 的同状态 action-scale 很小，说明在当前 policy samples 附近，改一点 simplex 分配对 episode return 的影响仍然很弱。

## SAT 短训验证

`sat_relay_processed` 已接入真实 reward mode，并做了 sat-only 20 update 对照：

```text
sat_relay_processed:
  run: runs/diagnostics/sat_relay_processed_satonly_u20
  env_reward_mean first5 -> last5: 0.775 -> 0.672
  entropy_sat first -> last: 7.06 -> 6.96
  explained_variance_sat: around 0

positive_weighted_workload_level baseline:
  run: runs/diagnostics/sat_positive_satonly_u20_baseline
  env_reward_mean first5 -> last5: 0.175 -> 0.159
  entropy_sat first -> last: 7.06 -> 4.28
  explained_variance_sat: around 0 to 0.16
```

结论：`sat_relay_processed` 的 true-grad 强，但 20 update PPO 短训没有显示 reward 上升。这个结果说明“true-grad 强”是必要但不充分；还需要检查 SAT PPO 更新是否真的在改变回传侧选择，以及训练日志需要直接记录 `x_rel / processed_ratio_eval / drop_ratio_eval / sat_overlap_eval`，不能只看 `bw_access_reward_mean`。

## 当前判断

```text
Accel:
  现有系统目标类 reward 太容易被状态难度淹没。
  增长/相对变化类 reward 有改善，但仍偏弱。

SAT:
  回传侧 stage-specific reward 的一阶信号最好。
  但短训没转成稳定上升，需要继续查 SAT update/指标记录。

BW:
  当前 simplex 小扰动对 return 的可见影响仍弱。
  仅改 reward 公式还没有解决 credit 信号弱的问题。
```

后续不要再用旧 proxy 排名判断 reward。reward 诊断看 `ranking_by_true_autograd_gradient_snr`，训练验证必须同时看 stage 直接影响的物理量。
