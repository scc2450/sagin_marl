# 训练与评估指标说明（更新于 2026-03-06）
训练指标写入 `runs/<log_dir>/metrics.csv`（并同步到 TensorBoard）。  
评估指标写入 `runs/<log_dir>/eval_trained.csv` / `runs/<log_dir>/eval_baseline.csv`。

## 1. 训练指标（`metrics.csv`）
### 1.1 PPO 与优化过程
- `episode_reward`, `policy_loss`, `value_loss`, `entropy`
- `approx_kl`, `clip_frac`
- `adv_raw_mean`, `adv_raw_std`, `adv_norm_mean`, `adv_norm_std`
- `reward_rms_sigma`, `reward_clip_frac`
- `imitation_loss`, `imitation_coef`
- `actor_lr`, `critic_lr`

### 1.2 奖励分解与中间量
- 比率/中间项：  
`r_service_ratio`, `r_drop_ratio`, `r_assoc_ratio`, `r_queue_pen`, `r_queue_topk`, `r_queue_delta`, `r_centroid`, `centroid_dist_mean`, `r_bw_align`, `r_sat_score`, `r_dist`, `r_dist_delta`, `r_energy`, `r_collision_penalty`, `r_battery_penalty`, `r_fail_penalty`
- 奖励项（`r_term_*`）：  
`r_term_service`, `r_term_drop`, `r_term_queue`, `r_term_topk`, `r_term_assoc`, `r_term_q_delta`, `r_term_dist`, `r_term_dist_delta`, `r_term_centroid`, `r_term_bw_align`, `r_term_sat_score`, `r_term_energy`, `r_term_accel`
- 队列/归一化相关：  
`reward_raw`, `arrival_sum`, `outflow_sum`, `service_norm`, `drop_norm`, `drop_sum`, `queue_total`, `queue_total_active`, `q_norm_active`, `prev_q_norm_active`, `q_norm_delta`, `arrival_rate_eff`
- 尾部惩罚与交叉退火相关：  
`q_norm_tail_q0`, `q_norm_tail_excess`, `queue_weight`, `q_delta_weight`, `crash_weight`, `centroid_transfer_ratio`
- 安全层相关：  
`collision_rate`, `avoidance_eta_eff`, `avoidance_eta_exec`, `avoidance_collision_rate_ema`, `avoidance_prev_episode_collision_rate`

### 1.3 队列、尾部与安全监测
- 均值与极值：  
`gu_queue_mean`, `uav_queue_mean`, `sat_queue_mean`, `gu_queue_max`, `uav_queue_max`, `sat_queue_max`
- 尾部分位（按 update 内步序列统计）：  
`q_norm_active_p95`, `q_norm_active_p99`, `queue_total_active_p95`, `queue_total_active_p99`
- 丢包与卫星流量：  
`gu_drop_sum`, `uav_drop_sum`, `sat_processed_sum`, `sat_incoming_sum`

### 1.4 运行性能
- `energy_mean`
- `update_time_sec`, `rollout_time_sec`, `optim_time_sec`
- `env_steps`, `env_steps_per_sec`, `update_steps_per_sec`
- `total_env_steps`, `total_time_sec`

## 2. 评估指标（`eval_trained.csv` / `eval_baseline.csv`）
- 回报与速度：  
`reward_sum`, `reward_raw`, `steps`, `episode_time_sec`, `steps_per_sec`
- 服务与队列：  
`service_norm`, `drop_norm`, `queue_total_active`, `centroid_dist_mean`
- 吞吐与丢包：  
`arrival_sum`, `outflow_sum`, `outflow_arrival_ratio`, `drop_sum`, `drop_ratio`, `drop_ratio_step_mean`
- 队列/丢包/卫星流量：  
`gu_queue_mean`, `uav_queue_mean`, `sat_queue_mean`, `gu_queue_max`, `uav_queue_max`, `sat_queue_max`, `gu_drop_sum`, `uav_drop_sum`, `sat_processed_sum`, `sat_incoming_sum`
- 关联与能量：  
`assoc_ratio_mean`, `assoc_dist_mean`, `energy_mean`

## 3. 常见“为 0 / 不变化”解释
- `energy_mean`, `r_energy`, `r_term_energy`：`energy_enabled=false` 时常为 0。
- `r_term_service`：`eta_service=0` 时为 0（即使 `service_norm` 非 0）。
- `r_term_bw_align`：`eta_bw_align=0` 时为 0。
- `r_term_sat_score`：`eta_sat_score=0` 时为 0。
- `drop_*` 长期为 0 可能是系统处于无丢包区间，不一定是 bug。
- `collision_rate` 长期接近 0 代表当前配置下几乎无碰撞；若与早停冲突，优先核查 `collision_event` 记录口径。

## 4. 口径说明（关键）
- `q_norm_active` 是逐步计算：  
`q_norm_active = clip(queue_total_active / queue_arrival_scale(arrival_sum), 0, 1)`。
- `q_norm_active_p95/p99` 与 `queue_total_active_p95/p99` 是“单个 update 内所有环境步样本”的分位数，不是跨 update 分位。
- `collision_rate` 是 update 内 `collision_event` 的步均值（0/1 事件平均）。
- 当启用尾部惩罚（`q_norm_tail_q0 > 0`）时，`queue_pen = max(q_norm_active-q0,0)^2`（active 分支）；`queue_weight` 反映该步实际惩罚权重。
- 启用交叉退火后，`centroid_transfer_ratio` 反映 centroid 退火进度，`queue_weight/q_delta_weight/crash_weight/avoidance_eta_exec` 为迁移后的有效权重。

## 5. 实验排查建议
1. 先看 `collision_rate` 与 `q_norm_active_p95/p99`，判断是否在“稳态+控尾”方向改进。  
2. 再看 `queue_total_active_p95/p99` 与 `drop_norm`，确认高拥塞尾部是否下降。  
3. 最后结合 `centroid_transfer_ratio` 与有效权重（`queue_weight` 等）检查交叉退火是否按预期发生。  
