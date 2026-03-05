# 训练与评估指标说明（更新于 2026-03-05）

本项目训练指标写入 `runs/<log_dir>/metrics.csv` 与 TensorBoard。  
评估指标写入 `runs/<log_dir>/eval_trained.csv` / `runs/<log_dir>/eval_baseline.csv`，并同步到 TensorBoard（默认 `runs/<log_dir>/eval_tb`，标签 `eval/trained` 与 `eval/baseline`）。

## 1. 训练指标（`metrics.csv`）

### 1.1 PPO 与优化过程
- `episode_reward`: 每次 update 的平均回报（总回报/步数）。
- `policy_loss`, `value_loss`, `entropy`: PPO 主损失与探索强度。
- `reward_rms_sigma`: reward RMS 的标准差 `sigma_R`（仅 `reward_norm_enabled=true` 时有效）。
- `reward_clip_frac`: reward 归一化后触发 clip 的比例（仅 `reward_norm_enabled=true` 且 `reward_norm_clip>0` 时有效）。
- `approx_kl`: 当前策略与旧策略的近似 KL。
- `clip_frac`: PPO clip 生效比例（`|ratio-1| > clip_ratio`）。
- `adv_raw_mean`, `adv_raw_std`: 标准化前优势函数统计。
- `adv_norm_mean`, `adv_norm_std`: 标准化并裁剪后的优势统计。
- `imitation_loss`, `imitation_coef`: 模仿学习损失与当前系数。
- `actor_lr`, `critic_lr`: 当前学习率。

### 1.2 奖励拆解（比率项与分项）
- 比率/中间项:  
`r_service_ratio`, `r_drop_ratio`, `r_assoc_ratio`, `r_queue_pen`, `r_queue_topk`, `r_queue_delta`, `r_centroid`, `centroid_dist_mean`, `r_bw_align`, `r_sat_score`, `r_dist`, `r_dist_delta`, `r_energy`, `r_collision_penalty`, `r_battery_penalty`, `r_fail_penalty`。
- 最终奖励各项（`term_*`）:  
`r_term_service`, `r_term_drop`, `r_term_queue`, `r_term_topk`, `r_term_assoc`, `r_term_q_delta`, `r_term_dist`, `r_term_dist_delta`, `r_term_centroid`, `r_term_bw_align`, `r_term_sat_score`, `r_term_energy`, `r_term_accel`。
- 归一化流量/队列项:  
`reward_raw`, `arrival_sum`, `outflow_sum`, `service_norm`, `drop_norm`, `drop_sum`, `queue_total`, `queue_total_active`, `arrival_rate_eff`。

### 1.3 系统状态与吞吐
- 队列: `gu_queue_mean`, `uav_queue_mean`, `sat_queue_mean`, `gu_queue_max`, `uav_queue_max`, `sat_queue_max`。
- 丢弃: `gu_drop_sum`, `uav_drop_sum`。
- 卫星流量: `sat_processed_sum`, `sat_incoming_sum`。
- 能耗: `energy_mean`（`energy_enabled=true` 才有意义）。
- 性能: `update_time_sec`, `rollout_time_sec`, `optim_time_sec`, `env_steps`, `env_steps_per_sec`, `update_steps_per_sec`, `total_env_steps`, `total_time_sec`。

## 2. 评估指标（`eval_trained.csv` / `eval_baseline.csv`）

- 回报与速度: `reward_sum`, `reward_raw`, `steps`, `episode_time_sec`, `steps_per_sec`。
- 服务质量: `service_norm`, `drop_norm`, `centroid_dist_mean`。
- 队列与丢弃: `gu_queue_mean`, `uav_queue_mean`, `sat_queue_mean`, `gu_queue_max`, `uav_queue_max`, `sat_queue_max`, `gu_drop_sum`, `uav_drop_sum`。
- 卫星流量与能耗: `sat_processed_sum`, `sat_incoming_sum`, `energy_mean`。
- 关联统计: `assoc_ratio_mean`, `assoc_dist_mean`。

## 3. 常见“为 0 / 不变化”解释

- `energy_mean`, `r_energy`, `r_term_energy`: `energy_enabled=false` 时为 0。
- `r_collision_penalty`: 仅发生碰撞时为负值，未碰撞时为 0。
- `r_battery_penalty`: 仅能量耗尽时为负值，`energy_enabled=false` 时恒为 0。
- `r_term_service`: `eta_service=0` 时为 0（即使 `service_norm` 非零）。
- `r_term_assoc`: 当前奖励实现中未启用，固定 0。
- `r_term_bw_align`: `eta_bw_align=0` 时为 0（`r_bw_align` 可能仍有值）。
- `r_term_sat_score`: `eta_sat_score=0` 时为 0（`r_sat_score` 可能仍有值）。
- `r_term_dist`, `r_term_dist_delta`, `r_dist`, `r_dist_delta`, `r_queue_topk`, `r_term_topk`: 当前实现保留位，固定 0。
- `gu_drop_sum`, `uav_drop_sum`, `drop_norm`, `drop_sum`: 容量与速率配置充足时可能长期为 0，这通常是系统处于“无丢弃”区间，不一定是 bug。
- `r_assoc_ratio` 或 `assoc_ratio_mean` 持续为 1: 表示 GU 始终成功关联（常见于覆盖与门限充足场景）。

## 4. 关于 `queue_total` 与 `gu_queue_max`

- `queue_total` 是“每步总队列（GU+UAV+SAT）”在一个 update 内的时间平均值。
- `gu_queue_max` 是“单个 GU 队列峰值”在该 update 内的最大值（峰值统计）。
- 两者统计口径不同，出现 `gu_queue_max` 远大于 `queue_total` 并不必然是错误。

## 5. 排查建议

1. 先看 `reward_sum/episode_reward` 与 `service_norm/drop_norm` 是否一致改善。  
2. 再看 `approx_kl` 与 `clip_frac`：若长期过大，通常表示策略更新过激。  
3. 看 `adv_raw_std` 与 `adv_norm_std`：过小可能学习信号不足，过大可能回报尺度或归一化不稳定。  
4. 对“长期为 0”的项，先核对对应开关与权重，再判断是否实现问题。  
