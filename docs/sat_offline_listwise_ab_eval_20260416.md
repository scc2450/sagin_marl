# SAT 离线 Listwise Actor A/B 评估

## 评估口径

固定：

- `accel = heuristic`
- `bw = heuristic`
- `heuristic_policy = cluster_center_queue_aware`

只让 `sat = policy`，比较两个 SAT actor：

1. 离线训练前：
   [runs/tmp_sat_clean_smoke/actor_final.pt](/d:/研三上/毕设/sagin_marl/runs/tmp_sat_clean_smoke/actor_final.pt:1)
2. 离线训练后：
   [runs/sat_offline_listwise_positivegap_v2_20260416/actor_offline_listwise.pt](/d:/研三上/毕设/sagin_marl/runs/sat_offline_listwise_positivegap_v2_20260416/actor_offline_listwise.pt:1)

评估命令都用同一批新 seed：

- `episodes = 20`
- `episode_seed_base = 72000`
- `policy_mode = deterministic`

结果文件：

- 训练前：
  [runs/sat_actor_ab_before_20260416/summary.json](/d:/研三上/毕设/sagin_marl/runs/sat_actor_ab_before_20260416/summary.json:1)
- 训练后：
  [runs/sat_actor_ab_after_20260416/summary.json](/d:/研三上/毕设/sagin_marl/runs/sat_actor_ab_after_20260416/summary.json:1)

## 结果

训练前：

- `reward_sum = -20679611.66`
- `processed_ratio_eval = 0.4769`
- `drop_ratio_eval = 0.4542`
- `pre_backlog_steps_eval = 37.9581`
- `x_rel_mean = 0.4304`
- `throughput_backhaul_norm_mean = 0.4304`
- `sat_processed_incoming_ratio_mean = 1.0540`

训练后：

- `reward_sum = -20837464.99`
- `processed_ratio_eval = 0.4465`
- `drop_ratio_eval = 0.4544`
- `pre_backlog_steps_eval = 41.6459`
- `x_rel_mean = 0.3999`
- `throughput_backhaul_norm_mean = 0.3999`
- `sat_processed_incoming_ratio_mean = 1.0770`

## 结论

在这次固定 `accel/bw`、只替换 SAT actor 的完整 episode A/B 里，离线 listwise 训练后的 SAT actor **没有带来更好的整体回报**，而且整体表现更差：

- 总回报更低
- `processed_ratio_eval` 更低
- `pre_backlog_steps_eval` 更高
- `x_rel_mean / throughput_backhaul_norm_mean` 更低

所以目前能得出的结论是：

- 离线 listwise 确实能把 SAT actor 往局部 oracle 候选推
- 但这种局部改进 **还没有转化成完整闭环里的更好总回报**
