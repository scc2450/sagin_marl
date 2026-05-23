# SAT Clean 当前方案与结果汇总（2026-04-16）

## 1. 当前主线方案

当前 `sat` 的主线不再使用旧的 `Beijing + sticky hotspot + res200` SAT clean 环境作为第一验证场景，而是先切回 **Lyapunov 主配置骨架**，只保留 `sat clean` 的训练机制。

当前主线配置：

- [configs/clean_sat/structured_sat_clean_joint_lyapunov_base.yaml](/d:/研三上/毕设/sagin_marl/configs/clean_sat/structured_sat_clean_joint_lyapunov_base.yaml:1)

这样做的原因是：

- 在旧的 `Beijing + sticky hotspot + res200` SAT clean 环境里，`queue_aware` 评估表现明显失真，前端 GU 侧过重，SAT 侧不是真正主瓶颈。
- 在 `lyapunov` 骨架上，固定 heuristic 的整体验证更健康，更适合先判断“SAT clean 训练方法本身有没有把 SAT 往好的方向推”。

## 2. 当前训练口径

### 2.1 环境骨架

保留 `lyapunov` 主配置的正式多 UAV 环境骨架：

- `num_uav = 3`
- `num_gu = 20`
- `T_steps = 250`
- `traffic_model = homogeneous`
- `b_acc = 1.0e7`
- `b_sat_total = 1.5e7`
- `sat_cpu_freq = 1.8e10`
- `fixed_satellite_strategy = false`

不再额外叠加：

- `Beijing + sticky hotspot`
- `resource_scale_enabled`
- `res200` 那套额外资源改动

### 2.2 训练/执行方式

当前 SAT clean 训练口径如下：

- 只训练 `sat`
  - `train_accel = false`
  - `train_sat = true`
  - `train_bw = false`
- 固定 partner
  - `exec_accel_source = cluster_center_queue_aware`
  - `exec_sat_source = policy`
  - `exec_bw_source = queue_aware`
- 走独立的 `sat_clean_joint` 路径
  - `sat_clean_joint_enabled = true`
- 不混旧 SAT 辅助信号
  - `sat_supervision_enabled = false`
  - `sat_counterfactual_credit_enabled = false`
- 真正 `critic-free`
  - 不建 critic
  - 不做 critic update

### 2.3 奖励口径

这里有一个实现约束需要单独说明：

- 当前 `sat_clean_joint_enabled` 在代码里要求 `reward_mode = weighted_workload_level`

所以当前主线是：

- 环境骨架按 `lyapunov`
- 训练奖励口径按 `weighted_workload_level`

这意味着：

- `reward_sum` 可以用于同口径前后比较
- 但**不能**和原始 `lyapunov` 配置里 `controllable_flow` 的 `reward_sum` 直接横比

### 2.4 `sat_clean_joint` 路径说明

这条路径不是普通 PPO 的 SAT 分支，而是一条独立的 `sat clean joint` 更新链路。代码入口和主要流向如下。

训练入口：

- 在 [train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1099) 里，先用 `sat_clean_joint_critic_free_enabled(...)` 判断这次 run 是否走 SAT clean critic-free。
- 满足条件时，`build_structured_modules_from_config(..., build_critic=not critic_free_build)` 会直接不建 critic，见 [train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1116)。
- 控制台也会打印 `Critic off | SAT clean joint path`，见 [train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1161)。

配置约束检查：

- 在 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:620) 初始化时，会先打开 `sat_clean_joint_enabled` 相关配置。
- 然后在 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:705) 开始做硬约束检查，当前实现要求：
- `train_accel = false`
- `train_sat = true`
- `train_bw = false`
- `num_uav > 1`
- `exec_accel_source = cluster_center_queue_aware`
- `exec_sat_source = policy`
- `exec_bw_source = queue_aware`
- `reward_mode = weighted_workload_level`
- 不能和 `reward_stage3_sat_overlap_enabled / sat_counterfactual_credit_enabled / sat_supervision_enabled` 混用

rollout 时如何缓存 SAT snapshot：

- 在每次环境 step 里，accel stage 执行完、sat stage 即将决策时，会先导出 `sat_stage_state`，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:1337)。
- 单环境 driver 用 [StructuredControlDriver.export_sat_stage_state()](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:528)。
- 多环境 `subproc` 版本走 [export_sat_stage_state_many()](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:1048)。
- 这个 snapshot 里包含：
- `env.export_runtime_state()` 导出的完整运行时状态
- 当前 step 是否打开
- `stage_assoc / stage_candidates / stage_bw_valid_mask`
- `stage_visible / stage_sat_pos / stage_sat_vel`
- `stage_cached_eta`
- `stage_world_cache`

更新入口：

- 在 `buffer.as_stage_batches()` 之后，`update()` 会直接分流到 [_update_sat_clean_joint(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3725)，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3959)。
- 这意味着走 `sat_clean_joint` 时，不再进入普通 PPO 三头 update 主线。

上下文筛选：

- `_update_sat_clean_joint(...)` 先只取 `stage_id == 1` 的 SAT 样本，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3728)。
- 然后从这些样本里取出：
- `sat_stage_states`
- `local_actor_states`
- 当前 rollout 里实际执行的 SAT `joint_actions`
- 接着调用 [_sat_clean_select_context_indices(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3644) 做上下文筛选。
- 这一步会先用当前 actor 的 [evaluate_sat_pair(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:2103) 算出当前执行动作的熵，再用合法 subset 数做归一化，最后按：
- 每个 env 保留 `sat_clean_entropy_topk_per_env` 个高熵样本
- 再补 `sat_clean_uniform_contexts_per_env` 个均匀随机样本
- 总数截到 `sat_clean_contexts_per_update`

候选 joint 动作如何构造：

- 对被选中的 context，会调用 [sat_topk_legal_subset_indices(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:2106) 取每个 UAV 当前 actor 下的 `top-M` 合法 subset，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3766)。
- 当前实现里，`M` 由 `sat_clean_topm_per_uav` 控制。
- 然后对各 UAV 的 `top-M` 做笛卡尔积，形成 candidate joint panel，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3788)。
- 当前 rollout 实际执行的 joint 动作不会强行并入这个 panel，而是单独记成 `current_pair_indices`，专门拿来做 reference reward。

snapshot replay 和打分：

- 每个 context 都会构成一个 task，包含：
- `snapshot_state`
- `current_pair_indices`
- `candidate_pair_indices`
- 打分时优先走并行版本 [_sat_clean_score_tasks_parallel(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3597)；如果并行条件不满足，就退回串行 [_sat_clean_score_tasks_serial(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3551)。
- 并行版本内部调用 [StructuredRemoteDriverGroup.rollout_sat_clean_tasks_many(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:1198)，也就是把这些 replay task 分发到 `subproc` worker。
- worker 里真正做事的是 [_worker_score_sat_pair_from_snapshot(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:478)：
- 先用 [load_sat_stage_state(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:583) 恢复同一个 SAT snapshot
- 再把 `pair_indices` 解码成真实 SAT 选择
- 执行 `run_sat_stage(...)`
- 然后基于当前状态重新调用固定 `queue_aware_bw` 产生 BW 动作
- 最后执行 `execute_stage_bw_and_prepare_next_accel(...)`
- 取 `bw_weighted_workload_level_reward` 作为评分；如果没有这个字段，就退回 step reward

为什么这一步是“只改 SAT、别的保持一致”：

- [load_sat_stage_state(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:583) 会恢复 `env_state`，而 `env_state` 里包含 runtime RNG 状态。
- 所以同一个 context 下，`current_pair_indices` 和各个 candidate 都是从同一个 snapshot、同一个随机状态出发重放的。
- 变化只来自这一步 SAT 动作不同；后续 BW continuation 固定为 `queue_aware`，但会对不同状态做出不同动作，这是刻意保留的闭环部分。

teacher 选择和参数更新：

- 每个 context 会先得到 `current_reward`，再得到整个 candidate panel 的 `candidate_rewards`，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3806)。
- 只要 `best(candidate_rewards) - current_reward > sat_clean_positive_gap_eps`，这个 context 才会进入正样本集合，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3814)。
- 也就是说，当前实现只用“teacher 明确优于当前执行动作”的 context 来训。
- 更新时，不再用 PPO advantage，而是直接对 teacher subset 做分类式更新：
- 先用 [evaluate_sat_pair(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:2103) 取 teacher 动作在旧参数下的 `logprob / logits`
- 再多轮小批次优化 `loss = -eval_out.logprob.mean()`，见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3854)
- 这本质上就是“把当前 actor 的概率往 teacher 选中的 SAT subset 上推”

训练日志里的关键诊断量：

- `clean_mean_target_gap`
- `clean_mean_update_shift`
- `clean_update_to_target_ratio`
- `clean_target_beats_ref_frac`

这些量都在 [_update_sat_clean_joint(...)](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3876) 里返回，并写进训练日志。它们分别表示：

- 当前 teacher 相对 reference 的平均正 gap
- 一次 update 后 teacher 动作概率平均抬高了多少
- update 吃掉了旧 target room 的多少比例
- 被选中 context 里，有多少比例真的存在正 gap teacher

## 3. 当前验证方法

### 3.1 固定 heuristic 基线

先跑固定 heuristic 参考，用于判断环境是否“通”：

- `accel = heuristic`
- `sat = heuristic`
- `bw = heuristic`
- `heuristic_policy = cluster_center_queue_aware`

结果文件：

- [runs/sat_clean_lyapunov_base_eval_20260416/summary.csv](/d:/研三上/毕设/sagin_marl/runs/sat_clean_lyapunov_base_eval_20260416/summary.csv:1)

### 3.2 训练前/训练后 A/B

再对 SAT actor 做固定 partner A/B：

- `accel = heuristic`
- `sat = policy`
- `bw = heuristic`
- `heuristic_policy = cluster_center_queue_aware`
- `policy_mode = deterministic`
- `episodes = 20`
- `episode_seed_base = 72000`

训练前初始 actor：

- [runs/sat_clean_lyapunov_base_probe_20260416/actor_init.pt](/d:/研三上/毕设/sagin_marl/runs/sat_clean_lyapunov_base_probe_20260416/actor_init.pt)

训练 run：

- [runs/sat_clean_lyapunov_base_probe_20260416](/d:/研三上/毕设/sagin_marl/runs/sat_clean_lyapunov_base_probe_20260416)

训练前评估：

- [runs/sat_clean_lyapunov_base_probe_before_eval_20260416/summary.csv](/d:/研三上/毕设/sagin_marl/runs/sat_clean_lyapunov_base_probe_before_eval_20260416/summary.csv:1)

训练 `40` updates 后评估：

- [runs/sat_clean_lyapunov_base_probe_after40_eval_20260416/summary.csv](/d:/研三上/毕设/sagin_marl/runs/sat_clean_lyapunov_base_probe_after40_eval_20260416/summary.csv:1)

## 4. 当前结果表

| 设置 | reward_sum | processed_ratio_eval | drop_ratio_eval | pre_backlog_steps_eval | D_sys_report | x_rel_mean | sat_processed_incoming_ratio_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全 heuristic 基线 | -559894.2783 | 1.005877 | 0.065629 | 8.533136 | 24.683970 | 0.959347 | 1.013570 |
| 训练前 SAT policy | -822032.8081 | 0.981845 | 0.071878 | 18.600642 | 163.826590 | 0.935314 | 1.763336 |
| 训练后 SAT policy（40 updates） | -539335.1892 | 1.005877 | 0.065629 | 10.500946 | 29.976328 | 0.959347 | 1.078772 |

## 5. 结果解读

可以明确确认的结论：

1. 这套 `lyapunov-base` 的 SAT clean 训练**不是空转**。
2. 训练 `40` updates 后，SAT policy 已经明显朝固定 heuristic 的好方向移动。
3. `processed_ratio_eval`、`drop_ratio_eval`、`x_rel_mean` 已基本追平 fixed heuristic。
4. `D_sys_report` 从 `163.83` 明显降到 `29.98`，改善很大。
5. `pre_backlog_steps_eval` 也从 `18.60` 降到 `10.50`，但还没有完全追到 heuristic 的 `8.53`。

更具体地说：

- `processed_ratio_eval`
  - 从 `0.981845` 提升到 `1.005877`
  - 已追平 heuristic
- `drop_ratio_eval`
  - 从 `0.071878` 降到 `0.065629`
  - 已追平 heuristic
- `x_rel_mean`
  - 从 `0.935314` 提升到 `0.959347`
  - 已追平 heuristic
- `pre_backlog_steps_eval`
  - 从 `18.600642` 降到 `10.500946`
  - 明显改善，但仍略高于 heuristic
- `D_sys_report`
  - 从 `163.826590` 降到 `29.976328`
  - 明显改善，但仍略高于 heuristic

## 6. 训练过程中的信号情况

训练日志在：

- [runs/sat_clean_lyapunov_base_probe_20260416/metrics.csv](/d:/研三上/毕设/sagin_marl/runs/sat_clean_lyapunov_base_probe_20260416/metrics.csv:1)

从后段更新看，teacher 信号不是没起作用：

- `clean_mean_target_gap` 持续为正
- `clean_target_beats_ref_frac` 大多在 `0.75 ~ 0.92`

这说明当前 `sat_clean_joint` 的 teacher 目标在训练后段仍然在提供正向更新信号。

## 7. 当前结论

截至目前，最稳妥的结论是：

- **SAT clean 训练方法本身可以把 SAT 动作往好的方向上推。**
- 这个结论目前已经在 `lyapunov-base` 环境上得到验证。
- 之前的问题更像是旧的 `Beijing + sticky hotspot + res200` SAT clean 环境把题目改坏了，而不是 “SAT 根本训不出来”。

当前最合理的下一步不是立刻回旧环境，而是：

1. 继续在 `lyapunov-base` 上把训练再拉长一点，观察 `pre_backlog` 和 `D_sys` 能否进一步追平。
2. 然后再把环境改动一项一项加回去，而不是一次跳回旧的 SAT clean 环境。
