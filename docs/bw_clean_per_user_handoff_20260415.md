# BW Clean Per-User 路线交接文档（2026-04-15）

这份文档用于把当前 `clean per-user BW teacher` 路线的实现状态、相关文件、验证结果和后续建议整理清楚，方便在新对话里继续。

---

## 1. 当前目标

当前在做的是一条**独立于 `branch_delta` / 旧 `proxy` / 旧 `per-slot PPO`** 的 BW 学习路线：

- 输入继续使用 `Reward-Obs`
- BW actor 改成 `score_only_softmax`
- `critic` 关闭
- clean teacher 用同一个 `bw_stage_state` 做多步 counterfactual
- teacher 内部分支已经改成：
  - 环境分支 `subproc`
  - policy follow forward 在主进程 `GPU`

公平对照对象是当前最强 baseline：

- [runs/structured/eight_gu_t40_beijing_res200_rewardobs_h30_u60/checkpoint_eval.csv](</d:/研三上/毕设/sagin_marl/runs/structured/eight_gu_t40_beijing_res200_rewardobs_h30_u60/checkpoint_eval.csv:1>)

也就是：

- `Reward-Obs + branch_delta h=30`

---

## 2. 已有方案文档

clean per-user 方案定义文档（中文）：

- [docs/bw_clean_per_user_teacher_20260415.md](</d:/研三上/毕设/sagin_marl/docs/bw_clean_per_user_teacher_20260415.md:1>)

这份文档描述的是“理想算法定义”。本交接文档描述的是“当前代码里已经做到什么、还差什么”。

---

## 3. 关键实现文件

### 3.1 配置与开关

- [sagin_marl/env/config.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/config.py:301>)

新增/相关项包括：

- `structured_bw_parameterization = "score_only_softmax"`
- `bw_clean_per_user_enabled`
- `bw_clean_per_user_horizon`
- `bw_clean_per_user_delta_probe`
- `bw_clean_per_user_beta`
- `bw_clean_per_user_loss`

并且这条 clean 路线被纳入了“BW actor-only signal -> critic-free”的判定。

---

### 3.2 BW actor：score-only softmax

- [sagin_marl/rl/structured_actor.py](</d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:1744>)

这里实现了 `score_only_softmax`：

- 只保留 per-user `score`
- 最终动作是 `masked softmax(score)`
- 不再走 `kappa/tau`
- 不再走 `stick-breaking`
- clean 路线下 `action = det_mean`
- `logprob = 0`
- `entropy = 0`

这条路线是 deterministic simplex policy。

---

### 3.3 clean per-user 主更新逻辑

- [sagin_marl/rl/structured_mappo.py](</d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:2375>)

这里新增/相关的核心函数有：

- `_bw_clean_rollout_return_with_driver`
- `_bw_clean_det_actions_from_snapshots`
- `_bw_clean_rollout_returns_parallel`
- `_bw_clean_target_actions_parallel`
- `_bw_clean_huber_loss`
- `_update_bw_clean_per_user`

当前 clean 路线特点：

- 独立主更新分支
- 不混 `branch_delta`
- 不混旧 `proxy_score`
- 不混旧 `per-slot PPO`
- 不用 aux loss
- 主损失是 `Huber(pred_action, target_action)`

### 3.4 critic 关闭逻辑

- [sagin_marl/rl/structured_mappo.py](</d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:345>)
- [scripts/train_structured.py](</d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1054>)

当前 clean 路线：

- `critic` 正确关闭
- `value_loss = 0`
- `critic_optimizer = None`

---

## 4. Reward-Obs 相关实现

### 4.1 Reward-Obs 的当前布局

reward-aligned 特征现在按位置放置：

- per-user：
  - `local_gu_service_cost`
  - `weighted_queue_cost`
  - `weighted_queue_cost_relative`
- ego_uav：
  - `assoc_uav_cost`
- sat-side：
  - `sat_cost`

相关代码：

- [sagin_marl/env/sagin_env.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py:844>)
- [sagin_marl/env/structured_driver.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1092>)
- [configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200_rewardobs.yaml](</d:/研三上/毕设/sagin_marl/configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200_rewardobs.yaml:1>)

### 4.2 Reward-Obs 和 stage cache 对齐

之前 reward-obs 有“读旧 `env.last_*`”的问题，已经修复。

现在 structured BW actor 输入使用当前 step 的：

- `_stage_assoc`
- `_stage_sat_selection`

而不是隐式回退到上一拍的 `env.last_*`。

相关代码：

- [sagin_marl/env/sagin_env.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py:401>)
- [sagin_marl/env/sagin_env.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py:883>)
- [sagin_marl/env/structured_driver.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1159>)
- [tests/test_structured_driver_buffer.py](</d:/研三上/毕设/sagin_marl/tests/test_structured_driver_buffer.py:575>)

---

## 5. teacher 分支：subproc + GPU

### 5.1 当前状态

现在 clean teacher 的多步 rollout 已经走到：

- 环境分支：`StructuredRemoteDriverGroup` 的 `subproc` worker
- BW follow policy 前向：主进程批量 `GPU`

也就是说：

- 环境动力学仍然是 NumPy/CPU
- 但 teacher 内部分支已经不再是本地串行 replay
- 现在是 `subproc` worker 上回放环境，主进程 GPU 上算 BW follow action

相关代码：

- [sagin_marl/env/structured_vec_env.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:493>)
- [sagin_marl/rl/structured_mappo.py](</d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:2291>)

### 5.2 当前限制

clean teacher 目前的 `subproc` 并行路径，依赖：

- `exec_accel_source == "zero"`
- `exec_sat_source == "zero"`

这样 worker 里就不需要额外做 accel/sat policy 前向，只要做：

- `begin_step`
- `run_accel_stage(zero)`
- `run_sat_stage(zero)`
- `build_bw_stage_snapshot`

然后主进程在 GPU 上批量算 BW follow action。

当前实验配置正好满足这个前提。

---

## 6. snapshot / replay 一致性修复

这是这次最关键的正确性修复之一。

### 6.1 之前为什么 replay 和原始 reward 不一致

之前 `load_bw_stage_state()` 会在载入 snapshot 后额外重建一部分 stage cache：

- 重算 `eta`
- 重算 `_stage_access_gain_matrix`

而在复杂信道配置下，这两步会吃随机信道采样（例如 fading），从而推进 `rng`，导致：

- 同一个 snapshot
- 同一个动作
- replay reward 和原始 reward 不一致

### 6.2 现在怎么修的

现在 `bw_stage_state` 里已经直接序列化了：

- `stage_cached_eta`
- `stage_access_gain_matrix`
- 以及整批 stage world cache

这样 `load_bw_stage_state()` 恢复时不再需要额外随机重建。

相关代码：

- [sagin_marl/env/structured_driver.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:477>)
- [sagin_marl/env/structured_driver.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:496>)
- [sagin_marl/env/sagin_env.py](</d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py:1429>)

### 6.3 当前验证结果

现在用真实配置做 probe：

- 同一 snapshot
- 同一动作
- 原始 reward vs replay reward

已经对齐到 bit-exact：

- `rng_before_equal = True`
- `reward diff = 0.0`
- `outflow_max_abs_diff = 0.0`

probe 脚本：

- [runs/tmp/probe_replay_debug.py](</d:/研三上/毕设/sagin_marl/runs/tmp/probe_replay_debug.py:1>)

---

## 7. 当前 probe / 临时脚本

这些是当前用于验证 clean 路线的临时脚本：

- clean teacher `subproc + GPU` probe：
  - [runs/tmp/probe_clean_teacher_subproc_gpu.py](</d:/研三上/毕设/sagin_marl/runs/tmp/probe_clean_teacher_subproc_gpu.py:1>)

- replay 一致性 probe：
  - [runs/tmp/probe_replay_debug.py](</d:/研三上/毕设/sagin_marl/runs/tmp/probe_replay_debug.py:1>)

这两个脚本都是为了当前调试，后续如果路线稳定，可以删掉或转成正式测试。

---

## 8. 当前测试状态

已通过：

- [tests/test_structured_action_modules.py](</d:/研三上/毕设/sagin_marl/tests/test_structured_action_modules.py:1>)
- [tests/test_structured_smoke_train.py](</d:/研三上/毕设/sagin_marl/tests/test_structured_smoke_train.py:1>)
- [tests/test_env_reset_shapes.py](</d:/研三上/毕设/sagin_marl/tests/test_env_reset_shapes.py:1>)
- [tests/test_structured_driver_buffer.py](</d:/研三上/毕设/sagin_marl/tests/test_structured_driver_buffer.py:1>)

这次新增/相关测试包括：

- `score_only_softmax` 动作形状与 simplex 性质
- clean route 下 critic 关闭、actor 能更新
- reward-obs 使用 stage cache 而不是旧 `env.last_*`
- `load_bw_stage_state()` 恢复 cached eta 不再重算

---

## 9. 真实配置下的当前状态

真实实验配置基础仍然是：

- [configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200_rewardobs.yaml](</d:/研三上/毕设/sagin_marl/configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200_rewardobs.yaml:1>)

clean route 目前还没有单独落一份正式训练配置文件；当前是通过在 probe / 构造 learner 时覆盖这些项：

- `structured_bw_parameterization = score_only_softmax`
- `bw_clean_per_user_enabled = true`
- `bw_clean_per_user_horizon`
- `bw_clean_per_user_delta_probe`
- `bw_clean_per_user_beta`
- `bw_clean_per_user_loss = huber`
- `bw_actor_advantage_override_mode = gae`
- `bw_actor_branch_parallel_envs = 4`
- `bw_actor_branch_parallel_backend = subproc`

---

## 10. 目前最重要的结论

1. clean per-user 这条线的**主结构已经搭起来了**
2. `critic` 已经**正确关闭**
3. `Reward-Obs` 输入继续保留，且和当前 stage cache 对齐
4. snapshot replay 现在已经做到：
   - 同一 snapshot
   - 同一动作
   - 原始/回放 reward 完全一致
5. teacher 内部分支已经改成：
   - `subproc` 环境回放
   - 主进程 `GPU` 上的 BW follow policy 前向

---

## 11. 下一步最自然的工作

如果在新对话里继续，最自然的顺序是：

### 第一步
先把 clean 路线落成一份**正式训练配置文件**，而不是继续靠 probe 脚本临时改。

建议从：

- [configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200_rewardobs.yaml](</d:/研三上/毕设/sagin_marl/configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200_rewardobs.yaml:1>)

复制一份 clean 版配置，例如：

- `..._rewardobs_clean_peruser.yaml`

### 第二步
先跑一条短训练验证 clean 路线是否真的能学起来：

- `num_envs=8`
- `vec_backend=subproc`
- `device=cuda`
- 先短跑若干 updates 看 `policy_loss / eval reward` 是否有正常变化

### 第三步
正式和 baseline 对照：

- baseline：
  - [runs/structured/eight_gu_t40_beijing_res200_rewardobs_h30_u60/checkpoint_eval.csv](</d:/研三上/毕设/sagin_marl/runs/structured/eight_gu_t40_beijing_res200_rewardobs_h30_u60/checkpoint_eval.csv:1>)
- clean per-user：
  - 用同一环境、同一 `Reward-Obs`
  - 只改 BW 参数化和训练目标

---

## 12. 新对话里建议直接说明的内容

如果在新对话接着做，建议直接说：

1. 先读：
   - [docs/bw_clean_per_user_teacher_20260415.md](</d:/研三上/毕设/sagin_marl/docs/bw_clean_per_user_teacher_20260415.md:1>)
   - [docs/bw_clean_per_user_handoff_20260415.md](</d:/研三上/毕设/sagin_marl/docs/bw_clean_per_user_handoff_20260415.md:1>)
2. 当前 clean 路线已经：
   - 带 `Reward-Obs`
   - critic off
   - teacher replay bit-exact
   - teacher branch = `subproc` + GPU policy forward
3. 下一步是：
   - 落正式 clean config
   - 跑第一次正式训练对照

---

## 13. 最后一句

当前最重要的未完成项已经不是“正确性修 bug”，而是：

**把 clean 路线真正作为一条正式训练配置跑起来，并和 `Reward-Obs + branch_delta h=30` 做第一轮干净对照。**

