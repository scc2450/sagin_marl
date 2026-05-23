# Structured `sat` Clean-Only 方案（2026-04-15）

## 1. 目标

当前目标不是继续在 joint PPO 上给 `sat` 头打补丁，而是做一条真正的 `structured sat clean-only` 路线，用来回答两个问题：

1. 在当前 `structured` 动作语义下，`sat` 本身能不能在不被 `accel / bw` 污染的前提下学出来。
2. 如果能，什么样的 teacher / replay / 并行框架最适合 `sat`。

这条线的核心不是“只训练 sat 参数”这么简单，而是：

- 训练信号只由 `sat` 决定。
- 比较不同 `sat` 动作时，环境随机性严格一致。
- 不依赖 critic。
- teacher 搜索必须并行，不能把整个训练拖成串行。

---

## 2. 基线配置怎么来

### 2.1 主骨架

主骨架从下面这条正式 `structured` 配置出发：

- [configs/phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995_lyapunov.yaml](/d:/研三上/毕设/sagin_marl/configs/phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995_lyapunov.yaml:1)

保留它的：

- `num_uav: 3`
- `num_gu: 20`
- `T_steps: 250`
- Ka / VSAT 场景
- `doppler / fading / interference / atm / rain`
- 当前 `structured` 三阶段控制
- 当前 `spawn / avoidance / boundary` 设定
- `fixed_satellite_strategy: false`

### 2.2 从 BW 最新 clean 配置搬什么

从下面这条 BW clean 配置借环境口径：

- [configs/clean_per_user/structured_bw_clean_per_user_beijing_res200_rewardobs_h30.yaml](/d:/研三上/毕设/sagin_marl/configs/clean_per_user/structured_bw_clean_per_user_beijing_res200_rewardobs_h30.yaml:1)

要搬的是：

- `Beijing` 几何锚点
  - `ref_lat_deg: 39.9042`
  - `ref_lon_deg: 116.4074`
  - `rain_lat_deg: 39.9042`
- `sticky_subset_hotspot` 流量模型
- `res200` 资源口径
  - `resource_scale_enabled: true`
  - `resource_scale_ref_num_uav: 3`
  - `resource_scale_ref_num_gu: 20`
  - `resource_scale_ref_task_arrival_rate: 4.9e5`
  - `resource_scale_ref_sat_active_count: 3.0`
  - `resource_scale_b_acc_multiplier: 0.45`
  - `b_acc / b_sat_total / sat_cpu_freq` 的 res200 数值
- reward / reward-aligned obs 口径
  - `reward_mode: weighted_workload_level`
  - `obs_own_include_assoc_uav_cost: true`
  - `obs_sat_include_sat_cost: true`

### 2.3 明确不搬什么

下面这些不要从 BW clean 单 UAV 配置里搬：

- `num_uav: 1`
- `num_gu: 8`
- `T_steps: 40`
- `uav_spawn_mode: gu_centroid`
- `uav_safe_random_init_enabled: false`
- `uav_init_speed_frac: 0.0`
- `fixed_satellite_strategy: true`
- `avoidance_enabled: false`
- `exec_accel_source: zero`
- `exec_sat_source: zero`

### 2.4 `Beijing hotspot` 怎么搬

“不要搬 `1 UAV / 8 GU`”不等于“不要搬 `Beijing hotspot`”。

推荐做法：

- 保留 `traffic_model: sticky_subset_hotspot`
- 保留 `hotspot_rho: 20.0`
- 保留 `hotspot_on_mean_steps: 20.0`
- 保留 `hotspot_off_mean_steps: 1.0`
- 保留 `arrival_mean_preserve: true`
- 把 `hotspot_subset_size` 从 `4/8=0.5` 按比例放大到 `10/20=0.5`
- 把 `hotspot_num_subsets` 提到和 `num_gu` 同阶，建议先用 `20`

这样保住的是“北京站点 + 热点切换流量”的语义，不是把单 UAV 小环境原样复制过来。

---

## 3. 当前 structured `sat` actor 结构

### 3.1 actor 是独立的，不是旧 shared trunk sat head

`structured` 这条线上，`sat` 不是旧 `policy.py` 那个共享 trunk sat 头，而是独立 actor：

- [sagin_marl/rl/structured_factory.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_factory.py:67)
- [sagin_marl/rl/structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:242)
- [sagin_marl/rl/structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:1944)

### 3.2 输入

sat actor 输入是 [LocalSatState](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_types.py:66)，由 [build_batched_local_sat_states_from_world](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_stage_builders.py:151) 构造。

字段包括：

- `ego_uav_after_accel`
- `sat_nodes`
- `sat_edges`
- `sat_mask`
- `subset_tokens`
- `subset_mask`
- `subset_members`

### 3.3 已经包含的关键信号

当前 sat actor 已经能看到：

- `sat_queue`：
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1225)
- `sat current load`：
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1226)
- `SE`：
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1288)
- `sat queue feature`：
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1289)
- `sat load feature`：
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1290)
- `projected bandwidth share`
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1291)
- `current selection flag`
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1294)

### 3.4 动作语义

当前 sat actor 不是逐星独立打分，而是对合法 subset 直接分类：

- subset 枚举在 [build_sat_subset_tokens](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:115)
- 空集在有可见星时会被 mask 掉：
  - [structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:138)
- sat actor 输出 subset logits：
  - [structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:308)
  - [structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:313)

如果 `visible_sats_max=6` 且 `sat_num_select=2`，常见合法动作数是：

- 单选 `C(6,1)=6`
- 双选 `C(6,2)=15`
- 合计 `21`

### 3.5 结论

`v1` 不需要重写 sat actor 主结构。最小必要修改只有：

1. 给 `ego_uav_after_accel` 增一个稳定身份特征 `uav_id_norm`
2. 给 sat actor 加一个 helper，返回 legal subset 的 `top-K index / logits`

---

## 4. 真正的 clean sat-only 定义

这条线的 clean 定义如下：

1. 只训练 sat：
   - `train_accel: false`
   - `train_sat: true`
   - `train_bw: false`
2. partner 固定：
   - `exec_accel_source: cluster_center_queue_aware`
   - `exec_bw_source: queue_aware`
   - 这里要按 `structured_mappo.py` 的命名来写，不是旧 `mappo.py` 的 `heuristic`
   - 见 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:303)
3. 所有 teacher 打分都用：
   - `reward_mode: weighted_workload_level`
4. 不混这些旧 sat 信号：
   - `reward_stage3_sat_overlap_enabled: false`
   - `sat_counterfactual_credit_enabled: false`
   - `sat_supervision_enabled: false`
5. 比较不同 `sat` 动作时，必须从同一个 `sat-stage` snapshot 出发
6. 改变 `sat` 动作时，环境随机性必须保持一致

这里第 5 条和第 6 条比“只更新 sat 参数”更重要。

---

## 5. 这条 clean 路线不需要 critic

### 5.1 判断

这条 `sat clean-only` 路线的 teacher 目标是直接从 replay 得到的离散 subset label，不是依赖 critic 估计 advantage。

所以这条线应该像 BW clean 一样走 **critic-free actor-only**：

- 不构建 critic 网络
- 不创建 critic optimizer
- update 时不跑 critic epoch

### 5.2 现有代码的相关入口

现在 `train_structured.py` 已经支持“构建时就不建 critic”：

- [scripts/train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1093)

当前 BW clean 的逻辑是：

- `build_critic=False`
- `critic_optimizer=None`
- 主进程用 `ZeroStructuredCritic`

参考：

- [structured_factory.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_factory.py:36)
- [train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1101)
- [train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1108)

### 5.3 需要补的开关

当前 `StructuredMAPPO.update()` 里只认 BW actor-only signal 的 critic-free 标志：

- [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3235)

因此 sat clean 需要补一条新的判定，例如：

- `sat_clean_joint_enabled`
- `sat_clean_joint_critic_free_enabled`

并让它满足时：

1. `build_structured_modules_from_config(..., build_critic=False)`
2. `critic_optimizer=None`
3. `StructuredMAPPO.update()` 不再要求 critic optimizer
4. 完全跳过 critic training / value refresh / GAE 相关分支

这不是“把 critic loss 设成 0”就够了，而是要真正从构建、优化器、update 路径三处都关掉。

---

## 6. replay 一致性：必须补 `sat-stage` 导出/恢复

### 6.1 现在缺什么

当前 `structured` 已经有：

- `BW stage` 导出 / 恢复
  - [export_bw_stage_state](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:477)
  - [load_bw_stage_state](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:528)

但 `sat` 只有：

- [build_sat_stage_snapshot](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:346)

没有：

- `export_sat_stage_state()`
- `load_sat_stage_state()`

### 6.2 为什么这一步是第一优先级

如果没有 sat-stage 导出/恢复，teacher 比较不同 sat action 时只能在 live env 上分叉推进，这会让：

- fading / doppler / stochastic arrival
- env RNG
- stage cache

全部混到 teacher 差异里。

这样得到的不是“只由 sat 决定的 teacher 信号”。

### 6.3 底层支持已经有了

环境底层已经支持完整运行时状态和 RNG 恢复：

- [export_runtime_state](/d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py:1387)
- [load_runtime_state](/d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py:1429)
- `rng_bit_generator_state`
  - [sagin_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py:1421)

所以 sat 这里应仿照 BW stage，序列化：

- `env_state`
- `stage_assoc`
- `stage_candidates`
- `stage_visible`
- `stage_sat_pos`
- `stage_sat_vel`
- sat world build cache
- sat obs cache

### 6.4 验收条件

`export_sat_stage_state / load_sat_stage_state` 做完以后，必须能通过：

1. 同一 snapshot
2. 同一 sat action
3. 重复 replay 多次

返回：

- 完全相同的 `reward`
- 完全相同的 `next world state`
- 完全相同的 `last_reward_parts`

---

## 7. context 怎么抽

### 7.1 原则

只从 **当前 rollout** 抽，不用跨 update 的旧 label bank。

原因：

- actor 会漂
- 旧 teacher label 会过期
- sat clean 的重点是“当前 policy 下哪里值得花 teacher 计算”

### 7.2 具体抽法

每遇到一个 sat-stage context，就记录：

- `sat_stage_state`
- 当前 rollout 实际执行的 sat `pair_index`
- 当前 sat actor 在该状态下的 entropy

entropy 直接用 sat actor 的 categorical entropy。

然后归一化：

- `H_norm = H / log(valid_subset_count)`

其中：

- `valid_subset_count = subset_mask.sum()`

每个 env 在当前 rollout 内保留：

- `top-2` 个最高 `H_norm` 的 context
- `+1` 个均匀随机 context

每个 update 真正送去 teacher 搜索的总数，建议先设：

- `16 ~ 24`

这样做：

- 不假设 overlap 一定坏
- 不用旧 bank
- 比纯随机更能把 teacher 预算集中到当前策略最不确定的 sat state

---

## 8. teacher 候选动作怎么生成

### 8.1 不加随机 subset

当前 sat actor 本来就在 legal subset 空间上直接输出 categorical logits。

因此 teacher panel 直接用：

- 每个 UAV 当前 sat actor 的 legal subset `top-M`

建议：

- `M = 4`

不要再额外加：

- `random legal subset`

原因：

- teacher panel 本身应尽量确定、可复现
- sat actor 已经在 subset 空间建模，不需要再靠随机 panel 增广

### 8.2 `current action` 的角色

这里要区分两个“当前动作”：

1. `当前 deterministic action`
   - 就是 policy argmax subset
   - 它天然就在 `top-M` 里
2. `当前 rollout 实际执行动作`
   - 如果 rollout 是 stochastic，它可能不是 argmax
   - 它不一定在 `top-M` 里

teacher 打分时，真正的 baseline 应该是：

- `score_current = 当前 rollout 实际执行 joint sat action 的分数`

不是 deterministic argmax。

---

## 9. joint teacher 打分流程

### 9.1 单个 context 的 panel

对一个 sat context：

1. 恢复同一个 `sat_stage_state`
2. 用当前 sat actor 对每个 UAV 算 legal subset logits
3. 每个 UAV 取 `top-4` subset
4. 组成 joint panel

如果 `3 UAV`、每个 `4` 个候选：

- joint 候选总数 = `4^3 = 64`

### 9.2 评分流程

对每个 joint sat action：

1. `load_sat_stage_state(snapshot)`
2. `run_sat_stage(candidate_joint_action)`
3. 构建 `BW stage`
4. 用固定 `queue_aware BW` 执行 `bw`
5. 只走 `1-step`
6. 读取 `weighted_workload_level`

把它记为：

- `score_candidate`

### 9.3 为什么 `v1` 先只做 `1-step`

`v1` 先只做单步，是为了尽量保证：

- 变化只来自当前 sat 决策
- 不把后续多步 sat 连锁效应混进来
- 并行 teacher 计算成本可控

多步 horizon 可以以后再做，不是第一版必要条件。

---

## 10. 损失函数为什么用交叉熵

sat clean 的 teacher 给的是：

- “这个 `LocalSatState` 下，第 `j*` 个 legal subset 是 teacher 选中的动作”

当前 sat actor 输出的是：

- 对 legal subset 的 categorical logits

所以最自然的监督损失就是分类损失：

- `L_sat_clean = CE(logits, teacher_subset_index)`

等价写法是：

- `L_sat_clean = -log pi(a_teacher | obs)`

这不是数学形式偏好，而是因为动作本身是**离散类别**。

BW clean 能用 Huber，是因为 BW teacher 给的是连续 simplex 目标。

sat 这里如果对 subset index 做 Huber / MSE，会把“类别编号”误当成连续量，不合适。

`v1` 建议：

- 先用普通 cross-entropy
- 不加 gap 权重
- 不做 soft target KL

等 replay 和 teacher 路线稳定后，再考虑更软的 distillation 版本。

---

## 11. 并行方案：必须走 `subproc + GPU`

### 11.1 原则

teacher 搜索不能串行在主进程里循环 `64` 个 joint action。

整个训练要继续保持：

- rollout 环境：`subproc`
- actor 前向：GPU
- teacher 打分：多 worker 并行环境回放

### 11.2 参考现有 BW clean 架构

现有 BW clean 已经有一套可复用的结构：

- `StructuredRemoteDriverGroup`
  - [structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:784)
- `subproc` worker 里做环境回放
- 主进程 GPU 上做 policy forward

相关现有接口：

- `begin_step_many`
  - [structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:889)
- `run_accel_stage_many`
  - [structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:902)
- `run_sat_stage_many`
  - [structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:949)
- `build_bw_stage_snapshot_many`
  - [structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:960)
- `load_bw_stage_state_many`
  - [structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:967)

### 11.3 sat clean 的并行拆分

sat clean 推荐的并行拆分是：

#### 主进程 GPU 做：

1. 对选中的 sat contexts 批量算 sat actor logits
2. 为每个 UAV 取 `top-M` legal subset
3. 组 joint candidate panel

#### subproc worker 做：

1. `load_sat_stage_state(snapshot)`
2. `run_sat_stage(candidate_joint_action)`
3. `build_bw_stage_snapshot`
4. 用固定 `queue_aware BW` 执行一步
5. 返回 `weighted_workload_level`

这样做的好处：

- GPU 只做 sat actor 前向
- 环境 replay 压到多进程 CPU worker
- 不需要在 worker 里同步 sat actor 参数去做 follow action
- 比 BW clean 更简单，因为 sat clean 的 `bw` 是固定 heuristic，不需要 worker 再调用 learned BW follow actor

### 11.4 需要新增的远程接口

建议在 `StructuredRemoteDriverGroup` 上新增：

- `export_sat_stage_state_many`
- `load_sat_stage_state_many`
- `rollout_sat_clean_tasks_many`

其中 `rollout_sat_clean_tasks_many` 的单个 task 至少包含：

- `snapshot_state`
- `candidate_joint_sat_actions`
- `bw_exec_source`
- `reward_mode`

worker 返回：

- 每个 candidate 的 `score`
- baseline `score_current`
- 可选的 `reward_parts`

### 11.5 配置建议

teacher 并行沿用现有 BW clean 风格的字段最方便：

- `sat_clean_parallel_envs`
- `sat_clean_parallel_backend: subproc`
- `sat_clean_tasks_per_worker`

运行时仍建议：

- `--vec_backend subproc`
- GPU 训练
- `torch_threads` 保持 `1~2`

---

## 12. 需要修改的代码

### 12.1 环境 / driver

- [sagin_marl/env/structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:346)
  - 新增 `export_sat_stage_state`
  - 新增 `load_sat_stage_state`
  - sat-stage cache 序列化 / 恢复

### 12.2 远程并行

- [sagin_marl/env/structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:784)
  - 新增 `export_sat_stage_state_many`
  - 新增 `load_sat_stage_state_many`
  - 新增 `rollout_sat_clean_tasks_many`

### 12.3 sat actor 小改动

- [sagin_marl/rl/structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:242)
  - legal subset `top-K` helper

### 12.4 structured trainer / learner

- [scripts/train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1093)
  - 按 sat clean 模式决定 `build_critic=False`
- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3235)
  - 增加 sat clean 的 critic-free 路径
  - 增加 sat clean context 收集
  - 增加 `_update_sat_clean_joint`

### 12.5 world-state 输入

- [sagin_marl/env/structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1297)
  - 把 `uav_nodes[..., 6]` 改成 `uav_id_norm`

---

## 13. `v1` 不做什么

第一版先不做：

- 回到 joint 训练
- 多步 sat teacher horizon
- overlap penalty
- queue-aware sat 候选注入
- random legal subset 候选
- soft-target KL distillation

`v1` 只回答：

- replay 是否严格一致
- sat clean-only 在固定 accel/bw partner 下能不能学出正信号

---

## 14. 验收标准

### 14.1 正确性

1. `sat_stage_state` replay 一致性通过
2. 同一 snapshot + 同一 sat action，多次 replay 完全一致
3. 不同 sat action 的差异只来自 sat 决策本身

### 14.2 训练信号

1. `score_teacher - score_current` 平均为正
2. 正 gap context 占比稳定，不是极少数样本
3. sat clean 的分类损失下降时，eval 同步改善

### 14.3 eval

主看：

- `weighted_workload_level`

辅助看：

- `processed_ratio_eval`
- `pre_backlog_steps_eval`

只记录、不作为目标：

- `sat_overlap_eval`

---

## 15. 推荐的推进顺序

1. 先做新配置：
   - `structured lyapunov` 主骨架
   - 搬入 `Beijing + hotspot + res200`
   - 切 `reward_mode=weighted_workload_level`
2. 补 `export_sat_stage_state / load_sat_stage_state`
3. 做 sat-stage replay 一致性测试
4. 做 sat clean 的 critic-free 框架
5. 做 current-rollout entropy 抽样
6. 做 `top-M` joint teacher panel
7. 做 `subproc + GPU` teacher 打分
8. 再接 sat clean cross-entropy 更新

在第 3 步 replay 一致性没过之前，不建议动 loss 和训练细节。
