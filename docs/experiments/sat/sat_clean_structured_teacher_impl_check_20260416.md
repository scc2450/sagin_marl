# Structured SAT Clean 实现对照（2026-04-16）

对照文档：
- [docs/sat_clean_structured_teacher_plan_20260415.md](/d:/研三上/毕设/sagin_marl/docs/sat_clean_structured_teacher_plan_20260415.md:1)

本文件只记录“已经落到代码里的东西”和“当前版本的有意简化”。

## 1. 已实现

### 1.1 配置基线

新增可运行配置：
- [configs/clean_sat/structured_sat_clean_joint_beijing_hotspot_res200.yaml](/d:/研三上/毕设/sagin_marl/configs/clean_sat/structured_sat_clean_joint_beijing_hotspot_res200.yaml:1)

这份配置满足当前方案口径：
- 主骨架来自 `structured_step250_gamma0995_lyapunov`
- 搬入 `Beijing + sticky_subset_hotspot + res200` 资源口径
- 保留 `3 UAV / 20 GU`
- 开启 `sat_clean_joint_enabled`
- `train_accel=false, train_sat=true, train_bw=false`
- `exec_accel_source=cluster_center_queue_aware`
- `exec_bw_source=queue_aware`
- `reward_mode=weighted_workload_level`
- 关闭 `reward_stage3_sat_overlap_enabled / sat_supervision_enabled / sat_counterfactual_credit_enabled`
- 开启 `obs_own_include_uav_id_norm`

### 1.2 Clean replay 基础设施

已补齐 SAT stage 的状态导出与恢复：
- [sagin_marl/env/structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:346)

新增能力包括：
- `export_sat_stage_state()`
- `load_sat_stage_state()`
- sat-stage world/cache 导出与恢复
- runtime state + RNG 一致恢复

这使得 teacher 比较可以从同一个 `sat-stage snapshot` 出发，避免 live env 分叉带来的随机性污染。

### 1.3 Structured SAT actor 口径

当前 SAT actor 仍然沿用 structured 子集动作语义，没有重写主体：
- [sagin_marl/rl/structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:242)

已补的仅是 teacher 所需辅助：
- `topk_legal_subset_indices(...)`
- `uav_id_norm` 观测支持

### 1.4 Critic-free SAT clean

SAT clean 这条更新路径已经做成真正的 critic-free：
- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:373)
- [scripts/train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1093)

当前行为：
- SAT clean 配置满足条件时，不构建 critic
- `critic_optimizer=None`
- update 时直接走 `_update_sat_clean_joint(...)`
- 不跑 critic update / value refresh

### 1.5 Context 抽样

当前实现只从 **当前 rollout** 抽 context，不复用旧标签：
- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3482)

抽样规则已经按方案落地：
- 基于当前 sat policy 的 entropy
- 用 `log(valid_subset_count)` 做归一化
- 每个 env 取 `top-k entropy + uniform`
- 最终裁到 `sat_clean_contexts_per_update`

### 1.6 Joint teacher 搜索与并行

当前 teacher 搜索已经支持：
- 当前 sat actor 的 per-UAV top-M legal subset 候选
- 联合笛卡尔积 panel
- `current executed action` 作为 baseline 单独评分
- 只保留正 gap 样本

并行执行已落地在 subproc worker 环境回放：
- [sagin_marl/env/structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:469)
- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3396)

当前分工是：
- 主进程 GPU：sat actor 前向、top-M panel 生成、loss/update
- subproc worker：从 snapshot 回放 candidate joint sat action，并用固定 `queue_aware_bw` 打分

### 1.7 Loss 形式

当前 SAT clean update 使用离散 subset 分类损失：
- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3679)

实现形式是：
- `teacher_subset_index` 作为 hard label
- `loss = -log pi(a_teacher | obs)`

这是当前 structured sat actor 的自然监督形式，因为动作空间本身就是合法 subset 的 categorical。

## 2. 当前版本的有意简化

这些点和方案一致的大方向不冲突，但当前版本先取了更保守的实现：

### 2.1 Teacher 打分先做 1-step

当前 worker 打分是：
- 固定 snapshot
- 执行 candidate sat action
- 再执行固定 `queue_aware_bw`
- 读当前 step 的 `bw_weighted_workload_level_reward`

还没有做多步 horizon rerank。

### 2.2 候选 panel 不加额外 heuristic / random 动作

当前 panel 只来自：
- 当前 sat actor 的 top-M legal subset

没有再额外并入：
- `queue_aware_sat`
- random legal subset

这和最后收敛下来的方案是一致的。

### 2.3 teacher 标签不跨 update 复用

当前版本没有 positive-gap bank。
只使用 fresh rollout context，避免 actor 漂移后旧标签过时。

## 3. 和方案的一致性结论

当前实现与方案的核心点是一致的：
- 正式环境上做，不走 `1 UAV / 8 GU`
- 北京热点和按规模资源口径已补进配置
- SAT clean 是 fixed-partner、critic-free、snapshot-replay 的独立路径
- teacher 搜索是 multi-UAV joint panel，不是 unilateral sat counterfactual
- teacher 回放走 subproc 并行，主进程保留 GPU actor 计算
- SAT actor 没有按老 shared trunk 思路误改

## 4. 已做检查

已通过的直接检查：
- [tests/test_structured_sat_clean.py](/d:/研三上/毕设/sagin_marl/tests/test_structured_sat_clean.py:1)

覆盖点包括：
- sat-stage replay 一致性
- subproc worker 回放一致性
- sat_stage_state 写入 rollout buffer
- critic-free sat clean 只更新 sat actor，不更新 accel/bw actor
