# Phase 1 训练过程与当前情况总结（更新至 2026-02-05）

**目标与问题**
Phase 1 目标：累计队列总积压尽可能小。当前主要问题是：full-action 的 RL 仍未超过 queue_aware 基线，reward 与队列目标存在偏离，候选机制对“是否学会靠近热点”影响显著。

**环境与动作空间概览**
环境：`SaginParallelEnv`（PettingZoo Parallel API），实体包含 GU/UAV/SAT。  
动作（每 UAV）：`accel`（2D 加速度）、`bw_logits`（带宽分配，`enable_bw_action=true` 生效）、`sat_logits`（卫星选择，`fixed_satellite_strategy=false` 生效）。  
观测包含：自身状态、用户特征、卫星特征、邻居 UAV 等。候选 GU 由 `candidate_mode` 与 `candidate_k` 决定（见下文）。

**奖励函数（当前采用）**
位置：`sagin_marl/env/sagin_env.py::_compute_reward`  
当前不使用 tanh 压缩，reward 直接由 raw_reward 与终止惩罚组成。  
基础项：  
`base_reward = eta_service * service_norm - eta_drop * drop_norm - omega_q * queue_term + eta_assoc * assoc_ratio`  
形状项：  
`shape_reward = eta_q_delta * queue_delta - eta_accel * accel_norm2 + eta_centroid * centroid_reward + eta_bw_align * bw_align + eta_sat_score * sat_score + eta_dist_delta * dist_delta`  
能量与终止：  
`reward = base_reward + shape_reward + omega_e * r_energy + fail_penalty`

关键定义：  
1. `queue_term`：对 GU/UAV/SAT 队列归一化后做 log 平滑（`log1p` 或 `log1p(k*q)/log1p(k)`）。  
2. `queue_delta`：默认使用总队列变化；可选 `queue_delta_use_active=true` 时使用 GU+UAV 总队列。  
3. `centroid_reward`：UAV 到“高队列 GU 重心”的距离指数奖励 `exp(-d/scale)`。  
4. `dist_delta`：`(prev_centroid_dist_mean - cur_centroid_dist_mean)/scale`，鼓励靠近重心（已修复并生效）。

**候选 GU 机制（必须澄清）**
1. `candidate_mode=assoc`（原默认）：GU 先选择最优 UAV（基于 pathloss），UAV 只看到“当前关联到自己的 GU”。`assoc_ratio_mean≈1` 只是“全局有接入”，不代表“每个 UAV 看到全部 GU”。  
2. `candidate_mode=nearest|radius`：候选集合按距离筛选，视野更广。**带宽分配仍只对 `assoc==u` 的 GU 生效**，不会出现多个 UAV 同时给同一 GU 分配带宽。  
3. `candidate_k`：仅限制候选数量，不改变 `users_obs_max`，网络输入维度不变，多余位置由 mask 置零。

**训练相关机制与改动（当前有效）**
1. MAPPO 训练：`sagin_marl/rl/mappo.py`  
2. Reward Normalization：`reward_norm_enabled`，`reward_norm_clip`  
3. 输入归一化：`input_norm_enabled`  
4. 队列热启动：`queue_init_frac`  
5. UAV 初始位置 curriculum：`uav_spawn_curriculum_enabled`  
6. Imitation loss：使用 queue_aware 作为教师  
7. 评估输出增强：`reward_raw`、`service_norm`、`drop_norm`、`centroid_dist_mean`

**当前推荐方案（按目标）**
1. 目标优先：队列尽可能小  
混合策略评估：accel 用 stage1 训练模型，bw/sat 用 queue_aware。  
结果（fixedload + nearest K=20，20 episodes 均值）：  
`gu_queue_mean ≈ 10,977`，`assoc_dist_mean ≈ 949.8`，`centroid_dist_mean ≈ 461.8`。
2. 目标均衡：靠近热点 + 队列也改善  
固定策略评估：accel 用 stage1 训练模型，bw/sat 固定策略（无 bw 动作、固定卫星）。  
结果（fixedload + nearest K=20，20 episodes 均值）：  
`gu_queue_mean ≈ 16,313`，`assoc_dist_mean ≈ 859.4`，`centroid_dist_mean ≈ 514.6`。

**最终三种策略对比（fixedload + nearest K=20，20 episodes）**
| 策略 | assoc_dist_mean | centroid_dist_mean | gu_queue_mean | service_norm |
| --- | --- | --- | --- | --- |
| 固定策略‑trained（accel=RL, bw/sat 固定） | 859.4 | 514.6 | 16,313 | 1.377 |
| 固定策略‑baseline（queue_aware） | 870.1 | 487.2 | 23,437 | 1.071 |
| 混合策略（accel=RL, bw/sat=queue_aware） | 949.8 | 461.8 | 10,977 | 1.068 |

**关键实验结果与问题定位（摘要）**
1. fixedload + nearest K=20 的 full-action RL 仍低于 queue_aware 基线（队列更高）。  
2. stage1（accel-only）明显优于固定策略 baseline（队列更低、热点更近）。  
3. stage2（full-action + 继承权重）未超过 queue_aware 基线（队列仍高）。  
4. candidate_k 过小（k15/k10）会导致队列显著恶化。  
5. reward 贡献长期由 `service_norm` 主导，`queue` 惩罚占比偏低，导致“reward 高但队列不优”。

**实验演进（按时间顺序整理，仅记录）**
1. v2/v3：引入 ramp、队列热启动、增强队列惩罚、centroid shaping。  
2. v3.1：加入 top-k 队列惩罚（后续证明无明显收益）。  
3. v3.2：加入 UAV 初始位置 curriculum（靠近 GU），去除 top-k。  
4. v3.3：加入 dist_delta 奖励（鼓励靠近重心）。  
5. v3.4/3.5：尝试更强聚集或仅模仿加速度，效果不稳定。  
6. Tune A/B：基于 v3.3 做小幅调参（imitation 系数、聚集权重）。  
7. 修复 dist_delta 未生效：`eta_dist_delta` 正式进入 `shape_reward`。  
8. 关闭 tanh 压缩（`reward_tanh_enabled=false`），避免奖励被压缩。  
9. fixedload 对照：证明训练卡住不是 ramp 造成的假象。  
10. local top-k：距离更近但队列更差，整体不优。  
11. candidate nearest + candidate_k 网格：k20 最稳，k15/k10 明显恶化。  
12. stage 训练：stage1(accel-only) 明显优于固定策略 baseline；stage2(full-action) 仍未超过 queue_aware。  
13. 混合策略评估：accel=RL + bw/sat=queue_aware，队列最小但距离略差。

**当前主要问题总结**
1. full-action RL 仍未超过 queue_aware 基线。  
2. reward 结构偏吞吐，队列惩罚相对不足。  
3. 候选机制过窄会显著恶化训练表现。

**下一步建议方向（简要）**
1. 若优先可交付结果，采用混合策略或固定策略方案。  
2. 若继续 full-action，建议冻结 bw/sat 或改为更温和的队列对齐方案。  
3. 保持 `candidate_mode=nearest` 且 `candidate_k` 不低于 20。
