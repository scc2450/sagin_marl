# PPT材料整理（2026-04-09）

说明：

- 除新增的 2 次 `Lyapunov` 基线评估外，其余内容均来自现有结果文件或现有分析文档。
- 新增结果文件：
  - `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/eval_lyapunov_seed42000_n20.csv`
  - `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/eval_lyapunov_retuned_seed42000_n20.csv`
  - `runs/structured_step250_gamma0995_u200_env8_subproc_gpu_20260403/eval_lyapunov_seed42000_n25_summary.csv`
  - `runs/structured_step250_gamma0995_u200_env8_subproc_gpu_20260403/eval_lyapunov_retuned_seed42000_n25_summary.csv`

## 1. 启用 GU 干扰与衰落：补齐 Lyapunov 基线

对应配置：`configs/phase1_actions_curriculum_joint_3heads_fading_interference.yaml`

表 1 结果汇总（同口径，可直接横比）

| 方法 | reward_sum | processed_ratio | drop_ratio | pre_backlog | D_sys |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练策略 | 397.95 | 1.0592 | 0.0000 | 0.8117 | 0.0003 |
| queue-aware | 105.61 | 1.1394 | 0.0047 | 17.8070 | 0.2132 |
| cluster-center-queue-aware | 290.22 | 0.9778 | 0.0269 | 17.3292 | 0.0570 |
| Lyapunov（默认尺度） | 75.86 | 1.1686 | 0.0337 | 12.2833 | 0.1480 |
| Lyapunov（retuned尺度） | 342.89 | 1.0372 | 0.0041 | 8.2964 | 0.0182 |

可直接讲的结论：

- 训练策略仍显著最优，`drop_ratio=0`，`D_sys` 也最低。
- `Lyapunov` 如果直接用默认尺度，吞吐推得很高，但掉包和系统时延明显偏差。
- 一旦换成 retuned 尺度，`Lyapunov` 会大幅改善：`reward_sum 75.86 -> 342.89`，`drop_ratio 0.0337 -> 0.0041`，`D_sys 0.1480 -> 0.0182`。
- 这说明第一组场景的问题主要是 **尺度参数没调对**，不是 Lyapunov 启发式本身无效。

数据来源：

- `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/eval_trained_best_seed42000_n20.csv`
- `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/eval_queue_aware_seed42000_n20.csv`
- `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/eval_cluster_center_queue_aware_seed42000_n20.csv`
- `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/eval_lyapunov_seed42000_n20.csv`
- `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/eval_lyapunov_retuned_seed42000_n20.csv`

### 1.1 为什么 `phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml` 下 Lyapunov 会更好

这里要先区分一件事：

- `configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml`
- `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/config_source.yaml`

这两者的环境主体其实基本一致；`lyapunov` 专用配置新增的核心差异，几乎全是 **Lyapunov 基线自己的超参**。

也就是说：

- 如果 Lyapunov 在这个专用配置下表现明显更好，
- 更直接的原因不是“任务换了”，
- 而是 **这组 Lyapunov 系数被专门调到了这个 fading + interference 场景上**。

表 1-补充 关键差异（`lyapunov` 专用配置 vs 当前默认值）

| 参数 | 默认值 | lyapunov专用配置 | 可能影响 |
| --- | ---: | ---: | --- |
| `baseline_assoc_bonus` | 0.3 | 1.0 | 更偏向保持已有关联，减少抖动 |
| `baseline_repulse_gain` | 1.0 | 0.6 | 安全斥力更柔和，避免过度把 UAV 从高价值用户簇推走 |
| `baseline_repulse_radius_factor` | 1.5 | 25.0 | 更早开始做稀疏化避让，减少近距离拥挤和干扰 |
| `baseline_lyapunov_urgency_alpha` | 1.0 | 30.0 | 用户责任划分更尖锐，UAV 更容易形成清晰分工 |
| `baseline_lyapunov_drift_weight` | 1.2 | 2.0 | 更强调虚拟队列，遇到拥塞时动作更激进 |
| `baseline_lyapunov_ema_beta` | 0.6 | 0.5 | 更快跟随瞬时压力变化 |
| `baseline_lyapunov_bw_service_scale` | 1.0 | 1.0e7 | 显著改变 service estimate 尺度，直接改变虚拟队列更新行为 |

从实现上看，最关键的是两条：

1. `urgency_alpha`
   `Lyapunov` 里有一项责任因子：
   `exp(alpha * (min_nbr_dist - dist_gu))`
   当 `alpha` 从 `1` 提到 `30` 后，UAV 对“这个用户更该我来服务还是邻机来服务”的划分会明显变尖。
   在有干扰的场景里，这会直接减少多机抢同一簇用户带来的重叠和无效竞争。

2. `bw_service_scale`
   `Lyapunov` 里 BW 的 service estimate 是：
   `service_est = service_scale * relay_gate * bw_alloc * eta_slot`
   `service_scale` 从 `1` 到 `1e7` 不是小修，而是把虚拟队列系统切到了完全不同的工作点。
   结合这套场景下本来就是千万量级的 `queue_ref_*_per_step`，这更像是在把 heuristic 调到“能对实际业务量级作出反应”的区间。

因此，对第一组 fading + interference 场景，更准确的判断应当是：

- `Lyapunov` 不是天然不好；
- 之前差，主要是 **默认系数没有调到这个场景上**；
- `phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml` 本质上是一版 **为该场景单独 retune 过的 Lyapunov 配置**。

### 1.2 为什么换到别的配置就会出问题

#### 情况 A：同样是 fading + interference，但不用专用 Lyapunov 配置

问题核心：

- 环境没变太多，但 Lyapunov 的关键系数退回默认值了。
- 这样一来，用户责任划分变钝、虚拟队列尺度失配、加速度避让和关联保持也不再匹配当前场景。

一句话：

- 这是 **同场景下的参数失配问题**，不是 Lyapunov 思路本身失效。

#### 情况 B：切到 Ka/VSAT + 多普勒 + 大气衰减 + 雨衰 + structured step250

这时问题就不只是调参了，而是出现了明显的 **分布迁移**：

| 变化项 | 影响 |
| --- | --- |
| `doppler_enabled / doppler_atten_enabled = true` | 回传链路质量随时间波动更强 |
| `atm_loss_enabled / rain_loss_enabled = true` | 卫星侧链路出现额外衰减，且更偏非平稳 |
| `carrier_freq = 30 GHz` | Ka 频段对天气与多普勒更敏感 |
| structured 三阶段控制 | 好动作不再只是“当前一步谁更急”，而是 `accel -> sat -> bw` 的前缀一致性 |

这会带来两层问题：

1. **手工系数失效**
   旧 Lyapunov 配置是按“只有 fading + interference”的 regime 调出来的。
   一旦回传链路开始受多普勒、雨衰和大气衰减共同影响，原先那组 `urgency / service / sat score` 系数就不再稳定。

2. **一步式 heuristic 变得更短视**
   现在真正有效的动作，需要前面 `accel` 先把几何位置摆对、`sat` 再选对回传链路、最后 `bw` 才能真正兑现。
   而 Lyapunov 仍然是一个一步式、当前观测驱动的 DPP controller，本质上没有显式建模这种多步一致性。

因此，对第二组 Ka/VSAT structured 配置，更准确的判断是：

- 一部分问题来自 **参数没有重新 retune**；
- 另一部分更本质的问题来自 **场景本身已经从“可用一步式启发式处理”变成了“需要多阶段一致控制”**。

数据来源：

- `configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml`
- `runs/phase1_actions/joint_3heads_fading_interference_u1200_20260327/config_source.yaml`
- `configs/phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995.yaml`
- `sagin_marl/env/config.py`
- `sagin_marl/rl/baselines.py`

## 2. 启用多普勒、大气衰减、雨衰：step250 三头联合结果 + Lyapunov 基线

对应主配置：`configs/phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995.yaml`

### 2.1 step250 训练过程（checkpoint_eval 口径）

| update | reward_sum | processed_ratio | drop_ratio | pre_backlog | D_sys |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 101.33 | 0.8542 | 0.1516 | 20.7929 | 0.1093 |
| 100 | 141.32 | 0.9292 | 0.1179 | 13.0062 | 0.0585 |
| 150 | 169.59 | 0.9891 | 0.0678 | 11.5050 | 0.0389 |
| 200 | 198.43 | 1.0465 | 0.0141 | 11.6693 | 0.0313 |

一句话：

- step250 三头联合训练是有效的，指标随训练持续改善，到 `u200` 时吞吐和掉包都明显优于早期 checkpoint。

### 2.2 补齐 Lyapunov 基线（与现有 structured_eval 结果对齐）

这次补的 `Lyapunov` 已显式纳入：

- 卫星链路频谱效率变化
- 归一化多普勒惩罚
- 卫星队列 / 负载

也就是会对多普勒、大气衰减、雨衰带来的回传链路退化做响应。

表 2 结果汇总（`structured_eval/checkpoint_eval` 口径；同表数值可横向参考，但 episode 数分别为 `5/5/25`）

| 方法 | reward_sum | processed_ratio | drop_ratio | pre_backlog | D_sys |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三头联合 policy（u200） | 198.43 | 1.0465 | 0.0141 | 11.6693 | 0.0313 |
| 固定 heuristic | 155.13 | 0.9652 | 0.1015 | 11.5814 | 0.0290 |
| Lyapunov（默认尺度） | 46.75 | 0.8607 | 0.1643 | 22.1801 | 0.2038 |
| Lyapunov（retuned尺度） | 181.57 | 1.0134 | 0.0220 | 13.3805 | 0.0588 |

可直接讲的结论：

- 默认尺度下，`Lyapunov` 在引入多普勒/大气/雨衰后表现很差。
- 换成 retuned 尺度后，`Lyapunov` 明显恢复：`reward_sum 46.75 -> 181.57`，`drop_ratio 0.1643 -> 0.0220`，`D_sys 0.2038 -> 0.0588`。
- 这说明第二组场景里也存在显著的 **尺度失配问题**；但即便如此，`pre_backlog` 和 `D_sys` 仍没回到第一组那样稳，说明这里除了调参，还叠加了更强的物理扰动和多阶段一致性问题。

数据来源：

- `runs/structured_step250_gamma0995_u200_env8_subproc_gpu_20260403/checkpoint_eval.csv`
- `runs/structured_step250_gamma0995_u200_env8_subproc_gpu_20260403/eval_lyapunov_seed42000_n25_summary.csv`
- `runs/structured_step250_gamma0995_u200_env8_subproc_gpu_20260403/eval_lyapunov_retuned_seed42000_n25_summary.csv`

### 2.3 为什么单独看 sat / bw 训练都不好

表 3 单头替换分析（`hybrid_eval_u0200` 口径；只用于分析单头影响，不与表 2 混口径）

| 组合 | reward_sum | processed_ratio | drop_ratio | pre_backlog | D_sys |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全 heuristic | 173.61 | 0.9935 | 0.0753 | 8.1079 | 9.8822 |
| 仅 accel 用 policy | 242.46 | 1.0902 | 0.0000 | 1.5165 | 2.3845 |
| 仅 sat 用 policy | 160.57 | 0.9915 | 0.0753 | 17.3727 | 63.1483 |
| 仅 bw 用 policy | 159.68 | 0.9682 | 0.0947 | 8.8078 | 10.7152 |
| 三头都用 policy | 189.60 | 1.0209 | 0.0303 | 14.4299 | 46.2634 |

可直接讲的结论：

- 新环境下，真正起主要正作用的是 `accel` 头；“只换 accel”为最强组合。
- `sat` 单独换成 learned policy 影响最差，`pre_backlog` 从 `8.11` 升到 `17.37`，`D_sys` 从 `9.88` 放大到 `63.15`。
- `bw` 单独换成 learned policy 也比 heuristic 差，但恶化幅度小于 sat。
- 三头联合最终能把 `drop_ratio` 压下来，但它不是靠 “sat 单头就学好了”，而更像是耦合后由 `accel` 主导、其余两头被一起带动。

### 2.4 原因分析：问题不只是“sat/bw 头自己弱”

证据 1：partner regime 很敏感

| 对比 | reward 变化 | processed 变化 | drop 变化 | backlog 变化 |
| --- | ---: | ---: | ---: | ---: |
| `bwonly`: cluster partner - joint partner | +125.62 | +0.1455 | -0.1083 | -11.7450 |
| `joint bw`: cluster partner - joint partner | +120.16 | +0.1391 | -0.1021 | -11.1646 |

结论：

- 同一个 `bw` 头，换不同 partner regime，结果差异非常大。
- 所以 `bw/sat` 看起来差，不是纯粹的单头能力问题，还包含强烈的上游分布依赖。

证据 2：主导项更像是 accel，而不是 sat

| 对比 | reward 变化 | processed 变化 | drop 变化 | backlog 变化 |
| --- | ---: | ---: | ---: | ---: |
| joint accel 相对 cluster accel（固定 cluster sat） | -115.47 | -0.1320 | +0.0983 | +10.5338 |
| joint sat 相对 cluster sat（固定 cluster accel） | +0.80 | +0.0007 | -0.0007 | -0.2717 |

结论：

- 在这组实验里，regime 的主导变化来自 `accel`，不是 `sat`。
- 也就是说，`sat/bw` 的问题很大程度上是被上游几何与候选分布放大的。

证据 3：sat 训练还会拖累整体

| 训练设置 | reward_sum | processed_ratio | drop_ratio | pre_backlog | sat_overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nobwtrain` | 84.06 | 0.6095 | 0.2186 | 49.1778 | 0.8554 |
| `nobwtrain + nosattrain` | 149.39 | 0.8169 | 0.2009 | 16.1294 | 0.6986 |

结论：

- 只去掉 `bw` 训练还不够；把 `sat` 训练也去掉，恢复幅度更大。
- 说明这里存在明显的 `sat/bw` 训练耦合与 credit 干扰。

数据来源：

- `runs/structured_step250_gamma0995_u200_env8_subproc_gpu_20260403/hybrid_eval_u0200/summary.csv`
- `runs/partner_swap_matrix_u0050_20260401/summary.json`
- `runs/partner_component_split_jointbw_u0050_20260401/summary.json`
- `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_u100_env12_subproc_20260401/checkpoint_eval.csv`
- `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_nosattrain_u100_env12_subproc_20260401/checkpoint_eval.csv`

## 3. 网络结构变化：从原始多头到 structured

### 3.1 一句话版本

- 原始网络：一个共享 actor trunk + 三个并列动作头，critic 仍是“全局状态 + 局部观测摘要 -> 单个团队价值”。
- structured 网络：把一个物理时间步拆成 `accel -> sat -> bw` 三阶段，每个阶段用自己的局部状态、自己的 actor readout，以及结构化 world-state critic。

### 3.2 旧版 vs structured 对照

| 项目 | 原始版本 | structured 版本 |
| --- | --- | --- |
| 控制方式 | 单步同时输出 `accel + bw + sat` | 一个物理步拆成三阶段：`accel -> sat -> bw` |
| actor 输入 | 每个 UAV 的展平局部观测 `obs_dim=259` | 阶段化局部状态 dataclass，不再先展平为一个统一向量 |
| actor 主体 | 共享 trunk，三头并列 | `AccelPolicy` / `SatSubsetPolicy` / `BwPolicy` 三个阶段 actor |
| critic 输入 | `global_state(142)` + 同步 `obs_step` 摘要 | `StructuredWorldState` 图结构：`uav/gu/sat` 节点 + `uav-gu/uav-sat/uav-uav` 边 + mask + `stage_id` |
| critic 输出 | 单个团队价值 `V_team` | `value_accel` / `value_sat` / `value_bw` 三个阶段 value head |
| sat 表达 | 6 个候选卫星槽位直接参与并列动作头 | 先构造可见卫星子集，再构造 subset token，再做 sat subset 选择 |
| bw 表达 | one-shot simplex / logits 头 | 在 `ego + visible sat + candidate users` 条件下单独做 BW 分配 |

### 3.3 actor 输入变化是这次改造的核心

原始 actor 输入：

- 一个统一的 flatten 观测，包含 `own + danger_nbr + users + bw_valid_mask + sats + sat_valid_mask + nbrs`。
- 三个动作头共享同一套上下文表示。

structured actor 输入：

- `accel`：`ego_uav + 邻机节点/边 + GU 节点/边 + 可见 SAT 节点/边`
- `sat`：`ego_uav_after_accel + 可见 SAT 节点/边 + subset tokens`
- `bw`：`ego_uav_after_sat + 可见 SAT 节点/边 + candidate user 节点/边 + bw_valid_mask`

直接含义：

- 旧版是“同一个上下文里并列出三个头”。
- structured 是“把上一步动作结果写回状态，再给下一阶段用”。
- 因此 structured 更符合真实决策顺序，也更能表达前缀动作对后续可选集合的影响。

### 3.4 critic 输入变化更大

旧 critic：

- 仍以 `global_state` 为主，再拼接由 `obs_step` 提炼出的局部摘要；
- 最终仍压成一个团队标量价值。

structured critic：

- 直接吃 `StructuredWorldState`；
- 保留 `UAV/GU/SAT` 节点和 `UAV-GU/UAV-SAT/UAV-UAV` 关系边；
- 先做关系编码，再按阶段读出 `accel/sat/bw` 三个 value。

直接含义：

- critic 不再依赖一个粗粒度的“伪全局向量”去概括所有阶段；
- actor 和 critic 的输入语义更一致，阶段之间的 credit 也更容易拆开。

数据来源：

- `docs/model_architecture.md`
- `sagin_marl/rl/policy.py`
- `sagin_marl/rl/critic.py`
- `sagin_marl/rl/structured_types.py`
- `sagin_marl/rl/structured_stage_builders.py`
- `sagin_marl/rl/structured_actor.py`
- `sagin_marl/rl/structured_critic.py`

## 4. BW 失败尝试汇总：每个尝试在解决什么问题

### 4.1 当前最准确的判断

- `global / broad leverage 不小` 这半句仍成立：one-step `reward` range 的 `p50≈0.313`，`K=2 reward gap` 的 `p50≈0.221`，`82.1%` 的状态满足 `gap>=0.1`。
- 真正卡住的是 `local learnable signal` 太浅而且方向有点脏：`best_reward_gain p50≈0.0194`，`top1_hit≈0.229`，`pairwise_acc≈0.609`；`loc` 与 `drop_gain` 的 Spearman `≈0.747`，明显高于它与 `reward_gain / processed_gain / backlog_gain` 的 `≈0.25~0.30`。
- 所以问题不是“BW 没 leverage”，而是“现有 PPO / update / credit / 参数化，没有把这些 leverage 稳定翻译成 deterministic BW 改善”。

### 4.2 主要失败/未解决尝试

| 尝试 | 想解决的问题 | 代表结果 | 结论 |
| --- | --- | --- | --- |
| 环境/目标清洗：`throughput_only`、`pure x_acc`、`no_backhaul`、broad-gap 审计 | 怀疑环境里 BW leverage 太小，或高价值状态太稀少 | `one-step reward range p50≈0.313`，`K=2 reward gap p50≈0.221`，`82.1%` 状态 `gap>=0.1` | 这条线最后证明了“leverage 其实不小”；诊断是成功的，但它不是最终解 |
| `runtime bank / high-gap bank` | 怀疑 PPO 很少见到高价值状态，想靠 bank 改善状态分布 | `highbank reward=207.93` vs `randombank=202.63`，但 `pre_backlog 10.59 > 10.34` | `bank` 有信息量，但收益很弱，没有变成 decisive 改善；问题不宜再归因成“样本不够好” |
| `bw_flow_proxy_aux` | 针对“local signal 太浅”，给 BW 更密的 proxy 排序信号 | `aux det 148.37 < noaux 156.94`；`aux stoch 146.80 < noaux 156.12` | 这类 local proxy 没救回来，反而更像把训练往偏 proxy 的方向带 |
| `bank -> dataset / preference + pull` | 想把 broad leverage 直接转译成 actor 可学的离线偏好信号 | `pairwise_acc 0.5189 -> 0.6970`，`spearman 0.0175 -> 0.4974`，但 `det_gap_to_best.mean 0.14724 -> 0.17274` 变差 | 排序学到了，但没有翻译成 deterministic 动作改善 |
| BW-specific counterfactual credit v0 | 怀疑 `shared advantage` 污染了 BW credit | 诊断里 `bad_local_reward_but_positive_adv_fraction=0.5391`；训练后 `u20 reward_sum=110.60`，固定参考 `162.73` | credit 污染判断成立，但当前 counterfactual 公式没兑现成训练收益 |
| 轻量 decomposition：`support + epsilon tail + support内Dirichlet` | 怀疑 one-shot simplex 太难，想先分 support 再分配 | 零步替换就回退：`det 100.568 < 108.933`，`stoch 188.844 < 214.557`；`u10 reward_sum=112.14 < 162.73` | 轻量 decomposition 不够，说明问题不是简单 top-k/support 就能解 |
| `AWR / V-MPO-lite` | 怀疑 `PPO clip` 更新族本身不适合 BW | `AWR u20=116.36`，`V-MPO-lite u20=116.48`，都低于固定参考 `162.73`；`V-MPO-lite collision_episode_fraction=0.05` | 只有弱正信号，而且不稳，不值得继续当主线 |

### 4.3 有帮助但没有彻底解决的尝试

| 尝试 | 想解决的问题 | 代表结果 | 结论 |
| --- | --- | --- | --- |
| `alpha-only` 参数化修补 | 怀疑旧参数化把排序、sharpness 和 deterministic readout 绑得太死 | 离线审计里 `det_gap_to_best 0.2871 -> 0.2141` | 说明“参数化确实是病灶之一”，但还不是最后主线 |
| `Dirichlet + per_simplex_dim` | 继续修正 PPO 几何/分布表达，让排序改善更容易落到 deterministic 动作上 | 离线审计 `pairwise_acc 0.6173 -> 0.7140`，`det_gap_to_best 0.2871 -> 0.1703`；matched eval `det 178.80 -> 212.81` | 这是目前最有价值的 all-policy BW 修补线，但仍未稳定打赢 heuristic BW |

### 4.4 最后问题落点（简略版）

- 现在更像是：系统里存在 broad leverage，但 `reward / return -> advantage -> sampled-action update -> deterministic BW readout` 这条转译链条不够稳。
- 结合第 5 节的简化模型结论，问题最终落在：当前训练/搜索一直在用“一步局部改进”，逼近一个本质上需要“多步一致性”的控制问题。
- 所以继续沿“pure on-policy PPO + 小修小补”硬推，性价比已经不高。

数据来源：

- `docs/bw_status_summary_20260407.md`
- `docs/bw_broad2local_v1.md`
- `docs/bw_possibility_followup_20260408.md`
- `runs/bw_broad2local_offline_noaux_u0120_gate1_subproc_fixed/summary.json`
- `runs/structured_local_signal_highbank_u0110_20260406/eval_trained.csv`
- `runs/structured_local_signal_randombank_u0110_20260406/eval_trained.csv`

## 5. 采用简化模型后，BW 问题最终落在什么地方

### 5.1 不是 actor 表达能力不够

模仿 sanity：

- learned deterministic reward `34.4164`
- fixed `queue_aware_bw` reward `34.5134`

结论：

- 当前 `BW actor` 基本能表示一个简单有效的老师策略；
- 所以主问题不再是 actor capacity。

### 5.2 高 gap benchmark 下，当前训练明确失败

| 方法 | reward_sum | processed | drop | backlog |
| --- | ---: | ---: | ---: | ---: |
| fixed `queue_aware_bw` | 25.9332 | 0.9109 | 0.0430 | 8.0802 |
| learned `u10` | 1.5819 | 0.7504 | 0.1720 | 10.7007 |

补充证据：固定策略 gap search 在强 benchmark 上有稳定大 gap：

- seed `52000`: `33.622`
- seed `53000`: `27.177`
- seed `54000`: `34.076`

结论：

- 现在不是“任务太弱、所以看不出来”；
- 而是任务已经有很强杠杆，但当前训练仍然学不出来。

### 5.3 最核心的定位：当前训练/搜索在用一步改进去逼近一个需要多步一致性的控制问题

`k-step splice` 结果（相对 `k=0` 的平均收益增量）：

| k | gain_over_k0 |
| --- | ---: |
| 1 | 0.1379 |
| 2 | 0.3364 |
| 5 | 0.6173 |
| 10 | 0.5725 |
| 20 | 1.6414 |
| 100 | 18.9638 |

局部 transport 上界：

- `transport_gap_mean = 0.4223`
- `good_gap_mean = 18.9638`

可直接讲的结论：

- 收益随 `k` 不是线性增长；`k=1/2/5` 只拿到极小一部分收益，真正的 gap 要到长前缀一致控制才会出现。
- “把质量局部挪一点”最多只能拿到 `0.42` 左右收益，但真正好策略和坏策略之间的 gap 接近 `18.96`。
- 因此当前问题更准确的说法不是“一步 reward 不够”，而是：
  - 当前训练/搜索一直在用一步或很短前缀的改进问题，
  - 去逼近一个本质上需要多步一致性的 `BW` 控制问题。

一句话收束：

- `BW` 的最终问题已经基本落在“当前 RL update 与短视搜索范式不适配”上，而不是单纯的 reward、小改动参数化或 bank 问题。

### 5.4 `T` sweep：分界点大约就在 `T=5`

这轮 `BW-only + full PPO + env_reward + monte_carlo` 的 `T=1/2/5/10` 可以直接总结成一句话：

- `T=1,2` 能学，`T=5` 开始掉队，`T=10` 明显不行。

同口径 `u0030` checkpoint eval：

| T | learned reward | heuristic reward | reward差值 | processed 对比 | drop 对比 | backlog 对比 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 0.682 | 0.576 | +0.106 | `1.632 > 1.427` | `0 = 0` | `4.368 < 4.573` |
| 2 | 1.133 | 1.048 | +0.085 | `1.401 > 1.319` | `0 = 0` | `4.372 < 4.467` |
| 5 | 2.212 | 2.249 | -0.036 | `1.152 < 1.163` | `0 = 0` | `4.390 > 4.314` |
| 10 | 3.344 | 3.939 | -0.595 | `0.949 < 1.046` | `0.0029 > 0` | `4.752 > 4.333` |

可直接讲的结论：

- 分界已经很清楚：大约从 `T=5` 开始，当前 full PPO 就从“能学”变成“开始输给固定规则”。
- 这说明问题不是“一步 BW 学不会”，而是随着多步闭环性增强，当前接口开始失效。

最后一轮训练指标也支持这个判断：

| T | env_reward_mean | approx_kl_bw | clip_frac_bw | bw_kappa_mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6388 | 0.0176 | 0.317 | 49.56 |
| 2 | 0.5441 | 0.0670 | 0.365 | 51.00 |
| 5 | 0.4546 | 0.0474 | 0.360 | 40.52 |
| 10 | 0.3252 | 0.00387 | 0.015 | 19.45 |

这里最值得注意的是：

- `T=10` 时 PPO 自己的更新强度已经明显塌了；`approx_kl_bw` 和 `clip_frac_bw` 都接近 0。
- 这更像是“更新已经很弱，但也没把策略带到好地方”，而不只是“它还在积极往坏方向更新”。

buffer / 训练接口的含义：

- 这四条 run 的 `buffer_size` 都是 `100`。
- 训练脚本默认 `rollout_env_steps = buffer_size`。
- 每个 env step 固定写 3 个 stage transition：`[accel, sat, bw]`。
- 所以每次 update 都是同样的 `100` 个 env steps、同样的 `300` 条 stage transitions。
- 真正变化的是：`T_steps` 不同以后，`done/reset` 频率不同，return 传播深度不同，同样是 100-step rollout，里面包含的 episode 数也不同。
- 也就是说：不是 buffer 容量变了，而是“同样的 buffer 深度，承载的时域难度变了”。

一句话收束：

- 这组 `T sweep` 进一步把问题定位清楚了：简化问题的难点不是“一步 BW 怎么学”，而是“当收益开始依赖连续多步都做对时，当前 full PPO 接口就开始失效”。

数据来源：

- `docs/bw_sanity_1uav_static_followup_20260408.md`
- `docs/bw_sanity_1uav_static_rl_update_followup_20260409.md`
- `runs/structured_bw_sanity_1uav_static_gap_debug_u10_20260408/checkpoint_eval.csv`
- `runs/structured_bw_kstep_splice_bad_to_good/summary.json`
- `runs/structured_bw_local_transport_bad_vs_good/summary.json`
- `runs/structured_bw_gap_onestep_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_t2_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_t5_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_t10_envreward_ppo_u30/checkpoint_eval.csv`
- `runs/structured_bw_gap_onestep_envreward_ppo_u30/metrics.csv`
- `runs/structured_bw_gap_t2_envreward_ppo_u30/metrics.csv`
- `runs/structured_bw_gap_t5_envreward_ppo_u30/metrics.csv`
- `runs/structured_bw_gap_t10_envreward_ppo_u30/metrics.csv`
