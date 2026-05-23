# `1 UAV + 2 GU` 轮流高负载/高紧急窗口问题排查记录

本文档整理本次对话中围绕以下目标所做的分析、代码修改、实验、诊断与当前结论：

> `1 UAV + 2 GU`，但让两个 `GU` 在 episode 里轮流进入高负载/高 urgency 窗口；每个窗口持续几步；希望动作收益明显，但策略仍然必须按状态切换，而不是学死规则。

整理时间：`2026-04-12`

---

## 1. 目标与核心问题

这次排查的核心，不只是“PPO 能不能训起来”，而是更具体的三层问题：

1. 现有 `2 GU` 强窗口环境里，PPO 是否真的学会按状态切换带宽分配。
2. 如果学得不好，问题更像是：
   - 环境本身不够尖锐。
   - 奖励/target 不够敏感。
   - critic / return / advantage 机制有问题。
   - PPO 每次更新的方向本身就不对。
3. 在“给该给的 GU 更多、给不该给的 GU 更少”这件事上，PPO 的每次更新到底是不是朝正确方向前进。

---

## 2. 第一阶段：直接短训现有 `2 GU` 强窗口版本

### 2.1 当时使用的原始配置

最初分析用的是已有的 `2 GU stronggap` 配置：

- `configs/structured_bw_sanity_1uav_2gu_t10_stronggap_ppo.yaml`

对应短训 run：

- `runs/structured_short/two_gu_stronggap_u60`

### 2.2 先得到的结论

最初的结论是：

- PPO 在这个版本上“能学出来”，但优势不大。
- 它不是纯粹学死规则，策略确实会随观测到的窗口状态切换。

### 2.3 原始短训结果

`u60` checkpoint eval（32 episodes）：

- fixed `queue_aware_bw`：
  - `reward 7.055`
  - `processed 1.467`
  - `pre_backlog 0.636`
- PPO：
  - `reward 7.242`
  - `processed 1.492`
  - `pre_backlog 0.534`

后续用 `256 episodes` 重评最终模型：

- PPO：
  - `reward 7.067`
  - `processed 1.4697`
  - `pre_backlog 0.7396`
- fixed heuristic：
  - `reward 6.9927`
  - `processed 1.4623`
  - `pre_backlog 0.7878`

### 2.4 当时观察到的“并非学死规则”的证据

对最终 actor 做了状态条件下的动作统计：

- `hot0` 时平均分配大约：
  - `a0=0.617`
  - `a1=0.383`
- `hot1` 时平均分配大约：
  - `a0=0.389`
  - `a1=0.611`
- 无 hotspot 时接近均分：
  - `a0=0.496`
  - `a1=0.504`
- `corr(action_gap, observed_rate_gap)=0.828`

这说明旧版本下 PPO 确实在跟随观测到的流量差切换，而不是固定偏某个 GU。

---

## 3. 第二阶段：发现原环境并不真正符合“严格轮流窗口”目标

### 3.1 原始窗口机制的问题

在原环境里，窗口并不是“严格交替、固定持续几步”，而是随机粘滞式 hotspot：

- `hotspot_num_subsets=2`
- `subset_size=1`
- `hotspot_rho=20`
- `hotspot_on_mean_steps=20`
- `hotspot_off_mean_steps=1`
- 默认 `arrival_mean_preserve=true`

这意味着：

- hotspot 是随机 on/off，不是严格 `GU0 -> GU1 -> GU0 -> ...`
- `T=10` 时常常只有一个长窗口
- `T=20` 时也经常只是一个长窗口，而不是多次明显轮换

我当时抽样了 `200` 个 episode，发现有 `123/200` 个 episode 根本没有发生切窗。

### 3.2 一拍时滞

原 traffic proxy 的更新顺序是：

1. 先按当前 hotspot 生成这一步的 arrival。
2. 再更新队列。
3. 最后才切到下一步 hotspot。

所以：

- 当前 `gu_queue` 是当前步后的真实队列。
- 但 `arrival_rate / recent_arrival` 这些 traffic proxy 对“下一次决策”来说是慢一拍的。

这并不是 bug，但如果任务想建模“立即看到窗口变化并切带宽”，它会让任务更绕。

### 3.3 观测里对窗口有较强泄露

当时配置中直接打开了这类用户特征：

- `arrival_rate`
- `recent_arrival`
- `recent_service`
- `queue_headroom`

因此 old version 更像“hot 窗口的后果半显式可见”，而不是纯隐藏状态切换任务。

---

## 4. 第三阶段：澄清状态特征与环境语义

这一步没有直接改代码，主要是为了弄清环境到底在给 actor/critic 什么信息。

### 4.1 显式 urgency 特征

在环境里进一步梳理了这些特征的语义：

- `obs_user_include_urgency_risk`
- `service-gap`
- `service-gap-risk`
- `deadline`
- `deadline_risk`

它们都是“已经把快溢出、拖太久、快到 deadline”编码好的特征，属于显式 urgency。

### 4.2 更原始的 proxy

也梳理了这几类更原始的 proxy：

- `arrival_rate`
- `recent_arrival`
- `recent_service`
- `queue_headroom`

它们更像原始症状，而不是人工整理好的“风险分数”。

### 4.3 发现 `task_arrival_poisson=false`

当前这批配置中：

- `task_arrival_poisson: false`

因此：

- `arrival_rate` 基本就是计划到达率
- `recent_arrival` 与其非常接近
- 没有额外泊松噪声

---

## 5. 第四阶段：解释为什么 `uniform/init` 没有看起来那么差

用户对这个问题抓得很准：如果 hot/cold 长期差异存在，为什么 `uniform-ish` 的初始化策略看起来没有很差，尤其是 `T=20` 似乎没有把差异放大很多。

### 5.1 直接比较 `uniform` 和 `queue_aware`

为了避免只看 PPO，我直接拿 `uniform` 和 fixed `queue_aware_bw` 跑了 episode，比真实效果。

#### `T=10`，256 episodes

- `uniform`
  - `reward 6.633`
  - `step throughput 1.4078`
  - `step backlog 1.1275`
- `queue_aware`
  - `reward 6.862`
  - `step throughput 1.4468`
  - `step backlog 0.9531`

#### `T=20`，256 episodes

- `uniform`
  - `reward 11.188`
  - `step throughput 1.1895`
  - `step backlog 1.0979`
- `queue_aware`
  - `reward 11.699`
  - `step throughput 1.2271`
  - `step backlog 0.6996`

这说明：

- `uniform` 确实更差。
- `T=20` 下 gap 实际上是更大的，这和用户直觉一致。

### 5.2 热窗口上具体发生了什么

以 `T=20` 为例，只看 hot step：

#### `uniform`

- hot 分配占比：`0.500`
- hot 到达：`1.5238e6`
- hot 实际服务：`1.5078e6`
- cold 到达：`7.619e4`
- cold 实际服务：`2.027e5`
- hot step 后 cold 为空比例：`93.7%`

#### `queue_aware`

- hot 分配占比：`0.559`
- hot 实际服务：`1.5843e6`
- cold 实际服务：`1.980e5`
- hot step 后 cold 为空比例：`66.1%`

所以：

- `cold 分太多` 是非常明显的。
- `hot 分不够` 也存在，但旧工作点里平均上仍然接近够用，所以不是每一步都爆。

### 5.3 hard seed 例子

在 `T=20, seed=5047` 上，差异很大：

#### `uniform`

- `step 5` 时 cold 就被清空
- 从 `step 6` 开始仍然坚持接近 `50/50`
- cold 每步只需 `76190`
- hot 每步到达 `1523810`
- hot 每步服务只有 `964132`
- hot 队列从 `step 10` 的 `9.43e6` 涨到 `step 19` 的 `1.447e7`
- episode reward：`4.897`

#### `queue_aware`

- 分配逐步接近 `0.05 / 0.95`
- hot 每步服务到 `1.82e6` 左右，已经高于 hot arrival
- hot 队列开始下降
- episode 末尾 hot 队列约 `1.50e6`
- episode reward：`9.915`

### 5.4 easy/hard episode 分布太散

进一步看 `T=20` 下 `queue_aware - uniform` 的每 episode reward gap：

- 均值：`0.512`
- 中位数：`0.0`
- `p90 = 3.76`
- `p95 = 4.67`
- `gap > 1` 的 episode 约 `19.9%`

这说明环境里大量 episode 仍然是“50/50 也差不多能过”，只有少数 hard episode 会真正把 gap 拉开。

---

## 6. 第五阶段：修改 `weighted_workload` 为逐设备版本

用户不满意“只按层级总和计算 workload”的语义，于是改成：

- 保留端到端层叠语义
- 但把 `EMA/cost` 从“每层一个标量”改成“每设备一个，再聚合”

### 6.1 修改内容

相关文件：

- `sagin_marl/env/structured_driver.py`
- `sagin_marl/env/sagin_env.py`
- `tests/test_structured_driver_buffer.py`

这次改动的核心是：

- `GU/UAV/SAT` 都改成逐设备 EMA
- `cost` 仍保留端到端嵌套语义
- `weighted_workload_delta`
- `weighted_workload_level`

都改为逐设备聚合版本

### 6.2 这次修改后的观察

在改完逐设备版本后，`T=20` 下 old 版本中“gap 没有看起来被放大”的问题缓和很多。

按新的 `weighted_workload_delta` 口径看，`init -> fixed` gap：

#### `T=10`

- `weighted_delta: 57.934 -> 59.355`，差 `+1.421`
- `reward: 6.610 -> 6.799`，差 `+0.189`
- `pre_backlog: 1.144 -> 1.010`，差 `-0.135`

#### `T=20`

- `weighted_delta: 94.125 -> 96.203`，差 `+2.078`
- `reward: 11.260 -> 11.668`，差 `+0.408`
- `pre_backlog: 1.085 -> 0.755`，差 `-0.330`

这更符合“窗口更长时错误分配应该更显著变坏”的直觉。

---

## 7. 第六阶段：调整资源口径，让任务难度更合理

用户指出得很重要：之前不少资源值还是按原来大规模系统的固定绝对值来的，不应该让 `1 UAV + 2 GU` 仍然沿用同样绝对资源。

### 7.1 资源缩放的改动

新增/修改的主要文件：

- `sagin_marl/env/config.py`
- `tests/test_config_parsing.py`

接入的配置文件：

- `configs/structured_bw_sanity_1uav_2gu_t10_stronggap_ppo.yaml`
- `configs/structured_bw_sanity_1uav_2gu_t20_stronggap_ppo.yaml`
- `configs/structured_bw_sanity_1uav_4gu_t10_stronggap_ppo.yaml`

### 7.2 缩放语义

现在资源缩放是显式 opt-in 的，重点是：

- `b_acc` 按 `task_arrival_rate * num_gu / num_uav` 相对参考系统缩放
- `b_sat_total` 与 `sat_cpu_freq` 按 `task_arrival_rate * num_gu / active_sat_count` 缩放
- `active_sat_count` 优先用：
  - `resource_scale_sat_active_count`
  - 否则 `queue_ref_sat_active_count`
  - 再否则才回退估算值

这一步专门避免了“按 144 颗总星缩放”的错误；因为实际活跃可用的卫星数远少于总星数。

### 7.3 参考系统

缩放参考系统为：

- `3 UAV`
- `20 GU`
- `task_arrival_rate = 4.9e5`
- `3 active sats`

### 7.4 资源缩放后的 quick check

在资源缩放开启后，先做了不训练的 quick check：

#### `T=10`

- `uniform -> queue_aware`
  - `reward 7.079 -> 7.147`
  - `pre_backlog 0.613 -> 0.549`

#### `T=20`

- `uniform -> queue_aware`
  - `reward 11.932 -> 12.060`
  - `pre_backlog 0.492 -> 0.371`

差异已经比最早阶段更稳定一些，但仍然不够把“hot 分不够”变成常态。

### 7.5 再进一步压低接入带宽

为了让 `uniform` 在 hot step 上更经常出现：

- `hot 不够`
- `cold 太多`

我又加入了：

- `resource_scale_b_acc_multiplier`

并把这几份 stronggap 配置都设成了：

- `resource_scale_b_acc_multiplier: 0.45`

也就是：

- `2GU` 的 `b_acc` 从 `4.898e6` 压到 `2.204e6`

### 7.6 压低后对 `uniform` 的直接检查

这一步是用户明确要求的：不看训练，只看 `uniform` 本身在当前工作点是否真的出现“hot 不够、cold 太多”。

结果：

#### `T=10`

- `hot_deficit_frac = 0.496`
- `hot_deficit_mean ≈ 2.52e5`
- `cold_unused_frac = 0.618`

#### `T=20`

- `hot_deficit_frac = 0.496`
- `hot_deficit_mean ≈ 2.52e5`
- `cold_unused_frac = 0.751`

到这里，工作点才真正变成了：

- `uniform` 在大量 hot step 上，确实既会让 hot 欠配，也会在 cold 侧浪费容量。

---

## 8. 第七阶段：统一 reward 语义

用户的要求非常明确：

- 不希望训练目标和环境 reward 各搞一套语义。
- 希望 reward 本身就能切到更自然的一致公式。

### 8.1 新增的 reward mode

在环境中新增了可直接选的 `reward_mode`：

- `weighted_workload_delta`
- `weighted_workload_level`
- `gu_queue_level`
- `system_queue_level`
- `gu_service_queue`

相关文件：

- `sagin_marl/env/sagin_env.py`
- `sagin_marl/env/structured_driver.py`

### 8.2 当前采用的统一语义

后来切到的统一口径是：

- `reward_mode: weighted_workload_level`

语义是：

- 当前步动作之后，系统还剩多少按设备服务能力归一化后的端到端工作量

它的优点是：

- `给 hot 不够` 会让它当步变差
- `给 cold 太多` 若真实挤占了 hot，也会体现在 `W_after` 里
- 不需要额外拼奇怪的 ad hoc shaping

---

## 9. 第八阶段：修正 `BW-only` 时错误的 return 递推

这一部分后来发现非常关键。

### 9.1 问题

虽然 `bw_step_reward` 已经切到了 `weighted_workload_level`，但在旧实现里，`BW-only + step_level` 下：

- `bw_return_mode=gae`

并不是标准的 BW-GAE，而是混入了：

- `next_step_accel_value`
- `next_step_return`

这对 `train_bw=True, train_accel=False` 的设置是不合理的。

### 9.2 改动

新增了真正的：

- `bw_return_mode: bw_gae`

它只在：

- `BW-only`
- `step_level`

时可用，并按标准 BW-only GAE 递推：

- `delta_bw = r_bw + gamma * V_bw(next) - V_bw(curr)`

相关文件：

- `sagin_marl/rl/structured_buffer.py`
- `sagin_marl/rl/structured_mappo.py`

### 9.3 同时切配置

以下配置也一起改成：

- `reward_mode: weighted_workload_level`
- `bw_return_mode: bw_gae`

文件：

- `configs/structured_bw_sanity_1uav_2gu_t10_stronggap_ppo.yaml`
- `configs/structured_bw_sanity_1uav_2gu_t20_stronggap_ppo.yaml`
- `configs/structured_bw_sanity_1uav_4gu_t10_stronggap_ppo.yaml`

### 9.4 smoke run

对应 smoke run：

- `runs/structured_short/two_gu_t20_bwgae_weightedlevel_smoke`

结果：

- update 1: `value_loss_bw = 15117.99`
- update 2: `value_loss_bw = 39903.92`

这里的 `v` 数值仍然很大，但此时已经不再是旧的“混错 return”问题，而更多是：

- `weighted_workload_level` 本身 target 尺度大
- `critic_loss_target_standardize = false`
- `critic_popart_enabled = false`

导致 raw MSE 数值很大

---

## 10. 第九阶段：critic 误差专项诊断

虽然用户后来明确说“critic 误差不是最根本的问题”，但这部分仍然做了比较深入的排查，因为它帮助定位到 return 机制的问题。

### 10.1 新增诊断脚本

- `scripts/diagnose_structured_critic_value_loss.py`

### 10.2 诊断 run

主要的相关 run 包括：

- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_diag`
- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_diag_file`
- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_diag_state`
- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_notime_diag`

### 10.3 旧实现下为什么 `value_loss` 很大

在旧实现的 `update 25`：

- `value_loss_bw = 16171.44`
- 但并不是所有样本都差
- SSE 被少数样本主导：
  - top1 样本占总 SSE 约 `10.2%`
  - top10 样本占约 `63.7%`

这些最坏样本的特点：

- 中段状态
- 一个 GU 几乎空
- 另一个 GU backlog 非常大
- `arrival_rate_proxy` 大约 `[0.095, 1.905]`
- `recent_service_proxy` 却还不够偏向 hot

最坏样本的典型形式：

- 真实 `return_target` 只在 `-58 ~ -216`
- 但 critic 预测成 `-512 ~ -677`

### 10.4 time_frac 去掉后仍然不解决

为了避免把人工 horizon 信息喂给模型，也临时把 `time_frac` 从 structured 输入里屏蔽了：

- `sagin_marl/env/structured_driver.py`

但去掉后 `value_loss` 仍然很高，因此 `time_frac` 不是主要矛盾。

### 10.5 更深一层的 return 分解

继续做 return decomposition 后，发现旧版本真正的问题是：

- `bw_step_reward` 虽然是 `weighted_workload_level`
- 但 `bw_step_return` 递推仍混进了 `next_step_accel_value`

这正是后来引出 `bw_gae` 修复的直接原因。

---

## 11. 第十阶段：去掉 `time_frac` 输入泄露

用户认为 continuing 语义下 `time_frac` 没必要喂给 actor/critic，我同意，因此做了简化：

- `sagin_marl/env/structured_driver.py`

处理方式：

- 先不改维度
- 直接把 `uav_nodes[..., 6]` 置 `0`

这样：

- 不破坏现有网络结构
- 但 actor/critic 再也看不到人工 episode 进度信号

---

## 12. 第十一阶段：增加 PPO 更新方向探针

用户后来把问题聚焦到最关键的地方：

> critic 误差不是最根本的，我想看到 PPO 更新为什么不正常；每一次更新是否是在朝“给更该给的用户更多、给不该给的用户更少”的方向前进。

### 12.1 新增 probe 配置

- `configs/structured_bw_sanity_1uav_2gu_t10_stronggap_probe_ppo.yaml`

这份配置主要用于：

- 每次 update 前后都在同一组固定 panel state 上重放 actor
- 同时做 branch delta / true MC probe

### 12.2 probe run

run 目录：

- `runs/structured_short/two_gu_t10_probe_u10_bwgae_weightedlevel`

### 12.3 probe 输出

主要输出：

- `update_direction_probe.csv`
- `update_direction_probe/`
- 各轮 `u0001.json ~ u0010.json`

### 12.4 probe 的总体结论

在这 10 次 update 里：

- 有些 update 会让固定 panel 上的 actor 变好
- 但很多 update 明显让它变坏
- 不是稳定朝正确方向走

按 panel 回报看：

- 好更新：`u3`, `u7`, `u8`
- 坏更新：`u1`, `u4`, `u5`, `u9`, `u10`

---

## 13. 第十二阶段：把每个 update 的动作直接解码成 hot/cold 分配

这是为了避免只看抽象的 probe 指标，而是直接看：

- 每一轮 checkpoint 的 actor
- 在同一组 panel state 上
- 到底给 hot 分了多少、给 cold 分了多少

### 13.1 新增离线分析脚本

- `scripts/analyze_bw_update_hot_cold_direction.py`

### 13.2 输出

- `runs/structured_short/two_gu_t10_probe_u10_bwgae_weightedlevel/update_hot_cold_direction.csv`
- `runs/structured_short/two_gu_t10_probe_u10_bwgae_weightedlevel/update_hot_cold_direction.json`

### 13.3 直接结论

固定 panel 共：

- `16` 个 state
- 其中 `11` 个是 hot state
- `5` 个是 off state

对这 `11` 个 hot state：

- heuristic 平均 `hot_share = 0.7320`
- PPO 每轮 actor 的平均 `hot_share` 只在 `0.4444 ~ 0.4894`
- 所有 update 中：
  - `hot_gt_cold_frac = 0.0`

也就是：

- 在所有 hot state 上，actor 从来没有做到“给 hot 的比给 cold 的更多”
- 它始终被卡在接近均分的盆地附近

### 13.4 每轮 update 的 `hot_share` 变化

简化后可读成：

| update | 平均 `hot_share` | 相对前一轮 | panel 回报变化 |
|---|---:|---:|---:|
| `u1` | `0.4875` | 初始 | `-1.813` |
| `u2` | `0.4876` | `+0.0001` | `-0.031` |
| `u3` | `0.4894` | `+0.0018` | `+0.309` |
| `u4` | `0.4739` | `-0.0155` | `-2.004` |
| `u5` | `0.4509` | `-0.0230` | `-2.362` |
| `u6` | `0.4447` | `-0.0062` | `-0.292` |
| `u7` | `0.4444` | `-0.0003` | `+0.706` |
| `u8` | `0.4583` | `+0.0139` | `+0.831` |
| `u9` | `0.4489` | `-0.0095` | `-1.378` |
| `u10` | `0.4449` | `-0.0040` | `-0.899` |

所以：

- 并不是每次都在往“hot 更多”走
- 反而多轮明显在往反方向走

### 13.5 两次最典型的更新

#### 坏更新：`u3 -> u4`

- `11/11` 个 hot state 的 `hot_share` 都下降
- `10/11` 个离更合理方向更远
- 平均 `hot_share: 0.4894 -> 0.4739`

#### 相对好的更新：`u7 -> u8`

- `9/11` 个 hot state 的 `hot_share` 上升
- `8/11` 个更接近更合理方向
- 平均 `hot_share: 0.4444 -> 0.4583`

---

## 14. 第十三阶段：为什么不能把 heuristic 当成标准答案

这部分是在 probe 后补充澄清的，非常重要。

### 14.1 heuristic 只是参考，不是标准答案

真正该当标准的是：

- `branch_delta_h10`
- `true_adv_mc`

即：

- 某个动作改动，在真实后续回报上到底是好还是坏

### 14.2 为什么会出现“hot_mask=GU1，但 heuristic 却给 GU1 分 0”

这个具体 state 为：

- `episode 0, t=8`
- `hot_mask = [0, 1]`
- `gu_queue = [3581, 0]`
- `last_gu_arrival = [7.6e4, 1.5238e6]`
- `last_gu_outflow = [1.06e5, 2.32e6]`
- heuristic 动作：`[1, 0]`

其含义是：

- `GU1` 虽然是 hot，但这一步决策时它已经被清空了
- 且上一拍给它的服务已经明显超过其到达
- `GU0` 虽然不是 hot，却还留着 backlog，而且 `service_gap` 更高

因此此时再给 `GU1` 更像空转；给 `GU0` 反而更合理。

所以规则不是：

- “只要 hot 就永远多给”

而是：

- “给当前更需要、且能真正用掉带宽的那个用户更多”

在大多数 hot state 上，这通常等价于多给 hot，但不是绝对规则。

---

## 15. 第十四阶段：样本级看 PPO 为什么会朝错误方向更新

这是当前最关键的调查结果。

### 15.1 核心问题不是单一的

坏更新里，至少存在三层问题：

1. actor 对正/负 advantage 的响应本身就不完全稳定
2. advantage 标准化会翻符号
3. 更根本地，raw advantage 自己就经常和真实动作价值排错序

### 15.2 `u4` 的坏更新细节

相关文件：

- `runs/structured_short/two_gu_t10_probe_u10_bwgae_weightedlevel/update_direction_probe/u0004.json`

#### 全局指标

- `corr_advantage_vs_delta_logprob = +0.262`
  - PPO 仍在“按自己的 advantage 更新”
- 但：
  - `corr_branch_delta_vs_delta_logprob = -0.294`
  - `corr_true_adv_mc_vs_delta_logprob = -0.220`

也就是：

- actor 的 logprob 变化，和真实动作价值整体上是反着的

#### 标准化翻符号

`u4` 中：

- `sign_agree_norm_raw_advantage = 0.5625`

也就是：

- `16` 个 probe 样本中有 `7` 个样本在标准化前后翻了符号

典型样本：

- `sample 5`
  - `raw_advantage = -4.092`
  - 标准化后 `advantage = +0.721`
  - `delta_logprob = +0.023`
  - 但：
    - `branch_delta_h10 = -1.881`
    - `true_adv_mc = -1.537`

也就是：

- 真实上是坏动作
- 但被标准化后的 advantage 当成正样本去鼓励了

#### raw advantage 自己也常常排错

`u4` 中：

- `corr_raw_advantage_vs_true_adv_mc = -0.054`
- `sign_agree_raw_advantage_true_adv_mc = 0.643`

这说明不仅标准化有问题，raw advantage 自己就经常和真实动作价值不一致。

典型样本：

- `sample 7`
  - `raw_advantage = -198.046`
  - `advantage = -2.951`
  - `delta_logprob = -0.187`
  - 但：
    - `branch_delta_h10 = +3.590`
    - `true_adv_mc = +3.845`

也就是：

- 一个真实上很好的动作
- 被 raw advantage 判成很坏
- 然后 PPO 确实去压低了它

### 15.3 `u8` 的相对好更新细节

相关文件：

- `runs/structured_short/two_gu_t10_probe_u10_bwgae_weightedlevel/update_direction_probe/u0008.json`

#### 全局指标

- `corr_branch_delta_vs_delta_logprob = +0.212`
- `corr_true_adv_mc_vs_delta_logprob = +0.313`

这次整体方向比 `u4` 更对。

#### 但依然不稳定

例如：

- `sample 3`
  - `advantage = +1.009`
  - `delta_logprob = +0.062`
  - `branch_delta_h10 = +1.723`
  - `true_adv_mc = +1.736`
  - 这是典型的“好样本被正确鼓励”

但也有：

- `sample 2`
  - `advantage = +0.664`
  - `delta_logprob = +0.291`
  - `branch_delta_h10 = -2.074`
  - `true_adv_mc = -0.948`
  - 这是“坏样本仍被鼓励”

所以：

- `u8` 只是净方向更好
- 并不是 PPO 已经稳定学会更新方向

---

## 16. 当前最重要的结论

把这次对话里的结果压缩成最核心的几条：

1. 旧环境下 PPO “能学”，但环境本身并不是真正严格轮流的窗口任务。
2. 旧工作点过松，`uniform` 在很多 episode 上也能混过去，因此不会稳定暴露策略问题。
3. 需要同时做两类修正：
   - 调整环境资源工作点
   - 调整 reward / return / update 诊断机制
4. 逐设备 `weighted_workload`、资源缩放、`b_acc` 压低、`reward_mode=weighted_workload_level`、`bw_gae` 这些修改后，环境与学习目标已经更接近我们真正想检验的问题。
5. 但当前最根本的问题已经不是“critic 误差大不大”，而是：
   - PPO 的更新方向并不稳定
   - 有些 update 会系统性把策略从“更该给的用户”那边拉开
6. 更具体地说：
   - 既有 raw advantage 排错序的问题
   - 也有标准化后翻符号的问题
   - 还存在共享参数更新后单样本 `delta_logprob` 不严格按 advantage 同号变化的问题

---

## 17. 本次对话中新增或修改过的主要文件

### 配置

- `configs/structured_bw_sanity_1uav_2gu_t10_stronggap_ppo.yaml`
- `configs/structured_bw_sanity_1uav_2gu_t20_stronggap_ppo.yaml`
- `configs/structured_bw_sanity_1uav_4gu_t10_stronggap_ppo.yaml`
- `configs/structured_bw_sanity_1uav_2gu_t10_stronggap_probe_ppo.yaml`

### 环境与 driver

- `sagin_marl/env/config.py`
- `sagin_marl/env/sagin_env.py`
- `sagin_marl/env/structured_driver.py`

### RL / return / probe

- `sagin_marl/rl/structured_buffer.py`
- `sagin_marl/rl/structured_mappo.py`
- `sagin_marl/rl/structured_bw_update_direction.py`
- `sagin_marl/rl/structured_factory.py`

### 脚本

- `scripts/evaluate_structured.py`
- `scripts/diagnose_structured_critic_value_loss.py`
- `scripts/analyze_bw_update_hot_cold_direction.py`

### 测试

- `tests/test_config_parsing.py`
- `tests/test_structured_driver_buffer.py`
- `tests/test_env_step_invariants.py`

---

## 18. 本次对话中重点相关的 run 目录

### 早期短训 / baseline

- `runs/structured_short/two_gu_stronggap_u60`
- `runs/structured_short/two_gu_t20_stronggap_u30_gpu4`
- `runs/structured_short/one_uav_four_gu_t10_stronggap_u30_gpu4`

### 逐设备 weighted workload 过渡阶段

- `runs/structured_short/two_gu_t10_stronggap_u30_gpu4_weighted_entity`
- `runs/structured_short/two_gu_t20_stronggap_u30_gpu4_weighted_entity`

### 资源压缩 + weighted_level

- `runs/structured_short/two_gu_t10_stronggap_u30_gpu4_weighted_level_access045`
- `runs/structured_short/two_gu_t20_stronggap_u30_gpu4_weighted_level_access045`

### critic 诊断

- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_diag`
- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_diag_file`
- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_diag_state`
- `runs/structured_short/two_gu_t20_stronggap_u25_gpu4_weighted_level_access045_notime_diag`

### `bw_gae + weighted_workload_level` smoke

- `runs/structured_short/two_gu_t20_bwgae_weightedlevel_smoke`

### PPO 更新方向 probe

- `runs/structured_short/two_gu_t10_probe_u10_bwgae_weightedlevel`

---

## 19. 当前仍待继续追的问题

从现在往下，最值得继续追的不是“critic 误差大不大”，而是：

1. `raw_advantage` 为什么会和 `branch_delta / true_adv_mc` 经常排错序。
2. 标准化为什么会在这批样本上频繁翻符号。
3. 在这些错样本里，`return_target / value_pred / values_for_advantage` 具体是如何导致错误排序的。
4. 是否需要进一步把 probe 样本表格化，逐样本列出：
   - `raw_adv sign`
   - `norm_adv sign`
   - `true_adv sign`
   - `branch sign`
   - `delta_logprob sign`

这样可以一眼定位：

- 错从哪一层开始出现
- 是 value 排序、return target、还是标准化在主导问题

