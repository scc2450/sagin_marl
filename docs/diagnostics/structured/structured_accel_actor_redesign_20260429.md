# Structured Accel Actor 重新设计

日期：2026-04-29

本文是 accel actor 的确定版设计规格。目标不是给出过渡方案，而是一次性定义清楚 accel actor 应该看什么、每个量怎么归一化、网络如何消费这些量、以及如何落到单 GPU native rollout 架构。

## 1. 设计目标

accel actor 是三阶段 actor 的第一阶段。按现有环境接口，它为每台 UAV 输出 2 维归一化加速度命令：

```text
a_norm_u = [a_x_norm, a_y_norm] in [-1, 1]^2
```

环境在 `sagin_marl/env/sagin_env.py::_apply_uav_dynamics()` 中把它映射为物理加速度：

```text
a_policy_u = clip(a_norm_u, -1, 1) * cfg.a_max
```

accel 的因果链是：

```text
UAV acceleration
  -> UAV position / velocity
  -> GU 和 UAV 的几何关系
  -> GU 固定连接最近 UAV
  -> BW actor 给已连接 GU 分带宽
  -> GU/UAV/SAT queue、drop、workload
  -> system reward
```

当前设定中：

```text
GU 连接离自己最近的 UAV。
UAV 可连接 GU 槽位等于 GU 数量。
所以所有 GU 都一定能进入某台 UAV 的可服务集合。
```

槽位足够不等于服务能力足够。一个 UAV 如果因为初始化位置离所有 GU 最近，所有 GU 都会连接它，随后它独占 access bandwidth 压力、UAV queue 压力和 backhaul 压力，其他 UAV 空闲。accel actor 的核心任务因此是：

```text
通过 UAV 运动塑造负载合理、链路可服务、安全可执行的 GU-UAV 几何分区。
```

它不是“朝高队列 GU 飞”的控制器，也不是“守住更多 GU”的贪心策略。

## 2. 符号和归一化基准

### 2.1 基本符号

```text
U = num_uav
G = num_gu
S = cfg.per_uav_visible_sat_token_max

p_u = UAV u 的二维位置 [m]
v_u = UAV u 的二维速度 [m/s]
p_g = GU g 的二维位置 [m]

D_ug = ||p_g - p_u||_2
D_uv = ||p_u - p_v||_2
```

本设计按“当前二维水平几何下的最近 UAV”定义 GU owner。代码中的唯一 owner 入口是 `_associate_users()` / native tensor association helper：

```text
D2_ug = ||p_g - p_u||_2^2
owner(g) = argmin_u D2_ug
```

tie 使用 `argmin` 的低 index 规则。access pathloss / access gain 仍然用于链路质量、弱链路、干扰和速率计算，但不参与 owner 判定。代码实现必须和环境 association 入口保持同一口径。`sagin_marl/env/sagin_env.py::_associate_users()`、batch numpy helper、torch helper 和 CUDA/native helper 都直接使用二维水平距离平方 argmin。`pl_threshold_db` 不再让 GU 变成未连接。accel builder 不能读取上一拍 `last_association` 作为当前 owner，也不能在 Python builder 和 native builder 各写一套不同的 owner 规则；由 parity test 锁定一致性。

本 redesign 规定当前 accel stage 的所有 GU 都会连接到某台 UAV。`_associate_users()` 不产生 `-1` association；`spec["assoc"]` / `owner(g)` 中检测到 `-1` 直接报错，accel actor schema 中不引入“当前未连接 GU”分支。

`last_association` 是上一拍执行历史，不等同于当前 owner。环境 reset 后、第一拍 access 执行前，`last_association` 仍可能全为 `-1`。因此所有 `last_association` 相关字段必须按历史字段处理：flag 为 0，`gu_last_bw_sum` 为 0，`env._bw_weighted_workload_device_costs(assoc_override=last_association, ...)` 继续使用现有 helper 对未连接历史的默认成本语义。不能因为上一拍历史为 `-1` 把当前 accel obs 构建判为非法。

配置必须满足：

```text
effective_candidate_k =
  cfg.users_obs_max
  if cfg.candidate_k is None or cfg.candidate_k <= 0
  else min(cfg.candidate_k, cfg.users_obs_max)

U > 0
G > 0
S > 0
cfg.num_sat > 0
cfg.users_obs_max >= G
effective_candidate_k >= G
cfg.candidate_mode in {"assoc", "nearest"}
sat_select_ref_count =
  min(
    cfg.num_sat,
    int(cfg.N_RF),
    int(cfg.sat_num_select)
      if cfg.sat_num_select is not None and cfg.sat_num_select > 0
      else int(cfg.N_RF)
  )
sat_select_ref_count > 0
cfg.N_RF > 0
cfg.map_size > 0
cfg.v_max > 0
cfg.a_max > 0
cfg.uav_energy_init > 0
cfg.queue_max_gu > 0
cfg.queue_max_uav > 0
cfg.queue_max_sat > 0
cfg.noise_density > 0
cfg.b_acc > 0
_effective_b_backhaul_per_sat_from_cfg(cfg) > 0
cfg.task_cycles_per_bit > 0
cfg.speed_of_light > 0
cfg.r_earth + cfg.sat_height > 0
```

不使用 `max(G, 1)` 这类兜底。配置非法时直接报错。

`U == 1` 是合法配置。此时没有 peer UAV，也没有几何分区竞争；所有 peer 相关 token 轴长度为 0，所有需要“其他 UAV”参与的分区敏感度按本文定义的退化规则计算。

### 2.2 物理归一化

```text
map_ref        = cfg.map_size
vel_ref        = cfg.v_max
accel_ref      = cfg.a_max
energy_ref     = cfg.uav_energy_init
orbit_pos_ref  = cfg.r_earth + cfg.sat_height
mu_earth       = 3.986004418e14
sat_vel_ref    = sqrt(mu_earth / orbit_pos_ref)
```

能量：

```text
energy_norm = uav_energy / energy_ref
```

位置、相对位置、距离：

```text
x_norm       = x / map_ref
rel_xy_norm  = rel_xy / map_ref
dist_norm    = dist / map_ref
```

速度：

```text
vx_norm       = vx / vel_ref
vy_norm       = vy / vel_ref
speed_norm    = sqrt(vx^2 + vy^2) / vel_ref
rel_v_norm    = rel_v / vel_ref
```

加速度历史：

```text
last_policy_accel_norm = last_policy_accel / accel_ref
last_exec_accel_norm   = last_exec_accel / accel_ref
last_intervention_dx   = (last_exec_accel_x - last_policy_accel_x) / accel_ref
last_intervention_dy   = (last_exec_accel_y - last_policy_accel_y) / accel_ref
last_intervention_l2   = ||last_exec_accel - last_policy_accel||_2 / accel_ref
```

`last_policy_accel` 是上一拍 policy 归一化输出经环境映射后的物理加速度。`last_exec_accel` 是经过 avoidance、energy safety、boundary hard filter、pairwise hard filter 等所有安全/约束处理后真正执行到动力学里的物理加速度。二者差值表达安全层或硬约束是否干预过策略。

### 2.3 Flow 和 queue 归一化

使用和当前 reward / critic 一致的 flow scale。当前代码中的语义入口是 `env._arrival_ref()` 和 `env._bw_weighted_workload_sat_active_ref_count()`：

```text
arrival_ref_step = env._arrival_ref()

gu_flow_ref  = arrival_ref_step / num_gu
uav_flow_ref = arrival_ref_step / num_uav

sat_active_ref_count =
  env._bw_weighted_workload_sat_active_ref_count()

sat_flow_ref = arrival_ref_step / sat_active_ref_count
```

这些 reference 在 config finalize 和 env 初始化后必须为正。`queue_ref_gu_per_step`、`queue_ref_uav_per_step`、`queue_ref_sat_per_step` 保持当前队列初始化用途，不进入 accel actor 的 flow scale，避免 actor 与当前 reward / critic 采用不同归一化口径。

队列、到达、出流、drop 都用对应层的 flow ref：

```text
gu_queue_steps            = gu_queue_bits / gu_flow_ref
gu_expected_arrival_steps = expected_gu_arrival_bits / gu_flow_ref
gu_last_arrival_steps     = last_gu_arrival / gu_flow_ref
gu_last_outflow_steps     = last_gu_outflow / gu_flow_ref
gu_last_drop_steps        = gu_drop / gu_flow_ref

uav_queue_steps           = uav_queue_bits / uav_flow_ref
uav_last_inflow_steps     = last_gu_to_uav_inflow_by_uav / uav_flow_ref
uav_last_outflow_steps    = last_uav_outflow / uav_flow_ref
uav_last_drop_steps       = uav_drop / uav_flow_ref

sat_queue_steps           = sat_queue_bits / sat_flow_ref
sat_last_incoming_steps   = last_sat_incoming / sat_flow_ref
sat_last_processed_steps  = last_sat_processed / sat_flow_ref
sat_last_drop_steps       = sat_drop / sat_flow_ref
```

其中：

```text
expected_gu_arrival_bits =
  env._current_expected_gu_arrival_rates()[g] * tau0

last_uav_outflow[u] =
  sum_s last_uav_to_sat_outflow_matrix[u, s]

gu_service_ema_steps  = gu_service_ema_bits_per_step / gu_flow_ref
uav_service_ema_steps = uav_service_ema_bits_per_step / uav_flow_ref
sat_service_ema_steps = sat_service_ema_bits_per_step / sat_flow_ref
```

`gu_service_ema_bits_per_step`、`uav_service_ema_bits_per_step`、`sat_service_ema_bits_per_step` 来自现有 `env._bw_weighted_workload_device_ema_vectors()`；本 redesign 不新增一套 EMA 统计。

容量填充率单独保留：

```text
gu_queue_fill  = gu_queue_bits / queue_max_gu
uav_queue_fill = uav_queue_bits / queue_max_uav
sat_queue_fill = sat_queue_bits / queue_max_sat
```

### 2.4 Cost 和 workload 归一化

服务 EMA 转换成 reward cost：

```text
service_floor = env._bw_weighted_workload_eps()

gu_local_cost  = 1 / max(gu_service_ema_bits_per_step, service_floor)
uav_local_cost = 1 / max(uav_service_ema_bits_per_step, service_floor)
sat_cost       = 1 / max(sat_service_ema_bits_per_step, service_floor)
```

cost reference：

```text
gu_local_cost_ref  = 1 / gu_flow_ref
uav_local_cost_ref = 1 / uav_flow_ref
sat_cost_ref       = 1 / sat_flow_ref

uav_total_cost_ref = uav_local_cost_ref + sat_cost_ref
gu_total_cost_ref  = gu_local_cost_ref + uav_total_cost_ref
```

actor 输入使用 log ratio：

```text
gu_local_cost_log_ratio  = log(gu_local_cost / gu_local_cost_ref)
uav_local_cost_log_ratio = log(uav_local_cost / uav_local_cost_ref)
sat_cost_log_ratio       = log(sat_cost / sat_cost_ref)
```

实现中使用当前代码已有的 log-ratio 保护口径：

```text
log_ratio(value, ref) =
  log(max(value, LOG_RATIO_EPS) / max(ref, LOG_RATIO_EPS))
```

这只防止 `log(0)`，不改变正值的量纲定义。

其中 `env._bw_weighted_workload_eps()` 由现有配置 `service_floor_bits_per_step` / `bw_weighted_workload_eps` 解析得到，actor 不新增 cost floor。

上一拍真实 route 下的总 cost 使用现有 reward helper 计算：

```text
last_gu_total_cost,
last_uav_total_cost,
last_sat_cost =
  env._bw_weighted_workload_device_costs(
    assoc_override=last_association,
    sat_selection_override=last_sat_selection
  )
```

actor 输入的 cost log ratio：

```text
gu_last_total_cost_log_ratio  = log(last_gu_total_cost / gu_total_cost_ref)
uav_last_total_cost_log_ratio = log(last_uav_total_cost / uav_total_cost_ref)
```

上一拍 workload level：

```text
gu_last_workload_log1p  = log1p(last_gu_total_cost * gu_queue_bits)
uav_last_workload_log1p = log1p(last_uav_total_cost * uav_queue_bits)
sat_last_workload_log1p = log1p(sat_cost * sat_queue_bits)
```

这些量不是人工 priority。它们来自 reward 的 workload 定义。

### 2.5 Access interference 归一化

access noise reference：

```text
access_noise_ref =
  noise_density * b_acc * access_noise_figure_linear

access_noise_figure_linear =
  channel.noise_figure_linear(access_noise_figure_db)
```

当前 ego-GU 参考接入 SNR：

```text
access_gain_ug = access_gain_matrix[g, ego]

ug_rx_snr_ref =
  gu_tx_power * access_gain_ug / access_noise_ref

ug_access_se_ref =
  if fading_enabled and access_fading_mode == "ergodic_rician":
    rician_ergodic_spectral_efficiency(
      ug_rx_snr_ref,
      rician_K,
      access_ergodic_rician_quadrature_points
    )
  else:
    log2(1 + ug_rx_snr_ref)
```

`ug_access_se_ref` 是当前动作前 ego 到 GU 的参考 spectral efficiency。它不包含当前 BW 分配，也不包含 post-accel 信息。
若 access fading mode 是 `iid_rician`，accel obs builder 不能在不同路径重复采样。stage spec 中必须携带本次 observation 使用的 `access_gain_matrix`；Python builder、native builder、critic world builder 使用同一份矩阵。

上一拍 ego 实际承受的 access interference：

```text
uav_last_access_interference_log1p =
  log1p(last_access_interference_by_uav / access_noise_ref)
```

每个 GU 对 ego 的上一拍非自身小区干扰贡献使用上一拍总 access BW fraction：

```text
gu_last_bw_sum =
  if last_association(g) in [0, U):
    max(sum_u last_bw_fraction_by_uav_gu[u, g], 0)
  else:
    0

ug_last_nonself_interference_power =
  gu_tx_power
  * access_gain_ug
  * gu_last_bw_sum
  * 1[last_association(g) != ego]

ug_last_nonself_interference_log1p =
  log1p(ug_last_nonself_interference_power / access_noise_ref)
```

这个量表达：如果 GU g 上一拍在 access 链路上传输，且它不是 ego 服务的 GU，那么它按上一拍占用的 access BW fraction 在 ego 接收端形成了多大潜在干扰。

这里按 `last_bw_fraction_by_uav_gu` 的总和缩放干扰源功率，和环境 `compute_access_interference_beta_continuous(...)` 的 `gu_band_fraction` 口径一致。因此 actor schema 中同时保留 ego 自己给该 GU 的 fraction 和该 GU 的全局总 fraction：

```text
last_bw_fraction_ego_gu
gu_last_bw_sum
```

## 3. Accel Actor 输入 Schema

accel actor 使用单独 schema，不复用 critic schema。每个 ego UAV 的 local state 包含：

```text
ego_features
ego_cell
gu_tokens[G]
peer_tokens[U - 1]
sat_tokens[S]
```

`ego_cell` 字段内容等于 `cell_summary[ego]`。

所有 token 轴固定长度，mask 区分 padding。

## 4. Ego Features

ego features 包含 ego 物理状态、自身 relay 状态、上一拍安全干预历史。

```text
ego_x_norm
ego_y_norm
ego_vx_norm
ego_vy_norm
ego_speed_norm
ego_energy_norm

boundary_left_norm   = ego_x / map_ref
boundary_right_norm  = (map_ref - ego_x) / map_ref
boundary_bottom_norm = ego_y / map_ref
boundary_top_norm    = (map_ref - ego_y) / map_ref
```

自身 UAV queue / workload：

```text
uav_queue_steps
uav_queue_fill
uav_last_inflow_steps
uav_last_outflow_steps
uav_last_drop_steps
uav_service_ema_steps = uav_service_ema_bits_per_step / uav_flow_ref
uav_local_cost_log_ratio
uav_last_total_cost_log_ratio
uav_last_workload_log1p
uav_last_access_interference_log1p
```

上一拍动作历史：

```text
last_policy_accel_x_norm
last_policy_accel_y_norm
last_exec_accel_x_norm
last_exec_accel_y_norm
last_intervention_dx
last_intervention_dy
last_intervention_l2
```

## 5. GU Tokens

每个 GU token 分成四类字段。

### 5.1 GU 原始 workload 字段

```text
gu_x_norm
gu_y_norm
gu_queue_steps
gu_queue_fill
gu_expected_arrival_steps
gu_last_arrival_steps
gu_last_outflow_steps
gu_last_drop_steps
gu_service_ema_steps
gu_local_cost_log_ratio
gu_last_total_cost_log_ratio
gu_last_workload_log1p
```

### 5.2 Ego-GU 几何和链路字段

```text
rel_gu_x_norm = (gu_x - ego_x) / map_ref
rel_gu_y_norm = (gu_y - ego_y) / map_ref
d_ego_gu_norm = D_ego,g / map_ref
ug_access_se_ref
```

上一拍 ego 对 GU 的服务历史：

```text
last_assoc_to_ego_flag = 1[last_association(g) == ego]
last_bw_fraction_ego_gu = last_bw_fraction_by_uav_gu[ego, g]
last_served_by_ego_flag =
  1[
    last_association(g) == ego
    and last_bw_fraction_ego_gu > NORMALIZATION_DENOM_EPS
  ]
```

`last_served_by_ego_flag` 使用执行历史中的带宽分配是否为正，是上一拍事实。这里的 `NORMALIZATION_DENOM_EPS` 是当前环境判断 access active 的数值保护口径，不是语义阈值。

### 5.3 当前几何分区字段

对每个 GU g，先计算：

```text
owner_g    = owner(g)
D_owner_g  = D_{owner_g,g}

if U > 1:
  D_second_g = min_{v != owner_g} D_vg
  D_best_other_ego_g = min_{v != ego} D_vg
else:
  D_second_g = D_owner_g + map_ref
  D_best_other_ego_g = D_ego,g + map_ref
```

输入 ego 的 GU token：

```text
pre_owner_is_ego = 1[owner_g == ego]

handoff_margin_ego =
  (D_best_other_ego_g - D_ego,g) / map_ref

owner_stability_margin =
  (D_second_g - D_owner_g) / map_ref

ego_takeover_gap =
  (D_ego,g - D_owner_g) / map_ref
```

含义：

```text
handoff_margin_ego > 0:
  ego 当前是这个 GU 的最近 UAV。

handoff_margin_ego < 0:
  其他 UAV 当前更近。

owner_stability_margin 越小:
  这个 GU 越靠近当前分区边界，轻微移动越可能改变 owner。

ego_takeover_gap = 0:
  ego 是 owner。

ego_takeover_gap > 0:
  ego 距离当前 owner 还有多少几何差距。

U == 1 时:
  pre_owner_is_ego = 1
  handoff_margin_ego = 1
  owner_stability_margin = 1
  ego_takeover_gap = 0
  这些值只表示单 UAV 退化状态，不表示存在可竞争边界。
```

### 5.4 平滑分区敏感度和干扰字段

不使用阈值定义“边界 GU”或“弱链路 GU”。使用连续权重：

```text
partition_boundary_weight =
  if U > 1:
    exp(-owner_stability_margin)
  else:
    0

ego_link_weakness =
  1 / (1 + ug_access_se_ref)
```

含义：

```text
partition_boundary_weight 接近 1:
  当前 owner 和第二近 UAV 距离很接近，GU 易换 owner。

ego_link_weakness 越大:
  ego 到该 GU 的参考链路越弱。

U == 1 时 partition_boundary_weight 固定为 0，因此 cell boundary workload 和 boundary moment 固定没有跨 UAV 重分配含义。单 UAV 场景下 accel actor 仍然通过 ego 状态、GU workload/link、SAT/backhaul 和 boundary safety 学习运动。
```

干扰字段：

```text
gu_last_bw_sum
ug_last_nonself_interference_log1p
```

`gu_last_bw_sum` 表示该 GU 上一拍获得的总 access BW fraction；若 `last_association(g)` 非法则为 0。干扰功率使用这个 fraction 缩放，保持和环境当前 access interference 计算口径一致。

## 6. Public Cell Summary

cell summary 是本设计解决负载集中问题的核心。它从公开 GU 状态和当前最近连接几何计算，不使用 peer UAV 私有 queue。

对每台 UAV u 的当前 cell：

```text
C_u = { g | owner(g) = u }
```

这里 `G` 必须大于 0。若 `num_gu <= 0`，环境配置非法，初始化时直接报错。

### 6.1 基本 cell 负载

```text
cell_gu_count_frac[u] =
  |C_u| / G

cell_queue_steps_sum[u] =
  sum_{g in C_u} gu_queue_steps[g] / G

cell_expected_arrival_steps_sum[u] =
  sum_{g in C_u} gu_expected_arrival_steps[g] / G

cell_last_arrival_steps_sum[u] =
  sum_{g in C_u} gu_last_arrival_steps[g] / G

cell_last_outflow_steps_sum[u] =
  sum_{g in C_u} gu_last_outflow_steps[g] / G

cell_last_drop_steps_sum[u] =
  sum_{g in C_u} gu_last_drop_steps[g] / G

cell_last_workload_log1p_sum[u] =
  sum_{g in C_u} gu_last_workload_log1p[g] / G
```

这些是 cell 内 GU workload 的公开聚合。除以 `G` 后，不会因为 GU 数量改变而改变量级。

### 6.2 cell share gap

系统期望不是所有 cell 数量完全相等，而是不要让公开 workload 全部压到一台 UAV。定义 workload share：

```text
total_cell_workload =
  sum_v cell_last_workload_log1p_sum[v]

cell_workload_share_gap[u] =
  if total_cell_workload > 0:
    cell_last_workload_log1p_sum[u] / total_cell_workload - 1 / U
  else:
    0
```

含义：

```text
cell_workload_share_gap > 0:
  该 UAV 当前几何 cell 承担的 GU workload 高于平均份额。

cell_workload_share_gap < 0:
  该 UAV 当前几何 cell 低负载或空闲。
```

### 6.3 平滑边界 workload

不用阈值筛选边界 GU。对所有 GU 连续加权：

```text
cell_boundary_workload_sum[u] =
  sum_{g in C_u}
    gu_last_workload_log1p[g] * partition_boundary_weight[g]
  / G
```

含义：

```text
高 workload 且 owner_stability_margin 小的 GU 权重大。
这个量表示 cell 中有多少 workload 位于容易重新分配的几何边界附近。
```

### 6.4 平滑弱链路 workload

不用阈值筛选弱链路 GU。对所有 GU 连续加权：

```text
se_owner_g =
  if fading_enabled and access_fading_mode == "ergodic_rician":
    rician_ergodic_spectral_efficiency(
      gu_tx_power * access_gain_matrix[g, owner(g)] / access_noise_ref,
      rician_K,
      access_ergodic_rician_quadrature_points
    )
  else:
    log2(1 + gu_tx_power * access_gain_matrix[g, owner(g)] / access_noise_ref)

owner_link_weakness_g =
  1 / (1 + se_owner_g)

cell_weak_link_workload_sum[u] =
  sum_{g in C_u}
    gu_last_workload_log1p[g] * owner_link_weakness_g
  / G
```

含义：

```text
高 workload 且当前 owner 到 GU 链路弱的 GU 权重大。
这个量表示 cell 中有多少 workload 虽然归当前 owner，但服务效率差。
```

### 6.5 cell access pressure

access pressure 按 GU 先计算未满足需求，再对 cell 求和。这里用上一拍 GU 实际出流作为服务反馈，不使用任意的分母常数，也不再用上一拍 interference 修正的参考 SE。链路弱和干扰暴露由 `cell_weak_link_workload_sum`、`cell_interference_exposure` 和 GU token 中的 interference 字段表达。

```text
gu_demand_steps[g] =
  gu_queue_steps[g]
  + gu_expected_arrival_steps[g]

gu_unserved_demand_steps[g] =
  max(gu_demand_steps[g] - gu_last_outflow_steps[g], 0)

gu_access_pressure_to_owner[g] =
  gu_unserved_demand_steps[g] + gu_last_drop_steps[g]

cell_access_pressure[u] =
  sum_{g in C_u} gu_access_pressure_to_owner[g] / G
```

含义：

```text
每个 GU 的当前队列和预期到达越高，pressure 越高。
上一拍该 GU 的真实出流越高，未满足需求越低。
上一拍 drop 是已经发生的损失信号，不被出流抵消，直接加回 pressure。
cell 中 GU 越多，sum 后的 pressure 自然越高。
```

### 6.6 cell interference exposure

对每台 UAV u，估计上一拍其他 cell 中活跃 GU 对它的 access 干扰暴露：

```text
cell_interference_power[u] =
  sum_{g: last_association(g) != u}
    gu_tx_power
    * access_gain_matrix[g, u]
    * gu_last_bw_sum[g]

cell_interference_exposure[u] =
  log1p(cell_interference_power[u] / access_noise_ref)
```

含义：

```text
该 UAV 接收端对其他小区 GU 的历史传输有多敏感。
这里先按环境当前 access interference 口径聚合上一拍干扰源功率，再做 log1p 归一化。
发射 GU 的贡献按上一拍总 BW fraction 缩放。
```

### 6.7 ego 和 peer 使用 cell summary 的方式

ego local state 中包含：

```text
ego_cell = cell_summary[ego]
```

peer token 中包含：

```text
peer_public_cell_summary = cell_summary[peer]
```

这不包含 peer UAV 私有 queue。它只来自 GU 公共状态、UAV/GU 位置、上一拍公开执行历史和当前几何 owner。

### 6.8 cell directional moments

accel action 是二维向量，因此 cell summary 不能只有标量压力，还必须保留压力相对 UAV 的方向信息。这里的方向量是 observation，不是手工 action。

对每个 GU 定义：

```text
rel_owner_gu_norm[g] =
  (p_g - p_owner(g)) / map_ref

boundary_weight[g] =
  partition_boundary_weight[g]

weak_link_weight[g] =
  owner_link_weakness_g
```

cell demand moment：

```text
cell_demand_moment_denom[u] =
  sum_{g in C_u} gu_demand_steps[g]

cell_demand_moment_xy[u] =
  if cell_demand_moment_denom[u] > 0:
    sum_{g in C_u}
      gu_demand_steps[g] * rel_owner_gu_norm[g]
    / cell_demand_moment_denom[u]
  else:
    [0, 0]
```

cell boundary moment：

```text
cell_boundary_moment_denom[u] =
  sum_{g in C_u} gu_last_workload_log1p[g] * boundary_weight[g]

cell_boundary_moment_xy[u] =
  if cell_boundary_moment_denom[u] > 0:
    sum_{g in C_u}
      gu_last_workload_log1p[g]
      * boundary_weight[g]
      * rel_owner_gu_norm[g]
    / cell_boundary_moment_denom[u]
  else:
    [0, 0]
```

cell weak-link moment：

```text
cell_weak_link_moment_denom[u] =
  sum_{g in C_u} gu_last_workload_log1p[g] * weak_link_weight[g]

cell_weak_link_moment_xy[u] =
  if cell_weak_link_moment_denom[u] > 0:
    sum_{g in C_u}
      gu_last_workload_log1p[g]
      * weak_link_weight[g]
      * rel_owner_gu_norm[g]
    / cell_weak_link_moment_denom[u]
  else:
    [0, 0]
```

含义：

```text
这些 if 只处理分母为 0 的情况，不是阈值筛选。分母为 0 表示该 cell 没有对应 demand/workload 权重，方向没有定义，因此填 [0, 0]。

cell_demand_moment_xy:
  cell 内需求重心相对该 UAV 的方向。

cell_boundary_moment_xy:
  容易被重新分配的 workload 相对该 UAV 的方向。

cell_weak_link_moment_xy:
  链路弱且 workload 高的 GU 相对该 UAV 的方向。
```

这些 moment 是 `cell_summary[u]` 的字段。ego 使用 `ego_cell = cell_summary[ego]`，peer token 使用 `peer_public_cell_summary = cell_summary[peer]`。网络中 `CellMLP` 和 `PeerMLP` 会编码这些方向量，最终进入 fusion head，作为输出 2D acceleration 的方向信息来源。

## 7. Peer UAV Tokens

peer token 只包含物理安全关系和 peer 的公开 cell summary，不包含 peer queue。

`U == 1` 时 peer token 轴长度为 0：

```text
peer_tokens: [row_count, 0, D_peer]
peer_mask:   [row_count, 0]
```

这种空轴是固定 ABI 的合法形状。Python forward、native history buffer、CUDA actor kernel 都必须接受该形状，并让 peer attention / mean / max 输出 zero vector。

对 ego i 和 peer j：

```text
rel_peer_pos = p_i - p_j
rel_peer_vel = v_i - v_j
```

这个方向是从 peer 指向 ego 的分离方向，必须与 safety layer 和 danger imitation 的方向一致。

字段：

```text
rel_peer_x_norm = (x_i - x_j) / map_ref
rel_peer_y_norm = (y_i - y_j) / map_ref
rel_peer_vx_norm = (vx_i - vx_j) / vel_ref
rel_peer_vy_norm = (vy_i - vy_j) / vel_ref

peer_dist_norm = D_ij / map_ref

closing_speed_norm =
  if D_ij > 0:
    max(0, - dot(rel_peer_pos, rel_peer_vel) / (D_ij * vel_ref))
  else:
    0

safe_distance_margin_norm =
  (D_ij - d_safe) / map_ref

unsafe_flag =
  1[D_ij < d_safe]

alert_flag =
  1[D_ij < avoidance_alert_factor * d_safe]

last_shared_sat_frac =
  sum_s last_selected_mask_by_uav_sat[i, s]
        * last_selected_mask_by_uav_sat[j, s]
  / sat_select_ref_count

peer_public_cell_summary
```

`closing_speed_norm > 0` 表示 ego 和 peer 正在接近。  
`safe_distance_margin_norm < 0` 表示已经低于安全距离。
`last_shared_sat_frac` 表示上一拍 ego 与该 peer 选择同一 SAT 的比例，来自公开执行历史，不包含 peer queue。

## 8. SAT / Backhaul Tokens

accel actor 不选择 SAT，但 UAV 位置会影响可见 SAT、doppler、elevation、backhaul SE。accel actor 使用 ego 当前可见/候选 SAT tokens，不使用 top1/topk 摘要。

SAT token 轴固定为每个 ego UAV 当前可见/候选卫星的本地上限：

```text
S = cfg.per_uav_visible_sat_token_max
```

`cfg.per_uav_visible_sat_token_max` 在 config finalize 阶段确定为正数。每个 ego UAV 只填自己当前的可见/候选 SAT，按环境已有 candidate order 排序，不足 padding。

### 8.1 SAT node 字段

```text
sat_x_norm = sat_ecef_x / orbit_pos_ref
sat_y_norm = sat_ecef_y / orbit_pos_ref
sat_z_norm = sat_ecef_z / orbit_pos_ref

sat_vx_norm = sat_vel_x / sat_vel_ref
sat_vy_norm = sat_vel_y / sat_vel_ref
sat_vz_norm = sat_vel_z / sat_vel_ref

sat_queue_steps
sat_queue_fill
sat_last_incoming_steps
sat_last_processed_steps
sat_last_drop_steps
sat_service_ema_steps
sat_cost_log_ratio
sat_last_workload_log1p

sat_last_selected_load_frac =
  number_of_UAVs_that_selected_this_SAT_last_step / num_uav

sat_proc_capacity_steps =
  env._effective_sat_cpu_freq() / task_cycles_per_bit * tau0 / sat_flow_ref
```

`sat_last_selected_load_frac` 是上一拍真实历史负载，不是当前 prefix。  
`sat_proc_capacity_steps` 表示该 SAT 每步能处理多少个 sat flow ref。若所有 SAT 同构，它是常量字段，但仍保留，因为 reward 中 SAT processing 是系统 workload 的一部分。
native 路径中该值使用现有 `_effective_sat_cpu_freq_from_cfg(cfg)` 计算，与 Python 的 `env._effective_sat_cpu_freq()` 保持一致。

### 8.2 Ego-SAT edge 字段

```text
us_rel_x_norm = (sat_ecef_x - ego_ecef_x) / orbit_pos_ref
us_rel_y_norm = (sat_ecef_y - ego_ecef_y) / orbit_pos_ref
us_rel_z_norm = (sat_ecef_z - ego_ecef_z) / orbit_pos_ref

us_rel_vx_norm = (sat_vel_x - ego_vel_ecef_x) / sat_vel_ref
us_rel_vy_norm = (sat_vel_y - ego_vel_ecef_y) / sat_vel_ref
us_rel_vz_norm = (sat_vel_z - ego_vel_ecef_z) / sat_vel_ref

us_range_norm =
  ||sat_ecef - ego_ecef|| / orbit_pos_ref

us_radial_velocity_norm =
  dot(us_rel_pos, us_rel_vel)
  / (||us_rel_pos|| * sat_vel_ref)

us_elevation_norm =
  elevation_rad / (pi / 2)

raw_doppler_hz =
  backhaul_carrier_freq
  / speed_of_light
  * dot(us_rel_pos, us_rel_vel)
  / ||us_rel_pos||

doppler_hz =
  _effective_doppler_array(
    ego,
    sat,
    raw_doppler_hz
  ) 的第一个返回值 nu_eff

us_doppler_ratio =
  if doppler_enabled or doppler_atten_enabled or doppler_observed:
    doppler_hz / nu_max
  else:
    0

us_doppler_abs_ratio =
  abs(us_doppler_ratio)

backhaul_noise_ref =
  noise_density * effective_b_backhaul_per_sat * backhaul_noise_figure_linear

backhaul_noise_figure_linear =
  channel.noise_figure_linear(backhaul_noise_figure_db)

backhaul_free_space_gain =
  (speed_of_light / (4 * pi * backhaul_carrier_freq))^2
  / ||us_rel_pos||^2

backhaul_loss_factor =
  所有已启用 atmospheric/rain loss 的线性增益因子乘积；
  若没有启用额外 backhaul loss，则为 1

backhaul_gain_ego_sat =
  backhaul_free_space_gain * backhaul_loss_factor

backhaul_snr_ref =
  uav_tx_power * backhaul_gain_ego_sat / backhaul_noise_ref

backhaul_snr_ref_for_se =
  if doppler_atten_enabled:
    backhaul_snr_ref * doppler_attenuation(doppler_hz, subcarrier_spacing)
  else:
    backhaul_snr_ref

us_backhaul_se_ref =
  log2(1 + backhaul_snr_ref_for_se)

us_visible_flag
us_valid_flag =
  1[
    elevation_rad >= theta_min_rad
    and (
      not doppler_enabled
      or abs(doppler_hz) <= nu_max
    )
  ]
us_last_selected_flag
us_last_outflow_steps =
  last_uav_to_sat_outflow_matrix[ego, sat] / uav_flow_ref
```

`||us_rel_pos||` 必须大于 0；SAT 和 UAV 不可能物理重合。若运行时出现 0，说明状态非法，应报错而不是用 eps 静默兜底。

`backhaul_gain_ego_sat` 与环境 `_compute_backhaul_rates()` 使用同一几何和损耗口径，包括自由空间距离项、atmospheric/rain loss 以及当前配置启用的其他 backhaul loss。
`doppler_hz` 使用环境 `_effective_doppler_array()` 之后的 effective Doppler，而不是 raw orbital Doppler。
native 路径中 `effective_b_backhaul_per_sat` 使用现有 `_effective_b_backhaul_per_sat_from_cfg(cfg)` 计算，与 Python 的 `env._effective_b_backhaul_per_sat()` 保持一致。

若 Doppler 启用、参与衰减或被观测，`nu_max` 必须为正；否则配置非法。Doppler 完全不参与环境和观测时，`us_doppler_ratio` 和 `us_doppler_abs_ratio` 都填 0。

`us_backhaul_se_ref` 使用当前几何链路和参考 per-SAT 带宽计算，不包含当前 SAT selection load。上一拍 ego 到该 SAT 的真实传输历史由 `us_last_selected_flag` 和 `us_last_outflow_steps` 表达；它们不能替代 `us_backhaul_se_ref`，因为上一拍是否选择该 SAT 不等于当前几何下是否仍然是好 backhaul opportunity。

accel actor 不输入：

```text
prefix_selected_sat
prefix_selected_load
prefix_backhaul_capacity
SAT top-k summary scalars
SAT selected-still-valid summary scalars
```

这些要么是当前 prefix 未知，要么是薄摘要，会丢掉 SAT token 结构。

## 9. 网络结构

网络结构固定如下。

### 9.1 Embedding 尺寸

```text
token_embed_dim = actor_embed_dim
hidden_dim      = actor_hidden_dim

本 redesign 的 structured actor 默认值：
actor_embed_dim  = 128
actor_hidden_dim = 256
```

当前 native actor ABI 使用共享的 `kActorHidden` 和 `kActorEmbed`，因此 accel、SAT、BW 三个 actor 子模块使用同一组 hidden/embed 尺寸。本 redesign 不拆分 per-head hidden/embed。

所有字段已经按物理量纲显式归一化。网络内部仍然使用 trainable LayerNorm，作用是稳定不同字段组合后的优化尺度，不替代前面的物理归一化。native actor kernel 已支持 LayerNorm，因此这不是单 GPU 架构障碍。

```text
ego_emb  = EgoMLP(LN(ego_features))
cell_emb = CellMLP(LN(ego_cell))
gu_emb   = GuMLP(LN(gu_token))
peer_emb = PeerMLP(LN(peer_token))
sat_emb  = SatMLP(LN(sat_token))
```

### 9.2 GU 聚合

GU 聚合使用 4 个 ego-and-cell-conditioned learned query slots：

```text
q_gu = Linear_gu_query([ego_emb, cell_emb]) -> [4, token_embed_dim]
```

这些 query slots 不做硬语义绑定。网络没有机制保证第 0 个 query 永远对应某个固定概念，因此文档不规定 query 与功能的一一对应关系。使用 4 个 slots 的原因是 accel actor 同时需要从 GU tokens 中读取多种互补信息：

```text
own-cell overload / release pressure
underloaded takeover opportunity
boundary-sensitive workload
weak-link workload
interference-sensitive workload
```

聚合结果：

```text
gu_attn_ctx[4] = masked_attention(q_gu, gu_emb, gu_mask)
gu_mean_ctx    = masked_mean(gu_emb, gu_mask)
gu_max_ctx     = masked_max(gu_emb, gu_mask)
```

同时使用 attention、mean、max 的原因是：

```text
attention 捕捉最关键 GU。
mean 捕捉整体需求分布。
max 捕捉极端 backlog / drop / 边界风险。
```

### 9.3 Peer 聚合

Peer 聚合使用 2 个 ego-and-cell-conditioned queries：

```text
q_peer = Linear_peer_query([ego_emb, cell_emb]) -> [2, token_embed_dim]

peer_attn_ctx[2] = masked_attention(q_peer, peer_emb, peer_mask)
peer_mean_ctx    = masked_mean(peer_emb, peer_mask)
peer_max_ctx     = masked_max(peer_emb, peer_mask)
```

这两个 peer query slots 不做硬语义绑定。使用 2 个 slots 的原因是 peer tokens 主要包含两类互补信息：

```text
safety / collision kinematics
public cell-pressure coordination
```

peer token 已包含安全方向、closing speed、safe margin、peer public cell summary，因此不再引入任何 peer 私有 queue。

### 9.4 SAT 聚合

SAT 聚合使用 2 个 ego-and-cell-conditioned queries：

```text
q_sat = Linear_sat_query([ego_emb, cell_emb]) -> [2, token_embed_dim]

sat_attn_ctx[2] = masked_attention(q_sat, sat_emb, sat_mask)
sat_mean_ctx    = masked_mean(sat_emb, sat_mask)
sat_max_ctx     = masked_max(sat_emb, sat_mask)
```

这两个 SAT query slots 不做硬语义绑定。使用 2 个 slots 的原因是 SAT tokens 主要包含两类互补信息：

```text
current geometric backhaul opportunity
historical SAT workload / last outflow risk
```

SAT 只表达当前 ego 的 backhaul opportunity 和上一拍 SAT workload 历史，不表达当前未选择的 prefix capacity。

### 9.5 Fusion 和 action head

拼接：

```text
z = [
  ego_emb,
  cell_emb,
  gu_attn_ctx[0..3],
  gu_mean_ctx,
  gu_max_ctx,
  peer_attn_ctx[0..1],
  peer_mean_ctx,
  peer_max_ctx,
  sat_attn_ctx[0..1],
  sat_mean_ctx,
  sat_max_ctx
]
```

`cell_emb` 是 ego 自己 cell 的信息入口；peer 的 cell 信息在 `peer_emb` 内。GU/SAT/peer token 中的 signed relative vectors，以及 `cell_*_moment_xy`，保留了把结构化信息映射到二维加速度所需的方向坐标。attention / mean / max 只做信息读取，不产生手工动作方向；最终 2D action 完全由 `FusionMLP + Linear -> R^2` 学出来。

action head：

```text
h = FusionMLP(z)
mean_raw = Linear(h) -> R^2
action_mean_norm = tanh(mean_raw)
```

策略分布：

```text
log_std = learnable parameter with shape [2]
std = exp(clamp(log_std, -5, 2))
z_action ~ Normal(mean_raw, std)
action_norm = tanh(z_action)
actor output = action_norm
env physical policy accel = clip(action_norm, -1, 1) * accel_ref
```

不使用 state-dependent std。探索噪声只由全局 2 维 `log_std` 控制。

所有 masked aggregation 的空 mask 行为固定：

```text
masked_attention(q, tokens, mask):
  score_k = dot(q, token_k) / sqrt(token_embed_dim)
  alpha = softmax(score over mask==1)
  return sum_k alpha_k * token_k

masked_mean(tokens, mask):
  return sum_{k: mask_k=1} token_k / count(mask)

masked_max(tokens, mask):
  return feature-wise max over k with mask_k=1

masked_attention(no valid token) = zero vector
masked_mean(no valid token)      = zero vector
masked_max(no valid token)       = zero vector
```

这用于 `U=1` 时的 peer token 轴，以及当前 ego 没有可见 SAT 时的 SAT token 轴。GU token 轴要求 `G>0`，不会为空。

## 10. Safety Layer 和 Danger Imitation 对齐

peer 方向固定：

```text
rel_peer_pos = ego_pos - peer_pos
```

这与 pairwise hard filter 对 ego 的分离方向一致。训练历史中的 danger imitation target 使用该 rollout 样本当前 step 经过 safety layer 之后的执行动作：

```text
danger_imitation_target = exec_accel_after_safety / accel_ref
```

注意区分两个量：

```text
actor input 中的 last_exec_accel:
  observation 时可见的上一拍执行加速度历史。

danger_imitation_target:
  rollout 当前 step 执行完 safety layer 后写入 history 的监督目标。
```

训练时：

```text
PPO loss:
  使用 policy action 的 logprob。

danger imitation loss:
  在 danger mask 激活的 UAV 上，让 tanh(mean_raw) 接近 target_accel_norm。
```

必须记录：

```text
intervention_rate
intervention_l2_mean
danger_imitation_active_rate
pairwise_hard_adjust_rate
boundary_hard_adjust_rate
```

这些统计不进入 actor 输入，除非它们已经是上一拍 per-UAV 执行历史的一部分。

## 11. 单 GPU Native Kernel 架构约束

accel actor state 必须能在单 GPU native rollout 中直接构建，不能依赖 Python 端动态对象。

### 11.1 固定 buffer 形状

native live accel observation 的物理 ABI 使用展平 row 维：

```text
row = env_id * U + ego_id
row_count = B * U

ego_features:      [row_count, D_ego]
ego_cell:          [row_count, D_cell]
gu_tokens:         [row_count, G, D_gu]
gu_mask:           [row_count, G]
peer_tokens:       [row_count, U - 1, D_peer]
peer_mask:         [row_count, U - 1]
sat_tokens:        [row_count, S, D_sat]
sat_mask:          [row_count, S]
```

训练历史也保存同样的 row 展平 actor-local tensors；critic 的 centralized `world_batch` 保持原来的 `[B, ...]` 语义。

PyTorch actor forward 使用同一 dataclass ABI：

```text
LocalAccelState(
  ego_features,
  ego_cell,
  gu_tokens,
  gu_mask,
  peer_tokens,
  peer_mask,
  sat_tokens,
  sat_mask,
)
```

### 11.2 native 构建顺序

在 accel stage 开始时，native kernel 按固定顺序构建：

```text
1. 读取 UAV/GU/SAT 当前状态和上一拍执行历史。
2. 计算 D_ug、access gain、owner(g)、D_second_g；owner 使用二维水平距离平方 argmin，access gain 不参与 owner 判定。
3. 计算 per-GU partition features。
4. 计算 per-cell public summary。
5. 为每个 ego 填 ego_features 和 ego_cell。
6. 为每个 ego 填完整 G 个 GU token。
7. 为每个 ego 按固定 off-diagonal 顺序填 peer token。
8. 为每个 ego 填 S 个 SAT candidate token。
```

off-diagonal peer 顺序固定：

```text
peer slot order for ego i:
  [0, 1, ..., i - 1, i + 1, ..., U - 1]
```

SAT candidate 顺序固定使用环境已有 `_visible_sats_sorted` 的排序，截断到 `S` 并 padding。

`U == 1` 时 peer 写入阶段不写任何 peer token，peer mask 轴长度为 0。kernel 中对应循环的 trip count 为 0，不能为了补 peer 空轴另起 Python 分支或额外 tensor。

### 11.3 schema 单一来源

定义独立文件：

```text
sagin_marl/rl/structured_accel_actor_schema.py
```

其中包含：

```text
ACCEL_EGO_DIM
ACCEL_CELL_DIM
ACCEL_GU_TOKEN_DIM
ACCEL_PEER_TOKEN_DIM
ACCEL_SAT_TOKEN_DIM

字段 index 常量
归一化 reference 名称
```

CUDA header 中的字段 enum 与这份 schema 保持一一对应，并由 schema parity test 校验。Python builder、native builder、history buffer、actor forward 的维度必须完全一致。

### 11.4 actor inference kernel

单 GPU native rollout 下，accel policy 不能回退到 Python actor callback。`exec_accel_source=policy` 时必须由 native actor kernel 完成 forward、采样、logprob 写回和 history 写回。

因此本 redesign 需要改造：

```text
native accel live obs builder
actor weight pack ABI
actor_accel_live_kernel
accel training history buffers
Python/native parity tests
```

`actor_accel_live_kernel` 需要支持：

```text
ego/cell/GU/peer/SAT LayerNorm
typed MLP encoders
fixed-count masked attention
masked mean
masked max
fusion MLP
2D Gaussian sampling with tanh squash
squashed logprob
```

这些都在现有 native actor path 内实现，不新开 Python 推理路径。

### 11.5 native hot path 调度约束

单 GPU native rollout 的热路径必须保持固定的少量 kernel launch，不把本 redesign 拆成 Python 端小 op 或每类 token 一个小 kernel。

accel stage 的热路径固定为：

```text
1. env/native stage kernel 写 accel live obs buffer。
2. actor_accel_live_kernel 对 row_count=B*U 的所有 ego UAV 做 policy forward、采样、logprob 和 action 写回。
3. 后续 env/native stage kernel 消费 live_accel_action。
```

其中：

```text
LayerNorm
MLP
query projection
masked attention
masked mean
masked max
fusion
Gaussian sampling
squashed logprob
```

都是 `actor_accel_live_kernel` 内的 device/helper 逻辑，不允许分别启动独立 CUDA kernel。obs 构建同理，GU/cell/peer/SAT 字段在同一个 native obs 构建段里写入，不允许按字段或按 token 类型反复调 Python 调度。

Python 端只负责：

```text
固定形状 buffer 初始化
权重打包
kernel launch 编排
parity/reference test
```

rollout step 内不能用 Python builder、PyTorch attention、PyTorch LayerNorm、PyTorch gather/scatter 小 op 代替 native hot path。CUDA Graph capture 需要看到固定 buffer、固定 kernel 序列和固定 ABI；本 redesign 的 token 轴固定为 `G`、`U-1`、`S`，满足这个约束。

### 11.6 native parity test

每次 schema 变化必须通过：

```text
Python accel local state builder
vs
native accel live obs builder
```

逐字段 parity test。比较内容包括：

```text
shape
mask
字段顺序
归一化尺度
owner / margin / cell summary
interference fields
SAT candidate order
peer direction
```

## 12. 设计排除项

accel actor 不包含：

```text
post-accel association
post-accel bw_valid
current selected SAT
current selected SAT load
prefix backhaul capacity
peer UAV 私有 queue / drop / inflow / outflow
top1/topk SAT 摘要
手工方向项
state-dependent std
```

accel actor 包含：

```text
当前 GU 原始 workload
当前 GU-UAV 几何分区
平滑 cell workload share / boundary workload / weak-link workload
历史 access interference
ego 自身 UAV relay 状态
peer 物理安全状态和 peer public cell summary
ego 当前 SAT/backhaul token
上一拍 policy/exec accel 干预历史
```

## 13. 关键场景下的行为要求

### 13.1 所有 GU 初始归同一 UAV

若 `owner(g) = 0` 对所有 GU 成立：

```text
UAV 0:
  cell_gu_count_frac 接近 1
  cell_queue_steps_sum 高
  cell_workload_share_gap > 0
  cell_access_pressure 高

其他 UAV:
  cell_gu_count_frac 接近 0
  cell_workload_share_gap < 0
  peer token 中能看到 UAV 0 的 public cell overload
  GU token 中能看到可接管 GU 的 ego_takeover_gap 和 owner_stability_margin
```

actor 不能被 `pre_owner_is_ego` 诱导成“UAV 0 守住所有 GU”。它应该从 cell workload share gap、boundary workload、peer public cell summary 中学习把部分 GU 重新分配给空闲 UAV。

### 13.2 高 interference 场景

若上一拍某些 GU 对 ego 造成强非自身小区干扰：

```text
ego uav_last_access_interference_log1p 高
相关 GU token 的 ug_last_nonself_interference_log1p 高
相关 cell_interference_exposure 高
```

actor 通过移动改变几何 owner 和 cross-gain，降低后续 access interference。

### 13.3 安全接近场景

若 ego 与 peer 接近：

```text
peer_dist_norm 低
closing_speed_norm 高
safe_distance_margin_norm 接近 0 或为负
unsafe_flag / alert_flag 反映安全状态
rel_peer_pos 与 safety layer 分离方向一致
```

danger imitation 的 target 与输入方向一致，actor 学到的是提前避让，而不是依赖 safety layer 事后修正。

## 14. 最终规格摘要

accel actor 固定采用：

```text
独立 actor schema
完整 GU tokens
公开 cell summary
peer physical + public cell tokens
ego SAT/backhaul tokens
interference-aware history features
typed MLP encoders
GU 4-query attention + mean + max
Peer 2-query attention + mean + max
SAT 2-query attention + mean + max
global 2D log_std
single-GPU native fixed-shape ABI
```

它的第一性任务是：

```text
通过 UAV 加速度塑造负载合理、链路可服务、安全可执行的 GU-UAV 几何分区，从而提高长期系统 reward。
```

## 15. 现有代码改造规格

本节按当前代码结构写出落地改法。实现时以本节为准，避免 Python actor、native rollout、CUDA actor kernel 和训练历史之间出现不同 schema。

### 15.1 新增 accel actor schema 文件

新增文件：

```text
sagin_marl/rl/structured_accel_actor_schema.py
```

这个文件是 accel actor local state 的单一来源。不要继续让 accel actor 复用 `structured_critic_schema.py` 的 `CRITIC_*_DIM`。

文件中定义：

```text
ACCEL_EGO_DIM = 27
ACCEL_CELL_DIM = 18
ACCEL_GU_TOKEN_DIM = 27
ACCEL_PEER_TOKEN_DIM = 10 + ACCEL_CELL_DIM = 28
ACCEL_SAT_TOKEN_DIM = 32

ACCEL_GU_QUERY_COUNT = 4
ACCEL_PEER_QUERY_COUNT = 2
ACCEL_SAT_QUERY_COUNT = 2
```

并为每个字段定义 index 常量。字段顺序必须与 Python builder、native builder、CUDA kernel 完全一致。

#### 15.1.1 Ego 字段

```text
EGO_X
EGO_Y
EGO_VX
EGO_VY
EGO_SPEED
EGO_ENERGY
EGO_BOUNDARY_LEFT
EGO_BOUNDARY_RIGHT
EGO_BOUNDARY_BOTTOM
EGO_BOUNDARY_TOP

EGO_UAV_QUEUE_STEPS
EGO_UAV_QUEUE_FILL
EGO_UAV_LAST_INFLOW_STEPS
EGO_UAV_LAST_OUTFLOW_STEPS
EGO_UAV_LAST_DROP_STEPS
EGO_UAV_SERVICE_EMA_STEPS
EGO_UAV_LOCAL_COST_LOG_RATIO
EGO_UAV_LAST_TOTAL_COST_LOG_RATIO
EGO_UAV_LAST_WORKLOAD_LOG1P
EGO_UAV_LAST_ACCESS_INTERFERENCE_LOG1P

EGO_LAST_POLICY_ACCEL_X
EGO_LAST_POLICY_ACCEL_Y
EGO_LAST_EXEC_ACCEL_X
EGO_LAST_EXEC_ACCEL_Y
EGO_LAST_INTERVENTION_DX
EGO_LAST_INTERVENTION_DY
EGO_LAST_INTERVENTION_L2
```

#### 15.1.2 Cell summary 字段

```text
CELL_GU_COUNT_FRAC
CELL_QUEUE_STEPS_SUM
CELL_EXPECTED_ARRIVAL_STEPS_SUM
CELL_LAST_ARRIVAL_STEPS_SUM
CELL_LAST_OUTFLOW_STEPS_SUM
CELL_LAST_DROP_STEPS_SUM
CELL_LAST_WORKLOAD_LOG1P_SUM
CELL_WORKLOAD_SHARE_GAP
CELL_BOUNDARY_WORKLOAD_SUM
CELL_WEAK_LINK_WORKLOAD_SUM
CELL_ACCESS_PRESSURE
CELL_INTERFERENCE_EXPOSURE

CELL_DEMAND_MOMENT_X
CELL_DEMAND_MOMENT_Y
CELL_BOUNDARY_MOMENT_X
CELL_BOUNDARY_MOMENT_Y
CELL_WEAK_LINK_MOMENT_X
CELL_WEAK_LINK_MOMENT_Y
```

#### 15.1.3 GU token 字段

```text
GU_X
GU_Y
GU_QUEUE_STEPS
GU_QUEUE_FILL
GU_EXPECTED_ARRIVAL_STEPS
GU_LAST_ARRIVAL_STEPS
GU_LAST_OUTFLOW_STEPS
GU_LAST_DROP_STEPS
GU_SERVICE_EMA_STEPS
GU_LOCAL_COST_LOG_RATIO
GU_LAST_TOTAL_COST_LOG_RATIO
GU_LAST_WORKLOAD_LOG1P

GU_REL_X
GU_REL_Y
GU_DIST
GU_ACCESS_SE_REF

GU_LAST_ASSOC_TO_EGO
GU_LAST_BW_FRACTION_EGO
GU_LAST_SERVED_BY_EGO

GU_PRE_OWNER_IS_EGO
GU_HANDOFF_MARGIN_EGO
GU_OWNER_STABILITY_MARGIN
GU_EGO_TAKEOVER_GAP

GU_PARTITION_BOUNDARY_WEIGHT
GU_EGO_LINK_WEAKNESS
GU_LAST_BW_SUM
GU_LAST_NONSELF_INTERFERENCE_LOG1P
```

#### 15.1.4 Peer token 字段

Peer token 取代当前 `nbr_uavs + nbr_edges` 的组合输入。peer token 里不放 peer UAV 私有 queue。

```text
PEER_REL_X
PEER_REL_Y
PEER_REL_VX
PEER_REL_VY
PEER_DIST
PEER_CLOSING_SPEED
PEER_SAFE_DISTANCE_MARGIN
PEER_UNSAFE_FLAG
PEER_ALERT_FLAG
PEER_LAST_SHARED_SAT_FRAC

PEER_CELL_*  # 完整追加 ACCEL_CELL_DIM 个 public cell summary 字段
```

`PEER_CELL_*` 的顺序必须与 `CELL_*` 完全一致。

#### 15.1.5 SAT token 字段

SAT token 合并当前 `sat_nodes + sat_edges`。每个 ego UAV 使用自己的可见/候选 SAT token 轴。

```text
SAT_X
SAT_Y
SAT_Z
SAT_VX
SAT_VY
SAT_VZ
SAT_QUEUE_STEPS
SAT_QUEUE_FILL
SAT_LAST_INCOMING_STEPS
SAT_LAST_PROCESSED_STEPS
SAT_LAST_DROP_STEPS
SAT_SERVICE_EMA_STEPS
SAT_COST_LOG_RATIO
SAT_LAST_WORKLOAD_LOG1P
SAT_LAST_SELECTED_LOAD_FRAC
SAT_PROC_CAPACITY_STEPS

SAT_REL_X
SAT_REL_Y
SAT_REL_Z
SAT_REL_VX
SAT_REL_VY
SAT_REL_VZ
SAT_RANGE
SAT_RADIAL_VELOCITY
SAT_ELEVATION
SAT_DOPPLER_RATIO
SAT_DOPPLER_ABS_RATIO
SAT_BACKHAUL_SE_REF
SAT_VISIBLE_FLAG
SAT_VALID_FLAG
SAT_LAST_SELECTED_FLAG
SAT_LAST_OUTFLOW_STEPS
```

### 15.2 修改 LocalAccelState dataclass

修改文件：

```text
sagin_marl/rl/structured_types.py
```

当前：

```python
class LocalAccelState:
    ego_uav
    nbr_uavs
    nbr_edges
    nbr_mask
    gu_nodes
    gu_edges
    gu_mask
    sat_nodes
    sat_edges
    sat_mask
```

改为：

```python
@dataclass
class LocalAccelState:
    ego_features: torch.Tensor        # [B*U, ACCEL_EGO_DIM]
    ego_cell: torch.Tensor            # [B*U, ACCEL_CELL_DIM]
    gu_tokens: torch.Tensor           # [B*U, G, ACCEL_GU_TOKEN_DIM]
    gu_mask: torch.Tensor             # [B*U, G]
    peer_tokens: torch.Tensor         # [B*U, U-1, ACCEL_PEER_TOKEN_DIM]
    peer_mask: torch.Tensor           # [B*U, U-1]
    sat_tokens: torch.Tensor          # [B*U, S, ACCEL_SAT_TOKEN_DIM]
    sat_mask: torch.Tensor            # [B*U, S]
```

旧字段名不要保留为 alias。否则后续容易出现旧 actor 和新 actor 混用。

同步修改所有读取 `LocalAccelState` 的代码：

```text
sagin_marl/rl/structured_actor.py
sagin_marl/rl/structured_stage_builders.py
sagin_marl/rl/structured_parallel_eval.py
sagin_marl/rl/structured_mappo.py
sagin_marl/env/structured_batch_env_core.py
sagin_marl/env/structured_gpu_rollout_runtime.py
```

### 15.3 修改 Python local accel builder

修改文件：

```text
sagin_marl/rl/structured_stage_builders.py
```

当前 `build_batched_local_accel_states_from_world(ws)` 只是把 `StructuredWorldState` 的 critic schema 切成 ego/nbr/gu/sat。新设计不能继续这样做，因为 accel actor schema 独立于 critic。

在当前代码结构下，改造路径固定为：

```text
1. 保留 StructuredWorldState 给 critic。
2. 在 StructuredControlDriver._prepare_accel_stage_spec() 中一次性生成 accel stage spec。
3. begin_step() 用同一份 spec 构造 critic StructuredWorldState。
4. build_local_accel_states() 用同一份 spec 构造新 LocalAccelState。
```

具体改动：

```text
sagin_marl/env/structured_driver.py
  新增 self._accel_stage_spec_cache。
  新增 _build_accel_actor_local_state(stage spec)。
  begin_step() 调用 _prepare_accel_stage_spec() 后缓存 spec，再构造 critic world。
  build_local_accel_states() 改为调用该函数，而不是从 critic world slice。
  _prepare_accel_stage_spec() 需要把 access_gain_matrix 放入 spec。
  run_accel_stage() 开始执行动作前清空 _accel_stage_spec_cache。

sagin_marl/rl/structured_stage_builders.py
  删除 build_batched_local_accel_states_from_world。
  所有 accel actor 输入都从 accel stage spec 构造，不能从 critic schema 派生。
```

训练历史保存新 actor-local tensors；native history 仍保留 critic 需要的 `world_batch`。actor-local tensors 不从 critic world 反推。

### 15.4 Python accel builder 的计算顺序

Python builder 与 native builder 必须按同一顺序计算。伪代码如下：

```python
def build_accel_local_state(env, spec):
    cfg = env.cfg
    U = int(cfg.num_uav)
    G = int(cfg.num_gu)
    S = int(cfg.per_uav_visible_sat_token_max)
    require cfg.num_uav > 0
    require cfg.num_gu > 0
    require cfg.per_uav_visible_sat_token_max > 0

    refs = compute_flow_cost_refs(env, cfg)
    map_ref = float(cfg.map_size)
    access_gain = spec["access_gain_matrix"]  # [G, U], stage spec 中固定的一份
    sat_pos, sat_vel = spec["sat_pos"], spec["sat_vel"]
    visible = spec["visible"]  # per-UAV visible/candidate SAT ids
    owner = spec["assoc"]      # [G], current accel-stage owner, same口径 as _associate_users()

    D = pairwise_distance_gu_uav(env.gu_pos, env.uav_pos)  # [G, U]
    require all owner[g] in [0, U)
    D_owner = D[np.arange(G), owner]
    second = min_distance_excluding_owner(D, owner) if U > 1 else D_owner + map_ref

    gu_base = build_gu_base_fields(...)
    cell_summary = build_cell_summary(owner, gu_base, access_gain, last history)

    for ego in range(U):
        row = ego  # batched builder 中 row = env_index * U + ego
        owner_features = build_partition_features(D, owner, second, ego)
        ego_features[row] = build_ego_features(ego, cell_summary)
        ego_cell[row] = cell_summary[ego]
        gu_tokens[row] = build_all_gu_tokens_for_ego(ego, owner_features)
        peer_tokens[row] = build_offdiag_peer_tokens(ego, cell_summary)
        sat_tokens[row] = build_visible_sat_tokens(ego, visible[ego])
```

注意：

```text
owner、cell summary 用 accel stage 动作前几何。
不能用 run_accel_stage 之后的 association。
不能用上一拍 last_association 替代当前 owner。
spec["assoc"] 必须保证每个 GU 有 owner；出现 -1 直接报错。
```

### 15.5 配置项

修改文件：

```text
sagin_marl/env/config.py
```

新增字段，并在 config finalize 阶段写入：

```python
per_uav_visible_sat_token_max: int
```

规则：

```text
per_uav_visible_sat_token_max =
  min(num_sat, visible_sats_max if visible_sats_max is not None else sats_obs_max)

sat_active_ref_count 不新增 config 字段，直接使用现有语义：
  env._bw_weighted_workload_sat_active_ref_count()
```

这些值在 config finalize 时确定并校验为正。直接构造 `SaginConfig(...)` 后进入 `SaginParallelEnv` 的测试/脚本路径，也必须在环境初始化时补齐这个派生字段，保证进入 actor builder 时已经是确定正数。actor builder、native shape spec、CUDA kernel 不再写运行时替代分支。

同时校验：

```text
effective_candidate_k =
  users_obs_max
  if candidate_k is None or candidate_k <= 0
  else min(candidate_k, users_obs_max)

users_obs_max >= num_gu
effective_candidate_k >= num_gu
candidate_mode in {"assoc", "nearest"}
num_sat > 0
N_RF > 0
sat_select_ref_count =
  min(
    num_sat,
    int(N_RF),
    int(sat_num_select)
      if sat_num_select is not None and sat_num_select > 0
      else int(N_RF)
  )
sat_select_ref_count > 0
```

如果直接传入的 `users_obs_max` 或正的 `candidate_k` 小于 `num_gu`，config finalize / env init 必须把它们提升到 `num_gu`，使进入 actor builder/native shape 前的最终配置满足上述不变量。

同时执行 2.1 中列出的所有正值约束；任何一个 reference 非正都直接报错。

accel actor 的 GU token 轴固定为 `G=num_gu`，不使用 `users_obs_max` 截断 GU。BW stage 的候选槽位覆盖所有 GU，保证“几何 owner 决定接入集合”这个假设成立。

同步修改：

```text
sagin_marl/env/sagin_env.py::_associate_users()
```

把当前 pathloss/阈值分支：

```python
best = np.argmin(pl, axis=1)
best_pl = pl[np.arange(num_gu), best]
assoc = np.where(best_pl <= cfg.pl_threshold_db, best, -1).astype(np.int32)
```

改为：

```python
dist2 = np.sum((gu_pos[:, None, :] - uav_pos[None, :, :]) ** 2, axis=2)
best = np.argmin(dist2, axis=1)
assoc = best.astype(np.int32)
```

native/tensor association helper 使用同一口径。`pl_threshold_db` 不再参与 structured accel redesign 的 owner 计算。

### 15.6 修改 structured_factory

修改文件：

```text
sagin_marl/rl/structured_factory.py
```

当前 `build_structured_modules_from_config(..., embed_dim=64, ...)` 的 actor 默认 embed 改为 128：

```python
def build_structured_modules_from_config(
    cfg,
    *,
    hidden_dim: int = 256,
    embed_dim: int = 128,
    ...
):
    ...
```

native actor ABI 仍使用共享 `kActorHidden/kActorEmbed`，所以 SAT/BW actor 也接收这个 `actor_embed`。

当前：

```python
shape = native_module_shape_spec_from_config(cfg)
accel_policy = AccelPolicy(
    ego_dim=shape.uav_node_dim,
    nbr_node_dim=shape.uav_node_dim,
    nbr_edge_dim=shape.uav_uav_edge_dim,
    gu_node_dim=shape.gu_node_dim,
    gu_edge_dim=shape.uav_gu_edge_dim,
    sat_node_dim=shape.sat_node_dim,
    sat_edge_dim=shape.uav_sat_edge_dim,
)
```

改为：

```python
from sagin_marl.rl import structured_accel_actor_schema as accel_schema

accel_policy = AccelPolicy(
    ego_dim=accel_schema.ACCEL_EGO_DIM,
    cell_dim=accel_schema.ACCEL_CELL_DIM,
    gu_token_dim=accel_schema.ACCEL_GU_TOKEN_DIM,
    peer_token_dim=accel_schema.ACCEL_PEER_TOKEN_DIM,
    sat_token_dim=accel_schema.ACCEL_SAT_TOKEN_DIM,
    gu_query_count=accel_schema.ACCEL_GU_QUERY_COUNT,
    peer_query_count=accel_schema.ACCEL_PEER_QUERY_COUNT,
    sat_query_count=accel_schema.ACCEL_SAT_QUERY_COUNT,
    hidden_dim=actor_hidden,
    embed_dim=actor_embed,
    action_scale=1.0,
)
```

SAT 和 BW actor 继续使用 `native_module_shape_spec_from_config(cfg)`，但 accel 不再从 critic dims 取形状。
accel actor 的 LayerNorm 固定启用；`input_norm_enabled` 不控制 accel actor 的这些 LayerNorm。
`action_scale` 固定为 `1.0`，因为现有环境把 actor 输出当作归一化加速度命令，并在 `_apply_uav_dynamics()` 中乘以 `cfg.a_max`。

### 15.7 重写 AccelPolicy

修改文件：

```text
sagin_marl/rl/structured_actor.py
```

当前 `AccelPolicy` 有：

```text
ego_encoder
nbr_encoder
gu_encoder
sat_encoder
query_proj_1 / query_proj_2
nbr_refine / gu_refine / sat_refine
fusion_1 / fusion_2
```

这些结构全部替换为新结构：

```python
class AccelPolicy(nn.Module):
    def __init__(
        self,
        ego_dim,
        cell_dim,
        gu_token_dim,
        peer_token_dim,
        sat_token_dim,
        gu_query_count=4,
        peer_query_count=2,
        sat_query_count=2,
        hidden_dim=256,
        embed_dim=128,
        action_scale=1.0,
    ):
        self.ego_norm = LayerNorm(ego_dim)
        self.cell_norm = LayerNorm(cell_dim)
        self.gu_norm = LayerNorm(gu_token_dim)
        self.peer_norm = LayerNorm(peer_token_dim)
        self.sat_norm = LayerNorm(sat_token_dim)

        self.ego_encoder = mlp(ego_dim, hidden_dim, embed_dim)
        self.cell_encoder = mlp(cell_dim, hidden_dim, embed_dim)
        self.gu_encoder = mlp(gu_token_dim, hidden_dim, embed_dim)
        self.peer_encoder = mlp(peer_token_dim, hidden_dim, embed_dim)
        self.sat_encoder = mlp(sat_token_dim, hidden_dim, embed_dim)

        self.gu_query = nn.Linear(2 * embed_dim, gu_query_count * embed_dim)
        self.peer_query = nn.Linear(2 * embed_dim, peer_query_count * embed_dim)
        self.sat_query = nn.Linear(2 * embed_dim, sat_query_count * embed_dim)

        fusion_dim = (
            2 * embed_dim
            + (gu_query_count + 2) * embed_dim
            + (peer_query_count + 2) * embed_dim
            + (sat_query_count + 2) * embed_dim
        )
        self.fusion = mlp(fusion_dim, hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
```

`_context(local_state)` 计算：

```python
ego_emb = ego_encoder(norm(local_state.ego_features))
cell_emb = cell_encoder(norm(local_state.ego_cell))

gu_emb = gu_encoder(norm(local_state.gu_tokens))
peer_emb = peer_encoder(norm(local_state.peer_tokens))
sat_emb = sat_encoder(norm(local_state.sat_tokens))

query_src = torch.cat([ego_emb, cell_emb], dim=-1)
gu_queries = gu_query(query_src).view(B, 4, E)
peer_queries = peer_query(query_src).view(B, 2, E)
sat_queries = sat_query(query_src).view(B, 2, E)

gu_attn = multi_query_attention(gu_queries, gu_emb, gu_mask)      # [B, 4, E]
peer_attn = multi_query_attention(peer_queries, peer_emb, peer_mask)
sat_attn = multi_query_attention(sat_queries, sat_emb, sat_mask)

z = concat(
    ego_emb,
    cell_emb,
    flatten(gu_attn),
    masked_mean(gu_emb),
    masked_max(gu_emb),
    flatten(peer_attn),
    masked_mean(peer_emb),
    masked_max(peer_emb),
    flatten(sat_attn),
    masked_mean(sat_emb),
    masked_max(sat_emb),
)
```

保留现有 `forward`、`act_into`、`evaluate_actions` 的外部行为：

```text
action: tanh-squashed 2D normalized accel command
logprob: squashed normal logprob
entropy: Normal entropy
mean: raw Gaussian mean
std: global 2D std
```

三个路径都使用同一套 std 口径：

```python
log_std = torch.clamp(self.log_std, -5.0, 2.0)
std = log_std.exp()
```

CUDA `actor_accel_live_kernel` 也必须使用同样 clamp 后的 `log_std` 计算采样和 logprob。

### 15.8 修改 native module shape spec

修改文件：

```text
sagin_marl/env/structured_batch_env_core.py
```

当前 `StructuredNativeModuleShapeSpec` 只有 critic/旧 actor 维度：

```text
uav_node_dim
gu_node_dim
sat_node_dim
uav_gu_edge_dim
uav_sat_edge_dim
uav_uav_edge_dim
...
```

新增 accel 专用维度：

```python
accel_ego_dim: int
accel_cell_dim: int
accel_gu_token_dim: int
accel_peer_token_dim: int
accel_sat_token_dim: int
accel_sat_width: int
accel_gu_query_count: int
accel_peer_query_count: int
accel_sat_query_count: int
```

`native_module_shape_spec_from_config(cfg)` 从 `structured_accel_actor_schema.py` 读取这些值：

```python
accel_ego_dim=accel_schema.ACCEL_EGO_DIM
accel_cell_dim=accel_schema.ACCEL_CELL_DIM
...
accel_sat_width=int(cfg.per_uav_visible_sat_token_max)
```

不要用 `critic_schema.CRITIC_UAV_NODE_DIM` 作为 accel ego dim。

### 15.9 修改 GPU accel obs view

修改文件：

```text
sagin_marl/env/structured_gpu_rollout_runtime.py
```

当前：

```python
class StructuredGpuAccelObsView:
    ego_uav
    nbr_uavs
    nbr_edges
    nbr_mask
    gu_nodes
    gu_edges
    gu_mask
    sat_nodes
    sat_edges
    sat_mask
```

改为：

```python
class StructuredGpuAccelObsView:
    _tensor_fields = (
        "ego_features",
        "ego_cell",
        "gu_tokens",
        "gu_mask",
        "peer_tokens",
        "peer_mask",
        "sat_tokens",
        "sat_mask",
    )
```

构造函数同步改名。所有引用旧字段的代码必须同步修改，不保留旧字段。

同步修改 native CUDA ABI 的 live accel obs tensor 顺序。`structured_batch_env_core.py::_build_native_cuda_runtime_abi(...)` 写入 runtime ABI 的顺序、`sagin_marl/env/native_cuda/kernels.cu` 的 runtime enum、以及 `sagin_marl/env/native_cuda/actor_kernels.cu` 中 `actor_accel_live_kernel` 的读取 offset 必须一致：

```text
float live accel obs order:
  ego_features
  ego_cell
  gu_tokens
  peer_tokens
  sat_tokens

bool live accel obs order:
  gu_mask
  peer_mask
  sat_mask
```

history accel obs 使用同一字段顺序。旧的 `nbr_*`、`gu_nodes/gu_edges`、`sat_nodes/sat_edges` slot 不保留。

### 15.10 修改 native runtime buffer 分配

修改文件：

```text
sagin_marl/env/structured_batch_env_core.py
```

当前分配：

```python
accel_obs_view = StructuredGpuAccelObsView(
    ego_uav=[row_count, uav_node_dim],
    nbr_uavs=[row_count, U-1, uav_node_dim],
    nbr_edges=[row_count, U-1, CRITIC_UAV_UAV_EDGE_DIM],
    gu_nodes=[row_count, G, user_node_dim],
    gu_edges=[row_count, G, CRITIC_UAV_GU_EDGE_DIM],
    sat_nodes=[row_count, sat_obs_width, sat_node_dim],
    sat_edges=[row_count, sat_obs_width, CRITIC_UAV_SAT_EDGE_DIM],
)
```

改为：

```python
accel_obs_view = StructuredGpuAccelObsView(
    ego_features=torch.empty((row_count, ACCEL_EGO_DIM), ...),
    ego_cell=torch.empty((row_count, ACCEL_CELL_DIM), ...),
    gu_tokens=torch.empty((row_count, G, ACCEL_GU_TOKEN_DIM), ...),
    gu_mask=torch.empty((row_count, G), dtype=torch.bool, ...),
    peer_tokens=torch.empty((row_count, U - 1, ACCEL_PEER_TOKEN_DIM), ...),
    peer_mask=torch.empty((row_count, U - 1), dtype=torch.bool, ...),
    sat_tokens=torch.empty((row_count, S, ACCEL_SAT_TOKEN_DIM), ...),
    sat_mask=torch.empty((row_count, S), dtype=torch.bool, ...),
)
```

同时修改 `_obs_accel(...)` 的 ABI 校验字段列表。

### 15.11 修改 native training history buffer

修改文件：

```text
sagin_marl/env/structured_batch_env_core.py
```

当前 `_NativeAccelTrainingHistoryOutBuffers` 字段和旧 obs 一样：

```text
world_batch, ego_uav, nbr_uavs, nbr_edges, nbr_mask, gu_nodes, gu_edges, ...
```

改为保留 `world_batch`，并把 actor-local 部分替换为新 `StructuredGpuAccelObsView` 字段：

```text
world_batch
ego_features
ego_cell
gu_tokens
gu_mask
peer_tokens
peer_mask
sat_tokens
sat_mask
danger_imitation_targets
danger_imitation_masks
```

在 native live/history buffer 中，`danger_imitation_targets` 和 `danger_imitation_masks` 使用 row 展平形状 `[row_count, 2]`；进入 PPO minibatch cache 后按现有 `structured_buffer.py` 组织成 `[num_samples, num_agents, 2]`。

`runtime.main.accel_history_out` 的绑定也同步更新。

### 15.12 修改 native accel obs 构建 kernel

修改文件：

```text
sagin_marl/env/native_cuda/kernels.cu
sagin_marl/env/structured_batch_env_core.py
```

当前 native 主 kernel 发布 accel obs 时写的是旧 live buffer：

```text
ego_uav
nbr_uavs / nbr_edges
gu_nodes / gu_edges
sat_nodes / sat_edges
```

新实现必须在 accel stage 开始时直接写：

```text
ego_features
ego_cell
gu_tokens
peer_tokens
sat_tokens
```

构建顺序固定：

```text
1. 校验 U > 0, G > 0, S > 0。
2. 计算 D_ug = ||p_g - p_u|| 和 access gain。
3. 对每个 GU 用 `_associate_users()` 同口径按二维水平距离平方 argmin 算 owner，并计算排除 owner 后的 second distance。
4. 读取 stage spec / stage fields 中固定的 access_gain_matrix。
5. 计算 GU base fields：
   queue/arrival/outflow/drop/EMA/cost/workload。
6. 用 access_gain_matrix 计算 ug_access_se_ref。
7. 计算 gu_last_bw_sum。
8. 计算 per-cell summary，包括 moments。
9. 每个 row=(env, ego) 写 ego_features 和 ego_cell。
10. 写 G 个 GU token。
11. 按 off-diagonal 顺序写 U-1 个 peer token。
12. 按 per-UAV visible/candidate order 写 S 个 SAT token。
```

这部分不要从 critic world tensors 拷贝。critic world 是 centralized value 输入，accel obs 是 actor 输入，二者 schema 不同。
`structured_batch_env_core.py` 中现有 `_build_local_accel_components_tensor_impl()` / `_build_local_accel_obs_from_stage_tensor_impl()` 也要同步改成同一 schema，用于 torch tensor reference path、history 写入和 parity reference。

native rollout 热路径不调用 `_build_local_accel_components_tensor_impl()` / `_build_local_accel_obs_from_stage_tensor_impl()`。这两个函数只作为测试 reference 和非 native 调试路径存在。正式 native rollout 的 accel obs 必须由 `sagin_marl/env/native_cuda/kernels.cu` 中的 fused stage kernel 写入 `StructuredGpuAccelObsView`，避免在 step 内产生 PyTorch 小 op 链。

### 15.13 修改 native actor weight pack ABI

修改文件：

```text
sagin_marl/rl/native_actor_cuda.py
sagin_marl/env/native_cuda/actor_kernels.cu
```

当前 `ACTOR_WEIGHT_NAMES` 中 accel 权重是旧结构：

```text
accel_policy.nbr_input_norm.*
accel_policy.query_proj_1.*
accel_policy.query_proj_2.*
accel_policy.nbr_refine.*
accel_policy.gu_refine.*
accel_policy.sat_refine.*
accel_policy.fusion_1.*
accel_policy.fusion_2.*
```

这些全部删除，替换为新结构权重：

```text
accel_policy.log_std
accel_policy.ego_norm.weight / bias
accel_policy.cell_norm.weight / bias
accel_policy.gu_norm.weight / bias
accel_policy.peer_norm.weight / bias
accel_policy.sat_norm.weight / bias

accel_policy.ego_encoder.*
accel_policy.cell_encoder.*
accel_policy.gu_encoder.*
accel_policy.peer_encoder.*
accel_policy.sat_encoder.*

accel_policy.gu_query.weight / bias
accel_policy.peer_query.weight / bias
accel_policy.sat_query.weight / bias

accel_policy.fusion.*
accel_policy.mu_head.weight / bias
```

所有通过 `_make_mlp(in_dim, hidden_dim, out_dim)` 创建的模块，CUDA ABI 中只登记其中两个 Linear 层：

```text
<module>.0.weight / bias
<module>.2.weight / bias
```

因此 `ego_encoder`、`cell_encoder`、`gu_encoder`、`peer_encoder`、`sat_encoder`、`fusion` 的权重名都按这个规则展开。`LayerNorm`、`gu_query`、`peer_query`、`sat_query`、`mu_head` 和 `log_std` 使用上面列出的显式名称。

`actor_kernels.cu` 中 `enum ActorWeightIndex` 的 `W_ACCEL_*` 必须与 `ACTOR_WEIGHT_NAMES` 顺序一一对应。不能只改 Python weight list，不改 CUDA enum。

`build_native_actor_cuda_binding(...)` 当前对缺失权重会放入 empty tensor。新 accel 权重不允许走这个路径：所有 `accel_policy.*` 名称必须在 `state_dict()` 中存在，且 shape 与 CUDA 期望一致；缺失时直接报错。empty tensor 只保留给 SAT/BW 里已有的条件结构。

### 15.14 重写 actor_accel_live_kernel

修改文件：

```text
sagin_marl/env/native_cuda/actor_kernels.cu
```

当前 `actor_accel_live_kernel` 是旧结构：

```text
ego0
nbr/gu/sat tokens
query_proj_1
fusion_1
token refine
query_proj_2
fusion_2
mu_head
```

新 kernel 直接实现新 `AccelPolicy._context`：

```text
1. 读取 row 对应：
   ego_features
   ego_cell
   gu_tokens
   peer_tokens
   sat_tokens

2. 分别 LayerNorm + MLP：
   ego_emb
   cell_emb
   gu_emb[G]
   peer_emb[U-1]
   sat_emb[S]

3. query_src = concat(ego_emb, cell_emb)

4. 生成 query slots：
   gu_queries[4]
   peer_queries[2]
   sat_queries[2]

5. 对每类 token 计算：
   masked attention for each query slot
   masked mean
   masked max

6. fusion input 拼接：
   ego_emb
   cell_emb
   gu_attn[4], gu_mean, gu_max
   peer_attn[2], peer_mean, peer_max
   sat_attn[2], sat_mean, sat_max

7. FusionMLP -> mu_head -> mean_raw[2]

8. 使用现有 tanh-squashed Gaussian：
   log_std_clamped = clamp(log_std, -5, 2)
   action = tanh(mean_raw + exp(log_std_clamped) * normal)
   logprob = squashed_normal_logprob(...)
```

这里写回 `live_accel_action` 的 action 是归一化加速度命令，不乘 `cfg.a_max`。物理加速度缩放仍由环境 dynamics 负责。

CUDA 中需要新增 helper：

```text
block_masked_mean(tokens, mask)
block_masked_max(tokens, mask)
block_multi_query_attention(queries, tokens, mask, query_count)
```

这些 helper 是 `__device__` / inline helper，不是单独的 `__global__` kernel。`actor_accel_live_kernel` 一次 launch 完成 accel actor 的全部 forward 和采样。`U == 1` 时 `peer_count = 0`，peer attention/mean/max 在 kernel 内直接写 zero vector，不启动 peer 专用 kernel。

旧的 `nbr_refine/gu_refine/sat_refine` 不再需要。

### 15.15 修改 actor scratch 大小估计

仍在：

```text
sagin_marl/env/native_cuda/actor_kernels.cu
```

当前 scratch 按旧结构估计：

```text
max_count = max(nbr_width, gu_count, visible)
max_in = max(uav_dim + uu_edge_dim, user_dim + ug_edge_dim, sat_dim + us_edge_dim, ...)
fusion_in = 4 * embed_dim
```

改为按新结构估计：

```text
max_count = max(G, U - 1, S)
max_in = max(
    ACCEL_EGO_DIM,
    ACCEL_CELL_DIM,
    ACCEL_GU_TOKEN_DIM,
    ACCEL_PEER_TOKEN_DIM,
    ACCEL_SAT_TOKEN_DIM,
    2 * embed_dim,
)

fusion_dim =
    2 * embed_dim
  + (ACCEL_GU_QUERY_COUNT + 2) * embed_dim
  + (ACCEL_PEER_QUERY_COUNT + 2) * embed_dim
  + (ACCEL_SAT_QUERY_COUNT + 2) * embed_dim
```

如果当前 `actor_row_scratch` 分配不足，需要同步修改 runtime scratch 分配逻辑。

### 15.16 修改 actor int params

修改文件：

```text
sagin_marl/rl/native_actor_cuda.py
sagin_marl/env/structured_batch_env_core.py
sagin_marl/env/native_cuda/kernels.cu
sagin_marl/env/native_cuda/actor_kernels.cu
```

runtime int params 先新增 accel 专用 SAT 轴宽度：

```text
kParamAccelSatWidth
```

`actor_accel_live_kernel` 读取 SAT token 时使用 `kParamAccelSatWidth`。不能复用 `kParamSatVisibleWidth`，因为 SAT/BW actor 仍使用现有 critic/world SAT token 轴，而 accel actor 使用每个 ego UAV 的本地 `S = cfg.per_uav_visible_sat_token_max`。

actor int params 当前只包含 hidden/embed、BW 架构等。新增：

```text
kActorAccelEgoDim
kActorAccelCellDim
kActorAccelGuTokenDim
kActorAccelPeerTokenDim
kActorAccelSatTokenDim
kActorAccelGuQueryCount
kActorAccelPeerQueryCount
kActorAccelSatQueryCount
```

CUDA kernel 不再从 runtime 的 `kParamUavNodeDim / kParamUserNodeDim / kParamUavGuEdgeDim` 推导 accel actor 输入维度。

### 15.17 修改 structured_mappo 训练历史读取

修改文件：

```text
sagin_marl/rl/structured_mappo.py
```

所有使用：

```text
accel_batch.ego_uav
accel_batch.nbr_uavs
accel_batch.nbr_edges
accel_batch.gu_nodes
accel_batch.gu_edges
accel_batch.sat_nodes
accel_batch.sat_edges
```

的地方改为新字段：

```text
accel_batch.ego_features
accel_batch.ego_cell
accel_batch.gu_tokens
accel_batch.peer_tokens
accel_batch.sat_tokens
```

文件顶部从 `structured_stage_builders` 导入的 `build_batched_local_accel_states_from_world` 也要删除。accel actor 主路径不再从 `StructuredWorldState` 构造 local state。

`_num_agents_from_accel_obs` 当前用 `accel_obs.ego_uav.shape[0]`，改为：

```python
accel_obs.ego_features.shape[0]
```

danger imitation 缓存逻辑不变。native buffer 写入 row 展平的 `[B*U, 2]` target/mask；`structured_buffer.py` 进入训练 batch 时按现有 stage batch 语义整理为 `[num_samples, num_agents, 2]`。

同步修改：

```text
sagin_marl/rl/structured_buffer.py
```

当前 `_flatten_accel_stage_local_batch(...)` 从 native history 读取旧字段：

```text
stage.ego_uav
stage.nbr_uavs
stage.nbr_edges
stage.gu_nodes
stage.gu_edges
stage.sat_nodes
stage.sat_edges
```

改为读取：

```text
stage.ego_features
stage.ego_cell
stage.gu_tokens
stage.peer_tokens
stage.sat_tokens
```

并构造新的 `LocalAccelState`。`_EnvStepBatchRecord.accel_local_batch` 和 `StructuredStageTrainingBatch.local_batch` 类型不需要改名，但其中保存的字段必须是新 schema。

### 15.18 修改 evaluation path

修改文件：

```text
sagin_marl/rl/structured_parallel_eval.py
sagin_marl/rl/structured_eval.py
```

当前 eval 通过：

```python
build_batched_local_accel_states_from_world(accel_world_batch)
```

构造 actor 输入。新设计下，eval 必须按运行路径取新 accel local state：

```text
Python eval:
  driver.begin_step()
  driver.build_local_accel_states()

batched eval:
  collate new LocalAccelState fields

native eval:
  使用 runtime.main.accel_live_obs_buffers[active_idx]
```

不要再从 critic `StructuredWorldState` 反推 accel obs。

### 15.19 修改 Python/native shape parity tests

新增测试文件：

```text
tests/test_structured_accel_actor_schema.py
tests/test_structured_accel_actor_native_parity.py
```

同步更新现有测试和 test utils 中所有旧 `AccelPolicy(...)` 构造：

```text
tests/structured_test_utils.py
tests/test_structured_action_modules.py
tests/test_structured_batch_core_rollout.py
tests/test_structured_mappo_rollout.py
```

必须覆盖：

```text
1. schema dim 与 LocalAccelState tensor shape 一致。
2. Python builder 输出字段顺序与 schema index 一致。
3. native live obs view shape 与 schema dim 一致。
4. Python builder 与 native builder 对同一环境状态逐字段接近。
5. peer token rel_pos = ego_pos - peer_pos。
6. peer offdiag 顺序为 [0..ego-1, ego+1..U-1]。
7. GU owner/margin/cell summary 与纯 numpy reference 一致。
8. interference fields 与环境当前口径一致：
   按上一拍 GU 总 BW fraction 缩放。
9. SAT token order 与 _visible_sats_sorted 一致。
10. accel actor Python forward 与 native actor_accel_live_kernel deterministic action 一致。
```

deterministic action parity 使用：

```text
deterministic=True
比较 live_accel_action 与 tanh(mean_raw)
```

随机采样 parity 不作为必测项，因为 RNG 实现可能不同。

### 15.20 删除旧 accel actor 依赖的标志

实现完成后，全仓库不应再出现 accel actor 主路径依赖：

```text
accel_policy.nbr_*
LocalAccelState.nbr_uavs
LocalAccelState.nbr_edges
build_batched_local_accel_states_from_world 作为 actor 主路径
shape.uav_node_dim 传给 AccelPolicy ego_dim
critic_schema.CRITIC_*_DIM 传给 accel actor
```

允许 critic、sat、bw 继续使用 critic/world schema；禁止 accel actor 继续使用这些维度。

### 15.21 验收命令

改完代码后运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_structured_accel_actor_schema.py
.\.venv\Scripts\python.exe -m pytest tests/test_structured_accel_actor_native_parity.py
.\.venv\Scripts\python.exe -m pytest tests/test_structured_action_modules.py
```

native parity 前必须确保 CUDA extension 已按当前源码重新编译；测试覆盖点不能减少。
