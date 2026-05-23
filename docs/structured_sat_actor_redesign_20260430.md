# Structured SAT Actor 重新设计

日期：2026-04-30

本文定义 `actor sat` 的重新设计规格。目标是把 SAT actor 从当前“基于 critic world 的可见 SAT subset scorer”改成一个语义清晰的、单 ego UAV 调用的、参数共享的多智能体 SAT 连接选择器。

一句话：

```text
SAT actor 是单 ego UAV 的参数共享回传连接选择器。
它在 accel 已执行、BW 未执行之前，为当前 UAV 从当前合法可见 SAT 中选择一个无序 SAT 子集。
它不能偷看当前 SAT selection 之后才产生的 selected load / prefix capacity。
```

## 1. 设计目标

SAT actor 是三阶段 actor 的第二阶段：

```text
accel stage:
  UAV 运动
  -> post-accel UAV/GU 几何
  -> GU -> UAV association / bw_valid

sat stage:
  ego UAV 选择连接 SAT
  -> selected SAT prefix
  -> selected SAT load
  -> ego backhaul capacity

bw stage:
  ego UAV 在当前连接有效 GU 上分配 access bandwidth
  -> GU/UAV/SAT queue、drop、workload
  -> system reward
```

SAT actor 的核心任务不是“选几何最近的卫星”，也不是“选当前 SAT queue 最小的卫星”。它要学习：

```text
在当前 ego UAV 的 relay pressure、候选 SAT 链路质量、SAT workload、
上一拍拥塞历史和多 UAV 共享风险下，
选择哪些 SAT 可以让后续 backhaul 成为更好的下游服务瓶颈。
```

SAT actor 是多智能体参数共享策略：

```text
同一套参数 theta_sat 被所有 UAV 复用。
一次 actor 语义调用只针对一台 ego UAV。
工程上可以 batch 多个 ego row，但不是 centralized actor 一次联合输出所有 UAV 的动作。
```

记为：

```text
pi_sat,theta(o_u) -> selected_sat_set_u
```

其中 `u` 是当前 ego UAV。

## 2. Stage 语义

### 2.1 调用 SAT actor 时已知

SAT actor 调用发生在 accel stage 之后，因此以下 prefix 已固定：

```text
post-accel UAV position / velocity / energy
post-accel GU -> UAV association
post-accel bw_valid_mask
当前 ego UAV 的 visible / candidate SAT list
当前 ego UAV 到各候选 SAT 的几何、elevation、Doppler、validity
当前 GU/UAV/SAT queue
当前 GU expected arrival
当前 EMA service scale / reward cost
上一拍真实执行历史 last_*
```

上一拍历史包括：

```text
last association
last sat selection
last BW fraction
last GU arrival/outflow/drop
last UAV inflow/outflow/drop
last UAV -> SAT outflow matrix
last SAT incoming/processed/drop
last access interference
last selected SAT load
```

### 2.2 调用 SAT actor 时未知

SAT actor 当前正在决定 SAT selection，因此不能把这些量作为输入：

```text
ego current selected SAT
other UAV current selected SAT
current selected SAT load
prefix_backhaul_capacity_steps
BW allocation
BW action 后的 access rate / interference / outflow
```

特别注意：

```text
backhaul_se_ref:
  可以输入。它是在固定 reference bandwidth 下、只依赖当前几何和链路配置的参考链路质量。

prefix_backhaul_capacity_steps:
  不能输入 SAT actor。它需要 current selected load，而 selected load 正是 SAT actor 动作之后才知道的。
```

### 2.3 last 与 prefix 必须分开

`last_*` 是上一拍真实执行结果，所有 stage 都可见。  
`prefix_*` 是当前 stage 已经由本轮动作固定的量。

SAT actor 处于 sat stage：

```text
association prefix known = 1
sat prefix known         = 0
bw prefix known          = 0
```

因此：

```text
last_selected_flag:
  可以作为历史输入。

current_selected_flag / prefix_selected_flag:
  不能作为 SAT actor 输入。

last_selected_load_frac:
  可以作为历史拥塞输入。

current_selected_load_frac:
  不能作为 SAT actor 输入。
```

## 3. 动作定义

### 3.1 选择数量

定义：

```text
sat_action_select_k =
  min(
    num_sat,
    N_RF,
    sat_num_select if sat_num_select is configured and > 0 else N_RF
  )
```

`sat_action_select_k` 必须在配置解析阶段固化。后续 schema、policy、history buffer、native weight ABI、kernel launch 都只使用这个固化值，不再分别读取 `N_RF` / `sat_num_select` 推导不同的 K。

配置必须满足：

```text
num_uav > 0
num_sat > 0
N_RF > 0
sat_action_select_k > 0
per_uav_visible_sat_token_max > 0
fixed_satellite_strategy == false for learned SAT actor
```

如果 `fixed_satellite_strategy == true`，SAT actor 被规则策略绕过，不属于本文定义的 learned SAT actor 路径。训练、PPO history、native actor parity 都按 `fixed_satellite_strategy == false` 的路径实现。

### 3.2 单 ego UAV 动作

对单台 ego UAV：

```text
selected_sat_indices: [sat_action_select_k]
```

约束：

```text
每个元素是 global SAT id 或 -1 padding。
非负 SAT id 必须属于当前 ego 的 visible/candidate list。
非负 SAT id 必须 link-valid。
非负 SAT id 不能重复。

if valid_count > 0:
  选择数量 in [1, min(sat_action_select_k, valid_count)]
else:
  全部为 -1
```

环境执行侧不再改变 SAT actor 动作语义。actor 输出的 SAT set 就是 executed SAT set。环境只做：

```text
finite / dtype check
range check
visibility check
validity check
duplicate check
max select count check
```

超出约束时直接报错或记录 hard diagnostic 并终止该 rollout。不要 fallback 到 nearest SAT，不要补 best SAT，不要做二次 top-k。

### 3.3 无序集合语义

SAT action 是无序集合：

```text
{sat_a, sat_b} == {sat_b, sat_a}
```

因此策略分布建模为 legal subset categorical，而不是 sequential categorical。否则同一个环境动作会被多个顺序编码表示，导致 logprob 和 entropy 引入人为噪声。

输出定义：

```python
@dataclass
class SatSubsetPolicyOutput:
    selected_sat_indices: torch.Tensor  # [R, sat_action_select_k], global SAT id, -1 padding
    subset_index: torch.Tensor          # [R], legal subset index, for PPO history
    subset_members: torch.Tensor        # [R, sat_action_select_k], local candidate slot, -1 padding
    logprob: torch.Tensor               # [R]
    entropy: torch.Tensor               # [R]
    logits: torch.Tensor                # [R, M]
```

其中：

```text
R = batch 展平后的 ego row 数，通常是 B * U
S = per_uav_visible_sat_token_max
K = sat_action_select_k
M = 1 + sum_{c=1..min(K,S)} C(S, c)
```

`M` 包含 empty subset。empty subset 只在 `valid_count == 0` 时合法；有 valid SAT 时必须由 `subset_mask` 屏蔽。

`subset_members` 是 local candidate slot。环境执行前再按 `candidate_sat_ids[row, slot]` 映射为 global SAT id。

`logits [R, M]` 是 Python policy / training evaluate 的自然返回形式。native live rollout 不要求把完整 `[R,M]` logits 持久写入 buffer；只要在 kernel 内完成 sample、logprob、entropy 和 action decode 即可。

## 4. 归一化

SAT actor 的归一化必须和 critic / accel / BW actor 保持同一套 reference。

### 4.1 flow reference

```text
arrival_ref_step = effective_task_arrival_rate * num_gu * tau0

gu_flow_ref  = arrival_ref_step / num_gu
uav_flow_ref = arrival_ref_step / num_uav

sat_workload_ref_count =
  queue_ref_sat_active_count if configured
  else min(num_sat, sat_action_select_k * num_uav)

sat_flow_ref = arrival_ref_step / sat_workload_ref_count
```

所有 queue / arrival / inflow / outflow / drop / EMA 使用对应层的 flow ref：

```text
bits / corresponding_flow_ref
```

容量填充率单独保留：

```text
queue_fill = queue_bits / queue_max
```

不要用 `queue_max_*` 替代 reward-aligned flow scale。

### 4.2 position / velocity reference

```text
map_ref       = cfg.map_size
vel_ref       = cfg.v_max
orbit_pos_ref = cfg.r_earth + cfg.sat_height
mu_earth      = 3.986004418e14
sat_vel_ref   = sqrt(mu_earth / orbit_pos_ref)
```

```text
UAV xy / GU xy / local rel xy: / map_ref
UAV vxy:                       / vel_ref
SAT ECEF pos / rel pos:         / orbit_pos_ref
SAT ECEF vel / rel vel:         / sat_vel_ref
```

### 4.3 Doppler reference

不要优先使用 `nu_max` 作为 Doppler 的唯一归一化。`nu_max` 是 validity threshold，不一定是自然观测尺度。

```text
doppler_ref    = backhaul_carrier_freq * sat_vel_ref / speed_of_light
doppler_norm   = effective_doppler_hz / doppler_ref
doppler_margin = abs(effective_doppler_hz) / nu_max
```

含义：

```text
doppler_norm:
  物理多普勒量级。

doppler_margin:
  距离配置有效性阈值多近，0 表示无 Doppler 压力，1 表示刚到 validity threshold。
```

若 Doppler 完全不参与环境和观测，则两者都填 0。若启用 Doppler，相关配置必须为正，非法配置直接报错。

### 4.4 cost / workload

```text
service_floor = service_floor_bits_per_step

gu_local_cost  = 1 / max(gu_service_ema_bits_per_step, service_floor)
uav_local_cost = 1 / max(uav_service_ema_bits_per_step, service_floor)
sat_cost       = 1 / max(sat_service_ema_bits_per_step, service_floor)

gu_cost_ref  = 1 / gu_flow_ref
uav_cost_ref = 1 / uav_flow_ref
sat_cost_ref = 1 / sat_flow_ref

uav_total_cost_ref = uav_cost_ref + sat_cost_ref
gu_total_cost_ref  = gu_cost_ref + uav_total_cost_ref
```

输入使用：

```text
*_cost_log_ratio = log(cost / cost_ref)
*_workload_log1p = log1p(max(cost * queue_bits, 0))
```

上一拍 route cost 使用：

```text
last_gu_total_cost,
last_uav_total_cost,
last_sat_cost =
  env._bw_weighted_workload_device_costs(
    assoc_override=last_association,
    sat_selection_override=last_sat_selection
  )
```

SAT actor 不输入当前 prefix total cost，因为当前 SAT selection 尚未确定。

## 5. LocalSatState

SAT actor 使用专用 local state，不复用 critic schema。

```python
@dataclass
class LocalSatState:
    ego_features: torch.Tensor        # [R, SAT_EGO_DIM]
    demand_features: torch.Tensor     # [R, SAT_DEMAND_DIM]
    role_features: torch.Tensor       # [R, 1], ego_role_norm
    sat_tokens: torch.Tensor          # [R, S, SAT_TOKEN_DIM]
    sat_mask: torch.Tensor            # [R, S]
    sat_valid_mask: torch.Tensor      # [R, S]
    subset_members: torch.Tensor      # [M, K]
    subset_mask: torch.Tensor         # [R, M]
    candidate_sat_ids: torch.Tensor   # [R, S], global SAT id, -1 padding
```

其中：

```text
R = B * U 或单环境下 U
S = cfg.per_uav_visible_sat_token_max
K = sat_action_select_k
M = 1 + sum_{c=1..min(K,S)} C(S, c)
```

字段语义：

```text
sat_mask:
  该 candidate slot 是否真实存在。

sat_valid_mask:
  该 candidate slot 是否 link-valid，且可被当前 ego 选择。

role_features:
  固定对称破缺信号。
  role_features[row,0] = ego_uav_index / max(num_uav - 1, 1)。
  U == 1 时为 0。

subset_members:
  固定 canonical local candidate slot 表，不是 global SAT id。
  该表只依赖 S/K，所有 row 共享。
  成员按 local slot 升序保存；不足 K 的位置填 -1。

subset_mask:
  该 subset 是否由真实且 valid 的候选 SAT 组成。

candidate_sat_ids:
  local slot -> global SAT id 的映射，仅用于 decode / diagnostics / parity。
```

`subset_mask` 规则：

```text
if valid_count == 0:
  empty subset 合法，其它 subset 非法。
else:
  empty subset 非法；
  所有 size in [1, min(K, valid_count)] 且成员 valid 的 subset 合法。
```

`subset_members` 不应该在 hot path 里按 row 重新生成。Python 侧可以作为 `[M,K]` tensor 挂在 `LocalSatState` 上；native 侧应作为常驻 device 常量表复用。

## 6. Ego Features

ego features 只放 ego UAV 自身 relay/backhaul 决策需要的当前状态和上一拍历史。

不放 ego 的绝对位置、速度、speed：

```text
ego_x_norm / ego_y_norm / ego_vx_norm / ego_vy_norm / ego_speed_norm
```

原因是 SAT 选择需要的几何已经在每个 ego-SAT token 的相对位置、相对速度、elevation、Doppler 和 `backhaul_se_ref` 里表达。再放 ego 绝对运动状态是重复信息，还容易让策略学习地图/坐标系 artifact。

`ego_energy_norm` 不进入本版主 schema。若未来 energy reward / safety 直接要求 SAT actor 用能量决定少选或多选，应另起 schema migration，而不是在实现时临时追加字段。

```text
0  uav_queue_steps
1  uav_queue_fill
2  uav_last_inflow_steps
3  uav_last_outflow_steps
4  uav_last_drop_steps
5  uav_service_ema_steps
6  uav_local_cost_log_ratio
7  uav_last_total_cost_log_ratio
8  uav_last_workload_log1p
9  uav_last_access_interference_log1p

10 last_selected_count_frac
11 last_backhaul_outflow_steps
```

说明：

```text
last_selected_count_frac:
  上一拍 ego 选择了多少 SAT / K。

last_backhaul_outflow_steps:
  sum_s last_uav_to_sat_outflow_matrix[ego,s] / uav_flow_ref。
```

`last_sat_switch_count_frac` 不进入主 schema。它只是动作平滑/抗抖动信号，不是当前物理状态；若要减少频繁切换，使用 action smoothing regularizer 或在 loss 中加入切换惩罚，而不是把上上拍历史塞进核心观测。

`uav_last_access_interference_log1p` 使用上一拍真实 access interference：

```text
uav_last_access_interference_log1p =
  log1p(last_access_interference_by_uav[ego] / access_noise_ref)
```

## 7. Demand Features

SAT actor 虽然不直接分配 BW，但它需要知道 ego 后续 backhaul 大概要承接多少 access 流入压力。association prefix 在 sat stage 已知，因此可以输入 ego cell demand summary。

对 ego UAV：

```text
C_ego = { g | association[g] == ego }
```

主 schema 字段：

```text
0 cell_gu_count_frac
1 cell_queue_steps_sum
2 cell_expected_arrival_steps_sum
3 cell_last_arrival_steps_sum
4 cell_last_outflow_steps_sum
5 cell_last_drop_steps_sum
6 cell_last_workload_log1p_sum
7 cell_access_rate_full_bw_ref_sum
```

其中：

```text
cell_gu_count_frac =
  |C_ego| / num_gu

cell_queue_steps_sum =
  sum_{g in C_ego} gu_queue_steps[g] / num_gu

cell_expected_arrival_steps_sum =
  sum_{g in C_ego} gu_expected_arrival_steps[g] / num_gu

cell_last_arrival_steps_sum =
  sum_{g in C_ego} gu_last_arrival_steps[g] / num_gu

cell_last_outflow_steps_sum =
  sum_{g in C_ego} gu_last_outflow_steps[g] / num_gu

cell_last_drop_steps_sum =
  sum_{g in C_ego} gu_last_drop_steps[g] / num_gu

cell_last_workload_log1p_sum =
  sum_{g in C_ego} gu_last_workload_log1p[g] / num_gu

cell_access_rate_full_bw_ref_sum =
  sum_{g in C_ego} access_rate_full_bw_ref_steps[g, ego] / num_gu
```

其中：

```text
access_rate_full_bw_ref_steps[g, ego] =
  b_acc * access_se_ref[g, ego] * tau0 / gu_flow_ref
```

它是“若该 GU 把完整 access bandwidth 给 ego UAV 时”的归一化 step service reference，只使用当前 access channel snapshot 和已知 association prefix，不使用 SAT action 或 BW action。

`cell_access_pressure_sum` 不单独放入，因为它可由 queue / expected arrival / last outflow / drop 组合得到。  
`cell_weak_link_workload_sum` 也不放入 SAT actor 主 schema；弱接入链路更直接影响 BW/access outflow，而 SAT actor 只需要估计 ego 后续进入 backhaul 的 relay pressure。

这些 demand 字段来自已知 association prefix，不是 SAT action 后的结果。

## 8. SAT Tokens

每个 ego UAV 只看自己的当前可见/候选 SAT：

```text
candidate_sat_ids[row] = _stage_visible[ego] 截断到 S 后 padding -1
```

候选顺序必须使用环境已有 `_visible_sats_sorted(...)` / stage cache 口径。Python builder、tensor builder、native builder 必须完全一致。

### 8.1 SAT workload 字段

```text
0  sat_queue_steps
1  sat_queue_fill
2  sat_last_incoming_steps
3  sat_last_processed_steps
4  sat_last_drop_steps
5  sat_service_ema_steps
6  sat_cost_log_ratio
7  sat_last_workload_log1p
8  sat_last_selected_load_frac
9  sat_proc_capacity_steps
```

不放 SAT 绝对 ECEF 位置和速度：

```text
sat_x_norm / sat_y_norm / sat_z_norm / sat_vx_norm / sat_vy_norm / sat_vz_norm
```

原因是当前 ego 选择 SAT 时需要的是“ego 到该 SAT 的相对链路状态”和“该 SAT 自身 workload”。绝对轨道状态已经通过 `us_rel_*`、elevation、Doppler、range、`backhaul_se_ref` 消化；再放绝对 ECEF 量会让网络学习不稳定的坐标系细节。

`sat_last_selected_load_frac` 是上一拍真实历史：

```text
number_of_UAVs_that_selected_this_SAT_last_step / num_uav
```

它不是 current prefix load。

`sat_proc_capacity_steps` 定义为：

```text
sat_proc_capacity_steps =
  effective_sat_cpu_freq * tau0 / task_cycles_per_bit / sat_flow_ref
```

### 8.2 Ego-SAT link 字段

```text
10 us_rel_x_norm
11 us_rel_y_norm
12 us_rel_z_norm
13 us_rel_vx_norm
14 us_rel_vy_norm
15 us_rel_vz_norm
16 us_range_norm
17 us_radial_velocity_norm
18 us_elevation_norm
19 us_doppler_norm
20 us_doppler_margin
21 us_backhaul_se_ref
22 us_visible_flag
23 us_valid_flag
24 us_last_selected_flag
25 us_last_outflow_steps
```

定义：

```text
us_backhaul_se_ref:
  固定 reference bandwidth 下的 spectral efficiency。
  不包含 current selected load。

us_last_selected_flag:
  上一拍 ego 是否选择该 SAT。

us_last_outflow_steps:
  last_uav_to_sat_outflow_matrix[ego, sat] / uav_flow_ref。
```

`us_valid_flag` 与 `sat_valid_mask` 使用同一口径：

```text
elevation >= theta_min_rad
and, if doppler_enabled:
  abs(effective_doppler_hz) <= nu_max
```

padding slot 的 `us_visible_flag`、`us_valid_flag` 均为 0。

### 8.3 相对关系不作为输入字段

SAT actor 的重点确实是不同 SAT 之间的相对关系，但这个关系不应作为 `sat_pair_edges / sat_pair_mask` 输入 schema 固化。

原因：

```text
1. pair tensor 会把 actor ABI 扩成 [R,S,S,D]，native kernel、buffer、history 成本明显增加。
2. 很多 pair 差值可由 per-SAT token 直接比较得到，例如 range、elevation、Doppler、backhaul_se、queue、last load。
3. 相对关系应该由网络里的 masked SAT self-attention 学习，而不是由 builder 预先手写一套 pair feature。
4. pair feature 容易和 per-SAT token 字段重复，增加 schema 维护和 Python/native parity 风险。
```

因此主 schema 只保留 per-SAT token。候选 SAT 之间的相对关系在网络结构中通过 self-attention 的 Q/K/V 交互体现，见第 9 节。

### 8.4 多 UAV 拥塞信号

以下字段不进入 SAT token 主 schema：

```text
candidate_peer_count_frac
candidate_peer_last_selected_frac
```

原因：

```text
candidate_peer_count_frac:
  如果 3 个 UAV 看到的候选 SAT 条件基本一致，它几乎是常数，没有决策信息。

candidate_peer_last_selected_frac:
  与 sat_last_selected_load_frac 高度重复。
  如果需要“去掉 ego 自己后的上一拍负载”，网络可由 sat_last_selected_load_frac 和 us_last_selected_flag 推出。
```

`candidate_peer_demand_sum` 也不进入本版最终 schema。原写法只看 peer cell demand，不看 peer UAV 自身 queue，会低估 peer 的 backhaul 压力；在“所有 UAV 看到几乎相同 SAT 集合”的场景里，它对每颗 SAT 仍可能接近常数，无法单独解决同 SAT 拥堵。

避免不同 UAV 堵在同一 SAT，不能只靠 peer token 字段解决；本版通过历史拥塞反馈和 `ego_role_norm` 破对称处理。见第 9.6 节。

### 8.5 不应输入的字段

SAT actor 不输入：

```text
prefix_selected_flag
prefix_selected_known
selected_sat_load_current
prefix_backhaul_capacity_steps
projected_bw_under_current_selection
projected SE using current selected load
BW allocation / beta
candidate subset hand-crafted score
```

尤其不要把 `backhaul_se_ref` 写成“按 current selected load 分带宽后的 SE”。那是 BW stage 才能知道的 prefix capacity。

## 9. 网络结构

本节是最终网络 ABI。实现时不要沿用旧 SAT actor 的 `query_proj_1/query_proj_2 + subset_tokens + scorer` 结构。

固定符号：

```text
E = embed_dim
H = hidden_dim
L = sat_competition_layers
heads = sat_attention_heads
```

配置硬约束：

```text
E % heads == 0
L >= 1
heads >= 1
```

SAT actor 使用 trainable LayerNorm；`input_norm_enabled` 不再关闭 SAT actor 的这些 LayerNorm。字段已经做物理归一化，LayerNorm 只用于优化稳定性，不替代前面的物理 reference。

模块形状固定如下：

```text
LayerNorm:
  ego_input_norm:    LayerNorm(SAT_EGO_DIM)
  demand_input_norm: LayerNorm(SAT_DEMAND_DIM)
  sat_input_norm:    LayerNorm(SAT_TOKEN_DIM)

MLP2(x_dim, y_dim):
  Linear(x_dim, H)
  ReLU
  Linear(H, y_dim)

ego_encoder:        MLP2(SAT_EGO_DIM, E)
demand_encoder:     MLP2(SAT_DEMAND_DIM, E)
role_encoder:       MLP2(SAT_ROLE_DIM, E)
sat_encoder:        MLP2(SAT_TOKEN_DIM, E)
ctx_encoder:        MLP2(3 * E, E)
sat_context_fusion: MLP2(2 * E, E)
sat_logit_head:     MLP2(E, 1)
count_logit_head:   MLP2(E, K + 1)
```

SAT actor 的 MLP 激活固定为 ReLU，以复用现有 Python `_make_mlp` / native `mlp2` helper 的语义，避免 SAT native ABI 额外引入一套激活函数。

`role_encoder` 直接读取 raw `ego_role_norm`。不要添加 `role_input_norm`。

`sat_self_attention_blocks` 固定为 L 个 masked Transformer-style block：

```text
attn_norm: LayerNorm(E)
qkv_proj:  Linear(E, 3 * E)
out_proj:  Linear(E, E)
ffn_norm:  LayerNorm(E)
ffn:       MLP2(E, E)
```

masked multi-head self-attention 固定为 scaled dot-product attention：

```text
q, k, v = split(qkv_proj(x_norm), 3)
attention_score[i,j] = dot(q_i, k_j) / sqrt(E / heads)
attention_score[i,j] = -inf if valid_sat_mask[j] == 0
attention_out[i] = 0 if valid_sat_mask[i] == 0
```

不使用 dropout。所有 invalid / padding query token 在 block 输出后继续置 0。

每个 block 的计算顺序：

```text
x1 = x + masked_multihead_self_attention(attn_norm(x), valid_sat_mask)
x2 = x1 + ffn(ffn_norm(x1))
x2[~valid_sat_mask] = 0
```

当 `valid_count == 0` 时，actor 直接走 empty subset 特殊行，不运行会产生 all-masked softmax 的 attention 分支。

### 9.1 编码

```text
ego_emb =
  ego_encoder(LayerNorm(ego_features))

demand_emb =
  demand_encoder(LayerNorm(demand_features))

role_emb =
  role_encoder(role_features)

sat_emb[k] =
  sat_encoder(LayerNorm(sat_tokens[k]))
```

`role_features` 是 1 维标量，不要使用 `LayerNorm(1)`。`LayerNorm(1)` 会把单个标量归一化成常量，抹掉 role 信息。`role_encoder` 使用第 9 节固定的 `MLP2(SAT_ROLE_DIM, E)`，直接读 raw `ego_role_norm`。

有效候选：

```text
valid_sat_mask = sat_mask & sat_valid_mask
```

### 9.2 Ego-conditioned token injection

`Ego-conditioned SAT context` 如果理解成“先把所有 SAT 摘成一个全局上下文，再让这个上下文主导决策”，是不合适的；它会在候选 SAT 还没有互相比较之前就压缩信息。

正确做法是：先把 ego/demand/role 注入每个 SAT token，然后让 SAT token 自己做 masked self-attention。

```text
ctx0 =
  ctx_encoder([ego_emb, demand_emb, role_emb])

sat_h0[k] =
  sat_context_fusion([sat_emb[k], ctx0])
```

这样 SAT token 仍保持逐候选结构，不会提前丢掉 SAT 间差异。

没有 valid SAT 时：

```text
sat_h0 可以正常计算；
后续 attention / pooling 必须用 valid_sat_mask 屏蔽。
```

实现时要保证 padding token 不泄漏。

### 9.3 Masked SAT self-attention

SAT actor 的重点是不同 SAT 之间的相对关系，所以主干必须包含候选 SAT 集合上的 masked self-attention。

```text
sat_h = sat_h0
for block in sat_self_attention_blocks:
  sat_h = block(sat_h, valid_sat_mask)
```

self-attention 的作用是让每颗 SAT 在打分前读到其它候选 SAT，并比较：

```text
谁的 backhaul_se_ref 更高
谁的 range / elevation / Doppler margin 更好
谁的 queue / last load 更低
哪些 SAT token 在当前候选集合里相对更互补
```

这里不需要 `sat_pair_edges`。相对关系来自 token embedding 之间的 Q/K/V 交互；每个 token 已包含 ego-SAT 相对链路和 SAT workload。

`valid_sat_mask` 必须进入 attention mask。padding 或 link-invalid SAT 不能作为 key/value，也不能得到可选 score。

不再额外做 self-attention 后的 ego-conditioned readout。`ctx0` 已经注入每个 SAT token，后续单 SAT 价值只读取 `sat_h[k]`；`ctx0` 只用于选择数量先验。再做一层全局 readout 会重复建模，而且可能把已经保留在 token 里的候选差异再次压缩掉。

### 9.4 Subset representation

SAT self-attention 后，每个 `sat_h[k]` 已经包含：

```text
ego/demand/role 条件
该 SAT 自身链路与 workload
它相对其它候选 SAT 的比较结果
```

因此后面不再把 `ctx0` 拼回单颗 SAT 打分，也不再做额外 subset interaction。每颗 SAT 只输出一个 contextual item logit：

```text
sat_item_logit[k] =
  sat_logit_head(sat_h[k])
```

本设计允许少选 SAT，因此必须有一个“选择数量”logit：

```text
count_logits =
  count_logit_head(ctx0)  # shape [K + 1], index c in [0, K]
```

`count_logits[c]` 表示选择 `c` 颗 SAT 的偏好。`c=0` 只在 `valid_count == 0` 时允许；正常有 valid SAT 时 empty subset 非法。

count mask 固定为：

```text
valid_count == 0:
  only c = 0 is legal

valid_count > 0:
  legal c in [1, min(K, valid_count)]
```

对每个 legal subset `A`：

```text
subset_logit[A] =
  count_logits[|A|]
  + sum_{k in A} sat_item_logit[k]
```

非法 subset 的 logit 必须写成 `-inf`，不能靠 softmax 后再清零。

说明：

```text
sat_item_logit:
  是 self-attention 后的上下文化单 SAT 价值。
  它已经知道其它候选 SAT 的相对情况，因此不需要再拼 ctx0。

sum_{k in A}:
  保留多 SAT 并行资源的可加语义。

count_logits:
  单独学习选择数量偏好，避免 item logit 的符号和尺度隐式决定“选几颗”。
```

这个分布是 order-invariant 的：

```text
同一个成员集合只有一个 subset_logit；
成员排列不改变 logit；
不用 flatten subset 成员。
```

最终方案固定采用可变 cardinality，因为它能表达“链路很差或拥塞压力高时少选”。实现时不要再退回固定满选 K 的 head；如果未来要改成固定 cardinality，应作为新的设计迁移处理。

### 9.5 Subset categorical

```text
score[m] = subset_logit[m]
score[m] = -inf if not subset_mask[m]
```

这里没有单独的 subset pooled representation，也不再对 subset 过一个额外 MLP。score 由 contextual item logits 和 cardinality logits 直接组成：

```text
SAT 间相对关系：由 self-attention 写进 sat_h[k]。
单颗 SAT 价值：由 sat_logit_head(sat_h[k]) 给出。
子集价值：由成员 item logits 求和得到。
选择数量：由 count_logit_head(ctx0) 给出。
```

确定性动作：

```text
subset_index = argmax(score)
```

随机策略：

```text
Categorical(logits = masked_score)
```

特殊行：

```text
valid_count == 0:
  subset_index = 0  # empty subset
  selected_sat_indices = all -1
  logprob = 0
  entropy = 0

only one legal subset:
  选择该 subset
  logprob = 0
  entropy = 0
```

### 9.6 避免多个 UAV 堵在同一 SAT

需要先明确一个边界：

```text
如果 SAT actor 是纯并行、纯局部、参数共享，并且多个 UAV 的观测完全相同，
那么 deterministic policy 必然输出相同动作。
这种设置下无法保证不同 UAV 不选择同一 SAT。
```

因此“避免同 SAT 拥堵”不能只靠多塞几个 peer feature。本版最终方案采用：

```text
1. 历史拥塞反馈。
2. ego_role_norm 显式打破对称。
```

不采用 autoregressive prefix 选择。

#### 9.6.1 历史拥塞反馈

主 schema 已保留：

```text
sat_last_selected_load_frac
us_last_selected_flag
us_last_outflow_steps
sat_queue_steps
sat_last_incoming_steps
sat_last_processed_steps
sat_last_drop_steps
```

这些字段让策略知道上一拍哪些 SAT 已经拥堵或服务不充分。它能减少长期挤在同一 SAT，但不能保证单拍内完全分流，也可能产生周期性摆动。

#### 9.6.2 使用 ego_role_norm 显式打破对称

如果实际场景中多个 UAV 的候选 SAT 和需求长期近似相同，并且需要 deterministic policy 也稳定分流，则必须给 actor 一个 symmetry breaker。本设计固定使用：

```text
ego_role_norm = ego_uav_index / max(num_uav - 1, 1)
```

本设计固定使用 `ego_role_norm`，并作为 `role_features` 输入 SAT actor。它不是物理状态，而是参数共享多智能体中的角色信号，用来在“多个 UAV 观测几乎相同”时允许 deterministic policy 输出不同选择。

注意：加入 `ego_role_norm` 后，actor 不再对 UAV index 完全 permutation-equivariant。这个取舍是有意的，因为当前任务更需要多个 UAV 在近似同质候选 SAT 上形成稳定分工。

#### 9.6.3 不采用顺序 prefix 选择

顺序 prefix 选择可以硬保证单拍内不拥堵，但不属于本版最终方案。它的形式是：

```text
UAV 按某个顺序依次选择 SAT。
第 t 个 UAV 可以看到前 t-1 个 UAV 的 current selected load prefix。
```

这样 SAT actor 可以输入：

```text
prefix_selected_load_so_far
prefix_selected_known_for_previous_uavs
```

这会改变 stage 语义、history logprob、native kernel 和训练接口，复杂度明显更高。因此本版不实现 autoregressive SAT actor；若以后确实需要硬约束，应单独重开设计文档，而不是在当前 schema 上临时追加 prefix input。

## 10. 为什么不使用 sequential categorical

旧 `MaskedSequentialCategorical` 或逐步 top-k 有两个问题：

```text
1. 同一个无序 SAT set 有多个排列编码。
   例如 [1,2] 和 [2,1] 对环境相同，但 logprob 不同。

2. 第二步选择条件依赖第一步，容易让 policy 学到 slot/order artifact。
```

因此最终主路径固定使用 legal subset categorical。不要为了降低实现成本保留 sequential categorical；否则 PPO logprob、entropy 和 history action 语义都会重新引入排列噪声。

## 11. 与 BW actor 的边界

SAT actor 输出 selected SAT 后，BW actor 才能看到：

```text
ego selected SAT
selected SAT prefix load
prefix_backhaul_capacity_steps
selected SAT workload/cost tokens
```

因此：

```text
SAT actor:
  看 candidate SAT 的当前 link/workload/history，选择连接集合。

BW actor:
  看 selected SAT 的 prefix capacity/workload，给 valid GU 分 access bandwidth。
```

不要让 SAT actor 提前输入 selected SAT capacity，也不要让 BW actor 重新决定 SAT。

## 12. 与 critic 的边界

SAT actor 不复用 `StructuredWorldState`。原因：

```text
1. critic 是 centralized system value，需要 GU/UAV/SAT 全局结构。
2. SAT actor 是单 ego local policy，只能消费当前 ego 可观测/允许的信息。
3. critic SAT token 是 visible union，SAT actor SAT token 是 per-ego visible list。
4. critic 有 prefix known flags，SAT actor 在 sat stage 固定为 sat prefix unknown。
```

允许共享底层计算 helper：

```text
flow/cost reference helper
visible SAT sorting helper
backhaul link quality helper
Doppler/elevation validity helper
last_selected/load/outflow runtime tensors
```

但不共享 actor 输入 tensor schema。

## 13. Native / tensor runtime 要求

native live rollout 不能通过 Python per-row 构造 `LocalSatState`，也不能把 actor forward 拆成许多小 PyTorch op / 小 kernel。

### 13.1 固定 ABI 与常驻表

以下量是 native actor ABI 的一部分：

```text
S = per_uav_visible_sat_token_max
K = sat_action_select_k
M = 1 + sum_{c=1..min(K,S)} C(S, c)
```

初始化时必须校验：

```text
M <= sat_subset_count_max
M <= native kernel 支持的 hard max
```

`subset_members_base [M,K]` 和 `subset_sizes [M]` 在初始化阶段按 canonical 顺序生成一次，并常驻 device。它们不是每个 step、每个 env、每个 ego row 重新生成的 live observation。

canonical 顺序固定为：

```text
index 0: empty subset
size 1 subsets, lexicographic local slot order
size 2 subsets, lexicographic local slot order
...
size K subsets, lexicographic local slot order
```

这保证：

```text
同一个 SAT set 只有一个 subset_index。
Python / tensor / native decode 完全一致。
PPO history 只需要保存 subset_index，不需要保存动作排列。
```

### 13.2 Live contiguous buffers

需要固定 contiguous buffers：

```text
float:
  live_sat_ego_features      [B*U, SAT_EGO_DIM]
  live_sat_demand_features   [B*U, SAT_DEMAND_DIM]
  live_sat_role_features     [B*U, SAT_ROLE_DIM]
  live_sat_tokens            [B*U, S, SAT_TOKEN_DIM]

bool:
  live_sat_mask              [B*U, S]
  live_sat_valid_mask        [B*U, S]

int64:
  live_sat_candidate_sat_ids [B*U, S]
  live_sat_action_indices    [B, U, K]
  live_sat_subset_index      [B, U]

device constants:
  sat_subset_members_base    int64 [M, K]
  sat_subset_sizes           int64 [M]

float diagnostics:
  live_sat_old_logprobs_per_agent [B, U]
  live_sat_entropy_per_agent      [B, U]

optional debug / parity:
  live_sat_subset_mask       [B*U, M]
```

native live 主路径不单独维护 `[B]` 聚合 logprob，避免为了求和额外引入小 kernel 或 Python 调度。
历史环中的 `sat_old_logprobs [B]` 由 main commit fused kernel 对
`live_sat_old_logprobs_per_agent[b,u]` 按 env 求和写入。

native 主路径不持久写 `live_sat_subset_mask [B*U,M]`。`actor_sat_live_kernel` 根据 `live_sat_valid_mask`、`sat_subset_members_base`、`sat_subset_sizes` 即时判断 subset legality。只有调试或 parity 需要时，才额外生成 `live_sat_subset_mask`。

### 13.3 Fused actor kernel

CUDA live actor 主路径要求：

```text
grid.x = B * U
每个 block 处理一个 ego row。
ego encoder、demand encoder、role encoder、SAT encoder
ctx fusion
masked SAT self-attention
sat_item_logit
count_logits
subset_logit 枚举
categorical sample / deterministic argmax
logprob / entropy
global SAT id decode
全部在 actor_sat_live_kernel 内完成。
```

不允许在 hot loop 中用多个小 PyTorch op 拼 actor forward。

native 主路径不持久 materialize `[B*U,M]` logits。固定做法是：

```text
1. 在 block 内遍历 M 个 canonical subset。
2. 根据 valid mask 判断 legality，非法直接跳过或赋 -inf。
3. 用 sat_item_logit 与 count_logits 计算 subset_logit。
4. 同一 kernel 内完成 max/logsumexp/sample/entropy。
5. 只写 action_indices、subset_index、logprob、entropy。
```

这样该设计只增加单个 fused kernel 内的算术量，不增加 Python 调度次数，也不引入大量小 kernel。

### 13.4 复杂度边界

每个 ego row 的主要计算复杂度为：

```text
O(S^2 * E)       # masked SAT self-attention
O(M * K)         # subset enumeration
```

因此本方案要求 `S` 和 `K` 是小而固定的配置上界。若未来把 `per_uav_visible_sat_token_max` 或 `sat_action_select_k` 扩得很大，`M` 会组合爆炸；那时应重新设计动作分布，例如固定选满 K、分阶段选择、Gumbel top-k、或其它近似集合分布。当前文档的最终方案只适用于小 `S/K` 的精确 legal subset categorical。

## 14. 代码迁移规格

### 14.1 新增 schema 文件

新增：

```text
sagin_marl/rl/structured_sat_actor_schema.py
```

只放：

```text
SAT_EGO_DIM
SAT_DEMAND_DIM
SAT_ROLE_DIM = 1
SAT_TOKEN_DIM
字段 index 常量
schema 字段名 tuple
subset objective 口径
```

不要从 critic shape 推导 SAT actor 输入维度。

### 14.2 修改 LocalSatState

当前旧字段：

```text
ego_uav_after_accel
sat_nodes
sat_edges
sat_mask
subset_tokens
subset_mask
subset_members
```

改为：

```text
ego_features
demand_features
role_features
sat_tokens
sat_mask
sat_valid_mask
subset_members
subset_mask
candidate_sat_ids
```

旧字段名不要保留 alias。否则后续容易出现旧 actor 和新 actor 混用。

### 14.3 Python builder

新增：

```text
build_batched_local_sat_states_from_spec(spec) -> LocalSatState
build_local_sat_states_from_spec(spec) -> list[LocalSatState]
```

`structured_driver.py` 暴露：

```text
build_local_sat_states()
```

它必须从 sat stage spec 构造，而不是从 critic world 反推。

spec 必须包含：

```text
uav_pos / uav_vel
gu_pos
association
bw_valid_mask
sat_pos / sat_vel
visible
access_gain_matrix
last runtime state
queue / EMA / drop tensors
```

如果 stage cache 中缺少当前 access/backhaul channel snapshot，应该在 stage 准备阶段补齐；builder 内不要重新采样。

旧 `build_batched_local_sat_states_from_world(...)` / `build_local_sat_states_from_world(...)` 只能在迁移期作为删除对象存在，新 SAT actor 的训练、eval、native parity 都不得再从 `StructuredWorldState` 反推 actor 输入。

### 14.4 SatPolicy 重写

新类名：

```text
SatSubsetPolicy
```

不保留 `SatPairPolicy = SatSubsetPolicy` 兼容 alias；代码主路径只使用 `SatSubsetPolicy` / `sat_subset_policy` 命名。

构造器：

```python
class SatSubsetPolicy(nn.Module):
    def __init__(
        self,
        *,
        ego_dim: int = SAT_EGO_DIM,
        demand_dim: int = SAT_DEMAND_DIM,
        role_dim: int = SAT_ROLE_DIM,
        sat_token_dim: int = SAT_TOKEN_DIM,
        hidden_dim: int,
        embed_dim: int,
        sat_competition_layers: int,
        sat_attention_heads: int,
        sat_action_select_k: int,
        per_uav_visible_sat_token_max: int,
    ) -> None:
        ...
```

native shape spec 中 `hidden_dim/embed_dim` 继续使用 structured actor 共享的 hidden/embed 宽度；SAT actor 不单独引入另一套 native hidden/embed 常量。

模块命名固定：

```text
ego_input_norm
demand_input_norm
sat_input_norm
ego_encoder
demand_encoder
role_encoder  # raw ego_role_norm, no LayerNorm(1)
sat_encoder
ctx_encoder
sat_context_fusion
sat_self_attention_blocks
sat_logit_head
count_logit_head
```

`sat_self_attention_blocks.{i}` 内部模块命名按第 9 节固定为：

```text
attn_norm
qkv_proj
out_proj
ffn_norm
ffn
```

`LayerNorm` 权重使用 `.weight/.bias`；`MLP2` 权重使用 `.0.weight/.0.bias/.2.weight/.2.bias`；attention 线性层使用 `.weight/.bias`。Python policy、`native_actor_cuda.py` weight pack 和 CUDA actor kernel 必须按同一名称和 shape 对齐，缺失 SAT actor 权重是 hard error。

删除旧主路径：

```text
query_proj_1 / query_proj_2
sat_refine
ego_fusion
subset_tokens as precomputed actor input
sat_nodes + sat_edges concat from critic world
```

### 14.5 decode / history

history 保存：

```text
sat_subset_index: [B,U]
sat_action_indices: [B,U,K]  # global SAT id, -1 padding
old_logprob_per_agent: [B,U]
```

训练 evaluate 时：

```text
用 sat_subset_index 计算 PPO logprob。
用 sat_action_indices 做 diagnostics / env parity。
```

由于 subset table 依赖 `S` 和 `K`，训练时必须保证 replay/history 中的 `S/K` 与当前 policy ABI 一致。改变 `per_uav_visible_sat_token_max` 或 `sat_action_select_k` 应视为 checkpoint 不兼容，除非写 migration。

## 15. 测试要求

新增：

```text
tests/test_structured_sat_actor_schema.py
tests/test_structured_sat_actor_native_parity.py
```

必须覆盖：

```text
1. LocalSatState shape 与 schema dim 一致。
2. candidate_sat_ids 顺序与 _stage_visible[ego][:S] 完全一致，padding 为 -1。
3. sat_mask 只表示真实 candidate slot。
4. sat_valid_mask 与 elevation/Doppler/link validity reference 一致。
5. role_features 等于 ego_role_norm，且 U == 1 时为 0。
6. subset_mask:
   valid_count == 0 时只有 empty subset 合法；
   valid_count > 0 时 empty subset 非法；
   所有合法 subset 成员均 valid 且 size <= K。
7. backhaul_se_ref 不随 current selected load 改变。
8. SAT actor 输入不包含 prefix_backhaul_capacity_steps。
9. last_selected_flag / last_outflow_steps 来自上一拍真实历史。
10. current prefix selected/load 不进入 SAT actor。
11. deterministic action 输出合法 global SAT id，且 -1 padding 正确。
12. stochastic action 输出合法 global SAT id。
13. only one legal subset 时 logprob=0、entropy=0。
14. no valid SAT 时 action 全 -1、logprob=0、entropy=0。
15. subset action permutation-invariant：同一成员集合只有一个 canonical subset index。
16. subset_members_base 顺序固定且无重复集合；decode / logprob 不依赖动作成员排列。
17. SAT self-attention mask 生效：padding/invalid token 改成随机值不影响输出。
18. valid_count == 0 时不进入 all-masked attention 分支，输出 empty subset。
19. sat_action_select_k / S / M 与 config finalize 和 subset_members_base 一致。
20. Python builder 与 native/tensor builder 对同一 runtime state allclose。
21. native deterministic action 与 Python deterministic action allclose。
22. native live rollout 使用 fused actor_sat_live_kernel，不回退到 Python actor 或 per-row LocalSatState list。
23. history old_logprob 是 per-ego scalar，并可按 env 求和。
24. env 执行合法 SAT action 时不做 fallback / 二次 top-k / 补 best SAT。
25. env 对不可见、无效、重复、越界 SAT action 报错。
```

## 16. 迁移顺序

按以下顺序：

```text
1. 新增 structured_sat_actor_schema.py。
2. 修改 LocalSatState / SatSubsetPolicyOutput。
3. 写 Python LocalSatState builder，并先用 schema test 锁字段和 shape。
4. 重写 Python SatSubsetPolicy。
5. 修改 structured_factory.py，SAT actor 不再读 critic dims。
6. 修改 structured_driver.py 的 SAT stage builder / decode / run_sat_stage。
7. 修改 buffer / MAPPO / eval 路径，history 保存 subset_index 和 global selected_sat_indices。
8. 修改 native sat obs view / history buffer。
9. 修改 native actor weight ABI。
10. 实现 actor_sat_live_kernel。
11. 加 native parity 和 live rollout smoke test。
12. 清理旧的 critic-world-to-SAT-actor 路径。
```

## 17. 最终链路

对单台 ego UAV：

```text
SAT prefix:
  post-accel association
  ego cell demand / relay pressure
  current candidate SAT geometry/link/workload
  last SAT selection/load/outflow history

-> LocalSatState:
  ego_features
  demand_features
  per-ego SAT tokens
  valid SAT mask
  legal subset mask

-> Shared SAT actor:
  encode ego / demand / SAT tokens
  inject ego/demand/role into each SAT token
  masked SAT self-attention over candidate SATs
  contextual item logits + cardinality k-subset distribution
  categorical over legal SAT subsets

-> Output:
  selected global SAT ids for this ego UAV

-> Environment:
  execute selected SAT set directly
  compute selected load
  compute prefix backhaul capacity
  enter BW stage
```

最终要求：

```text
SAT actor 只决定当前 ego UAV 的 SAT 连接集合。
它看当前候选链路和历史拥塞，但不看当前动作之后才产生的 selected load/capacity。
它的动作是无序合法子集，环境不二次修正。
它和 accel/BW 一样拥有专用 local schema，不再复用 critic world。
```
