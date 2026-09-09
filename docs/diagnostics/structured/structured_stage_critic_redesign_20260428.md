# StructuredWorldState Critic 重新设计

日期：2026-04-29

目标：把 centralized critic 做成能预测当前 `weighted_workload_level` return 的系统级 value model。

这不是“BW-specific critic”。reward 是系统 reward：

```text
R_level = - workload_after - drop_cost

workload_after =
  sum_g gu_cost[g]  * gu_queue[g]
+ sum_u uav_cost[u] * uav_queue[u]
+ sum_s sat_cost[s] * sat_queue[s]

drop_cost =
  sum_g gu_cost[g]  * gu_drop[g]
+ sum_u uav_cost[u] * uav_drop[u]
+ sum_s sat_cost[s] * sat_drop[s]
```

所以 critic 必须看懂：

```text
GU arrival/queue/drop
GU -> UAV access
UAV queue/drop/backhaul
UAV -> SAT selected backhaul
SAT queue/drop/processing
EMA cost
interference
stage prefix
```

当前最大问题不是“少加几个 feature”，而是：

```text
当前 critic 是 UAV-centered mean readout；
应该改成 GU-flow / UAV-relay / SAT-service centered 的 system value。
```

## 1. 先定因果边界

critic 估计的是 `V(s_stage)`，不是 `Q(s_stage, a)`。因此输入要分三类。

### 1.1 当前状态量

这些是当前 stage 的状态，可以直接输入：

```text
position / velocity / energy
queue
expected arrival
EMA service scale
geometry
link large-scale quality
current association / visibility / validity flags
```

### 1.2 当前 prefix 量

这些只在当前 stage 已经固定时输入为 known：

```text
post-accel association / bw_valid
current selected SAT
current selected SAT load
prefix cost under current fixed association/SAT
```

stage 规则：

```text
accel stage:
  accel 未定，sat 未定。
  prefix association / prefix sat selection 都不能假装已知。

sat stage:
  accel 已执行。
  post-accel association / bw_valid 已知。
  current sat selection 仍未知。

bw stage:
  accel 已执行，sat selection 已执行。
  association / bw_valid / selected SAT 全部是当前 prefix 已知。
  当前 BW allocation 仍未知。
```

### 1.3 上一拍执行历史

这些不是当前动作结果，而是上一拍已经真实发生的结果，所有 stage 都可输入：

```text
last association
last sat selection
last BW fraction
last GU arrival
last GU outflow
last UAV inflow/outflow
last SAT incoming/processed
last GU/UAV/SAT drop
last access interference
last cost under last executed route
```

不要把 `last_*` 填到 `prefix_*` 里冒充当前已知，也不要因为 current prefix 未定就把 last history 清空。

## 2. 当前代码是什么样

### 2.1 `StructuredWorldState`

当前 dataclass：

```text
uav_nodes
gu_nodes
sat_nodes
uav_gu_edges
uav_sat_edges
uav_uav_edges
gu_mask
sat_mask
uav_gu_mask
uav_sat_mask
uav_uav_mask
stage_id
```

### 2.2 当前 `uav_nodes`

来自 `structured_driver.py::_build_world_state()`：

```text
0: uav_x / map_size
1: uav_y / map_size
2: uav_vx / v_max
3: uav_vy / v_max
4: uav_energy / uav_energy_init
5: uav_queue / queue_max_uav
6: 0.0, 原来 horizon feature 已禁用
7: 可选 assoc_uav_cost
8: 可选 uav_id_norm
```

问题：

- queue 用 `queue_max_uav`，和 reward 的“几步 workload”尺度不一致。
- 没有 UAV 入流：上一拍 GU 到 UAV 的 inflow。
- 没有 UAV 出流：上一拍 UAV 到 SAT 的 outflow。
- 没有 UAV drop。
- cost 是可选 actor obs feature，不是 critic 固定契约。
- 没有 last access interference。

### 2.3 当前 `gu_nodes`

当前基础字段：

```text
0: gu_x / map_size
1: gu_y / map_size
2: gu_queue / queue_max_gu
3+: 由 actor obs 开关动态拼接的 proxy features
```

可选 proxy 包括 arrival、recent arrival、recent service、reward-aligned cost 等。

问题：

- critic 输入维度依赖 actor obs 配置。
- queue 尺度和 reward 不一致。
- expected arrival 对 sticky hotspot 很重要，但现在是可选项。
- drop 没有固定暴露。
- cost 混在 actor proxy 里，且命名不区分 local/current-prefix/last。

### 2.4 当前 `sat_nodes`

当前字段：

```text
0: sat_ecef_x / (r_earth + sat_height)
1: sat_ecef_y / (r_earth + sat_height)
2: sat_ecef_z / (r_earth + sat_height)
3: sat_vel_x / (r_earth + sat_height)
4: sat_vel_y / (r_earth + sat_height)
5: sat_vel_z / (r_earth + sat_height)
6: sat_queue / queue_max_sat
7: sat_load / num_uav
8: 可选 sat_cost
```

问题：

- 速度除以长度尺度，量纲不对。数值未必爆，但语义不对。
- 没有 SAT incoming / processed / drop。
- 没有 processing capacity。
- `sat_load` 没说明是 current prefix load 还是 last load。现在在 BW stage 表达当前 selection count，在更早 stage 常常是 0。

### 2.5 当前 `uav_gu_edges`

当前字段：

```text
0: rel_x / map_size
1: rel_y / map_size
2: assoc flag
3: candidate flag
4: bw_valid flag
5: prev_assoc flag
6: eta_ref_feature
```

问题：

- mask/flag 混成 edge float feature。
- `assoc/candidate/bw_valid/prev_assoc` 没有明确 prefix 与 last 的区别。
- `eta_ref_feature` 是 slot/candidate 口径，不是完整 `(u,g)` cross-gain 口径。
- 没有上一拍 BW fraction。
- 没有足够表达 interference 的 GU-to-all-UAV receiver cross quality。

### 2.6 当前 `uav_sat_edges`

当前字段：

```text
0: rel_pos_x / orbit_radius
1: rel_pos_y / orbit_radius
2: rel_pos_z / orbit_radius
3: rel_vel_x / orbit_radius
4: rel_vel_y / orbit_radius
5: rel_vel_z / orbit_radius
6: doppler / nu_max
7: spectral efficiency
8: sat_queue feature
9: sat_load feature
10: projected_bw / effective_b_backhaul_per_sat
11: visible flag
12: valid flag
13: current selected flag
```

问题：

- rel velocity 除以 length，不合理。
- doppler 用 `nu_max` 归一化会把“物理观测尺度”和“validity threshold”混在一起。
- `SE` 当前用 projected bandwidth/load 算，已经混入 selection/load 假设；如果作为纯链路质量，应使用固定 reference bandwidth。
- visible/valid/selected 作为 edge float 和 mask 重复。
- 没有 last selected 与 prefix selected 的清晰区分。
- 没有 last backhaul outflow 信息。

这里的 `projected bandwidth/load` 指当前代码里的：

```text
projected_count = max(sat_loads[active_sat_ids], 1)
projected_bw    = effective_b_backhaul_per_sat / projected_count
snr             = P_uav * gain / (noise_density * projected_bw * NF)
se              = log2(1 + snr)
```

`sat_loads` 是“当前传入 `sat_selection` 下，每颗 SAT 被多少 UAV 选中”。所以这个 `SE` 不是纯几何/链路质量，它会随着 selected load 变。BW stage 里这可能是当前 prefix 已知量；但在更早 stage，sat selection 未定，用这个值会把“尚未选择的负载假设”混进 edge。

目标拆法是：

```text
backhaul_se_ref:
  固定 reference bandwidth 下的链路质量。

prefix_backhaul_capacity_steps:
  只有 sat prefix 已知时才计算的服务能力。
```

### 2.7 当前 `uav_uav_edges`

当前字段：

```text
0: rel_x / map_size
1: rel_y / map_size
2: rel_vx / v_max
3: rel_vy / v_max
4: dist / map_size
5: alert flag
6: unsafe flag
```

这部分安全几何基本有，但缺少共享 SAT 历史/prefix 这类通信耦合信息。

### 2.8 当前 critic 结构

当前 `StructuredCritic`：

```text
node/edge encoder
每个 UAV query attend GU/SAT/UAV pairs
得到 uav_ctx
team_ctx = mean(uav_ctx)
stage-specific value head
```

关键问题：

```python
team_ctx = uav_ctx.mean(dim=1)
```

它最后输出 scalar value，但系统表示是 UAV 平均，不适合 `weighted_workload_level` 这种由 GU/UAV/SAT 队列、drop 和瓶颈共同决定的 reward。

## 3. 归一化 reference

### 3.1 flow / queue / service reference

统一用 reward 同量纲的 flow scale：

```text
arrival_ref_step      = effective_task_arrival_rate * num_gu * tau0
sat_select_k          = sat_num_select if configured else N_RF
sat_workload_ref_count = queue_ref_sat_active_count
                         if configured
                         else min(num_sat, sat_select_k * num_uav)

gu_flow_ref           = arrival_ref_step / num_gu
uav_flow_ref          = arrival_ref_step / num_uav
sat_flow_ref          = arrival_ref_step / sat_workload_ref_count
```

`effective_task_arrival_rate`、`num_gu`、`num_uav`、`num_sat`、`sat_select_k`、`sat_workload_ref_count`、`tau0` 必须在 config finalize 阶段校验为正。这里不用 `max(..., eps)` 兜底；配置非法就直接报错。

所有 queue / arrival / inflow / outflow / drop / EMA 都用：

```text
bits / corresponding_flow_ref
```

不要再用 `queue_max_*` 作为 critic 主尺度。`queue_max_*` 只作为容量/overflow 相关字段，例如：

```text
queue_fill = queue_bits / queue_max
```

但 reward 主尺度应是 flow steps。

### 3.2 position / velocity reference

局部平面量：

```text
uav_xy, gu_xy, uav-gu rel_xy: / map_size
uav_vxy, uav-uav rel_vxy:     / v_max
```

卫星 ECEF 量：

```text
orbit_pos_ref = r_earth + sat_height
sat_speed_ref = sqrt(mu_earth / orbit_pos_ref), mu_earth = 3.986004418e14
```

解释：

- `sat_pos` 和 `uav_ecef` 都是 ECEF 米，算 `sat_pos - uav_ecef` 后仍然是长度，所以用同一个 length scale：`orbit_pos_ref`。
- `sat_vel` 和 `uav_vel_ecef` 都是 m/s，算 `sat_vel - uav_vel_ecef` 后仍然是速度；本设计统一用 `sat_speed_ref` 归一化完整 `rel_vel_xyz`。
- 当前位置代码把 `rel_vel / orbit_pos_ref`，数值可能不坏，但单位是 `1/s`，不是无量纲速度。critic 设计里应改掉。
- critic 应保留完整 `rel_vel_xyz`，因为它不只影响 Doppler，也表达相对运动趋势。同时保留 `radial_velocity_norm` 和 `doppler_norm`，但不要用它们替代完整相对速度。

### 3.3 doppler reference

不要优先用 `nu_max` 做观测归一化。`nu_max` 是 validity threshold，不一定是自然观测尺度。

定义：

```text
doppler_ref = backhaul_carrier_freq * sat_speed_ref / speed_of_light
doppler_norm = doppler_hz / doppler_ref
doppler_margin = doppler_hz / nu_max   # 表达离有效性阈值多近
```

也就是说：

- `doppler_norm` 表达物理多普勒量级。
- `doppler_margin` 表达是否接近配置阈值。
- `valid_mask` 仍然由 `abs(doppler_hz) <= nu_max` 决定。
- 启用 Doppler 时，`backhaul_carrier_freq`、`sat_speed_ref`、`speed_of_light`、`nu_max` 必须为正；未启用 Doppler 时，`doppler_norm` 和 `doppler_margin` 都填 0。

### 3.4 cost / weighted backlog reference

reward cost 的单位大致是 `1 / bits_per_step`，`cost * queue_bits` 是“几步 workload”。

定义：

```text
gu_local_cost_ref  = 1 / gu_flow_ref
uav_local_cost_ref = 1 / uav_flow_ref
sat_cost_ref       = 1 / sat_flow_ref
uav_total_cost_ref = uav_local_cost_ref + sat_cost_ref
gu_total_cost_ref  = gu_local_cost_ref + uav_total_cost_ref
```

统一命名：

```text
*_cost_log_ratio = log(cost / cost_ref)
*_workload_steps = cost * queue_or_drop_bits
*_workload_log1p = log1p(max(workload_steps, 0))
```

注意：`*_workload_steps` 不是两个归一化后的数相乘。应先用 reward 同一套原始物理量计算：

```text
workload_steps = raw_cost[1 / bits_per_step] * raw_queue_or_drop_bits
```

这个乘积本身已经是“步数”量纲，再做 `log1p` 或直接输入。不要写成：

```text
(cost / cost_ref) * (queue / flow_ref)
```

后者会把 reference 重复引入，和 reward 实际计算不一致。

不要混用 `cost_norm`、`weighted_queue_norm`、`weighted_queue_cost_relative` 这类不清楚名字。

### 3.5 eps / floor 规则

不能到处裸加 `1e-9/1e-12`。不同分母的语义不一样：

```text
配置尺度:
  arrival_ref、bandwidth、map_size、noise_density 这类应为正。
  如果 <=0，直接报配置错误，不用 eps 修。

mask 平均:
  用 where(count > 0, sum / count, 0)，不要靠 eps。

log / probability:
  用明确的 log_floor/prob_floor，只是数值域保护。

workload EMA cost:
  这里的 floor 不是数值 eps，而是 reward 模型里的 service floor。
  必须命名为 service_floor_bits_per_step，并且 Python/native/reward/critic 完全一致。
```

当前代码里的旧字段是 `bw_weighted_workload_eps`，语义上就是这个 service floor。迁移时固定规则：

```text
service_floor_bits_per_step:
  新配置名，reward / critic / native 统一读取它。

bw_weighted_workload_eps:
  legacy alias。若新字段未配置，则用它初始化 service_floor_bits_per_step。
```

不要让 reward 继续读 `bw_weighted_workload_eps`、critic 另读 `service_floor_bits_per_step`，否则 cost/log ratio 会不一致。

如果 `service_floor_bits_per_step` 大到经常超过真实 EMA，它确实会淹没实际差异。这不是 critic 归一化问题，而是 reward cost 定义本身把小服务率全部截成同一个 cost。需要在诊断里统计：

```text
fraction(ema < service_floor_bits_per_step)
```

如果这个比例很高，就应该调小 floor 或改 cost 形式，而不是在 critic 输入里偷偷用另一个 eps。

## 4. 新 StructuredWorldState 字段

保留类型名 `StructuredWorldState`，但 critic schema 固定，不再跟 actor obs 开关走。

### 4.1 masks

基础 masks：

```text
gu_mask:
  GU slot 是否真实存在。当前固定 num_gu 时基本全 true，但保留用于 padding/zero-GU/未来变长。

sat_mask:
  critic SAT token slot 是否真实存在。
  它不是 visible mask，也不是 selected mask。
```

pair masks / dataclass masks：

```text
uav_gu_mask:
  结构 pair mask。语义固定为 GU g 存在并且 UAV u 存在。
  当前固定 num_uav 时等价于 gu_mask broadcast 到 [U,G]。
  它不是 candidate mask，也不是 bw_valid mask。

uav_sat_mask:
  结构 pair mask。语义固定为 critic SAT token s 存在并且 UAV u 存在。
  当前等价于 sat_mask broadcast 到 [U,S_crit]。
  它不是 visible mask、valid mask，也不是 selected mask。

uav_uav_mask:
  结构 pair mask。语义固定为 i != j。
```

action/prefix/last 关系不再作为 dataclass mask 增加字段，而是作为 edge flag + known flag 放进 edge features：

```text
uav_gu_edges[..., last_served_flag]
uav_gu_edges[..., prefix_bw_valid_flag]
uav_gu_edges[..., prefix_bw_valid_known]

uav_sat_edges[..., visible_flag]
uav_sat_edges[..., valid_flag]
uav_sat_edges[..., last_selected_flag]
uav_sat_edges[..., prefix_selected_flag]
uav_sat_edges[..., prefix_selected_known]
```

这样做的原因是：accel stage / sat stage 里很多 prefix 还未知，但 critic 仍然需要看见完整 GU-UAV / UAV-SAT 物理关系来估计未来 return。不能因为当前 `bw_valid` 或 `selected SAT` 未知，就把整条 message passing edge mask 掉。

SAT token 口径固定为当前 active / 可参与连接的 bounded SAT set，不使用全 `num_sat`。当前主配置里 `num_sat` 常见为 72/144，而每拍真正可能参与选择的 SAT 很少；全量 token 会把 critic 变成大星座图，成本高且多数 token 长期无关。

`critic_sat_ids` 只由当前 SAT actor candidate union 构造、稳定去重、补零到固定 `critic_sat_token_max`：

```text
visible_to_any_uav:
  当前每个 UAV 的 SAT actor candidate list 的去重 union。
  这里的 candidate list 必须和 sat action builder 使用同一份 `_stage_visible[u]`。
  实际数量 <= num_uav * sat_actor_candidate_max，通常明显更少，因为不同 UAV 会看到同一批 SAT。
```

正常情况下不需要截断：`critic_sat_token_max = min(num_sat, num_uav * sat_actor_candidate_max)` 已经覆盖 union 的理论上界。如果实现里发现 `len(visible_to_any_uav) > critic_sat_token_max`，这说明 `sat_actor_candidate_max_from_cfg(cfg)` 和 `_visible_sats_sorted(...)` 的截断口径不一致，应直接报错并修 helper，而不是静默截断。

BW stage 额外校验：

```text
prefix_selected_sat ⊆ critic_sat_ids
```

如果不满足，说明 sat action / visible cache 口径不一致，应报错或在 parity test 里暴露；不应靠额外 token 容量兜底。

固定配置：

```text
critic_sat_token_max =
  min(
    num_sat,
    num_uav * sat_actor_candidate_max
  )
```

其中：

```text
sat_actor_candidate_max =
  visible_sats_max if visible_sats_max is configured
  else sats_obs_max
```

这和当前 `_visible_sats_sorted(...)` 的候选截断口径一致。不要写死 `sats_obs_max`，否则当 `visible_sats_max > sats_obs_max` 且 structured SAT action 使用更长 `_stage_visible` 时，critic SAT 轴会漏掉合法可选 SAT。

这里的 `critic_sat_token_max` 是“最坏情况下所有 UAV 的 SAT actor candidates 完全不重合”的固定张量上限，不是每步实际 SAT token 数。每步实际有效 token 数为：

```text
len(unique(visible_to_any_uav))
```

然后 padding 到 `critic_sat_token_max`，用 `sat_mask` 区分真实 token 和 padding。

这样做的语义是：

```text
SAT tokens:
  只负责表达当前可能被 UAV 选择或已经由当前 prefix 选择的 SAT。

global_scalars:
  负责表达所有非 active SAT 的 residual queue/drop/processed。
```

上一拍 selected 但当前已经不 active 的 SAT 不进入 token。它们当前不能被 UAV 选择，和当前 action 没有链路边；如果还有队列残余，只会继续由自身 SAT CPU 处理，所以通过 residual global scalars 输入即可。对于当前 active 且上一拍也 selected 的 SAT，`last_selected_flag` 仍然写在对应 `uav_sat_edges` 上。

未进入 `critic_sat_ids` 的 SAT 不能通过 edges 参与当前选择，但它们的 queue/drop/processed residual 必须进入 global scalar，避免系统 reward 中的 SAT backlog 被漏掉。

`uav_gu_candidate_mask` 不作为 critic 主语义字段。candidate 是 actor 局部观测/动作支持集的实现概念；reward 里真正能服务的是当前 prefix 下的 `bw_valid`。candidate 只能留在 actor/local sparse builder 内部，不能进入 critic schema，也不能生成 candidate demand sum 这类 reward-relevant 统计。

### 4.2 `gu_nodes`

目标字段：

```text
0  gu_x                         = gu_pos_x / map_size
1  gu_y                         = gu_pos_y / map_size
2  gu_queue_steps               = gu_queue_bits / gu_flow_ref
3  gu_queue_fill                = gu_queue_bits / queue_max_gu
4  gu_expected_arrival_steps    = expected_arrival_bits_per_step / gu_flow_ref
5  gu_last_arrival_steps        = last_gu_arrival_bits / gu_flow_ref
6  gu_last_outflow_steps        = last_gu_outflow_bits / gu_flow_ref
7  gu_last_drop_steps           = gu_drop_bits / gu_flow_ref
8  gu_service_ema_steps         = gu_workload_ema_bits_per_step / gu_flow_ref
9  gu_local_cost_log_ratio      = log((1 / max(gu_service_ema_bits_per_step, service_floor_bits_per_step)) / gu_local_cost_ref)
10 gu_last_total_cost_log_ratio = log(last_gu_cost / gu_total_cost_ref)
11 gu_prefix_total_cost_log_ratio = log(prefix_gu_cost / gu_total_cost_ref), unknown 时 0
12 gu_prefix_cost_known         = 1 if prefix_gu_cost 当前 stage 完整可算 else 0
13 gu_last_workload_log1p       = log1p(last_gu_cost * gu_queue_bits)
14 gu_prefix_workload_log1p     = log1p(prefix_gu_cost * gu_queue_bits), unknown 时 0
```

说明：

- `last_gu_cost` 用上一拍真实 `last_association + last_sat_selection + current EMA` 算。它不是当前动作结果。
- `prefix_gu_cost` 只有在当前 prefix 已经足够完整时才 known。BW stage 一般完整；sat stage 缺 sat selection；accel stage 缺 post-accel association。
- drop 是 reward 的直接项，必须输入上一拍 `gu_drop`。
- `gu_last_workload_log1p` 和 `gu_prefix_workload_log1p` 是 reward-aligned backlog 特征，命名明确为 workload，不再叫 weighted_queue。

### 4.3 `uav_nodes`

目标字段：

```text
0  uav_x                         = uav_pos_x / map_size
1  uav_y                         = uav_pos_y / map_size
2  uav_vx                        = uav_vel_x / v_max
3  uav_vy                        = uav_vel_y / v_max
4  uav_energy                    = uav_energy / uav_energy_init
5  uav_queue_steps               = uav_queue_bits / uav_flow_ref
6  uav_queue_fill                = uav_queue_bits / queue_max_uav
7  uav_last_inflow_steps         = last GU->UAV inflow / uav_flow_ref
8  uav_last_outflow_steps        = last_uav_outflow_bits / uav_flow_ref
9  uav_last_drop_steps           = uav_drop_bits / uav_flow_ref
10 uav_service_ema_steps         = uav_workload_ema_bits_per_step / uav_flow_ref
11 uav_local_cost_log_ratio      = log((1 / max(uav_service_ema_bits_per_step, service_floor_bits_per_step)) / uav_local_cost_ref)
12 uav_last_total_cost_log_ratio = log(last_uav_cost / uav_total_cost_ref)
13 uav_prefix_total_cost_log_ratio = log(prefix_uav_cost / uav_total_cost_ref), unknown 时 0
14 uav_prefix_cost_known         = 1 if current prefix sat 已固定 else 0
15 uav_last_workload_log1p       = log1p(last_uav_cost * uav_queue_bits)
16 uav_prefix_workload_log1p     = log1p(prefix_uav_cost * uav_queue_bits), unknown 时 0
17 uav_prefix_bw_valid_count_frac = sum_g prefix_bw_valid_flag[u,g] / num_gu, unknown 时 0
18 uav_last_access_interference_log1p = log1p(last_interference_by_uav / access_noise_full_band)
```

说明：

- UAV 必须有入流和出流。当前代码有 `last_uav_outflow`；入流按上一拍 `last_gu_outflow` 和 `last_association` 在 transition 汇总处计算，并显式缓存为 `last_gu_to_uav_inflow_by_uav`。
- prefix cost known 的含义只针对 sat downstream 是否已知；不要用 last cost 填 prefix cost。
- `uav_prefix_bw_valid_count_frac` 是 prefix 已知后的服务集合大小；accel stage 填 0，并由 `uav_gu_edges[..., prefix_bw_valid_known]` 和 stage head 表示 unknown。
- `access_noise_full_band = noise_density * b_acc * 10^(access_noise_figure_db/10)`，其中 `b_acc` 是单个 UAV 的 access 总带宽。

### 4.4 `sat_nodes`

目标字段：

```text
0  sat_x                         = sat_ecef_x / orbit_pos_ref
1  sat_y                         = sat_ecef_y / orbit_pos_ref
2  sat_z                         = sat_ecef_z / orbit_pos_ref
3  sat_vx                        = sat_vel_x / sat_speed_ref
4  sat_vy                        = sat_vel_y / sat_speed_ref
5  sat_vz                        = sat_vel_z / sat_speed_ref
6  sat_queue_steps               = sat_queue_bits / sat_flow_ref
7  sat_queue_fill                = sat_queue_bits / queue_max_sat
8  sat_last_incoming_steps       = last_sat_incoming_bits / sat_flow_ref
9  sat_last_processed_steps      = last_sat_processed_bits / sat_flow_ref
10 sat_last_drop_steps           = sat_drop_bits / sat_flow_ref
11 sat_service_ema_steps         = sat_workload_ema_bits_per_step / sat_flow_ref
12 sat_cost_log_ratio            = log(sat_cost / sat_cost_ref)
13 sat_last_workload_log1p       = log1p(sat_cost * sat_queue_bits)
14 sat_prefix_selected_load_frac = prefix selected UAV count / num_uav, unknown 时 0
15 sat_prefix_load_known         = 1 if current prefix sat 已固定 else 0
16 sat_last_selected_load_frac   = last selected UAV count / num_uav
17 sat_proc_capacity_steps       = sat_processing_capacity_bits_per_step / sat_flow_ref
```

说明：

- SAT 也必须有入流、处理量、drop。当前代码已有 `last_sat_incoming`、`last_sat_processed`、`sat_drop`。
- `sat_cost` 只依赖 SAT EMA，所有 stage 可算。
- `prefix_selected_load` 和 `last_selected_load` 分开。

### 4.5 `uav_gu_edges`

边方向：

```text
edge[u,g] = GU_g - UAV_u
```

目标字段：

```text
0 rel_x                       = (gu_x - uav_x) / map_size
1 rel_y                       = (gu_y - uav_y) / map_size
2 horizontal_dist             = ||gu_xy-uav_xy|| / map_size
3 elevation_norm              = atan2(uav_height, horizontal_dist_m) / (pi/2)
4 access_se_ref               = SE under reference access bandwidth and no current BW action
5 last_bw_fraction            = 上一拍 UAV u 的接入带宽中分给 GU g 的比例
6 last_served_flag            = 1 if 上一拍 GU g 由 UAV u 服务 else 0
7 prefix_bw_valid_flag        = 1 if 当前 prefix 已知且 GU g 可由 UAV u 做 BW 分配 else 0
8 prefix_bw_valid_known       = 1 if 当前 stage 已经确定 association/bw_valid else 0
```

`last_bw_fraction` 的含义：

```text
它是“UAV u 自己的 access bandwidth b_acc 的分配比例”，
不是全系统带宽比例。
```

如果上一拍 GU g 不在 UAV u 的候选/服务集合，取 0。

`prefix_bw_valid_flag` 是当前 prefix 下的服务关系，不是 actor candidate。accel stage 里 association 未定，所以 `prefix_bw_valid_known=0`、flag 填 0；sat/bw stage 里 post-accel association 已定，所以 known=1，并按当前 `stage_bw_valid_mask` 填 flag。

噪声公式：

```text
NF_lin = 10^(access_noise_figure_db / 10)
access_noise_ref = noise_density * B_ref * NF_lin
SNR = P_gu * gain / access_noise_ref
SE = log2(1 + SNR)
```

这里 `B_ref` 固定为：

```text
B_ref = b_acc
```

这表示“如果参考整段 access bandwidth 看链路质量”。它仍不是当前 service。

`access_se_ref` 不考虑当前 BW action 后的 interference；它是 no-current-action 的 cross-link quality。因为它对所有 `(u,g)` 都计算，所以非服务 UAV 上的 `access_se_ref` 也能告诉 critic “这个 GU 如果发射，会被别的 UAV 接收得多强”，这是潜在干扰信息。

`access_se_ref` 本身已经是 `log2(1+SNR)`，属于压缩过的无量纲链路质量，不再额外归一化。`access_rx_power_log1p` 和它高度重复，不纳入这版固定 schema。

### 4.6 `uav_sat_edges`

边方向：

```text
edge[u,s] = SAT_s - UAV_u
```

目标字段：

```text
0  rel_x                       = (sat_ecef_x-uav_ecef_x) / orbit_pos_ref
1  rel_y                       = (sat_ecef_y-uav_ecef_y) / orbit_pos_ref
2  rel_z                       = (sat_ecef_z-uav_ecef_z) / orbit_pos_ref
3  rel_vx                      = (sat_vel_x-uav_vel_ecef_x) / sat_speed_ref
4  rel_vy                      = (sat_vel_y-uav_vel_ecef_y) / sat_speed_ref
5  rel_vz                      = (sat_vel_z-uav_vel_ecef_z) / sat_speed_ref
6  radial_velocity_norm        = dot(rel_pos, rel_vel) / (||rel_pos|| * sat_speed_ref)
7  range_norm                  = ||rel_pos|| / orbit_pos_ref
8  elevation_norm              = elevation_rad / (pi/2)
9  doppler_norm                = doppler_hz / doppler_ref
10 doppler_margin              = doppler_hz / nu_max, Doppler disabled 时填 0
11 backhaul_se_ref             = SE under reference backhaul bandwidth
12 visible_flag                = 1 if 当前几何下 UAV u 可见 SAT s else 0
13 valid_flag                  = 1 if 当前几何/约束下 UAV u 到 SAT s 有效 else 0
14 last_selected_flag          = 1 if 上一拍 UAV u 选 SAT s else 0
15 prefix_selected_flag        = 1 if 当前 prefix 已知且 UAV u 选 SAT s else 0
16 prefix_selected_known       = 1 if current sat prefix 已固定 else 0
17 prefix_backhaul_capacity_steps
                               = 当前 prefix 已知时，这条 UAV->SAT 共享回传容量 / uav_flow_ref，否则 0
```

噪声公式：

```text
NF_lin = 10^(backhaul_noise_figure_db / 10)
backhaul_noise_ref = noise_density * B_ref * NF_lin
SNR = P_uav * gain / backhaul_noise_ref
SE = log2(1 + SNR)
```

`B_ref` 固定为：

```text
B_ref = b_backhaul_per_sat
```

`b_backhaul_per_sat` 是“每颗 SAT 可被选中 UAV 分享的总回传带宽”。旧配置名 `b_sat_total` 容易被误解成全系统所有 SAT 的总回传带宽，因此只作为 legacy alias 保留。不要再除以 `sat_workload_ref_count` 或 visible/active SAT 数，否则会把“每颗 SAT 的带宽池”错误地再平均一次。

不要再用当前 selected load 去算 `backhaul_se_ref`，否则更早 stage 会混入尚未选择的东西。当前 selected load 信息已经在 edge flags / sat node 里；真正服务能力由另一个字段计算：

```text
prefix_backhaul_capacity_steps =
  (b_backhaul_per_sat / selected_uav_count_on_sat) * SE_at_that_bandwidth * tau0 / uav_flow_ref
```

这个量只有 `prefix_selected_known=1` 时才能作为 prefix capacity 输入。

`backhaul_se_ref` 和 access 一样，是 `log2(1+SNR)`，不再额外归一化。`backhaul_rx_power_log1p` 与它高度重复，不纳入这版固定 schema。

实现时必须保证 `backhaul_se_ref` 和 `prefix_backhaul_capacity_steps` 是两个不同字段：前者是固定 reference bandwidth 下的纯链路质量，后者是当前 prefix 已知后按 selected load 算出的可服务容量。

### 4.7 `uav_uav_edges`

目标字段：

```text
0 rel_x                       = (uav_j_x-uav_i_x) / map_size
1 rel_y                       = (uav_j_y-uav_i_y) / map_size
2 rel_vx                      = (uav_j_vx-uav_i_vx) / v_max
3 rel_vy                      = (uav_j_vy-uav_i_vy) / v_max
4 dist_norm                   = ||rel_xy|| / map_size
5 closing_speed_norm          = -dot(rel_pos,rel_vel) / max(||rel_pos||*v_max, geometry_eps)
6 alert_flag                  = 1[dist < avoidance_alert_factor*d_safe]
7 unsafe_flag                 = 1[dist < d_safe]
8 last_shared_sat_frac        = 上一拍共同选择 SAT 的比例
9 prefix_shared_sat_frac      = 当前 prefix 已知时共同选择 SAT 的比例，否则 0
10 prefix_shared_sat_known    = 1 if current sat prefix 已固定 else 0
```

`closing_speed_norm > 0` 表示正在靠近。这个符号必须和 safety/danger imitation 使用的相对量一致。

## 5. last / prefix / EMA 的统一规则

每个 reward-relevant 量只按下面规则出现，不要随意复制：

```text
EMA:
  当前状态里的历史服务尺度。
  用于 local cost，所有 stage 可用。

last:
  上一拍真实执行后的结果。
  用于告诉 critic 近期系统实际如何运行。
  所有 stage 可用。

prefix:
  当前 stage 已经固定的前序动作后果。
  只有因果上已知时 known=1。
  unknown 时数值填 0，但必须有 known flag。
```

不要做：

```text
prefix unknown 时用 last value 顶上。
```

应该做：

```text
last_value 单独输入；
prefix_value 单独输入；
prefix_known 告诉网络该 prefix 是否有效。
```

## 6. token 结构

top-level token 应该是：

```text
GU flow tokens
UAV relay tokens
SAT service tokens
system token
```

不是每个 `(u,g)` pair 一个 top-level route token。pair edge 是 message 的通道，不是系统 readout 的主体。

### 6.1 typed graph message passing 是否必须

因为 reward 的依赖是 typed flow：

```text
GU queue/drop/cost
  depends on access service and associated UAV

UAV queue/drop/cost
  depends on GU->UAV inflow and UAV->SAT backhaul

SAT queue/drop/cost
  depends on UAV->SAT incoming and SAT processing

interference
  depends on GU-to-all-UAV receiver cross-gain and BW fractions
```

typed graph message passing 不是为了“更深更花”，而是为了匹配这个 value function 的因果结构。`weighted_workload_level` 的 value 依赖 GU/UAV/SAT 三层队列和服务链路；如果只把节点做一次编码再全局池化，网络需要自己发明“GU 的 backlog 会流到哪个 UAV、UAV 又被哪个 SAT 限制”这条计算。

本方案直接使用 typed relational encoder：

```text
GU/UAV/SAT tokens
+ typed edge features
+ explicit physical sums
+ 2 层 residual typed message passing
+ system readout
```

它不是唯一能表达这个函数的网络。一个带 typed edge bias 的全 token Transformer 理论上也能表达。但在这个环境里，typed message passing 更贴近实际计算图：GU 通过 access edge 影响 UAV，UAV 通过 selected backhaul edge 影响 SAT，UAV-UAV 通过安全/干扰耦合互相影响。因此它不是实现成本问题，而是 inductive bias 是否对齐的问题。

typed message passing 不应该丢原始信息。每层都要做 residual update：

```text
message_{a->b} = MLP([source_token, target_token, edge_feature])
agg_b          = masked_sum(message_{a->b})
new_token_b    = old_token_b + typed_update_mlp_b(...)
```

也就是说，它不是先把所有 GU/UAV/SAT 压成一个池化向量，而是在保留每个 GU/UAV/SAT token 的前提下沿 typed edges 交换信息。相比当前 `UAV attend 一圈 -> mean(UAV)`，它丢的信息更少。

每层 typed update 固定为：

```text
GU  <- UAV messages through uav_gu_edges
UAV <- GU messages through uav_gu_edges
UAV <- SAT messages through uav_sat_edges
SAT <- UAV messages through uav_sat_edges
UAV <- UAV messages through uav_uav_edges
system <- GU/UAV/SAT messages + physical global scalars
```

固定做 2 层，让 GU/UAV/SAT 之间能形成系统瓶颈表示，同时避免 value model 在小样本 PPO 更新里过深。

### 6.2 为什么还需要 physical sums

只用 attention pooling 不够。attention 更像“挑重点”，但 reward 里很多东西是总量：

```text
sum queue
sum drop
sum cost * queue
sum expected arrival
sum last outflow
sum selected load
```

所以每个 UAV/SAT token update 应同时有：

```text
learned typed messages
fixed local summary vectors
```

例如 UAV relay token 不应该只写 `AttnPool(GU tokens)`，而要显式有：

```text
bw_valid_gu_queue_sum
bw_valid_expected_arrival_sum
last_served_gu_outflow_sum
last_served_gu_drop_sum
```

这些 sum 不是乱加字段，而是 reward 和流量守恒直接需要的量。`bw_valid_*` 是当前 prefix 已知时按 `prefix_bw_valid_flag` 汇总的服务集合；prefix unknown 时填 0，并依赖 known flag / stage head 告诉网络“不是没有需求，而是当前阶段还不能知道服务集合”。全局 GU queue / expected arrival 仍然通过 GU tokens 和 `physical_global_scalars` 输入。

`candidate_*_sum` 不作为 critic schema 保留。candidate 是 actor/local sparse builder 的动作支持集概念，不是 centralized critic 的 reward 主语义。

本设计把 local physical sums 固定成两个 summary 向量。GU 不单独加 local summary，因为 GU 节点自身已经包含 queue/arrival/outflow/drop/cost；GU 的瓶颈信息通过 UAV->GU typed message 进入。

```text
uav_local_summary[u]  # 8 维
0 prefix_bw_valid_gu_queue_sum_steps
1 prefix_bw_valid_expected_arrival_sum_steps
2 prefix_bw_valid_last_gu_outflow_sum_steps
3 prefix_bw_valid_last_gu_drop_sum_steps
4 last_served_gu_queue_sum_steps
5 last_served_expected_arrival_sum_steps
6 last_served_gu_outflow_sum_steps
7 last_served_gu_drop_sum_steps

sat_local_summary[s]  # 8 维
0 prefix_selected_uav_queue_sum_steps
1 prefix_selected_uav_last_inflow_sum_steps
2 prefix_selected_uav_last_outflow_sum_steps
3 prefix_selected_uav_last_drop_sum_steps
4 prefix_backhaul_capacity_sum_steps
5 last_selected_uav_queue_sum_steps
6 last_selected_uav_last_outflow_sum_steps
7 last_selected_uav_last_drop_sum_steps
```

所有 `prefix_*` summary 在对应 prefix unknown 时填 0；不要用 last summary 顶替。`last_*` summary 对所有 stage 可用。实现时这些 summary 由 `StructuredCritic.forward()` 根据已经定义好的 node/edge 字段和 flags 计算，不作为新的 `StructuredWorldState` 顶层字段保存；进入网络前分别过：

```text
uav_local_summary_encoder: [8] -> H -> E
sat_local_summary_encoder: [8] -> H -> E
```

system value 不再做一层 `concat(masked_sum/max(GU/UAV/SAT tokens))` 作为最终读出。系统级总量进入两条路径：

```text
1. physical_global_scalars -> global_scalar_encoder -> system token 初始化/更新
2. GU/UAV/SAT tokens -> typed messages -> system token
```

也就是说，sum/mean 这类聚合是 token update / system-token message 的内部操作，不是最终绕过 system token 的 readout。

### 6.3 system readout 输出什么

critic 输出的是：

```text
one scalar system value
```

即当前 stage state 的 expected return / baseline。

system readout 的输入应该是：

```text
final system token from graph message passing
stage-specific head
```

`physical_global_scalars` 固定为 26 维，不再写成“至少包括”。当前 schema 按下面索引实现：

```text
0  total_gu_queue_steps
1  total_uav_queue_steps
2  total_sat_queue_steps
3  total_gu_drop_steps
4  total_uav_drop_steps
5  total_sat_drop_steps
6  total_expected_arrival_steps
7  total_last_gu_outflow_steps
8  total_last_uav_outflow_steps
9  total_last_sat_processed_steps
10 total_last_weighted_workload_steps
11 total_prefix_weighted_workload_steps, unknown 时 0
12 prefix_workload_known
13 last_interference_mean
14 last_interference_max
15 selected_sat_load_mean, unknown 时 0
16 selected_sat_load_max, unknown 时 0
17 selected_sat_load_known
18 last_selected_sat_load_mean
19 last_selected_sat_load_max
20 non_token_sat_count_frac
21 non_token_sat_queue_steps
22 non_token_sat_drop_steps
23 non_token_sat_last_processed_steps
24 non_token_sat_workload_steps
25 non_token_sat_drop_workload_steps
```

`last_interference_mean/max` 是从 `uav_last_access_interference_log1p` 汇总来的，保留是有意义的：它直接告诉 critic 上一拍多 UAV access 干扰是否严重。`selected_sat_load_mean/max` 对 BW stage 有意义，因为 sat prefix 已固定；更早 stage 置 0 并配合 known flag。`last_selected_sat_load_mean/max` 对所有 stage 有意义，因为它是历史状态。

`total_last_weighted_workload_steps` 和 `total_prefix_weighted_workload_steps` 的队列部分都使用当前 world state 里的当前 queue，不是上一拍 queue：

```text
total_last_weighted_workload_steps =
  sum current_queue * last_route_cost

total_prefix_weighted_workload_steps =
  sum current_queue * prefix_route_cost
```

其中 `last_route_cost` 来自上一拍真实执行后的 association/sat selection 和当前 EMA；`prefix_route_cost` 来自当前 stage 已固定的 prefix。drop workload 不混进这两个字段，drop 已由 `total_*_drop_steps` 和 residual drop workload 字段单独表达。

`non_token_sat_*` 是 bounded SAT token 方案的补偿项，只统计没有进入 `critic_sat_ids` 的 SAT：

```text
non_token_sat_count_frac = count(non_token_sat) / num_sat
non_token_sat_queue_steps = sum queue / sat_flow_ref
non_token_sat_drop_steps = sum drop / sat_flow_ref
non_token_sat_last_processed_steps = sum last_processed / sat_flow_ref
non_token_sat_workload_steps = sum(sat_cost * sat_queue)
non_token_sat_drop_workload_steps = sum(sat_cost * sat_drop)
```

如果某个 SAT 已进入 token 集合，它的 queue/drop/processed/cost/workload 由 SAT token 和 typed messages 表达；如果没有进入 token 集合，它只能通过这些 residual scalars 进入 system token。注意 residual 不能只放 unweighted queue/drop，因为 `weighted_workload_level` reward 直接使用 `sat_cost * sat_queue/drop`。

当前三阶段下，global known flags 固定为：

```text
accel world:
  prefix_workload_known = 0
  selected_sat_load_known = 0

sat world:
  prefix_workload_known = 0
  selected_sat_load_known = 0

bw world:
  prefix_workload_known = 1
  selected_sat_load_known = 1
```

`total_prefix_weighted_workload_steps` 表示“当前 prefix 已经足够完整时，按 prefix 派生出的系统 weighted workload”。它不是“已经知道一部分就先填一部分”的混合量。accel/sat world 里即使已有部分前序动作，也必须填 0 并把 `prefix_workload_known=0`。本版 schema 不给 sat stage 单独输入“局部 prefix workload”；避免同一个字段在不同 stage 下变成不同含义。

stage conditioning 固定用 `stage-specific readout/head` 作为主机制，而不是只给 trunk 加一个 stage embedding 后共用同一个 head。原因不是实现成本，而是目标函数本身就是三个条件 value：

```text
V_accel = E[G | accel 未定]
V_sat   = E[G | accel 已固定，sat 未定]
V_bw    = E[G | accel/sat 已固定，BW 未定]
```

它们共享同一个物理系统表示，但读出的条件 value 不同。固定结构是：

```text
shared typed relational encoder
+ shared physical global scalars
+ head_accel / head_sat / head_bw
```

本方案不使用 stage embedding 作为 encoder 输入。stage 条件由两部分表达：prefix known flags 写进 state，最终读出使用 stage-specific value heads。这样避免同一个 head 同时拟合三种条件 value，也避免 stage token 和 prefix flags 互相重复。

system token 的形式固定为：

```text
z_sys^0 = learned_system_token + global_scalar_encoder(physical_global_scalars)

for each typed relational block:
  GU/UAV/SAT tokens 通过 typed edges 更新
  z_sys 通过来自 GU/UAV/SAT 的 typed aggregate messages 更新

V_stage = head_stage(z_sys^L)
```

因为本版固定 `Eg == Es == 128`，`global_scalar_encoder(...)` 可以直接加到 `learned_system_token` 上。实现时应 assert `critic_global_embed_dim == critic_system_token_dim`；不要静默广播或截断。

这里不使用 `alpha_i = softmax(q_stage · token_i)` 的单次 attention pooling 作为最终 readout，因为 softmax 加权和可能丢总量。`learned_system_token` 也不是和 sum/max 并列拼接的一个额外 feature，而是系统表示本身。

## 7. 要改成什么

### 7.1 输入构造

改造 `StructuredWorldState`：

- 固定 critic node/edge/mask schema。
- 新增 last/prefix edge flags 和 known flags。
- 新增 inflow/outflow/drop/interference 历史状态。
- cost 统一按 `*_cost_log_ratio` 和 `*_workload_log1p` 命名。
- queue 同时给 `steps` 和 `fill`，但 steps 是主 reward scale。

### 7.2 环境状态

Python env 需要新增或明确缓存：

```text
last_access_interference_by_uav
last_bw_fraction_by_uav_gu
last_gu_to_uav_inflow_by_uav
last_uav_to_sat_outflow_matrix
last_selected_mask_by_uav_sat
```

其中一些可由已有字段推导，但为了 native parity，本设计要求在 runtime state 里显式维护。

原则：

```text
critic 要用的 last_* 历史量都应该建成显式持久字段，
并在 Python env 和 native runtime state 中同步维护；
不要每个 builder 临时用不同口径重算。
```

### 7.3 native 路径

同步修改：

```text
sagin_marl/env/structured_driver.py
sagin_marl/env/structured_batch_env_core.py
sagin_marl/rl/structured_buffer.py
sagin_marl/rl/structured_stage_builders.py
sagin_marl/rl/structured_types.py
```

要求：

- Python world builder 和 native GPU world builder 字段顺序完全一致。
- 写 world-state parity test，逐字段检查 shape、mask、关键统计值。

### 7.4 critic 网络

替换当前：

```text
per-UAV context -> mean(UAV) -> value
```

改成：

```text
fixed node/edge/mask encoders
GU flow tokens
UAV relay tokens
SAT service tokens
typed graph message passing
system token + physical global scalars
stage-specific scalar value head
```

这才是和 `weighted_workload_level` 对齐的 centralized system critic。

## 8. critic 是否需要单独 kernel

需要把“critic 输入构造”和“critic 网络 forward”分开。

### 8.1 不应该做单独 kernel 的部分

`StructuredCritic` 本身不应该写成 env/native kernel。

原因：

```text
critic 是 PyTorch nn.Module。
它需要 autograd、optimizer、PopArt、torch.compile、checkpoint、AMP 等训练工具。
```

如果把 value forward 写进 native CUDA/env kernel，会带来三个问题：

```text
1. 梯度和 optimizer 不再自然接入 PyTorch。
2. 每次改网络结构都要改 kernel ABI。
3. 调试 EV / value loss / feature attribution 会更困难。
```

所以 critic neural network 应保持：

```text
sagin_marl/rl/structured_critic.py
  StructuredCritic(nn.Module)
```

### 8.2 应该做单独 kernel 的部分

critic 的输入构造应该有单独的 native builder/kernel。

这里的“单独”指的是**契约和实现模块独立**，不是指训练热路径里从 Python 额外调度一个小 kernel。单 GPU 原生架构的原则仍然是：

```text
不要 per-env / per-stage / per-field Python 调度。
不要把 node、edge、global scalar 拆成很多小 torch op / 小 kernel。
```

所以正确理解是：

```text
逻辑上:
  critic world builder 独立于 actor obs builder，有自己的 schema 和 parity test。

物理执行上:
  native rollout 热路径里融合到已有 main kernel / captured graph，
  一次性批量写出 accel/sat/bw/next 的 critic world tensors。
```

也就是说，“单独 builder”不是“多一次 Python 小调用”，而是“不要继续复用 actor obs builder 的动态 proxy schema”。

当前 native rollout 已经会预分配并写入：

```text
history.accel_stage.world_batch
history.sat_stage.world_batch
history.bw_stage.world_batch
history.terminal_next_world
```

相关代码：

```text
sagin_marl/env/structured_batch_env_core.py
  _NativeTrainingWorldTensorFields
  _allocate_native_training_world_tensor_fields(...)
  _build_world_from_packed_specs_tensor_impl(...)
  _build_world_from_stage_fields_direct_tensor_impl(...)
  StructuredBatchEnvCore._bind_native_main_kernel_history_outputs(...)

sagin_marl/env/structured_gpu_rollout_runtime.py
  StructuredGpuRolloutRuntime.history
```

这部分改造成 critic world builder。GPU 热路径固定为：

```text
runtime/env state
  -> native main rollout kernel / captured fused builder
  -> StructuredWorldState tensors on GPU
  -> StructuredCritic(nn.Module)
```

也就是说：

```text
需要单独 kernel:
  StructuredWorldState / physical_global_scalars / typed structural masks / edge flags 的 fused builder contract。

不需要单独 kernel:
  StructuredCritic.forward()。
```

正式实现只有一条热路径：

```text
在现有 native main-kernel 阶段写 history.world_batch 时，
直接按 critic schema 填好 StructuredWorldState。
这在执行上不是额外 kernel，只是 main kernel 多写一组正确字段。

Python builder 只用于 debug / parity，不能作为正式 native rollout 热路径。
```

不接受：

```text
for each env:
  for each stage:
    Python 调一次 builder
    再用很多小 torch op 拼 node/edge/scalar
```

这会重新引入之前单 GPU 原生架构特意避免的 Python 调度和小 kernel 问题。

不新建 `CriticStageState`。按本设计，保留 `StructuredWorldState` 这个类型名，但它的含义改成：

```text
centralized critic world state
```

而不是当前这种“actor obs proxy 拼出来的 world state”。

## 9. 按当前代码应该怎么改

这一节是代码改造蓝图。目标是实现者照着改，不再让实现和文档各说各话。

### 9.1 先固定 schema 和维度常量

当前：

```text
sagin_marl/rl/structured_types.py
  StructuredWorldState 只有 node/edge/mask/stage_id 字段，没有 global_scalars。

sagin_marl/env/structured_driver.py::_build_world_state(...)
  根据 actor obs 开关动态改变 node dim。

sagin_marl/env/structured_batch_env_core.py::_build_world_from_packed_specs_tensor_impl(...)
  native builder 也按当前动态 dim 写旧 schema。
```

目标：

新增一个集中 schema 定义位置，放在：

```text
sagin_marl/rl/structured_critic_schema.py
```

内容包括：

```python
CRITIC_GU_NODE_DIM = 15
CRITIC_UAV_NODE_DIM = 19
CRITIC_SAT_NODE_DIM = 18
CRITIC_UAV_GU_EDGE_DIM = 9
CRITIC_UAV_SAT_EDGE_DIM = 18
CRITIC_UAV_UAV_EDGE_DIM = 11
CRITIC_GLOBAL_SCALAR_DIM = 26
CRITIC_UAV_LOCAL_SUM_DIM = 8
CRITIC_SAT_LOCAL_SUM_DIM = 8

CRITIC_EMBED_DIM = 128
CRITIC_EDGE_EMBED_DIM = 128
CRITIC_GLOBAL_EMBED_DIM = 128
CRITIC_SYSTEM_TOKEN_DIM = 128
CRITIC_HIDDEN_DIM = 256
CRITIC_MESSAGE_LAYERS = 2
CRITIC_VALUE_HEAD_HIDDEN = 256

CRITIC_STAGE_ACCEL = 0
CRITIC_STAGE_SAT = 1
CRITIC_STAGE_BW = 2
```

`critic_sat_token_max` 不是模块级常量，因为它依赖配置。放在同一 schema 文件里做 config-derived helper：

```python
def sat_actor_candidate_max_from_cfg(cfg) -> int:
    candidate_max = int(cfg.visible_sats_max if cfg.visible_sats_max is not None else cfg.sats_obs_max)
    if candidate_max <= 0:
        raise ValueError("visible_sats_max/sats_obs_max must define a positive SAT candidate count")
    num_sat = int(cfg.num_sat)
    if num_sat <= 0:
        raise ValueError("num_sat must be positive")
    return min(num_sat, candidate_max)


def critic_sat_token_max_from_cfg(cfg) -> int:
    num_uav = int(cfg.num_uav)
    if num_uav <= 0:
        raise ValueError("num_uav must be positive")
    return min(int(cfg.num_sat), num_uav * sat_actor_candidate_max_from_cfg(cfg))
```

native shape / Python builder / tests 必须都调用同一个 helper，不要各自手写一遍公式。

并给每个 index 写常量或 enum，不要在 builder/critic 里散落魔法数字。例如：

```python
GU_QUEUE_STEPS = 2
UAV_LAST_ACCESS_INTERFERENCE_LOG1P = 18
UG_PREFIX_BW_VALID_FLAG = 7
US_PREFIX_BACKHAUL_CAPACITY_STEPS = 17
GLOBAL_TOTAL_PREFIX_WEIGHTED_WORKLOAD_STEPS = 11
GLOBAL_NON_TOKEN_SAT_QUEUE_STEPS = 21
```

`StructuredWorldState` 在 `structured_types.py` 里保留类型名，但改成固定 critic schema：

```python
@dataclass
class StructuredWorldState:
    uav_nodes: ArrayLike          # [B,U,19]
    gu_nodes: ArrayLike           # [B,G,15]
    sat_nodes: ArrayLike          # [B,S_crit,18]
    sat_ids: ArrayLike            # [B,S_crit], actual SAT id, padding 为 -1
    uav_gu_edges: ArrayLike       # [B,U,G,9]
    uav_sat_edges: ArrayLike      # [B,U,S_crit,18]
    uav_uav_edges: ArrayLike      # [B,U,U,11]
    global_scalars: ArrayLike     # [B,26]

    gu_mask: ArrayLike            # [B,G]
    sat_mask: ArrayLike           # [B,S_crit]
    uav_gu_mask: ArrayLike        # [B,U,G], 语义固定为结构 pair mask，不是 bw_valid
    uav_sat_mask: ArrayLike       # [B,U,S_crit], 语义固定为结构 pair mask，不是 visible/valid/selected
    uav_uav_mask: ArrayLike       # [B,U,U], 语义固定为 i != j
    stage_id: ArrayLike           # [B]
```

注意：

```text
uav_gu_mask / uav_sat_mask 保留旧字段名，避免 buffer/index/collate 大改。
但它们的语义必须在 schema 常量里固定：
  uav_gu_mask = gu_mask broadcast 到 [U,G]
  uav_sat_mask = sat_mask broadcast 到 [U,S_crit]

sat_ids 必须随 world state 一起进入 buffer/native history：
  sat_ids[b, slot] = 真实 SAT id
  padding slot = -1

critic forward 不需要把 sat_ids 当 feature 输入；它用于 builder/parity/debug，以及确保 Python/native 对同一 slot 使用同一颗 SAT。

其他 mask 不再另起 dataclass 字段，放在 edge flags：
  uav_gu_edges[..., last_served_flag]
  uav_gu_edges[..., prefix_bw_valid_flag]
  uav_gu_edges[..., prefix_bw_valid_known]
  uav_sat_edges[..., visible_flag]
  uav_sat_edges[..., valid_flag]
  uav_sat_edges[..., last_selected_flag]
  uav_sat_edges[..., prefix_selected_flag]
  uav_sat_edges[..., prefix_selected_known]
```

当前方案为了降低迁移风险，保留 `uav_gu_mask / uav_sat_mask` 字段名并固定语义，保证 actor/local-state/buffer 索引工具不被一次性打散。后续重命名不属于本设计范围。

`SaginConfig` 也要增加对应 critic 网络字段，不能继续让 structured critic 默认继承 actor 的 `embed_dim=64`：

```python
critic_embed_dim: int = 128
critic_edge_embed_dim: int = 128
critic_global_embed_dim: int = 128
critic_system_token_dim: int = 128
critic_hidden: int = 256
critic_message_layers: int = 2
critic_value_head_hidden: int = 256
```

`structured_factory.build_structured_modules_from_config(...)` 应从这些 critic 字段读取维度；actor 仍可保留 `actor_set_embed_dim=64`。不要再用同一个 `embed_dim` 同时控制 actor 和 critic。

### 9.2 改 Python env 持久状态

当前 Python env 已有一部分历史量：

```text
sagin_marl/env/sagin_env.py
  last_gu_arrival_rate_vec
  last_gu_arrival
  last_gu_outflow
  last_uav_outflow
  last_sat_processed
  last_sat_incoming
  gu_drop / uav_drop / sat_drop
  last_sat_selection
  last_sat_connection_counts
  gu_workload_ema / uav_workload_ema / sat_workload_ema
```

但 critic schema 还需要更明确的持久字段。应该在 `SaginParallelEnv.reset()` 初始化，并在真实 transition 里维护：

```text
last_access_interference_by_uav: [U]
last_bw_fraction_by_uav_gu: [U,G]
last_gu_to_uav_inflow_by_uav: [U]
last_uav_to_sat_outflow_matrix: [U,S]
last_selected_mask_by_uav_sat: [U,S]
```

具体改法：

```text
sagin_marl/env/sagin_env.py::_compute_access_rates(...)
```

当前这里会算：

```text
interference_by_u = _compute_access_interference_power(...)
exec_bw[u, slot] = beta
```

要在 `record_exec=True` 的真实执行路径写：

```text
self.last_access_interference_by_uav = interference_by_u
self.last_bw_fraction_by_uav_gu[u, gu_id] = beta
```

`record_exec=False` 的诊断/branch replay 不应污染这些历史字段。

```text
sagin_marl/env/sagin_env.py::_update_uav_queues(...)
```

当前这里会算：

```text
total_rate = sum(rate_matrix, axis=1)
last_uav_outflow = outflow
outflow_matrix = rate_matrix / total_rate * outflow
```

要把 `outflow_matrix` 持久化：

```text
self.last_uav_to_sat_outflow_matrix = outflow_matrix
```

并显式维护：

```text
self.last_gu_to_uav_inflow_by_uav[u] = sum_g last_gu_outflow[g] where assoc[g] == u
```

如果 `_update_uav_queues()` 当前没有 `assoc` 参数，就不要在 builder 里临时重算；应把 `assoc` 或上一拍 association 传进去，或者在调用它的 transition 函数里计算并存。

```text
sagin_marl/env/sagin_env.py::_apply_sat_selection / _compute_backhaul_rates / transition 汇总处
```

应维护：

```text
self.last_selected_mask_by_uav_sat[u,s] = 1 if s in last_sat_selection[u] else 0
```

虽然可由 `last_sat_selection` 推导，但 native runtime 已经是 tensor matrix 口径；Python 也必须显式存 matrix，以减少 Python/native builder 不一致。

### 9.3 改 native runtime state

当前 native runtime state 已有很多 `last_*`，但还不够完整。相关位置：

```text
sagin_marl/env/structured_batch_env_core.py
  _NativeRuntimeTensorState
  _NativeBwRuntimeStateBuffers
  StructuredBatchEnvCore._init_runtime_tensor_state(...)
  StructuredBatchEnvCore._build_native_cuda_runtime_abi(...)

sagin_marl/env/structured_gpu_rollout_runtime.py
  StructuredGpuNativeMainKernelBuffers
```

需要新增与 Python env 同名同形状的 tensor：

```text
last_access_interference_by_uav: [B,U]
last_bw_fraction_by_uav_gu: [B,U,G]
last_gu_to_uav_inflow_by_uav: [B,U]
last_uav_to_sat_outflow_matrix: [B,U,S]
last_selected_mask_by_uav_sat: [B,U,S]
```

如果 native 当前只有 drop sums，也要确认是否有 per-entity：

```text
gu_drop: [B,G]
uav_drop: [B,U]
sat_drop: [B,S]
```

critic node 字段需要 per-entity drop，只有 sum 不够。

这些字段要同步进入：

```text
runtime state dataclass / NamedTuple
state initialization
reset path
main kernel output/update path
Python fallback sync path
native CUDA ABI if main kernel needs direct access
```

关键原则：

```text
critic builder 只能读 runtime state。
不能在 builder 里用另一套公式临时重算 last_*。
```

### 9.4 改 Python `StructuredWorldState` builder

当前 Python builder：

```text
sagin_marl/env/structured_driver.py::_build_world_state(...)
```

问题：

```text
1. sat/uav/gu node dim 受 actor obs 开关影响。
2. gu_nodes 会调用 _gu_proxy_feature_arrays(...)。
3. uav_sat_edges[...,7] 是 projected-load SE，不是 backhaul_se_ref。
4. uav_sat_edges[...,10] 是 projected_bw / effective_b_backhaul_per_sat。
5. stage prefix 和 last history 混在旧 flags 里。
```

目标改法：

保留 `_build_world_state(...)` 函数名，但函数契约改成 critic schema builder：

```text
_build_world_state(...)
  -> build_critic_world_state(...)
```

实现时不要再读 actor obs 开关：

```text
obs_own_include_assoc_uav_cost
obs_own_include_uav_id_norm
obs_sat_include_sat_cost
_gu_proxy_feature_arrays(...)
```

这些是 actor/local obs 概念，不应该影响 centralized critic schema。

builder 应按 stage 明确 known flag：

```text
stage_id == ACCEL:
  association prefix unknown
  sat prefix unknown
  prefix_*_known = 0

stage_id == SAT:
  post-accel association / bw_valid known
  sat prefix unknown
  uav_gu prefix_bw_valid fields 可按 post-accel association 算
  full downstream prefix cost 仍 unknown
  SAT/backhaul prefix fields known = 0

stage_id == BW:
  association known
  sat selection known
  prefix cost / prefix selected / prefix backhaul capacity known
```

builder 还必须先构造 bounded `critic_sat_ids`，并用它填 `sat_ids / sat_nodes / uav_sat_edges / sat_mask / uav_sat_mask`。不要直接使用旧 `_active_sat_ids(visible, sat_selection)` 作为 critic SAT 轴；旧函数会把 `sat_selection` 也并入 active set，这不是本方案的 token 规则。critic SAT 轴只能来自 `visible_to_any_uav`，然后在 BW stage 校验 selected 是它的子集。

```text
candidate order:
  visible_to_any_uav

deduplicate preserving order
pad with -1 to critic_sat_token_max

sat_mask[slot] = critic_sat_ids[slot] >= 0
world.sat_ids[slot] = critic_sat_ids[slot]
```

BW stage 需要额外校验：`prefix_selected_sat` 必须属于 `critic_sat_ids`。如果不属于，说明 visible/selection cache 或 action validity 有 bug；不要为了这个异常路径扩大 `critic_sat_token_max`。

所有 `uav_sat_edges[u, slot, :]` 用 `sat_id = critic_sat_ids[slot]` 查真实 SAT；padding slot 全 0，mask 为 false。

`uav_gu_edges[...,7:9]` 按 stage 写：

```text
accel:
  prefix_bw_valid_flag = 0
  prefix_bw_valid_known = 0

sat/bw:
  prefix_bw_valid_flag = stage_bw_valid_mask[u,g]
  prefix_bw_valid_known = 1
```

`uav_sat_edges[...,12:17]` 按 stage 写：

```text
visible_flag / valid_flag:
  所有 stage 都按当前几何/约束写。

last_selected_flag:
  所有 stage 都按上一拍真实选择写。

prefix_selected_flag / prefix_selected_known:
  accel/sat stage: flag=0, known=0
  bw stage:        flag=current sat selection, known=1
```

`uav_sat_edges[...,11] = backhaul_se_ref` 应使用：

```text
B_ref = cfg.b_backhaul_per_sat
selected load 不参与 backhaul_se_ref
```

`uav_sat_edges[...,17] = prefix_backhaul_capacity_steps` 才使用：

```text
selected_uav_count_on_sat
B_share = b_backhaul_per_sat / selected_uav_count_on_sat
SE_at_that_bandwidth
tau0 / uav_flow_ref
```

并且只有 `prefix_selected_known=1` 时填，否则为 0。

### 9.5 改 native critic world builder

当前 native builder：

```text
sagin_marl/env/structured_batch_env_core.py
  _NativeTrainingWorldTensorFields
  _allocate_native_training_world_tensor_fields(...)
  _build_world_from_packed_specs_tensor_impl(...)
  _build_world_from_stage_fields_direct_tensor_impl(...)
```

目标：

```text
_NativeTrainingWorldTensorFields 增加 global_scalars。
所有 tensor shape 改为固定 critic schema dims。
```

例如：

```python
uav_nodes:      [B,U,CRITIC_UAV_NODE_DIM]
gu_nodes:       [B,G,CRITIC_GU_NODE_DIM]
sat_nodes:      [B,S_crit,CRITIC_SAT_NODE_DIM]
uav_gu_edges:   [B,U,G,CRITIC_UAV_GU_EDGE_DIM]
uav_sat_edges:  [B,U,S_crit,CRITIC_UAV_SAT_EDGE_DIM]
uav_uav_edges:  [B,U,U,CRITIC_UAV_UAV_EDGE_DIM]
global_scalars: [B,CRITIC_GLOBAL_SCALAR_DIM]
```

`_build_world_from_packed_specs_tensor_impl(...)` 不应再接收 actor proxy feature 作为 critic 关键字段：

```text
gu_proxy_features_t
uav_assoc_uav_cost_t
sat_cost_norm_active_t
```

兼容期保留这些旧参数以减少调用链改动，但它们不能写入 critic schema；critic 字段只能从 runtime state / typed params 直接构造 reward-aligned 字段。

native builder 同样要生成 `critic_sat_ids: [B,S_crit]`，并用它 gather SAT queue/position/velocity/cost/drop/last fields。不要把 `active_sat_ids` 当成 critic SAT 轴；`active_sat_ids` 只作为 `visible_to_any_uav` 的候选输入。

`_NativeTrainingWorldTensorFields` 也要保存 `sat_ids: [B,S_crit]`。否则 native world 和 Python world 即使 shape 相同，也无法证明第 k 个 SAT token 对应同一颗卫星。

`_build_world_from_stage_fields_direct_tensor_impl(...)` 即使已经有 `stage_id` 参数，也不能只在最后写 `world.stage_id`。它必须在填 node/edge/global 字段时就使用 stage，按 stage 写 prefix known flags；否则 node/edge 中的 prefix 字段和 `stage_id` 会不一致。

### 9.6 改 buffer 和 rollout history

当前 buffer 是 dataclass 泛型拼接，相关位置：

```text
sagin_marl/rl/structured_buffer.py
  _collate_dataclass(...)
  _index_dataclass_items(...)
  _where_world_state(...)
  add_env_step_batch(...)

sagin_marl/env/structured_batch_env_core.py
  runtime.preallocate_native_main_kernel_training_ring_buffers(...)
  _bind_native_main_kernel_history_outputs(...)
```

`StructuredWorldState` 新增 `global_scalars` 后，这些函数必须同步支持新字段。

检查点：

```text
1. _collate_dataclass 能拼 global_scalars。
2. _index_dataclass_items 能切 global_scalars。
3. _where_world_state 能在 terminal_next_world 和 next_actor_world 之间选择 global_scalars。
4. native history ring 里 accel/sat/bw/terminal_next_world 都预分配 global_scalars。
5. CPU/Python rollout 和 native rollout 的 world_batch 字段完全一致。
```

### 9.7 改 `StructuredCritic`

当前：

```text
sagin_marl/rl/structured_critic.py
  _relational_tokens(...)
  _uav_context_base(...)
  _team_context(...)

核心路径：
  UAV query attends GU/SAT/UAV pairs
  uav_ctx.mean(dim=1)
  optional _global_features(old indices)
  stage head
```

要删除或废弃：

```text
_global_features(...) 里按旧 index 手写统计。
_uav_context_base(...)
_team_context(...) 里的 uav_ctx.mean(dim=1)。
stage_embedding 作为主要 stage conditioning。
```

目标模块：

```text
gu_encoder
uav_encoder
sat_encoder
uav_gu_edge_encoder
uav_sat_edge_encoder
uav_uav_edge_encoder
global_scalar_encoder

typed relational blocks x 2
system readout
value_accel_head
value_sat_head
value_bw_head
```

网络维度固定为：

```text
E  = 128   # GU/UAV/SAT token dim
Ee = 128   # UG/US/UU edge token dim
Eg = 128   # global scalar embedding dim
Es = 128   # system token dim
H  = 256   # all message/update/head hidden dim
L  = 2     # typed relational blocks
```

本版默认 `E == Ee == Eg == Es == 128`，这样实现最简单。但代码里仍按各自名字建层，层输入宽度按下面公式写死到模块构造里，不要在 forward 里猜。

具体层宽：

```text
node encoders:
  gu_nodes[15]  -> H -> E
  uav_nodes[19] -> H -> E
  sat_nodes[18] -> H -> E

edge encoders:
  uav_gu_edges[9]   -> H -> Ee
  uav_sat_edges[18] -> H -> Ee
  uav_uav_edges[11] -> H -> Ee

global scalar encoder:
  global_scalars[26] -> H -> Eg

local summary encoders:
  uav_local_summary[8] -> H -> E
  sat_local_summary[8] -> H -> E

typed edge message MLP:
  [source_token, target_token, edge_token] = 2E + Ee = 384
  (2E + Ee) -> H -> E

node update MLP:
  GU update:
    [gu_token, agg_U_to_G] = 2E = 256
    256 -> H -> E

  UAV update:
    [uav_token, agg_G_to_U, agg_S_to_U, agg_U_to_U, uav_local_summary_embed] = 5E = 640
    640 -> H -> E

  SAT update:
    [sat_token, agg_U_to_S, sat_local_summary_embed] = 3E = 384
    384 -> H -> E

system message MLP:
  使用 G->SYS / U->SYS / S->SYS 三个 typed MLP
  [token, z_sys] = E + Es = 256
  (E + Es) -> H -> Es

system update MLP:
  [z_sys, sum(G->SYS), sum(U->SYS), sum(S->SYS), global_embed] = 4Es + Eg = 640
  (4Es + Eg) -> H -> Es

stage value heads:
  z_sys[128] -> H -> 1
```

这里没有 `readout_hidden=512`，也没有最终 `concat(masked_sum/max(GU/UAV/SAT tokens), z_sys, global_embed)`。如果需要总量信息，放进 `global_scalars`；如果需要局部瓶颈信息，通过 typed messages 写入 `z_sys`。

typed relational block 的输入输出：

```text
GU tokens:  [B,G,E]
UAV tokens: [B,U,E]
SAT tokens: [B,S_crit,E]
edge tokens:
  UG [B,U,G,Ee]
  US [B,U,S,Ee]
  UU [B,U,U,Ee]
masks:
  gu_mask, sat_mask, uav_gu_mask, uav_sat_mask, uav_uav_mask
```

typed aggregate 的维度和 mask 固定为：

```text
msg_U_to_G[u,g] = MLP_UG_to_G([uav_token[u], gu_token[g], ug_edge[u,g]])
agg_U_to_G[g]   = sum_u msg_U_to_G[u,g] * uav_gu_mask[u,g]

msg_G_to_U[u,g] = MLP_UG_to_U([gu_token[g], uav_token[u], ug_edge[u,g]])
agg_G_to_U[u]   = sum_g msg_G_to_U[u,g] * uav_gu_mask[u,g]

msg_S_to_U[u,s] = MLP_US_to_U([sat_token[s], uav_token[u], us_edge[u,s]])
agg_S_to_U[u]   = sum_s msg_S_to_U[u,s] * uav_sat_mask[u,s]

msg_U_to_S[u,s] = MLP_US_to_S([uav_token[u], sat_token[s], us_edge[u,s]])
agg_U_to_S[s]   = sum_u msg_U_to_S[u,s] * uav_sat_mask[u,s]

msg_U_to_U[i,j] = MLP_UU_to_U([uav_token[j], uav_token[i], uu_edge[i,j]])
agg_U_to_U[i]   = sum_j msg_U_to_U[i,j] * uav_uav_mask[i,j]

msg_G_to_SYS[g] = MLP_G_to_SYS([gu_token[g], z_sys])
msg_U_to_SYS[u] = MLP_U_to_SYS([uav_token[u], z_sys])
msg_S_to_SYS[s] = MLP_S_to_SYS([sat_token[s], z_sys])

agg_G_to_SYS = sum_g msg_G_to_SYS[g] * gu_mask[g]
agg_U_to_SYS = sum_u msg_U_to_SYS[u]
agg_S_to_SYS = sum_s msg_S_to_SYS[s] * sat_mask[s]
```

当前 schema 里 UAV 是固定数量的真实实体，不存在 padded UAV slot，所以 `agg_U_to_SYS` 对所有 UAV 求和，不再引入 `uav_mask`。GU/SAT 有 padding 或 inactive slot，必须用 `gu_mask` / `sat_mask`。

这里使用 `sum` 而不是 final pooling，因为总量本身对 `weighted_workload_level` 有物理含义；不存在有效元素时对应 aggregate 填 0。

每层 residual update：

```text
GU <- aggregate messages from UAV through UG edges
UAV <- aggregate messages from GU through UG edges
UAV <- aggregate messages from SAT through US edges
SAT <- aggregate messages from UAV through US edges
UAV <- aggregate messages from UAV through UU edges
SYSTEM <- aggregate typed messages from GU/UAV/SAT tokens + global scalar embedding

token_new = token_old + typed_update_mlp([...])
```

system token update 单独写成：

```text
z_sys^0 = learned_system_token + global_scalar_encoder(global_scalars)

z_sys^{l+1} =
  z_sys^l
  + MLP_system([
      z_sys^l,
      agg_G_to_SYS,
      agg_U_to_SYS,
      agg_S_to_SYS,
      global_scalar_encoder(global_scalars),
    ])
```

这里同样要求 `global_scalar_encoder(global_scalars)` 输出 `Es` 维。按本版默认配置 `Eg=Es=128`；构造网络时应显式 assert 两者相等。

最终 value 只读 system token：

```text
V_accel = head_accel(z_sys^L)
V_sat   = head_sat(z_sys^L)
V_bw    = head_bw(z_sys^L)
```

不要在最终 readout 再拼 `masked_sum/max(GU/UAV/SAT tokens)`。如果某个总量对 value 是硬需求，它应该进入 `global_scalars`；如果某个瓶颈需要从局部关系里判断，它应该通过 typed messages 写进 `z_sys`。

保留现有 PopArt 接口，作为可配置 value target 归一化机制。实现要求：

```text
popart_mean / popart_var 仍然按 stage 维护 3 份。
stage-specific head 的 PopArt rescale 逻辑保留。
```

### 9.8 改 MAPPO value 调用处

当前 value 调用入口：

```text
sagin_marl/rl/structured_mappo.py
  _evaluate_world_batch_for_stage(...)
  _stage_value_eval_from_batch(...)
```

这部分原则上不用改接口：

```text
critic.value_accel(world_batch)
critic.value_sat(world_batch)
critic.value_bw(world_batch)
```

但要确认：

```text
world_batch.global_scalars 在所有 minibatch/index/replay bank 路径里存在。
```

尤其检查：

```text
fixed-bank / replay-bank value fitting
clean teacher / update direction probe 如果用 world_batch
native rollout bootstrap
terminal_next_world
```

### 9.9 测试必须先写

实现前先补测试，避免又出现“文档和代码各一套”。

必须有：

```text
tests/test_structured_critic_world_schema.py
```

覆盖：

```text
1. Python builder 输出 shape:
   sat_nodes.shape[1] == critic_sat_token_max_from_cfg(cfg)
   sat_ids.shape == sat_mask.shape
   gu_nodes[-1] == 15
   uav_nodes[-1] == 19
   sat_nodes[-1] == 18
   uav_gu_edges[-1] == 9
   uav_sat_edges[-1] == 18
   uav_uav_edges[-1] == 11
   global_scalars[-1] == 26

2. stage known flags:
   accel world:
     prefix_bw_valid_known == 0
     prefix_selected_known == 0
     prefix_workload_known == 0
     selected_sat_load_known == 0
   sat world:
     prefix_bw_valid_known == 1
     prefix_selected_known == 0
     prefix_workload_known == 0
     selected_sat_load_known == 0
   bw world:
     prefix_bw_valid_known == 1
     prefix_selected_known == 1
     prefix_workload_known == 1
     selected_sat_load_known == 1

3. backhaul_se_ref:
   不随 selected_uav_count_on_sat 改变。

4. prefix_backhaul_capacity_steps:
   只在 BW stage / sat prefix known 时非零。
   会随 selected_uav_count_on_sat 变化。

5. last_* parity:
   Python env 执行一步后，last_access_interference_by_uav、
   last_bw_fraction_by_uav_gu、last_uav_to_sat_outflow_matrix
   与 world 字段一致。

6. local summary:
   StructuredCritic.forward 里由 node/edge/flag 计算出的
   uav_local_summary[8]、sat_local_summary[8] 与手算一致。
   prefix unknown 时 prefix summary 四项为 0。
   last summary 四项不受 prefix known flag 影响。

7. critic_sat_ids:
   有效 SAT token 等于 unique(_stage_visible[u]) 的稳定去重结果。
   sat_ids[sat_mask] 与该结果逐项一致。
   sat_ids[~sat_mask] == -1。
   当 visible_sats_max > sats_obs_max 时，使用 visible_sats_max 口径。
   BW stage 的 prefix_selected_sat 必须已经在 critic_sat_ids 中，否则测试失败。

8. non-token SAT residual:
   不在 critic_sat_ids 中的 SAT queue/drop/processed 分别进入 global_scalars[21]、[22]、[23]。
   不在 critic_sat_ids 中的 sat_cost * queue/drop 分别进入 global_scalars[24]、[25]。
```

native parity 测试：

```text
tests/test_structured_critic_world_native_parity.py
```

覆盖：

```text
同一个 seed / 同一个 stage / 同一个 runtime state 下：
Python builder 和 native builder 的关键字段 allclose。
至少检查：
  queue_steps
  expected_arrival_steps
  last_outflow_steps
  drop_steps
  access_se_ref
  backhaul_se_ref
  prefix_backhaul_capacity_steps
  global_scalars
  sat_ids / sat_mask
  masks
```

critic forward 测试：

```text
tests/test_structured_critic_system_readout.py
```

覆盖：

```text
1. StructuredCritic.forward 返回 accel/sat/bw 三个 value。
2. batch size / fixed S_crit SAT axis 正确。
3. mask 掉一个 GU/SAT 后 value 不受该 padded token 随机值影响。
4. system token 对总量敏感：
   复制一个高 queue GU 后，global_scalars 变化，并且 `z_sys` / value 输入确实变化。
```

### 9.10 迁移顺序

固定执行顺序：

```text
1. 加 schema constants 和 StructuredWorldState.global_scalars。
2. 加 Python env 缺失的 last_* 持久字段。
3. 改 Python _build_world_state 为 critic schema，先让非 native 路径过测试。
4. 改 native runtime state 和 _NativeTrainingWorldTensorFields。
5. 改 native world builder，跑 Python/native parity。
6. 替换 StructuredCritic 网络结构。
7. 跑 value fitting / PPO 小场景验证 EV。
```

不要先改 critic 网络再改 world schema。否则新 critic 仍然吃旧 world，EV 结论没有意义。
