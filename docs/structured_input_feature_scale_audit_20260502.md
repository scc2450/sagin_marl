# Structured Actor/Critic 输入语义与尺度审计

日期：2026-05-02

这份文档只汇总当前代码的真实输入口径和后续应统一修改的尺度方案。本文不代表代码已经完成这些尺度修改；除“已修正项”外，其余 `log1p` 等处理是下一步待改。

## 已确认的已修正项

- `remaining_horizon_frac` 已加入：actor accel/sat/bw 放在各自 `ego`，critic 放在 `global_scalars`。定义为 `(T_steps - 1 - t) / max(T_steps - 1, 1)`，范围 `[0, 1]`。
- critic 的 prefix SAT load 已改成按当前 stage prefix `sat_selection_matrix` 重算，不再用上一拍 `last_sat_connection_counts` 代替当前 prefix load。
- selected backhaul rate / capacity 路径已统一使用包含 atmospheric + rain loss 的 backhaul gain，并按当前 stage selection load 分摊每颗 SAT 的回传带宽。
- SAT actor 的 `US_BACKHAUL_SE_REF` 是选星前 reference bandwidth 下的纯 spectral efficiency，包含 Doppler attenuation 和 atmospheric/rain loss；它不是 selected capacity，也不是 bps。

## Raw LayerNorm 口径

- raw physical feature input LayerNorm 由 `structured_actor_input_norm_enabled` 和 `structured_critic_input_norm_enabled` 控制，默认关闭。
- actor 内部仍有 token embedding 后的 LayerNorm：SAT 的 `SatSelfAttentionBlock.attn_norm/ffn_norm`，BW 的 `BwCompetitionBlock.norm_attn/norm_ffn`。
- 这些 LayerNorm 作用在 encoder 后的 latent token embedding 上，不是直接把 `queue_steps`、`SE`、`flag` 等原始物理列混在一起标准化。

## Flow Ref 定义

当前 actor/critic 输入里，除下文明确列出的 weighted workload 例外外，`*_STEPS` 都不是裸 bits，而是除过对应实体尺度的 steps。

代码中的 reference 是：

```text
gu_flow_ref  = arrival_ref / num_gu
uav_flow_ref = arrival_ref / num_uav
sat_flow_ref = arrival_ref / active_sat_ref_count
```

其中 `arrival_ref` 是环境业务量尺度，`active_sat_ref_count` 来自 weighted workload 口径下的 active SAT 参考数量。

## 当前 `*_STEPS` 字段实际是什么

可以确定：actor/critic 输入中表示 queue、arrival、outflow、drop、service EMA、processed、incoming 的 `*_STEPS` 字段，目前都已经除过对应 `flow_ref`，不是裸 bits。

### Accel actor

- `EGO_UAV_QUEUE_STEPS`、`EGO_UAV_LAST_INFLOW_STEPS`、`EGO_UAV_LAST_OUTFLOW_STEPS`、`EGO_UAV_LAST_DROP_STEPS`、`EGO_UAV_SERVICE_EMA_STEPS`：`uav_bits / uav_flow_ref`。
- `CELL_QUEUE_STEPS_SUM`、`CELL_EXPECTED_ARRIVAL_STEPS_SUM`、`CELL_LAST_ARRIVAL_STEPS_SUM`、`CELL_LAST_OUTFLOW_STEPS_SUM`、`CELL_LAST_DROP_STEPS_SUM`：先算每个 GU 的 `gu_bits / gu_flow_ref`，再对 cell 内 GU 求和并除以 `num_gu`。
- `GU_QUEUE_STEPS`、`GU_EXPECTED_ARRIVAL_STEPS`、`GU_LAST_ARRIVAL_STEPS`、`GU_LAST_OUTFLOW_STEPS`、`GU_LAST_DROP_STEPS`、`GU_SERVICE_EMA_STEPS`：`gu_bits / gu_flow_ref`。
- `SAT_QUEUE_STEPS`、`SAT_LAST_INCOMING_STEPS`、`SAT_LAST_PROCESSED_STEPS`、`SAT_LAST_DROP_STEPS`、`SAT_SERVICE_EMA_STEPS`：`sat_bits / sat_flow_ref`。
- `SAT_LAST_OUTFLOW_STEPS`：对某个 UAV-SAT link 的上一拍 outflow，用 `uav_flow_ref` 归一。

### SAT actor

- `EGO_UAV_QUEUE_STEPS`、`EGO_UAV_LAST_INFLOW_STEPS`、`EGO_UAV_LAST_OUTFLOW_STEPS`、`EGO_UAV_LAST_DROP_STEPS`、`EGO_UAV_SERVICE_EMA_STEPS`：`uav_bits / uav_flow_ref`。
- `EGO_LAST_BACKHAUL_OUTFLOW_STEPS`：该 UAV 上一拍回传 outflow 总量除以 `uav_flow_ref`。
- `DEMAND_CELL_QUEUE_STEPS_SUM`、`DEMAND_CELL_EXPECTED_ARRIVAL_STEPS_SUM`、`DEMAND_CELL_LAST_ARRIVAL_STEPS_SUM`、`DEMAND_CELL_LAST_OUTFLOW_STEPS_SUM`、`DEMAND_CELL_LAST_DROP_STEPS_SUM`：先算每个 GU 的 `gu_bits / gu_flow_ref`，再对 cell 内 GU 求和并除以 `num_gu`。
- `SAT_QUEUE_STEPS`、`SAT_LAST_INCOMING_STEPS`、`SAT_LAST_PROCESSED_STEPS`、`SAT_LAST_DROP_STEPS`、`SAT_SERVICE_EMA_STEPS`：`sat_bits / sat_flow_ref`。
- `US_LAST_OUTFLOW_STEPS`：某个 UAV-SAT link 的上一拍 outflow 除以 `uav_flow_ref`。

### BW actor

- `uav_queue_steps`、`uav_last_inflow_steps`、`uav_last_outflow_steps`、`uav_last_drop_steps`、`uav_service_ema_steps`：`uav_bits / uav_flow_ref`。
- `gu_queue_steps`、`gu_expected_arrival_steps`、`gu_last_arrival_steps`、`gu_last_outflow_steps`、`gu_last_drop_steps`、`gu_service_ema_steps`：`gu_bits / gu_flow_ref`。
- `sat_queue_steps`、`sat_last_incoming_steps`、`sat_last_processed_steps`、`sat_last_drop_steps`、`sat_service_ema_steps`：`sat_bits / sat_flow_ref`。

### Critic

- GU node 的 `GU_QUEUE_STEPS`、`GU_EXPECTED_ARRIVAL_STEPS`、`GU_LAST_ARRIVAL_STEPS`、`GU_LAST_OUTFLOW_STEPS`、`GU_LAST_DROP_STEPS`、`GU_SERVICE_EMA_STEPS`：`gu_bits / gu_flow_ref`。
- UAV node 的 `UAV_QUEUE_STEPS`、`UAV_LAST_INFLOW_STEPS`、`UAV_LAST_OUTFLOW_STEPS`、`UAV_LAST_DROP_STEPS`、`UAV_SERVICE_EMA_STEPS`：`uav_bits / uav_flow_ref`。
- SAT node 的 `SAT_QUEUE_STEPS`、`SAT_LAST_INCOMING_STEPS`、`SAT_LAST_PROCESSED_STEPS`、`SAT_LAST_DROP_STEPS`、`SAT_SERVICE_EMA_STEPS`：`sat_bits / sat_flow_ref`。
- `GLOBAL_TOTAL_*_QUEUE_STEPS`、`GLOBAL_TOTAL_*_DROP_STEPS`、`GLOBAL_TOTAL_EXPECTED_ARRIVAL_STEPS`、`GLOBAL_TOTAL_LAST_*_OUTFLOW/PROCESSED_STEPS`：系统总量除以对应 `flow_ref`。它们不是裸 bits，但由于是 total sum，会随实体数量变大。
- `GLOBAL_NON_TOKEN_SAT_QUEUE_STEPS`、`GLOBAL_NON_TOKEN_SAT_DROP_STEPS`、`GLOBAL_NON_TOKEN_SAT_LAST_PROCESSED_STEPS`：non-token SAT 总量除以 `sat_flow_ref`。

### `*_STEPS` 的例外

以下字段名字带 `STEPS`，但实际不是简单 `bits / flow_ref`：

- `GLOBAL_TOTAL_LAST_WEIGHTED_WORKLOAD_STEPS`
- `GLOBAL_TOTAL_PREFIX_WEIGHTED_WORKLOAD_STEPS`
- `GLOBAL_NON_TOKEN_SAT_WORKLOAD_STEPS`
- `GLOBAL_NON_TOKEN_SAT_DROP_WORKLOAD_STEPS`

它们实际是 `cost * queue_or_drop_bits` 的 weighted workload 总和。目前没有再除以 workload ref，也没有 `log1p`。这些字段命名容易误导，后续应改名或改成明确的 `log1p(workload / workload_ref)`。

## 当前 `SE` 字段是否纯 SE

可以确定：当前 schema 中所有名字叫 `SE_REF` / `ACCESS_SE` / `BACKHAUL_SE` 的 actor/critic 输入字段，写入的都是纯 spectral efficiency，单位语义是 bps/Hz，不是 `bandwidth * SE`，也不是 bits/s。

当前普通 SE 公式是：

```text
SE = log2(1 + SNR)
```

access 若启用 `ergodic_rician`，公式是：

```text
SE = E_h[log2(1 + SNR_large_scale * h)]
```

因此 SE 本身已经包含链路物理公式里的 log。后续不应因为“SE 没有 log”再对 SE 做 `log1p(SE)`。若确实要压缩 SE 输入尺度，应先看范围；必要时用明确命名的线性归一，例如 `SE / se_ref`，不要改成 capacity，也不要乘 bandwidth。

### 当前纯 SE 字段清单

- Accel actor `GU_ACCESS_SE_REF`：`access_spectral_efficiency(SNR_access_ref)`，纯 SE。
- Accel actor `SAT_BACKHAUL_SE_REF`：`spectral_efficiency(SNR_backhaul_ref)`，纯 SE。
- SAT actor `US_BACKHAUL_SE_REF`：`spectral_efficiency(SNR_backhaul_ref)`，纯 SE。
- Critic `UG_ACCESS_SE_REF`：`access_spectral_efficiency(SNR_access_ref)`，纯 SE。
- Critic `US_BACKHAUL_SE_REF`：`spectral_efficiency(SNR_backhaul_ref)`，纯 SE。
- Legacy/local BW `eta_ref_feature` / `user_edges[...,6]`：虽然名字里叫 `eta`，当前代码写入的是 reference access SE，不是 rate。
- Native live SAT branch feature `feature == 7` 读取的是 `kFLiveSatObs + 3` 的 `US_BACKHAUL_SE_REF`，也是纯 SE。

未发现当前仍被使用的 `SE` 命名字段写入 `bandwidth * SE`。`write_sat_obs_row(...)` 这个旧 helper 中的 `sat_edges[...,7]` 也写 `spectral_efficiency(snr)`，且当前没有发现它被 hot path 调用。

## 当前确实是 capacity/rate 的字段

这些字段不是 SE，而是 `bandwidth * SE * tau0 / flow_ref` 或计算能力 bits 除以 `flow_ref`。它们当前是线性 steps，后续应统一改成 `log1p(capacity_steps)` 或新增并行 `*_LOG1P` 字段。

- Accel actor `SAT_PROC_CAPACITY_STEPS`：`effective_sat_cpu_freq / task_cycles_per_bit * tau0 / sat_flow_ref`。
- SAT actor `SAT_PROC_CAPACITY_STEPS`：同上。
- SAT actor `DEMAND_CELL_ACCESS_RATE_FULL_BW_REF_SUM`：`sum(b_acc * access_SE_ref * tau0 / gu_flow_ref for GU in cell) / num_gu`。
- BW actor `access_rate_full_bw_ref_steps`：`b_acc * access_SE_ref * tau0 / gu_flow_ref`。
- BW actor `prefix_backhaul_capacity_steps`：`selected_backhaul_bw_share * backhaul_SE_prefix * tau0 / uav_flow_ref`。
- Critic `SAT_PROC_CAPACITY_STEPS`：`effective_sat_cpu_freq / task_cycles_per_bit * tau0 / sat_flow_ref`。
- Critic `US_PREFIX_BACKHAUL_CAPACITY_STEPS`：`selected_backhaul_bw_share * backhaul_SE_prefix * tau0 / uav_flow_ref`。
- Legacy/native SAT branch feature `feature == 10`：读取 SAT processing capacity，属于 capacity steps。

## Accel Actor 输入逐字段处理建议

### Ego

| 字段 | 语义 | 当前/建议处理 |
|---|---|---|
| `EGO_X`, `EGO_Y` | UAV 位置 | `pos / map_size`，保留 |
| `EGO_VX`, `EGO_VY`, `EGO_SPEED` | UAV 速度/速率 | `vel / v_max`，保留 |
| `EGO_ENERGY` | 剩余能量比例 | `energy / energy_init`，保留 |
| `EGO_BOUNDARY_LEFT`, `RIGHT`, `BOTTOM`, `TOP` | 到边界距离比例 | `[0,1]`，保留 |
| `EGO_UAV_QUEUE_STEPS`, `EGO_UAV_QUEUE_FILL` | UAV 队列存量 | `queue_steps = queue / uav_flow_ref`；把 steps 改为 `log1p(queue_steps)` ；fill 保留 |
| `EGO_UAV_LAST_INFLOW_STEPS`, `OUTFLOW_STEPS`, `DROP_STEPS`, `SERVICE_EMA_STEPS` | 上一拍流入/流出/丢弃/服务 EMA | 当前都是 `bits / uav_flow_ref`；统一考虑 `log1p(steps)` |
| `EGO_UAV_LOCAL_COST_LOG_RATIO`, `EGO_UAV_LAST_TOTAL_COST_LOG_RATIO` | 服务成本相对参考 | 已是 log ratio，保留 |
| `EGO_UAV_LAST_WORKLOAD_LOG1P` | 上一拍 workload | 已 `log1p` |
| `EGO_UAV_LAST_ACCESS_INTERFERENCE_LOG1P` | access 干扰相对噪声 | 已 `log1p(I/N)`，保留 |
| `EGO_LAST_POLICY_ACCEL_X/Y`, `EGO_LAST_EXEC_ACCEL_X/Y`, `EGO_LAST_INTERVENTION_DX/DY/L2` | 上一拍策略/执行/安全层修正 | `accel / a_max` 或对应归一，保留 |
| `EGO_REMAINING_HORIZON_FRAC` | 剩余 episode horizon | `[0,1]`，保留 |

### Cell Summary

| 字段 | 语义 | 当前/建议处理 |
|---|---|---|
| `CELL_GU_COUNT_FRAC` | cell 内 GU 比例 | `[0,1]`，保留 |
| `CELL_QUEUE_STEPS_SUM` | cell 队列汇总 | 当前是 `sum(in_cell gu_queue_steps) / num_gu`，且已有 count fraction；后续改为 `log1p(sum_steps/num_gu)`  |
| `CELL_EXPECTED_ARRIVAL_STEPS_SUM`, `CELL_LAST_ARRIVAL_STEPS_SUM`, `CELL_LAST_OUTFLOW_STEPS_SUM` | cell 流量汇总 | 当前同样是 `sum(in_cell steps) / num_gu`；保留汇总语义，并统一考虑 `log1p` |
| `CELL_LAST_DROP_STEPS_SUM` | cell drop 汇总 | 当前是 `sum(drop_steps) / num_gu`； `log1p` |
| `CELL_LAST_WORKLOAD_LOG1P_SUM` | workload 汇总 | 名称已经是 log1p sum，保留，但应避免和线性 sum 混用 |
| `CELL_WORKLOAD_SHARE_GAP` | workload share 偏差 | signed ratio，保留 |
| `CELL_BOUNDARY_WORKLOAD_SUM`, `CELL_WEAK_LINK_WORKLOAD_SUM` | 风险 workload 汇总 | 后续用 `log1p` |
| `CELL_ACCESS_PRESSURE`, `CELL_INTERFERENCE_EXPOSURE` | access 压力/干扰 | ratio 或 `log1p(I/N)`，保留 |
| `CELL_DEMAND_MOMENT_X/Y`, `CELL_BOUNDARY_MOMENT_X/Y`, `CELL_WEAK_LINK_MOMENT_X/Y` | cell 内 demand moment | `relative_position / map_size`，保留 |

### GU Token

| 字段 | 语义 | 当前/建议处理 |
|---|---|---|
| `GU_X`, `GU_Y`, `GU_REL_X`, `GU_REL_Y`, `GU_DIST` | GU 位置/相对几何 | 用 map scale，保留 |
| `GU_QUEUE_STEPS`, `GU_QUEUE_FILL` | GU 队列存量 | `queue / gu_flow_ref`；后续 `log1p(queue_steps)`；fill 保留 |
| `GU_EXPECTED_ARRIVAL_STEPS`, `GU_LAST_ARRIVAL_STEPS`, `GU_LAST_OUTFLOW_STEPS` | 每步 bit-volume | 当前是 `bits / gu_flow_ref`；统一 `log1p(steps)` |
| `GU_LAST_DROP_STEPS` | drop bit-volume | 当前是 `drop / gu_flow_ref`； `log1p` |
| `GU_SERVICE_EMA_STEPS` | 服务 EMA bit-volume | 当前是 `ema / gu_flow_ref`； `log1p`，且 service floor 语义要统一 |
| `GU_LOCAL_COST_LOG_RATIO`, `GU_LAST_TOTAL_COST_LOG_RATIO` | cost | log ratio，保留 |
| `GU_LAST_WORKLOAD_LOG1P` | workload | 已 `log1p`，保留 |
| `GU_ACCESS_SE_REF` | access 链路 SE | 当前是纯 SE，不是 rate；保留纯 SE，不乘 bandwidth |
| `GU_LAST_ASSOC_TO_EGO`, `GU_LAST_BW_FRACTION_EGO`, `GU_LAST_SERVED_BY_EGO`, `GU_PRE_OWNER_IS_EGO` | 上一拍归属/带宽比例 | flag/fraction，保留 |
| `GU_HANDOFF_MARGIN_EGO`, `GU_OWNER_STABILITY_MARGIN`, `GU_EGO_TAKEOVER_GAP` | handoff 比较量 | signed normalized margin，保留但需保证 ref 一致 |
| `GU_PARTITION_BOUNDARY_WEIGHT`, `GU_EGO_LINK_WEAKNESS`, `GU_LAST_BW_SUM` | 分区/弱链路/BW 汇总 | ratio/fraction，保留 |
| `GU_LAST_NONSELF_INTERFERENCE_LOG1P` | 非本 UAV 干扰 | `log1p(I/N)`，保留 |

### Peer Token

`PEER_REL_X/Y`, `PEER_REL_VX/VY`, `PEER_DIST`, `PEER_CLOSING_SPEED` 走几何/速度 reference；`PEER_SAFE_DISTANCE_MARGIN` 是安全距离 margin；`PEER_UNSAFE_FLAG`, `PEER_ALERT_FLAG` 是 flag；`PEER_LAST_SHARED_SAT_FRAC` 是 fraction；`PEER_CELL_OFFSET` 是 cell summary 偏移。整体可保留，关键是相对速度必须存在且不能是脏值。

### Accel SAT Token

位置、速度、range、radial velocity、elevation、doppler 用物理 reference；`SAT_QUEUE_STEPS`, `SAT_LAST_*_STEPS`, `SAT_SERVICE_EMA_STEPS`, `SAT_LAST_OUTFLOW_STEPS` 当前都是除过对应 `flow_ref` 的 steps，后续统一 `log1p`；`SAT_PROC_CAPACITY_STEPS` 是 processing capacity steps，应改为 `log1p(capacity_steps)`；`SAT_COST_LOG_RATIO`, `SAT_LAST_WORKLOAD_LOG1P` 保留；`SAT_LAST_SELECTED_LOAD_FRAC`, `SAT_VISIBLE_FLAG`, `SAT_VALID_FLAG`, `SAT_LAST_SELECTED_FLAG` 保留；`SAT_BACKHAUL_SE_REF` 是 loss 后 reference bandwidth 下纯 SE，保留纯 SE。

## SAT Actor 输入逐字段处理建议

### Ego / Demand / Role

`SAT_EGO_*` 与 accel ego 中 UAV queue、flow、cost、workload、interference 的处理一致；`EGO_REMAINING_HORIZON_FRAC` 保持 `[0,1]`。`SAT_DEMAND_*` 中 count 保留；queue/arrival/outflow/drop/workload 都是 cell 汇总 bit-volume steps，当前实现为 `sum(in_cell steps) / num_gu`，已有 count fraction，因此保留汇总语义并统一 `log1p`。`DEMAND_CELL_ACCESS_RATE_FULL_BW_REF_SUM` 是 access full-BW capacity steps 汇总，不是 SE，应改为 `log1p(capacity_steps_sum)` 。

### SAT Token

`SAT_QUEUE_STEPS`, `SAT_LAST_*_STEPS`, `SAT_SERVICE_EMA_STEPS`, `US_LAST_OUTFLOW_STEPS` 当前都已经除过 `flow_ref`，后续统一 `log1p`；`SAT_PROC_CAPACITY_STEPS` 是 capacity steps，应改为 `log1p`；`SAT_COST_LOG_RATIO`, `SAT_LAST_WORKLOAD_LOG1P` 保留；`SAT_LAST_SELECTED_LOAD_FRAC` 保留。`US_REL_*`, `US_RANGE_NORM`, `US_RADIAL_VELOCITY_NORM`, `US_ELEVATION_NORM`, `US_DOPPLER_NORM`, `US_DOPPLER_MARGIN` 保留物理 reference。`US_BACKHAUL_SE_REF` 当前是 loss 后 reference SE，不乘 bandwidth；`US_VISIBLE_FLAG`, `US_VALID_FLAG`, `US_LAST_SELECTED_FLAG` 保留。

## BW Actor 输入逐字段处理建议

### Ego

`uav_queue_steps` 当前是 `uav_queue / uav_flow_ref`，后续 `log1p`；`uav_queue_fill` 保留；`uav_last_inflow/outflow/service_ema/drop_steps` 当前都已除以 `uav_flow_ref`，统一 `log1p`；`uav_local_cost_log_ratio`, `uav_last_total_cost_log_ratio`, `uav_last_workload_log1p`, `uav_last_access_interference_log1p` 保留；`remaining_horizon_frac` 保留。

### Selected SAT Token

`prefix_backhaul_capacity_steps` 是已选星后的 capacity，不是 SE，应改为 `log1p(capacity_steps)`；`sat_queue_steps`、`sat_last_incoming/processed/drop/service_ema_steps` 当前都已除以 `sat_flow_ref`，建议统一考虑 `log1p`；`sat_queue_fill` 保留；`sat_cost_log_ratio`, `sat_last_workload_log1p` 保留。

### GU Token

`gu_queue_steps` 当前是 `gu_queue / gu_flow_ref`，后续 `log1p`；`gu_queue_fill` 保留；`gu_expected_arrival/last_arrival/last_outflow/drop/service_ema_steps` 当前都已除以 `gu_flow_ref`，统一 `log1p`；`gu_local_cost_log_ratio`, `gu_last_total_cost_log_ratio`, `gu_last_workload_log1p` 保留；`access_rate_full_bw_ref_steps` 是 capacity steps，应改为 `log1p`; `cross_interference_mean/max_log1p` 保留。

## Critic 输入逐字段处理建议

### GU Node

`GU_X/Y` 保留；`GU_QUEUE_STEPS` 当前是 `gu_queue / gu_flow_ref`， `log1p`；`GU_QUEUE_FILL` 保留；`GU_EXPECTED_ARRIVAL_STEPS`, `GU_LAST_ARRIVAL_STEPS`, `GU_LAST_OUTFLOW_STEPS`, `GU_SERVICE_EMA_STEPS`, `GU_LAST_DROP_STEPS` 当前都已除以 `gu_flow_ref`，统一 `log1p`；`GU_LOCAL_COST_LOG_RATIO`, `GU_LAST_TOTAL_COST_LOG_RATIO`, `GU_PREFIX_TOTAL_COST_LOG_RATIO` 保留；`GU_PREFIX_COST_KNOWN` 保留；`GU_LAST_WORKLOAD_LOG1P`, `GU_PREFIX_WORKLOAD_LOG1P` 保留。

### UAV Node

位置、速度、能量保留；`UAV_QUEUE_STEPS` 当前是 `uav_queue / uav_flow_ref`，后续 `log1p`；`UAV_QUEUE_FILL` 保留；`UAV_LAST_INFLOW/OUTFLOW/SERVICE_EMA/DROP_STEPS` 当前都已除以 `uav_flow_ref`，统一 `log1p`；`UAV_LOCAL_COST_LOG_RATIO`, `UAV_LAST_TOTAL_COST_LOG_RATIO`, `UAV_PREFIX_TOTAL_COST_LOG_RATIO` 保留；`UAV_PREFIX_COST_KNOWN`, `UAV_PREFIX_BW_VALID_COUNT_FRAC` 保留；`UAV_LAST_WORKLOAD_LOG1P`, `UAV_PREFIX_WORKLOAD_LOG1P`, `UAV_LAST_ACCESS_INTERFERENCE_LOG1P` 保留。

### SAT Node

位置、速度保留；`SAT_QUEUE_STEPS` 当前是 `sat_queue / sat_flow_ref`，后续 `log1p`；`SAT_QUEUE_FILL` 保留；`SAT_LAST_INCOMING/PROCESSED/SERVICE_EMA/DROP_STEPS` 当前都已除以 `sat_flow_ref`，统一 `log1p`；`SAT_COST_LOG_RATIO`, `SAT_LAST_WORKLOAD_LOG1P` 保留；`SAT_PREFIX_SELECTED_LOAD_FRAC`, `SAT_PREFIX_LOAD_KNOWN`, `SAT_LAST_SELECTED_LOAD_FRAC` 保留；`SAT_PROC_CAPACITY_STEPS` 是 capacity steps，应改为 `log1p`。

### UAV-GU Edge

相对几何保留；`UG_ACCESS_SE_REF` 当前是 access 纯 SE，不是 capacity；`UG_LAST_BW_FRACTION`, `UG_LAST_SERVED_FLAG`, `UG_PREFIX_BW_VALID_FLAG`, `UG_PREFIX_BW_VALID_KNOWN` 保留。

### UAV-SAT Edge

相对位置、速度、range、radial velocity、elevation、doppler 保留；`US_BACKHAUL_SE_REF` 当前是 reference 纯 SE；`US_VISIBLE_FLAG`, `US_VALID_FLAG`, `US_LAST_SELECTED_FLAG`, `US_PREFIX_SELECTED_FLAG`, `US_PREFIX_SELECTED_KNOWN` 保留；`US_PREFIX_BACKHAUL_CAPACITY_STEPS` 是 selected capacity steps，应改为 `log1p`。

### UAV-UAV Edge

`UU_REL_X/Y`, `UU_REL_VX/VY`, `UU_DIST_NORM`, `UU_CLOSING_SPEED_NORM` 保留；`UU_ALERT_FLAG`, `UU_UNSAFE_FLAG` 保留；`UU_LAST_SHARED_SAT_FRAC`, `UU_PREFIX_SHARED_SAT_FRAC`, `UU_PREFIX_SHARED_SAT_KNOWN` 保留。

### Global Scalars

全局 total queue/drop/workload 是系统总量，随 GU/UAV/SAT 数变化。若训练场景规模固定，可以保留；若跨规模训练，建议改为 mean per entity 加显式 count/fraction。`GLOBAL_TOTAL_*_QUEUE_STEPS`, `GLOBAL_TOTAL_*_DROP_STEPS`, `GLOBAL_TOTAL_EXPECTED_ARRIVAL_STEPS`, `GLOBAL_TOTAL_LAST_*_OUTFLOW/PROCESSED_STEPS`, `GLOBAL_NON_TOKEN_SAT_QUEUE_STEPS`, `GLOBAL_NON_TOKEN_SAT_DROP_STEPS`, `GLOBAL_NON_TOKEN_SAT_LAST_PROCESSED_STEPS` 当前都不是裸 bits，而是 total steps；后续 `log1p`。`GLOBAL_TOTAL_LAST_WEIGHTED_WORKLOAD_STEPS`, `GLOBAL_TOTAL_PREFIX_WEIGHTED_WORKLOAD_STEPS`, `GLOBAL_NON_TOKEN_SAT_WORKLOAD_STEPS`, `GLOBAL_NON_TOKEN_SAT_DROP_WORKLOAD_STEPS` 当前是 weighted workload 总量，不是普通 flow steps，应 `log1p`。`GLOBAL_*KNOWN`, `GLOBAL_SELECTED_SAT_LOAD_*`, `GLOBAL_LAST_SELECTED_SAT_LOAD_*`, `GLOBAL_REMAINING_HORIZON_FRAC` 保留。

## 当前待统一修改的尺度原则

这些是下一步代码修改目标，目前并未全部实现：

1. 所有 queue/drop/backlog 类 `*_STEPS`：虽然已经除过 `flow_ref`，但仍是线性 steps；改成 `log1p(steps)` ，避免跨步积压压过 flag/ratio。
2. arrival/outflow/service/processed/EMA 类 `*_STEPS`：也是非负 bit-volume steps；如果作为 magnitude 输入，统一 `log1p(steps)`。
3. capacity/rate 类字段：必须改成 `log1p(capacity_steps)`，因为它们是能力上界，可能比当前业务流量大很多。
4. SE 字段：保持纯 SE，不乘 bandwidth；暂不默认 `log1p`。如后续实测 SE 范围仍压过其它字段，再用明确的 `SE / se_ref`。
5. workload 字段：统一 `log1p(workload)`。
6. global total 类字段：如果只在固定规模训练，可以保留 total。
