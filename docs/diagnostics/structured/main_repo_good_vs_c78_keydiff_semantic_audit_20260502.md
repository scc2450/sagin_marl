# 主仓库对照 c78/good 关键差异语义审计（2026-05-02）

本文只检查主仓库当前代码是否仍存在 `D:\研三上\毕设\process_compare\good_vs_c78_update1_process_diff.md` 中 “2026-05-02 关键差异极简版” 的同类问题。这里的“同类问题”按语义判断，不要求字段名、位置、旧 schema 完全一致。

本次只读代码并跑了两个 smoke：

- `configs/tmp/structured_joint_vs_ref_3uav_20gu_t250.yaml` 实例化结构化 actor/critic，检查 raw input norm。
- 同配置做 1-step native rollout，抓实际进入训练 buffer 的 actor/critic 输入范围。
- 用 1000 次采样比较 numpy `thomas_cluster_process(...)` 与当前 torch/native reset tape 的 Thomas cluster 分布。

## 总览

| 序号 | 当前结论 | 简要判断 |
|---:|---|---|
| 1 | 已关 raw input LayerNorm | actor 内部 attention LayerNorm 仍在，但 raw input norm 为 Identity；critic 无 LayerNorm。 |
| 2 | 部分仍需警惕 | 没有发现 `bw * SE` 这种几百/几千量级错误；但有 raw SE、step-ratio、global sum 列达到 5~20，可能影响初始梯度尺度。 |
| 3 | 基本无旧的边界偏置 | 当前 torch/native Thomas center 已采 `[0.1, 0.9] * map_size`；smoke 未见更靠边/更易裁剪。 |
| 4 | 原关键错误已消失 | local/live `bw_valid_mask` 不是全 1，当前 smoke 为约 `1 / num_uav`；stage/world 的 candidate/prefix 语义另列为低优先级 caveat。 |
| 5 | 已修 | history world 在各 stage action 前写入，不再明显写 BW action 后 state。 |
| 6 | active SAT env offset 未见旧问题 | 当前 active SAT 写入带 env offset；1-step buffer 未见 0/脏值型异常。 |
| 7 | 需要按字段区分 | last-load 字段应保留上一拍；prefix-load 字段应反映当前 prefix selection。当前 BW actor 的 prefix capacity 用当前 selected count，critic 的 prefix load fraction 仍疑似用上一拍 load。 |
| 8 | 仍需复核/修正 selected-rate 口径 | Python 路径有 atm+rain；native `backhaul_rate_for_selected_us(...)` 当前从裸 `gain_const / dist2` 算 selected rate。候选排序路径含 loss，但不是 reward/rate 路径。 |
| 9 | 仍有同类问题 | SAT actor live obs 的 `US_BACKHAUL_SE_REF` 仍用裸 backhaul gain/full-band noise；旧 helper 目前无调用，主要影响是 active live SAT actor writer。 |
| 10 | 已有 | accel local peer token 有 UAV-UAV 相对速度。 |
| 11 | 已有 | critic world `uav_sat_edges`/`sat_nodes` 已包含 rel_vel、doppler、SE/capacity、queue/load/selected 等语义信息。 |
| 12 | 已有 | critic world `uav_uav_edges` 已包含相对速度、安全/碰撞 flag、shared-sat 信息。 |
| 13 | 当前已移除 explicit time feature | 旧 feature 同时进入过旧 local actor ego 和 critic world；当前结构化 schema 不再暴露 `t/T_steps`。是否恢复应重新设计为 named horizon feature。 |

## 逐条审计

### 1. raw input LayerNorm

当前配置层面：

- `input_norm_enabled=True` 仍是全局旧字段。
- 结构化路径实际使用 `structured_actor_input_norm_enabled=False` 和 `structured_critic_input_norm_enabled=False`。
- `structured_factory.py` 用这两个结构化专用开关传给 `AccelPolicy / SatSubsetPolicy / BwPolicy / StructuredCritic`。

smoke 结果：

```text
actor LayerNorm count = 8
actor LayerNorm names = sat/bw self-attention block norms only
critic LayerNorm count = 0
```

解释：

- actor 里剩下的是 attention block 内部的 token LayerNorm，不是 raw physical feature input LayerNorm。
- `ego_norm / gu_norm / sat_input_norm / bw_policy.*_input_norm` 这些属性仍存在，但当前实例化为 `Identity()`。

结论：第 1 条在主仓库当前结构化路径下已关。

### 2. 所有 actor/critic 输入列的量级混杂

这条不能只看 `simple_access_gain`，所以做了 1-step native rollout，检查实际进入训练 buffer 的 stage local input 和 critic world input。

本次 smoke 中超过 `abs > 5` 的列主要是：

```text
stage0.local.gu_tokens[GU_ACCESS_SE_REF] max_abs ~= 15.16
stage1.local.sat_tokens[US_BACKHAUL_SE_REF] max_abs ~= 5.69
stage2.local.gu_tokens[access_rate_full_bw_ref_steps] max_abs ~= 15.08
stage2.local.gu_tokens[cross_interference_max_log1p] max_abs ~= 5.85
world.global_scalars[GLOBAL_TOTAL_EXPECTED_ARRIVAL_STEPS] max_abs = 20.0
```

没有看到旧 c78 那种 `eta_ref_feature ~= 697` 或未初始化 live buffer 进入训练的情况。`simple_access_gain(...)` 当前也不是旧的裸 `1 / d^2`，而是使用 A2G pathloss / LoS-NLoS / carrier frequency / loss 参数，并且 `simple_eta_from_gain(...)` 返回 SE/eta 量级，不再返回 `bandwidth * SE` rate 量级。

但仍要注意：

- `GU_ACCESS_SE_REF` 是 raw SE，和位置、mask、queue fill 等 O(1) 特征同层进入 actor token encoder。
- `access_rate_full_bw_ref_steps` 是“满带宽可服务多少步流量”的 step-ratio，不是 SE；在当前配置下也能到 15 左右。
- `GLOBAL_TOTAL_EXPECTED_ARRIVAL_STEPS=20` 是系统总和，不是单 GU 归一化值；它进入 critic global scalar embedding。

结论：旧的灾难性 rate 量级错误没有复现，但主仓库仍存在“同层混入 O(1) 与 5~20 量级特征”的设计事实。它不一定是 bug，但会影响第一层线性层的初始梯度尺度；如果某些 10~20 量级列长期稳定占主导，可能拖慢或偏置训练。它和旧 `bw * SE` 错误不是同一量级的问题。

### 3. Thomas cluster 生成方式

当前 torch/native reset tape 的 GU cluster center 已经是：

```text
center_low = 0.1 * map_size
center_span = 0.8 * map_size
centers = rand * center_span + center_low
```

这和 numpy `thomas_cluster_process(...)` 的 center 范围一致，不再是旧的 `torch.rand * map_size`。

1000 次分布 smoke：

```text
clip_coord_frac: numpy 0.00295, torch 0.00308
gu_edge5pct_coord_frac: numpy 0.02170, torch 0.02310
center_outside_[0.1,0.9]_frac: numpy 0.0, torch 0.0
min_center_dist_mean: numpy 798.15, torch 798.24
cluster_count_std: numpy 1.94, torch 1.90
```

结论：

- 没看到“系统性更靠边、更分散、更容易边界裁剪”的旧问题。
- 当前 torch 版本不是 bit-exact 复刻 numpy RNG，但分布层面已接近。
- 以后不要为了对齐倒回 numpy；如果要进一步收敛，应在 GPU/native 上复刻同一分布语义。

### 4. BW valid 语义

当前 local/live BW obs 路径：

```cpp
const int owner = assoc[g];
const bool is_valid = owner == u;
a.b[kBLiveBwObs + 2][row * gu + g] = is_valid;
```

这条是正确语义。

另一个低优先级 caveat 是 `refresh_stage_derived_parallel(...)` 里：

```cpp
bool valid = gid >= 0 && gid < gu && c < valid_count;
a.f[kSfBwValidMask][candidate_slot] = valid ? 1 : 0;
a.f[kSfBwValidFlag][u,gid] = 1;
```

这里仍然是 candidate-valid 语义，没有再次检查 `assoc[gid] == u`。在当前配置 `candidate_mode=assoc` 下，candidate 本身只从关联 GU 里选，所以不会暴露；并且这不是当初 “local/live `bw_valid_mask` 全 1” 的关键错误。

结论：当初最关键的 local/live `bw_valid_mask` 全 1 问题已经没有。stage/world 的 prefix flag 在非 assoc candidate mode 下是否需要强制 owner check，是另一个可单独处理的问题。

### 5. history world 写入时序

当前 native 主路径：

- `prepare_initial_accel_live_kernel(...)`：准备 accel stage 后，先写 accel world，再写 accel obs。
- `accel_to_sat_live_kernel(...)`：执行 accel 后准备 sat stage，先写 sat world，再写 sat obs。
- `sat_to_bw_live_kernel(...)`：写入 sat selection 到 BW stage 后，先写 BW world，再写 BW obs。
- `finish_commit_prepare_live_kernel(...)`：之后才执行 arrival/access/backhaul/queue/reward。

结论：critic history world 现在是 stage-prefix/action-before state，不再明显偷看 BW action 执行后的 queue state。

### 6. UAV-SAT active feature env offset / 脏值

当前 active SAT 相关写入使用了 env 维 offset，例如：

```cpp
active_us_idx = e * num_uav * active + idx
kSfUsRelPosActive[active_us_idx * 3 + d]
kSfUsRelVelActive[active_us_idx * 3 + d]
kSfUsGainActive[active_us_idx]
kSfUsNuEffActive[active_us_idx]
kSfVisibleFlagActive[active_us_idx]
kSfUsValidFlagActive[active_us_idx]
```

1-step buffer smoke 中 world/local 输入 finite 比例正常，没有看到 env1..N 读到全 0 或脏值的模式。

结论：旧的 active feature 缺 env offset 问题未复现。

### 7. BW stage SAT load

这条要拆开看。系统里应该同时存在两类量：

- `last_*`：上一拍选星/负载，给 actor/critic 作为历史状态。
- `prefix_*`：当前 stage 已经固定的动作前缀，例如 BW stage 中 sat selection 已固定，应该反映当前选择。

当前代码里：

- BW actor selected SAT token 的 `prefix_backhaul_capacity_steps` 调 `backhaul_rate_for_us(...)`，这个路径会用 `sat_selected_count(...)`，语义上能反映当前 prefix selection 的共享数。
- critic 的 `SAT_LAST_SELECTED_LOAD_FRAC` 明确读 `kFStateLastSatConnectionCounts`，这是上一拍，合理。
- critic 的 `SAT_PREFIX_SELECTED_LOAD_FRAC` 读 `kSfSatLoadActive`，而 `kSfSatLoadActive` 来自 `kSfSatLoads`；当前看到 `kSfSatLoads` 在 stage base 中来自上一拍 `kFStateLastSatConnectionCounts`，没有看到按当前 `kSlSatSelectionMatrix` 重算的写入。

结论：不是“所有 BW stage load 都错”。更精确地说：last-load 应保留上一拍；prefix-capacity 似乎已用当前 selected count；但 critic 的 prefix-load fraction 疑似仍沿用上一拍 load，需要改成当前 prefix selection load。

### 8. backhaul loss

Python env 路径 `_backhaul_loss_factor(...)` 同时包含：

```python
atm_loss_enabled
rain_loss_enabled
```

native CUDA 里 `atmospheric_loss_factor_device(...)` 也包含 atmospheric + rain；候选排序路径 `sat_score_se_for_stage_us(...)` 会用它。这里“候选排序路径”指 `refresh_stage_derived_parallel(...)` 在 `sat_candidate_mode=score` 时给可见卫星排序/选候选，不是最终 reward/rate。

关键 selected backhaul rate 函数：

```cpp
backhaul_rate_for_selected_us(...)
```

当前计算：

```cpp
gain = backhaul_gain_const / dist2
snr = tx_power * gain / noise
```

当前没有乘 `atmospheric_loss_factor_device(a, elevation)`。这个函数被 `backhaul_rate_for_us(...)` 调用，并进入：

- reward / relay outflow
- `kFMainBwLinkRateMatrix`
- BW selected SAT token `prefix_backhaul_capacity_steps`
- critic `US_PREFIX_BACKHAUL_CAPACITY_STEPS`

结论：如果 native selected rate 应与 Python backhaul 物理口径一致，这里仍需修正。之前文档说“部分候选打分路径”容易误导，准确说法是：候选排序路径含 loss，但 selected reward/rate 路径仍从裸 backhaul gain 算。

### 9. local/live SAT obs 的 backhaul SE

当前 accel actor 的 SAT token `SAT_BACKHAUL_SE_REF` 路径会先：

```cpp
gain *= atmospheric_loss_factor_device(...)
```

其中 `atmospheric_loss_factor_device(...)` 名字虽然只写 atmospheric，但实现里也包含 rain。SAT actor live obs 的 `US_BACKHAUL_SE_REF` 当前仍是：

```cpp
gain = backhaul_gain_const / range2
snr = tx_power * gain / backhaul_noise_ref
st[21] = spectral_efficiency(snr)
```

它缺少：

- atmospheric loss
- rain loss
- projected bandwidth/load 或至少明确命名为 full-band/no-load reference

旧 helper `write_sat_obs_row(...)` 目前没有调用，所以对当前 hot path 没直接影响；只是如果以后复用，会带回旧口径。

结论：第 9 条在 SAT actor live obs 语义上仍存在。主仓库字段已变成 `SAT_TOKEN_FIELDS[US_BACKHAUL_SE_REF]`，主要影响 active live SAT actor writer。

### 10. accel local UAV-UAV 相对速度

当前 accel actor schema:

```python
PEER_REL_VX = 2
PEER_REL_VY = 3
```

native writer:

```cpp
peer[kAccelPeerRelVx] = relvx / vmax;
peer[kAccelPeerRelVy] = relvy / vmax;
```

Python builder 也写了 `peer_tokens[..., PEER_REL_VX:PEER_REL_VY+1] = rel_vel / vel_ref`。

结论：相对速度信息存在。

### 11. critic world UAV-SAT 信息

当前 critic schema `CRITIC_UAV_SAT_EDGE_DIM=18`，包含：

- rel position: `US_REL_X/Y/Z`
- rel velocity: `US_REL_VX/Y/Z`
- radial velocity / range / elevation
- doppler norm / doppler margin
- `US_BACKHAUL_SE_REF`
- visible / valid
- last selected
- prefix selected / known
- prefix backhaul capacity steps

SAT 节点和 global scalar 还包含 queue、drop、processed、load、non-token SAT 汇总。

结论：第 11 条要求的信息在主仓库 critic world 中存在，虽然部分 backhaul SE/capacity 的 loss 口径受第 8/9 条影响。

### 12. critic world UAV-UAV 信息

当前 critic schema `CRITIC_UAV_UAV_EDGE_DIM=11`，包含：

- relative position
- relative velocity
- distance
- closing speed
- alert flag
- unsafe flag
- last/prefix shared SAT fraction

结论：第 12 条要求的信息存在。

### 13. time feature

旧版本中，time feature 确实不只出现在 critic：

- good repo `structured_driver.py` 写过 `uav_nodes[0, :, 6] = env.t / T_steps`，这是 critic world。
- c78 native `write_ego_uav(...)` 写过 `out[..., 6] = time_frac`，并被 accel/sat/bw local obs 复用，所以旧 local actor ego 也用过。
- c78 `structured_critic.py` 还显式取 `world_state.uav_nodes[..., 6].mean(...)`。

当前主仓库：

- `structured_accel_actor_schema.py`、`structured_sat_actor_schema.py`、`structured_bw_actor_schema.py` 中没有 explicit time field。
- `structured_critic_schema.py` 的 UAV node 第 6 维现在是 `UAV_QUEUE_FILL`，不是 time。
- native `write_ego_uav(...)` 里第 6 列没有写 time；当前 dedicated actor writer 也没有 `t/T_steps` 输入。
- `kIStateT` 仍存在，但用于 orbit sync、terminal 判断、random tape，不作为 actor/critic feature 暴露。

结论：第 13 条旧 feature 在主仓库当前结构化 actor/critic 输入里已经没有。如果恢复，不应再塞回匿名 `uav_nodes[...,6]`，而应作为 named horizon feature：

- critic：放进 `global_scalars`，例如 `remaining_horizon_frac = (T_steps - 1 - t) / T_steps`。
- accel actor：放进 accel ego feature，因为每个 UAV 的运动/避障策略会受剩余 horizon 影响。
- sat actor：放进 sat ego 或 demand/global feature，因为选星策略也可能受剩余 horizon 影响。
- bw actor：放进 bw ego feature，因为临近 terminal 时清队列优先级不同。

不要用 raw `t`，用 `[0, 1]` 范围的 `time_frac` 或 `remaining_horizon_frac`。

## 本次最重要的未修语义风险

1. `backhaul_rate_for_selected_us(...)` 当前没乘 atmospheric/rain loss。若 native selected rate 要和 Python 物理链路一致，这会影响 native reward/backhaul outflow、BW capacity token、critic prefix capacity。

2. SAT actor live obs 的 `US_BACKHAUL_SE_REF` 仍是裸 gain/full-band SE，不是 projected bandwidth/load + Doppler + atmospheric/rain 后的 SE。

3. critic prefix-load fraction 疑似仍像上一拍 load，没有按当前 stage `sat_selection` 重算。BW actor 的 prefix capacity 另走 selected-count rate 路径，不应混为一谈。

4. `kSfBwValidFlag` 在非 `candidate_mode=assoc` 时仍可能把 candidate 当 BW valid。当前配置不触发，但代码语义没有彻底封死。

5. 输入量级上没有旧的灾难性 `bw * SE`，但 raw SE、step-ratio、system sum 仍可到 5~20，和 O(1) 特征同层共存。这不一定是 bug，但解释了为什么 raw input LayerNorm 不能随便打开，也提示后续如果要做输入尺度规范，需要按字段语义单独处理。
