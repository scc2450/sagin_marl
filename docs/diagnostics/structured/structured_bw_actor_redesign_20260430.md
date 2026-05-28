# Structured BW Actor 重新设计

日期：2026-04-30

本文是 BW actor 的最终设计规格。BW actor 是多智能体参数共享策略：

```text
同一套参数 θ 被所有 UAV 复用；
一次 actor 语义调用只针对一台 ego UAV；
工程上可以 batch 多个 ego row，但不是 centralized actor 一次联合输出所有 UAV 动作。
```

记：

```text
π_bw,θ(o_u) -> beta_u[0:G]
```

其中 `u` 是当前 ego UAV，`G = num_gu`。

## 1. Stage 语义

调用 BW actor 时，以下 prefix 已固定：

```text
post-accel UAV/GU 几何
GU -> UAV association
ego UAV selected SAT
selected SAT prefix load
bw_valid_mask
```

BW actor 只决定：

```text
ego UAV 的 access bandwidth 在当前连接有效 GU 之间的比例。
```

BW actor 不决定：

```text
GU 连接哪台 UAV
UAV 选择哪颗 SAT
其他 UAV 的 BW allocation
UAV 运动
```

## 2. 动作定义

对单台 ego UAV：

```text
beta_by_gu: [G]
```

约束：

```text
beta[g] >= 0
beta[g] = 0 if not bw_valid_mask[g]

if valid_count > 0:
  sum_{g: bw_valid} beta[g] = 1
else:
  beta[:] = 0
```

`bw_valid_mask[g]` 的语义：

```text
GU g 存在，并且当前 association[g] == ego UAV。
```

不再单独输入：

```text
owner_is_ego
valid_known
candidate_flag
```

这些与 `bw_valid_mask` 重复或属于旧 candidate-slot 实现细节。

环境执行侧不再改变动作语义。actor 输出的 `beta_by_gu` 就是 executed beta。环境只做：

```text
finite check
non-negative check
invalid mass check
valid sum check
```

数值契约固定为：

```text
value_check_eps = 1e-7
fp32_eps = 1.1920929e-7
sum_check_eps(valid_count) =
  max(1e-5, 8 * fp32_eps * max(valid_count, 1))

invalid_mass <= sum_check_eps(valid_count)
min(beta) >= -value_check_eps
abs(valid_sum - 1) <= sum_check_eps(valid_count)   if valid_count > 0
abs(valid_sum) <= sum_check_eps(valid_count)       if valid_count == 0
```

环境不做二次 softmax、不做二次 renorm、不按 support 重分配。超出 tolerance 时直接报错或记录 hard diagnostic 并终止该 rollout；在 tolerance 内的微小残差只允许用于检查通过，不改变实际执行语义。actor/native live kernel 必须在写出 action 前把 invalid beta 写成 0。

## 3. Access 干扰模型必须同步改为 beta-continuous

BW 动作是带宽比例，因此 access interference 也必须随带宽比例连续变化。

### 3.1 最终执行模型

令：

```text
beta_g = ego/serving UAV 给 GU g 的带宽比例
B_g = beta_g * b_acc
```

对被 UAV `u` 服务的 GU `g`，使用：

```text
full_interference_by_u =
  sum_{h: association[h] != u}
    gu_tx_power * beta_h * access_gain[h, u]

effective_interference_for_g =
  beta_g * full_interference_by_u

effective_noise_for_g =
  noise_density * (beta_g * b_acc) * access_noise_figure_linear

signal_power_g =
  gu_tx_power * access_gain[g, u]

SINR_g =
  signal_power_g
  / (effective_noise_for_g + effective_interference_for_g)

rate_g =
  beta_g * b_acc * spectral_efficiency(SINR_g)
```

等价地，`beta_g > 0` 时：

```text
SINR_g =
  gu_tx_power * access_gain[g, u]
  / (
      beta_g * noise_density * b_acc * NF
      + beta_g * sum_{h: assoc[h] != u} gu_tx_power * beta_h * access_gain[h,u]
    )
```

这个模型保留了当前代码“victim bandwidth 越小，noise/interference 积分带宽越小”的口径，同时让 interferer 的贡献随 `beta_h` 连续变化。

### 3.2 必须同步修改的地方

所有使用 access interference 的路径都必须与 beta-continuous 模型一致。

环境执行：

```text
sagin_marl/env/sagin_env.py
  _compute_access_interference_power(...)
  _compute_access_rates(...)
```

要求：

```text
_compute_access_interference_power 不再用 beta_h > eps 作为二值 active；
干扰贡献必须乘 beta_h。
```

native / tensor 环境：

```text
sagin_marl/env/structured_batch_env_core.py
sagin_marl/env/native_cuda/kernels.cu
```

要求：

```text
batched access rate
access interference
bw post stats
native main kernel
```

全部使用同一公式。

critic world：

```text
last_access_interference_by_uav
uav_gu_edges 中与 last BW / interference 相关字段
global access interference summaries
```

全部解释为 beta-continuous 执行结果。

accel actor：

旧语义中类似：

```text
gu_last_access_active_flag
ug_last_nonself_interference_log1p
cell_interference_exposure
```

必须改为 beta-weighted：

```text
last_gu_access_beta[g] =
  sum_u last_bw_fraction_by_uav_gu[u,g]

last_nonself_interference_to_ego_from_g =
  gu_tx_power
  * last_gu_access_beta[g]
  * access_gain[g, ego]
  * 1[last_association[g] != ego]
```

BW actor：

`cross_interference_*` 是 per-unit-beta potential interference，实际外部性由当前 `beta_g` 决定。

测试：

```text
beta_h = 0.01 的干扰必须明显小于 beta_h = 1.0；
beta_h = 0 时无该 GU 的干扰贡献。
```

## 4. 归一化

### 4.1 flow reference

```text
arrival_ref_step = effective_task_arrival_rate * num_gu * tau0
gu_flow_ref      = arrival_ref_step / num_gu
uav_flow_ref     = arrival_ref_step / num_uav

sat_select_k =
  sat_num_select if configured
  else N_RF

sat_workload_ref_count =
  queue_ref_sat_active_count if configured
  else min(num_sat, sat_select_k * num_uav)

sat_flow_ref = arrival_ref_step / sat_workload_ref_count
```

所有 queue / arrival / inflow / outflow / drop / EMA：

```text
bits / corresponding_flow_ref
```

容量填充率：

```text
queue_fill = queue_bits / queue_max
```

### 4.2 access full-band reference

BW actor 固定保留：

```text
access_rate_full_bw_ref_steps
```

定义：

```text
access_noise_ref =
  noise_density * b_acc * access_noise_figure_linear

access_snr_full_ref[g, ego] =
  gu_tx_power * access_gain_matrix[g, ego] / access_noise_ref

access_se_full_ref[g, ego] =
  if fading_enabled and access_fading_mode == "ergodic_rician":
    rician_ergodic_spectral_efficiency(access_snr_full_ref, K, quadrature_points)
  else:
    log2(1 + access_snr_full_ref)

access_rate_full_bw_ref_steps[g, ego] =
  b_acc * access_se_full_ref[g, ego] * tau0 / gu_flow_ref
```

这里不使用当前 BW action 后产生的实际 interference。

### 4.3 selected backhaul capacity reference

Selected SAT token 中的 `prefix_backhaul_capacity_steps` 定义为当前 prefix 下 ego UAV 到 selected SAT 的可用 backhaul 服务量：

```text
selected_load[s] =
  count of UAVs selecting SAT s in current SAT prefix

b_share[s] =
  effective_b_backhaul_per_sat / max(selected_load[s], 1)

backhaul_noise[s] =
  noise_density * b_share[s] * backhaul_noise_figure_linear

backhaul_snr[ego,s] =
  uav_tx_power * backhaul_gain[ego,s] / backhaul_noise[s]

if doppler_observed and doppler_atten_enabled:
  backhaul_snr[ego,s] *= doppler_attenuation(nu_eff[ego,s], subcarrier_spacing)

backhaul_se[ego,s] =
  spectral_efficiency(backhaul_snr[ego,s])

prefix_backhaul_capacity_steps[ego,s] =
  backhaul_se[ego,s] * b_share[s] * tau0 / uav_flow_ref
```

只对 ego 当前 selected SAT 写真实值；padding SAT 和未 selected SAT 写 0，并由 `selected_sat_mask` 屏蔽。

`selected_sat_mask[row,k] = 1` 当且仅当该 slot 对应 ego 当前 prefix 中实际 selected 且 link-valid 的 SAT；padding、未选中或 link-invalid slot 都为 0。

### 4.4 workload / cost

```text
service_floor = service_floor_bits_per_step

gu_local_cost  = 1 / max(gu_service_ema_bits_per_step, service_floor)
uav_local_cost = 1 / max(uav_service_ema_bits_per_step, service_floor)
sat_cost       = 1 / max(sat_service_ema_bits_per_step, service_floor)

gu_flow_cost_ref  = 1 / gu_flow_ref
uav_flow_cost_ref = 1 / uav_flow_ref
sat_cost_ref      = 1 / sat_flow_ref

uav_total_cost_ref = uav_flow_cost_ref + sat_cost_ref
gu_total_cost_ref  = gu_flow_cost_ref + uav_total_cost_ref

cost_log_ratio(value, ref) =
  log(max(value, LOG_RATIO_EPS) / max(ref, LOG_RATIO_EPS))

gu_local_cost_log_ratio =
  cost_log_ratio(gu_local_cost, gu_flow_cost_ref)

uav_local_cost_log_ratio =
  cost_log_ratio(uav_local_cost, uav_flow_cost_ref)

sat_cost_log_ratio =
  cost_log_ratio(sat_cost, sat_cost_ref)

gu_last_total_cost_log_ratio =
  cost_log_ratio(last_gu_total_cost, gu_total_cost_ref)

uav_last_total_cost_log_ratio =
  cost_log_ratio(last_uav_total_cost, uav_total_cost_ref)

gu_last_workload_log1p[g] =
  log1p(max(last_gu_total_cost[g] * gu_queue_bits[g], 0))

uav_last_workload_log1p[u] =
  log1p(max(last_uav_total_cost[u] * uav_queue_bits[u], 0))

sat_last_workload_log1p[s] =
  log1p(max(sat_cost[s] * sat_queue_bits[s], 0))
```

其中 `last_gu_total_cost` 和 `last_uav_total_cost` 使用现有 `_bw_weighted_workload_device_costs(...)` 口径，但其 access interference 输入必须已经是 beta-continuous。

## 5. LocalBwState

BW actor 使用专用 local state，不复用 critic schema。

```python
@dataclass
class LocalBwState:
    ego_features: torch.Tensor          # [R, BW_EGO_DIM]
    selected_sat_tokens: torch.Tensor   # [R, K_sat, BW_SAT_TOKEN_DIM]
    selected_sat_mask: torch.Tensor     # [R, K_sat]
    gu_tokens: torch.Tensor             # [R, G, BW_GU_TOKEN_DIM]
    gu_mask: torch.Tensor               # [R, G]
    bw_valid_mask: torch.Tensor         # [R, G]
```

其中：

```text
R 是 batch 展开 row 数；
每个 row 仍然表示一个 ego UAV；
K_sat = sat_select_k。
```

### 5.1 Ego Features

BW actor 的 ego features 只保留与 relay/backpressure 和上一拍 access 干扰有关的量：

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
```

其中：

```text
uav_last_access_interference_log1p =
  log1p(last_access_interference_by_uav[ego] / access_noise_ref)
```

`last_access_interference_by_uav` 必须来自上一拍 beta-continuous access 执行结果。

不放 ego 位置/速度。BW 的几何影响由：

```text
GU token 的 access_rate_full_bw_ref_steps
GU token 的 cross_interference_*
selected SAT token 的 prefix_backhaul_capacity_steps
```

表达。

### 5.2 Selected SAT Tokens

BW actor 不需要 selected SAT 的原始几何：

```text
rel_pos
rel_vel
doppler
backhaul_se_ref
selected_sat_load_frac
```

这些已经被当前 BW stage 已知的 backhaul capacity 和 SAT workload/cost 消化。

每个 selected SAT token 固定字段：

```text
0  prefix_backhaul_capacity_steps
1  sat_queue_steps
2  sat_queue_fill
3  sat_last_incoming_steps
4  sat_last_processed_steps
5  sat_last_drop_steps
6  sat_service_ema_steps
7  sat_cost_log_ratio
8  sat_last_workload_log1p
```

`selected_sat_load_frac_mean` 不作为输入。它原本表示 selected SAT 被多少 UAV 共享的平均负载比例，但：

```text
prefix_backhaul_capacity_steps 已经包含 selected load 对带宽均分的影响；
再输入 load fraction 是重复信息。
```

### 5.3 GU Tokens

每个 GU 一个 token，全量 `G`，但语义计算只对 valid GU 生效。

固定字段：

```text
0  gu_queue_steps
1  gu_queue_fill
2  gu_expected_arrival_steps
3  gu_last_arrival_steps
4  gu_last_outflow_steps
5  gu_last_drop_steps
6  gu_service_ema_steps
7  gu_local_cost_log_ratio
8  gu_last_total_cost_log_ratio
9  gu_last_workload_log1p

10 access_rate_full_bw_ref_steps

11 cross_interference_mean_log1p
12 cross_interference_max_log1p
```

不放：

```text
prefix_gu_total_cost_log_ratio
prefix_gu_workload_log1p
```

原因：

```text
BW stage 虽然 association 和 selected SAT 已知，但当前 BW allocation 未知；
reward 的 workload 是 GU/UAV/SAT 分层 queue cost，不是严格 per-GU route cost；
多 selected SAT 下，一个 GU 的 downstream cost 还涉及 UAV queue 混合流和后续 backhaul 分流。
```

因此这两个量最多是启发式近似，不进入最终 actor schema。BW actor 通过：

```text
GU 自身 queue/arrival/drop/cost
ego UAV relay state
selected SAT capacity/workload tokens
```

来学习当前 GU 的边际价值。

不放：

```text
rel_gu_x_norm
rel_gu_y_norm
d_ego_gu_norm
```

原因：

```text
BW actor 需要的是链路服务能力和干扰外部性；
相对位置/距离已经通过 access_rate_full_bw_ref_steps 和 cross_interference_* 表达。
```

不放：

```text
gu_demand_steps
gu_unserved_demand_steps
```

原因：

```text
它们可由 gu_queue_steps、gu_expected_arrival_steps、gu_last_outflow_steps 算出；
不是新信息。
```

不放：

```text
last_bw_fraction_ego_gu
last_assoc_to_ego_flag
last_served_by_ego_flag
gu_last_access_active_flag
```

原因：

```text
当前决策主要由当前 backlog、arrival、last outflow/drop、链路和 downstream bottleneck 决定；
上一拍分配比例不是必要物理状态。
若需要抗抖动，用 action smoothing regularizer；不要把上一拍动作混进 core schema。
```

### 5.4 cross_interference 字段

这两个字段表示 GU g 对其他 UAV receiver 的 per-unit-beta 潜在干扰。

对 `v != ego`：

```text
cross_power_unit[g, ego -> v] =
  gu_tx_power * access_gain_matrix[g, v]

cross_log1p[g, ego -> v] =
  log1p(cross_power_unit[g, ego -> v] / access_noise_ref)
```

输入：

```text
cross_interference_mean_log1p =
  mean_{v != ego} cross_log1p[g, ego -> v]

cross_interference_max_log1p =
  max_{v != ego} cross_log1p[g, ego -> v]
```

`U == 1` 时两项为 0。

不保留 `sum`，因为固定 `U` 下：

```text
sum = mean * (U - 1)
```

## 6. 网络结构

### 6.1 语义原则

BW actor 的网络语义必须是：

```text
只对 valid GU 进行竞争、聚合、score 和动作分布；
invalid GU 只因 full-G 固定张量形状存在，不参与决策。
```

语义上输入和输出都是 full-G；实现上允许在 native kernel 内部把 valid GU 临时 compact 到 scratch buffer 以减少无效 GU 的 MLP/attention 计算，但这种 compact 只能发生在单个 fused kernel 内，不能恢复旧的 Python candidate-slot actor 接口。无论是否内部 compact，都必须满足：

```text
invalid GU token 在 attention/pooling/readout 前 mask 掉或不进入 compact scratch；
invalid score = -inf；
invalid beta = 0。
```

### 6.2 编码

```text
ego_emb =
  ego_encoder(LayerNorm(ego_features))

sat_emb[k] =
  sat_encoder(LayerNorm(selected_sat_tokens[k]))

gu_emb[g] =
  gu_encoder(LayerNorm(gu_tokens[g]))
```

定义：

```text
valid_gu_mask = gu_mask & bw_valid_mask
```

### 6.3 Downstream Context

BW actor 必须有 downstream context，因为 access outflow 会进入 UAV relay queue，再由 selected SAT backhaul 消化。

对 selected SAT tokens 使用 ego-conditioned masked attention readout：

```text
Q_down =
  num_down_queries

down_queries =
  down_query_proj(ego_emb).view(R, Q_down, E)

down_attn =
  multi_query_attention(
    queries = down_queries,
    keys    = sat_emb,
    values  = sat_emb,
    mask    = selected_sat_mask
  )

sat_add_pool =
  masked_sum(sat_add_proj(sat_emb), selected_sat_mask)

down_ctx =
  MLP([flatten(down_attn), sat_add_pool])
```

`down_queries` 是由当前 ego UAV 状态生成的一组 readout query，不是 SAT token，也不是动作。它的作用是让 ego UAV 用自己的 relay queue、last service、last interference 等状态去读取 selected SAT 集合：同一组 selected SAT，在不同 ego backlog 和 relay 状态下，应该关注的 downstream bottleneck 可以不同。

`Q_down` 是固定的小整数超参，只决定 downstream readout 有几个查询槽。实现里 tensor shape 是 `[R, Q_down, E]`，其中 `R` 是当前 batch 里的 ego-UAV 行数，`E` 是 embedding 维度。

`sat_add_pool` 使用 masked sum，不使用 mean，原因是 selected SAT 提供的是并行 downstream 资源和负载：

```text
1. prefix_backhaul_capacity_steps 是可加资源，总 capacity 应随 selected SAT 数和各链路容量增加。
2. SAT queue / incoming / processed / drop 也是系统负载量，aggregate scale 对 BW 决策有意义。
3. mean 会抹掉“一个可用 SAT”和“多个可用 SAT”的总量差异；sum 保留总 downstream 规模。
4. attention 输出负责 ego-conditioned bottleneck readout；additive pool 负责补充总资源/总负载尺度。
```

这里的 sum 必须带 `selected_sat_mask`，padding SAT 不参与聚合。`sat_add_proj` 是一个共享 Linear projection，用来让网络学习哪些 SAT embedding 通道适合按加法聚合。

若没有 selected SAT：

```text
down_attn = 0
sat_add_pool = 0
```

这里仍然需要 `selected_sat_mask`。`K_sat` 是固定张量宽度，但实际 selected SAT 数可能小于 `K_sat`，padding token 不能参与 attention 或 additive pool。

不用把 `selected_sat_tokens` 直接 flatten 后送入一个大 MLP，原因是：

```text
1. flatten+MLP 对 slot 顺序敏感；同一组 selected SAT 换一个排列会变成不同输入。
2. 它把参数形状强绑定到 K_sat，后续调整 sat_num_select 时更容易破坏 checkpoint 和 native ABI。
3. ego-conditioned attention 能表达“当前 ego UAV 在这个 downstream 状态下应该关注哪些 selected SAT”。
4. mask 是 attention 的一等输入，padding SAT 不会泄漏到 downstream context。
```

### 6.4 Valid-GU Competition

BW allocation 是同一 ego UAV 下 valid GU 之间的相对分配。网络必须让 valid GU 互相比较。

先把 ego/downstream context 注入每个 valid GU：

```text
ctx0 = MLP([ego_emb, down_ctx])
gu_h[g] = MLP([gu_emb[g], ctx0])
```

然后只在 valid GU 上运行 competition blocks。competition block 就是 masked Transformer/self-attention block：

```text
for block in competition_blocks:
  gu_h = block(gu_h, valid_gu_mask)
```

### 6.5 Score Head

```text
score[g] = score_head(gu_h[g]).squeeze(-1)
score[g] = -inf if not valid_gu_mask[g]
```

`score_head` 是共享 per-GU MLP，不是手工 priority。

## 7. Action Distribution

### 7.1 Deterministic mean

```text
tau =
  tau_min
  + (tau_max - tau_min) * sigmoid(tau_head(ctx0))

det_mean =
  masked_softmax(score / tau, valid_gu_mask)
```

固定口径：

```text
tau_min = 0.5
tau_max = 2.0
```

不保留 fixed-tau / learnable-tau 两套分支。实现必须使用 `tau_head(ctx0)`，并通过上式限制在 `[tau_min, tau_max]`。

`masked_softmax` 必须显式处理特殊行：

```text
valid_count == 0: det_mean = 0
valid_count == 1: det_mean = one_hot(valid GU)
```

不能对全 `-inf` score 直接调用普通 softmax。

### 7.2 Stochastic policy

使用 masked mean-concentration Dirichlet：

```text
kappa =
  kappa_min
  + (kappa_max - kappa_min) * sigmoid(kappa_head(ctx0))

alpha[g] =
  det_mean[g] * kappa
  for valid GU g

beta ~ MaskedMeanConcentrationDirichlet(
  mean = det_mean,
  kappa = kappa,
  mask = valid_gu_mask
)
```

约束：

```text
invalid beta = 0
valid beta sum = 1
```

特殊行：

```text
valid_count == 0:
  beta = 0
  logprob = 0
  entropy = 0

valid_count == 1:
  beta唯一有效GU = 1
  logprob = 0
  entropy = 0
```

### 7.3 valid_count 尺度

Dirichlet simplex latent dimension 为：

```text
latent_count = max(valid_count - 1, 0)
```

训练统计必须记录：

```text
valid_count
latent_count
approx_kl_bw_raw
approx_kl_bw_per_latent
clip_frac_bw
delta_exec_l1
delta_exec_l1_per_valid
entropy_bw_raw
entropy_bw_per_latent
```

BW PPO objective 固定使用 per-latent-dim 口径：

```text
structured_bw_objective_norm = "per_latent_dim"
```

具体：

```text
latent_denom = max(valid_count - 1, 1)

logprob_obj = logprob_raw / latent_denom
old_logprob_obj = old_logprob_raw / latent_denom

ratio_bw =
  exp(logprob_obj - old_logprob_obj)

entropy_bw_obj =
  entropy_raw / latent_denom
```

同时继续记录 raw logprob / raw KL 作为诊断，但 PPO ratio、clip 和 BW entropy bonus 使用 per-latent-dim 口径。

## 8. Policy Output

```python
@dataclass
class BwPolicyOutput:
    action: torch.Tensor        # [R,G], final executed beta
    logprob: torch.Tensor       # [R], per-latent objective logprob
    entropy: torch.Tensor       # [R], per-latent objective entropy
    logprob_raw: torch.Tensor   # [R], raw Dirichlet logprob
    entropy_raw: torch.Tensor   # [R], raw Dirichlet entropy
    score: torch.Tensor         # [R,G]
    det_mean: torch.Tensor      # [R,G]
    alpha: torch.Tensor         # [R,G], invalid entries 0
    kappa: torch.Tensor         # [R]
    valid_count: torch.Tensor   # [R], int64
    latent_count: torch.Tensor  # [R], int64
    tau: torch.Tensor           # [R]
```

不输出：

```text
support_members
support_mask
support_count
```

本设计要求环境改为 beta-continuous 干扰模型，因此不使用 support-sparse 动作。

## 9. 代码迁移规格

本节按当前代码结构写成实现清单。实现时以本节的字段名、shape、函数职责为准，避免继续沿用旧 BW actor 的候选 GU slot、support sparse、二次 renorm 和二值 active 干扰语义。

### 9.1 新增 schema 文件

新增文件：

```text
sagin_marl/rl/structured_bw_actor_schema.py
```

文件内只放 BW actor 输入 schema、字段索引和 objective 口径，不放 `embed_dim`、`hidden_dim`、layer 数等网络宽度超参，也不从 critic shape 推导：

```python
BW_EGO_DIM = 10
BW_SAT_TOKEN_DIM = 9
BW_GU_TOKEN_DIM = 13

BW_OBJECTIVE_NORM = "per_latent_dim"

BW_EGO_FIELDS = (
    "uav_queue_steps",
    "uav_queue_fill",
    "uav_last_inflow_steps",
    "uav_last_outflow_steps",
    "uav_last_drop_steps",
    "uav_service_ema_steps",
    "uav_local_cost_log_ratio",
    "uav_last_total_cost_log_ratio",
    "uav_last_workload_log1p",
    "uav_last_access_interference_log1p",
)

BW_SAT_TOKEN_FIELDS = (
    "prefix_backhaul_capacity_steps",
    "sat_queue_steps",
    "sat_queue_fill",
    "sat_last_incoming_steps",
    "sat_last_processed_steps",
    "sat_last_drop_steps",
    "sat_service_ema_steps",
    "sat_cost_log_ratio",
    "sat_last_workload_log1p",
)

BW_GU_TOKEN_FIELDS = (
    "gu_queue_steps",
    "gu_queue_fill",
    "gu_expected_arrival_steps",
    "gu_last_arrival_steps",
    "gu_last_outflow_steps",
    "gu_last_drop_steps",
    "gu_service_ema_steps",
    "gu_local_cost_log_ratio",
    "gu_last_total_cost_log_ratio",
    "gu_last_workload_log1p",
    "access_rate_full_bw_ref_steps",
    "cross_interference_mean_log1p",
    "cross_interference_max_log1p",
)
```

实现要求：

```text
1. 字段 tuple 长度必须 assert 等于对应 DIM。
2. 所有 builder、policy、native ABI 都引用这个文件的常量。
3. 不再使用 shape.sat_node_dim / shape.user_node_dim / critic edge dim 作为 BW actor 输入维度。
```

### 9.2 structured_types.py

修改：

```text
sagin_marl/rl/structured_types.py
```

把现有 `LocalBwState`：

```python
ego_uav_after_sat
sat_nodes
sat_edges
sat_mask
user_nodes
user_edges
user_mask
bw_valid_mask
```

替换为：

```python
@dataclass
class LocalBwState:
    ego_features: torch.Tensor          # [R, BW_EGO_DIM]
    selected_sat_tokens: torch.Tensor   # [R, K_sat, BW_SAT_TOKEN_DIM]
    selected_sat_mask: torch.Tensor     # [R, K_sat]
    gu_tokens: torch.Tensor             # [R, G, BW_GU_TOKEN_DIM]
    gu_mask: torch.Tensor               # [R, G]
    bw_valid_mask: torch.Tensor         # [R, G]
```

同步修改 `BwPolicyOutput` 为第 8 节定义的字段。旧输出中的 `support_members`、`support_mask`、`support_count` 不再作为主路径字段出现。若调试代码临时需要旧字段，只能在 compatibility wrapper 里生成，不能进入新 policy 的 dataclass。

所有引用旧字段的代码都必须改名或删除。允许保留旧类名 `LocalBwState`，因为训练、buffer、eval 已经大量依赖这个类型名；字段语义必须换成新 schema。

`BwStageSnapshot` 若保留，不能再把 `candidate_indices` / candidate-slot `bw_valid_mask` 作为 actor 输入核心字段。最终形态为：

```python
@dataclass
class BwStageSnapshot:
    world_state: StructuredWorldState
    assoc: ArrayLike
    selected_sat_indices: ArrayLike
    selected_sat_mask: ArrayLike
    access_gain_matrix: ArrayLike
    bw_valid_mask_full: ArrayLike  # [U, G]
```

eval/probe 可以通过 snapshot 重建 `LocalBwState`，但重建结果必须仍是 full-G schema。

### 9.3 Python local BW builder

当前旧路径主要在：

```text
sagin_marl/rl/structured_stage_builders.py
  build_batched_local_bw_states_from_snapshot(...)

sagin_marl/env/structured_driver.py
  build_bw_valid_context(...)
  build_bw_stage_snapshot(...)
```

改法：

```text
1. 新增主路径函数 build_batched_local_bw_states_from_spec(...)。
2. structured_driver.py 暴露 build_local_bw_states()，返回新的 LocalBwState。
3. build_bw_valid_context(...) 若仍保留，只返回 bw_valid_mask 和必要 stage index，不再构造旧 candidate slot actor obs。
4. build_bw_stage_snapshot(...) 若 eval/probe 仍需要，可以保留 snapshot 容器，但其中 actor 输入必须是 full-G 新 schema。
```

最终函数签名与当前 accel builder 保持同一风格：

```python
def build_batched_local_bw_states_from_spec(
    spec: dict,
    *,
    device: torch.device | str | None = None,
) -> LocalBwState:
    ...
```

`spec` 至少包含：

```text
env
driver
assoc                    # [G]
sat_selection            # list[list[int]] 或 [U,K_sat]
sat_loads
access_gain_matrix       # [G,U]
```

因此 `structured_driver.py::_prepare_bw_stage_spec()` 也要补充：

```python
"access_gain_matrix": np.asarray(self._stage_access_gain_matrix, dtype=np.float32),
```

如果 `_stage_access_gain_matrix is None`，先调用 `_refresh_world_build_cache(...)`，不能在 BW builder 里重新 sample access channel。BW stage 必须使用 accel stage 已经固定的同一个 access gain snapshot。

单环境 driver spec 输出 `R = U`。batch/tensor/native 路径输出 `R = B * U`，行顺序固定为：

```text
row = b * num_uavs + ego_uav
```

mask 规则：

```python
gu_mask[row, g] = gu_exists[b, g]
bw_valid_mask[row, g] = gu_exists[b, g] & (association[b, g] == ego_uav)
valid_gu_mask = gu_mask & bw_valid_mask
```

builder 必须写入 full-G token，不再按 `candidate_indices` 压缩 GU。`users_obs_max` 只允许存在于旧兼容代码，不能决定新 actor 的动作维度。

`access_rate_full_bw_ref_steps` 的计算口径：

```text
这是链路能力 reference，不是当前 BW action 下的实际受干扰 rate：

  access_noise_ref =
    noise_density * b_acc * access_noise_figure_linear

  access_snr_full_ref[g, ego] =
    gu_tx_power * access_gain_matrix[g, ego] / access_noise_ref

  access_se_full_ref[g, ego] =
    channel.spectral_efficiency(access_snr_full_ref)
    或 ergodic_rician spectral efficiency

  access_rate_full_bw_ref_steps[g, ego] =
    b_acc * access_se_full_ref[g, ego] * tau0 / gu_flow_ref
```

实现注意：

```text
1. 该字段只对 valid GU 写真实值，invalid GU 可写 0。
2. 使用当前 stage 的 access_gain_matrix，不重新计算几何。
3. 不加入当前 BW action、last BW action 或 beta-continuous interference。
4. 不能再使用 backhaul_se_ref、rel_pos、doppler 作为 BW actor token。
```

`cross_interference_mean_log1p` 和 `cross_interference_max_log1p` 的 builder 口径：

```python
for row=(b, ego), gu=g:
    other_uavs = [v for v in range(U) if v != ego]
    unit_power[v] = gu_tx_power_watt * access_gain_matrix[b, g, v]
    x[v] = log1p(unit_power[v] / access_noise_ref_watt)
    mean = mean(x over other_uavs)
    maxv = max(x over other_uavs)
```

若 `U == 1`，两个字段都写 0。不要加入 `sum` 字段；`sum` 与 `mean` 在固定 UAV 数下重复，在可变 UAV 数下会把系统规模混进 token。

### 9.4 环境执行与 access 干扰

当前需要重点同步的文件：

```text
sagin_marl/env/sagin_env.py
sagin_marl/env/structured_batch_env_core.py
sagin_marl/env/native_cuda/*.cu
```

把 access 干扰统一改成 beta-continuous，并抽出 Python reference helper 作为测试基准：

```python
def compute_access_interference_beta_continuous(
    association: np.ndarray,       # [G]
    access_gain_matrix: np.ndarray,# [G, U]
    gu_band_fraction: np.ndarray,  # [G]
) -> np.ndarray:                  # [U]
    ...
```

语义：

```python
interference_by_uav[u] =
    sum(
        gu_tx_power_watt * gu_band_fraction[h] * access_gain_matrix[h, u]
        for h in range(G)
        if association[h] >= 0 and association[h] != u
    )
```

执行 rate 时使用同一个 `gu_band_fraction`：

```python
gu_band_fraction[g] = beta_by_uav_gu[association[g], g]
effective_bandwidth[g] = gu_band_fraction[g] * access_bandwidth_hz
effective_interference[g] = gu_band_fraction[g] * interference_by_uav[association[g]]
effective_noise[g] = noise_density * effective_bandwidth[g] * noise_figure
rate[g] = effective_bandwidth[g] * spectral_efficiency(signal / (effective_noise + effective_interference))
```

这里保留 `effective_interference[g] = beta_g * interference_by_uav[u]`，含义是 GU g 只在自己占用的频段上承受同频干扰；其它 GU h 的贡献已经通过 `gu_band_fraction[h]` 缩放。这样小带宽 GU 只产生小干扰，也只在小带宽上接收干扰。

环境动作 shape 改为 full-G：

```text
single env: beta_by_uav_gu [U, G]
batched env: beta_by_uav_gu [B, U, G]
```

执行前只做数值合法性检查：

```text
1. invalid GU beta 必须为 0。
2. 每个 (B,U) 的 valid beta sum 必须为 1，valid_count==0 时 sum 为 0。
3. 环境不做二次 softmax / renorm / support 重分配；数值 tolerance 只用于合法性检查。
```

`last_exec_bw_alloc`、`last_bw_fraction_by_uav_gu`、history ring 和 native history 都应以 `[B,U,G]` full-G beta 为主。若旧统计还需要 candidate slot 视图，只能由 full-G beta 临时 gather 出来。

### 9.5 accel actor 与 critic 的联动修改

因为 access 干扰从二值 active 改为 beta-continuous，以下所有路径必须同步：

```text
sagin_marl/rl/structured_accel_actor.py
sagin_marl/rl/structured_accel_actor_schema.py
sagin_marl/rl/structured_critic.py
sagin_marl/rl/structured_stage_builders.py
sagin_marl/env/structured_batch_env_core.py
sagin_marl/env/native_cuda/*.cu
```

accel actor 中凡是表示 last access interference、cell interference exposure、GU last access active 的字段，都要从旧口径：

```text
active = beta > eps
```

改为：

```text
beta_weight = last_bw_fraction_by_uav_gu.sum(axis=U)
interference contribution = gu_tx_power * beta_weight[g] * gain[g, receiver]
```

accel schema 中与上拍 access activity 相关的 actor 输入应改为连续字段 `gu_last_bw_fraction`。若为了日志兼容仍保留 `gu_last_access_active_flag`，它只能表示诊断用 `beta_weight > eps`，不能参与干扰功率计算，也不能作为新 actor 的核心输入。

critic world state 中的 `last_access_interference`、UAV-GU edge 中的 last interference、history cost 中由 access rate 推出的服务量，都必须来自同一套 beta-continuous helper。critic 不应该看到一套二值干扰，而 actor/env 执行另一套连续干扰。

### 9.6 StructuredGpuBwObsView 和 native history buffer

当前旧 view 在：

```text
sagin_marl/env/structured_gpu_rollout_runtime.py
  StructuredGpuBwObsView
```

把 `_tensor_fields` 改成：

```python
_tensor_fields = (
    "ego_features",
    "selected_sat_tokens",
    "selected_sat_mask",
    "gu_tokens",
    "gu_mask",
    "bw_valid_mask",
)
```

构造参数和属性 shape：

```text
ego_features:        [R, 10]
selected_sat_tokens: [R, K_sat, 9]
selected_sat_mask:   [R, K_sat]
gu_tokens:           [R, G, 13]
gu_mask:             [R, G]
bw_valid_mask:       [R, G]
```

删除旧字段：

```text
ego_uav_after_sat
sat_nodes
sat_edges
sat_mask
user_nodes
user_edges
user_mask
```

当前 native training history buffer 在：

```text
sagin_marl/env/structured_batch_env_core.py
  _NativeBwTrainingHistoryOutBuffers
```

把旧 obs 字段替换为新字段，同时保留 reward、terminated、truncated、done、value、logprob 等训练字段。分配 live obs 和 history obs 的地方也同步改 shape：

```text
live_bw_obs.ego_features:        [B*U, 10]
live_bw_obs.selected_sat_tokens: [B*U, K_sat, 9]
live_bw_obs.selected_sat_mask:   [B*U, K_sat]
live_bw_obs.gu_tokens:           [B*U, G, 13]
live_bw_obs.gu_mask:             [B*U, G]
live_bw_obs.bw_valid_mask:       [B*U, G]
```

`live_bw_action`、`live_bw_ref_action`、`live_bw_flow_proxy_override_action` 等动作张量改成 full-G 口径：

```text
action: [B, U, G] 或 flatten 后 [B*U, G]
```

logprob / entropy 不再是 per-slot 张量，而是每个 ego row 一个标量。沿用现有 rollout 命名时，`old_logprob` 表示采样时旧策略的 logprob：

```text
live_bw_old_logprobs_per_agent: [B, U], per-latent objective old logprob
live_bw_old_logprob:            [B], sum_u live_bw_old_logprobs_per_agent[b,u]

live_bw_entropy_per_agent:      [B, U], per-latent objective entropy
live_bw_logprob_raw_per_agent:  [B, U], raw Dirichlet logprob
live_bw_entropy_raw_per_agent:  [B, U], raw Dirichlet entropy
```

旧 `live_bw_old_logprobs_per_slot`、`live_bw_support_mask`、candidate slot old logprob、support residual 相关 buffer 从主路径删除。若为了对照旧实验保留，命名必须带 `legacy_`，并且不能被新训练入口读取。

### 9.7 tensor/native local obs builder

当前 tensor builder 重点位置：

```text
sagin_marl/env/structured_batch_env_core.py
  _build_local_bw_obs_from_stage_tensor_impl(...)
  _build_local_bw_user_components_tensor_impl(...)
```

改法：

```text
1. _build_local_bw_obs_from_stage_tensor_impl(...) 直接写新 LocalBwState 六个字段。
2. 删除或停止主路径调用 _build_local_bw_user_components_tensor_impl(...) 旧 user_nodes/user_edges 输出。
3. 不再读取 candidate_indices_t 来决定 actor GU 维度。
4. selected SAT token 从当前 stage 的 selected_sat_indices / selected_sat_mask gather。
5. GU token 对所有 G 写入；invalid GU 只靠 bw_valid_mask 屏蔽，不靠移除 token。
```

native CUDA builder 的输出布局必须和 Python builder 逐字段一致。先写 Python reference test，再让 CUDA kernel 对齐 reference，不要反过来用现有 CUDA 布局牵引 Python schema。

### 9.8 structured_buffer.py

当前旧转换函数：

```text
sagin_marl/rl/structured_buffer.py
  _bw_local_from_flat_history_ring(...)
```

改为从 history ring 读取：

```python
LocalBwState(
    ego_features=...,
    selected_sat_tokens=...,
    selected_sat_mask=...,
    gu_tokens=...,
    gu_mask=...,
    bw_valid_mask=...,
)
```

所有 flatten、mini-batch、to(device)、pin_memory 逻辑都只认识新六字段。旧字段访问应通过测试全部清掉：

```text
rg "ego_uav_after_sat|sat_nodes|sat_edges|user_nodes|user_edges|user_mask" sagin_marl tests
```

保留 `sat_mask` 名称会造成歧义，因此 BW actor 路径统一叫 `selected_sat_mask`。

### 9.9 BwPolicy 重写

当前旧类：

```text
sagin_marl/rl/structured_actor.py
  class BwPolicy
```

保留类名，替换内部结构，避免 factory、checkpoint loader、MAPPO 调用链大面积改名。新构造器接收 BW schema 输入维度和配置里的网络超参：

```python
class BwPolicy(nn.Module):
    def __init__(
        self,
        *,
        ego_dim: int = BW_EGO_DIM,
        sat_token_dim: int = BW_SAT_TOKEN_DIM,
        gu_token_dim: int = BW_GU_TOKEN_DIM,
        embed_dim: int,
        hidden_dim: int,
        down_query_count: int,
        num_competition_layers: int,
        num_heads: int,
        tau_min: float,
        tau_max: float,
        kappa_min: float,
        kappa_max: float,
    ) -> None:
        ...
```

模块命名固定为：

```text
ego_input_norm
sat_input_norm
gu_input_norm
ego_encoder
sat_encoder
gu_encoder
down_query_proj
sat_add_proj
down_context_encoder
ctx0_encoder
gu_context_fusion
competition_blocks
score_head
tau_head
kappa_head
```

其中 `down_attention` 是无参数 masked dot-product readout，不作为 state_dict 模块。其余名字要和 native actor weight ABI 一致。

模块形状固定如下，避免 Python 与 native 对 MLP 层数理解不一致：

```text
LayerNorm:
  ego_input_norm: LayerNorm(BW_EGO_DIM)
  sat_input_norm: LayerNorm(BW_SAT_TOKEN_DIM)
  gu_input_norm:  LayerNorm(BW_GU_TOKEN_DIM)

MLP2(x_dim, y_dim):
  Linear(x_dim, hidden_dim)
  SiLU
  Linear(hidden_dim, y_dim)

ego_encoder:          MLP2(BW_EGO_DIM, embed_dim)
sat_encoder:          MLP2(BW_SAT_TOKEN_DIM, embed_dim)
gu_encoder:           MLP2(BW_GU_TOKEN_DIM, embed_dim)
down_query_proj:      Linear(embed_dim, down_query_count * embed_dim)
sat_add_proj:         Linear(embed_dim, embed_dim)
down_context_encoder: MLP2((down_query_count + 1) * embed_dim, embed_dim)
ctx0_encoder:         MLP2(2 * embed_dim, embed_dim)
gu_context_fusion:    MLP2(2 * embed_dim, embed_dim)
score_head:           MLP2(embed_dim, 1)
tau_head:             MLP2(embed_dim, 1)
kappa_head:           MLP2(embed_dim, 1)
```

`competition_blocks` 使用现有 masked self-attention block 形式：multi-head self-attention、attention residual norm、FFN、FFN residual norm。`embed_dim % bw_attention_heads == 0` 是硬校验。

不要沿用旧 BW policy 的：

```text
loc_head
priority_head
alpha_head
log_scale_head
support_logits
residual_head
lowdim_*
sat_refine
query_proj_1 / query_proj_2 旧语义
```

新 forward 主流程：

```python
def _params(self, state: LocalBwState):
    valid = state.gu_mask.bool() & state.bw_valid_mask.bool()
    valid_count = valid.sum(dim=-1)
    latent_count = torch.clamp(valid_count - 1, min=0)

    ego_emb = self.ego_encoder(self.ego_input_norm(state.ego_features))
    sat_emb = self.sat_encoder(self.sat_input_norm(state.selected_sat_tokens))
    gu_emb = self.gu_encoder(self.gu_input_norm(state.gu_tokens))

    q = self.down_query_proj(ego_emb).view(R, self.down_query_count, E)
    down_attn = masked_cross_attention(q, sat_emb, sat_emb, state.selected_sat_mask)
    sat_add_pool = masked_sum(self.sat_add_proj(sat_emb), state.selected_sat_mask)
    down_ctx = self.down_context_encoder(torch.cat([down_attn.flatten(1), sat_add_pool], dim=-1))

    ctx0 = self.ctx0_encoder(torch.cat([ego_emb, down_ctx], dim=-1))
    gu_h = self.gu_context_fusion(torch.cat([gu_emb, ctx0[:, None, :].expand_as(gu_emb)], dim=-1))
    gu_h = run_masked_self_attention_blocks(gu_h, valid)

    raw_score = self.score_head(gu_h).squeeze(-1)
    score = raw_score.masked_fill(~valid, -torch.inf)
    tau = tau_min + (tau_max - tau_min) * torch.sigmoid(self.tau_head(ctx0)).squeeze(-1)
    det_mean = masked_softmax(score / tau[:, None], valid)
    kappa = kappa_min + (kappa_max - kappa_min) * torch.sigmoid(self.kappa_head(ctx0)).squeeze(-1)
    alpha = det_mean * kappa[:, None]
    alpha = alpha.masked_fill(~valid, 0.0)
    return score, det_mean, alpha, kappa, tau, valid_count, latent_count
```

特殊行必须显式处理：

```text
valid_count == 0:
  action = 0
  logprob_raw = 0
  entropy_raw = 0

valid_count == 1:
  action = one_hot(valid GU)
  logprob_raw = 0
  entropy_raw = 0
```

随机策略使用 `MaskedMeanConcentrationDirichlet`：

```text
sagin_marl/rl/distributions.py
  MaskedMeanConcentrationDirichlet
```

`forward()`、`act_into()`、`evaluate_actions()` 都必须返回 per-latent objective logprob/entropy：

```python
latent_denom = torch.clamp(valid_count - 1, min=1).to(logprob_raw.dtype)
logprob = logprob_raw / latent_denom
entropy = entropy_raw / latent_denom
```

raw logprob/entropy 只作为 `BwPolicyOutput.logprob_raw`、`BwPolicyOutput.entropy_raw` 和 diagnostics 保存。

### 9.10 structured_factory.py 和配置

当前工厂里 BW policy 仍按旧 critic-like dims 构造：

```text
sagin_marl/rl/structured_factory.py
```

改为：

```python
from sagin_marl.rl import structured_bw_actor_schema as bw_schema

bw_policy = BwPolicy(
    ego_dim=bw_schema.BW_EGO_DIM,
    sat_token_dim=bw_schema.BW_SAT_TOKEN_DIM,
    gu_token_dim=bw_schema.BW_GU_TOKEN_DIM,
    down_query_count=config.bw_down_query_count,
    embed_dim=config.actor_embed_dim,
    hidden_dim=config.actor_hidden_dim,
    num_competition_layers=config.bw_competition_layers,
    num_heads=config.bw_attention_heads,
    tau_min=config.bw_tau_min,
    tau_max=config.bw_tau_max,
    kappa_min=config.bw_kappa_min,
    kappa_max=config.bw_kappa_max,
)
```

配置收敛要求：

```text
1. structured_bw_objective_norm 固定校验为 "per_latent_dim"。
2. bw_down_query_count 默认 2，必须 >= 1。
3. bw_competition_layers 默认 2，必须 >= 1。
4. bw_attention_heads 默认 4，且必须整除 actor_embed_dim。
5. bw_tau_min 默认 0.5，bw_tau_max 默认 2.0。
6. bw_kappa_min / bw_kappa_max 使用现有 Dirichlet concentration 配置口径，但必须满足 0 < min < max。
7. 旧 actor_arch、loc_readout、score_model、parameterization、support_count、residual_action 相关配置从新 BW 主路径删除。
8. 若旧 checkpoint 需要迁移，显式写 conversion script；不要让新 BwPolicy 静默加载旧权重。
```

### 9.11 MAPPO、rollout、eval 调用链

需要检查并修改：

```text
sagin_marl/rl/structured_mappo.py
sagin_marl/rl/structured_eval.py
sagin_marl/rl/structured_parallel_eval.py
sagin_marl/env/structured_driver.py
```

规则：

```text
1. 训练 PPO ratio 使用 BwPolicyOutput.logprob，它已经是 per-latent objective logprob。
2. diagnostics 同时记录 logprob_raw、entropy_raw、valid_count、latent_count。
3. eval/probe 不再调用旧 build_batched_local_bw_states_from_snapshot(...) 得到 candidate slot obs。
4. teacher/ref action 若保留，必须输出 full-G beta，并满足 invalid=0、valid sum=1。
5. 所有 action queue、history action ring 的 BW 动作维度改为 G。
6. old_logprob ring 改为每个 ego row 一个 per-latent objective 标量，并保留按 env 聚合的 sum；raw-logprob/raw-entropy 作为 per-ego diagnostics，不再保留 per-slot logprob。
```

### 9.12 native actor CUDA ABI

当前 native actor 相关文件：

```text
sagin_marl/rl/native_actor_cuda.py
sagin_marl/env/native_cuda/actor_kernels.cu
```

把 `ACTOR_WEIGHT_NAMES` 中 BW 段替换为新 `BwPolicy.state_dict()` 名称，顺序固定，Python 和 CUDA 共享同一份枚举。BW 权重名必须覆盖第 9.9 节模块：

```text
bw_policy.ego_input_norm.*
bw_policy.sat_input_norm.*
bw_policy.gu_input_norm.*
bw_policy.ego_encoder.*
bw_policy.sat_encoder.*
bw_policy.gu_encoder.*
bw_policy.down_query_proj.*
bw_policy.sat_add_proj.*
bw_policy.down_context_encoder.*
bw_policy.ctx0_encoder.*
bw_policy.gu_context_fusion.*
bw_policy.competition_blocks.*
bw_policy.score_head.*
bw_policy.tau_head.*
bw_policy.kappa_head.*
```

`LayerNorm` 权重使用 `.weight/.bias`；`MLP2` 权重使用 `.0.weight/.0.bias/.2.weight/.2.bias`；`down_query_proj` 和 `sat_add_proj` 使用 `.weight/.bias`。native binding 在 BW actor 启用时必须把缺失的 `bw_policy.*` 权重视为 hard error，不能继续沿用当前“缺失 BW 权重就跳过”的兼容行为。

删除或隔离旧 BW native ABI：

```text
W_BW_LOC_*
W_BW_PRIORITY_*
W_BW_ALPHA_*
W_BW_LOG_SCALE_*
W_BW_SUPPORT_*
W_BW_RESIDUAL_*
W_BW_LOWDIM_*
```

`actor_kernels.cu` 顶部的 `ActorIntParamIndex` / `ActorFloatParamIndex` 也要同步收敛。新 BW actor 只需要：

```text
ActorIntParamIndex:
  kActorHidden
  kActorEmbed
  kActorCompetitionHeads
  kActorCompetitionLayers
  kActorBwDownQueryCount
  kActorRngSeedLo
  kActorRngSeedHi

ActorFloatParamIndex:
  kActorBwTauMin
  kActorBwTauMax
  kActorBwKappaMin
  kActorBwKappaMax
```

旧 BW mode 参数从新主路径删除或忽略：

```text
kActorBwArch
kActorBwScore
kActorBwLoc
kActorBwParam
kActorTauEnabled
kActorSupportMinK
kActorSupportMaxK
kActorFixedLogScalePresent
kActorFixedKappaPresent

kActorBwAlphaMax
kActorBwFixedLogScale
kActorBwFixedKappa
kActorBwResidualTransferCap
kActorBwResidualFloor
kActorBwResidualAssocBonus
kActorBwSupportFrac
kActorBwSupportTailMass
kActorBwLocResidualMaxScale
```

`native_actor_cuda.py` 负责把 `config.bw_down_query_count` 写入 `kActorBwDownQueryCount`，把 `config.bw_competition_layers` / `config.bw_attention_heads` 写入 competition params，并在 build binding 时校验这些参数与 Python `BwPolicy` 实例一致。

`actor_bw_live_kernel` 输入输出改为：

```text
input:
  ego_features        [R, 10]
  selected_sat_tokens [R, K_sat, 9]
  selected_sat_mask   [R, K_sat]
  gu_tokens           [R, G, 13]
  gu_mask             [R, G]
  bw_valid_mask       [R, G]

output:
  action              [R, G]
  logprob             [R]
  entropy             [R]
  logprob_raw         [R]
  entropy_raw         [R]
  det_mean            [R, G]
  tau                 [R]
  kappa               [R]
  valid_count         [R]
  latent_count        [R]
```

对应 runtime 字段命名固定为：

```text
float tensors:
  live_bw_action                 [B,U,G]
  live_bw_ref_action             [B,U,G]
  live_bw_flow_proxy_override_action [B,U,G]
  live_bw_det_mean               [B,U,G]
  live_bw_old_logprobs_per_agent [B,U], per-latent objective old logprob
  live_bw_old_logprob            [B], sum_u live_bw_old_logprobs_per_agent[b,u]
  live_bw_entropy_per_agent      [B,U], per-latent objective entropy
  live_bw_logprob_raw_per_agent  [B,U], raw Dirichlet logprob
  live_bw_entropy_raw_per_agent  [B,U], raw Dirichlet entropy
  live_bw_tau                    [B,U]
  live_bw_kappa                  [B,U]

int64 tensors:
  live_bw_valid_count            [B,U]
  live_bw_latent_count           [B,U]
```

native live runtime 不要求把 Python `BwPolicyOutput` 的所有字段都常驻写入 buffer。`score` 和 `alpha` 可以只存在于 kernel scratch；上表字段是 rollout、诊断和 parity 必需输出。`valid_count` / `latent_count` 必须按 int64 写出，避免 Python、tensor backend、native runtime 对计数张量口径不一致。

CUDA 随机采样可复用现有 Dirichlet helper：

```text
block_dirichlet_log_prob_fast
gamma_sample_mt
```

但 `alpha` 的来源必须是新 mean-concentration：

```text
alpha[g] = det_mean[g] * kappa
```

native parity 第一阶段只要求 deterministic allclose，第二阶段再测 Dirichlet logprob/entropy 与 Python 一致。随机采样由于 RNG 不同，只测合法性和统计范围。

### 9.13 单 GPU native 性能约束

本设计不能破坏当前单 GPU 原生 rollout 的目标：减少 Python 调度、减少小 PyTorch op、减少小 CUDA kernel。实现时分清三条路径：

```text
Python builder / Python BwPolicy:
  只作为 reference、单环境 debug、训练 evaluate_actions 路径。

tensor backend:
  可以用 PyTorch batched tensor ops，但不能在 hot loop 里 per-UAV / per-GU Python 循环调度小 op。

native CUDA live rollout:
  必须使用 fused CUDA 路径构造 BW obs、执行 BW actor、写 action/logprob/entropy。
```

native live actor 的硬要求：

```text
1. 一次 BW actor live 调用仍是 actor_bw_live_kernel 一个 row-block grid：
     grid.x = B * U
     block 内处理该 ego row 的 selected SAT 和 GU。
2. ego encoder、selected SAT attention、sat_add_pool、GU context fusion、competition blocks、score、tau/kappa、Dirichlet sample/logprob 全部在 actor_bw_live_kernel 内完成。
3. 不允许在 live rollout 中用 Python 调用若干 torch LayerNorm/Linear/attention/softmax 小 kernel 来拼出 BW action。
4. BW obs tensor 必须由 native/tensor stage builder 批量写入 contiguous buffer，不允许 Python per-row 构造 LocalBwState list 再搬到 GPU。
5. 权重同步只发生在 policy update 后或显式 sync 时；step hot path 不做 state_dict 遍历或 Python 权重拼接。
6. diagnostics 如 valid_count、latent_count、det_mean、raw/per-latent logprob 必须在同一个 live kernel 写出，不能为每个诊断再发小 kernel。
```

scratch / complexity 约束：

```text
max_item_count = max(G, K_sat)
max_in_dim = max(BW_EGO_DIM, BW_SAT_TOKEN_DIM, BW_GU_TOKEN_DIM, 2*embed_dim, (down_query_count+1)*embed_dim)
```

full-G schema 会把 BW action/obs 宽度从旧 `users_obs_max` 改为 `G`。这会增加固定宽度张量，但不会要求更多 Python 调度；native kernel 内部可以为 valid GU 建立 compact scratch index 来跳过 invalid GU 的重计算，同时最终 action 仍写回 full-G `[R,G]`。

native 常量边界必须显式校验：

```text
actor_embed_dim <= kMaxEmbed
actor_hidden_dim <= kMaxHidden
G <= kMaxItems
K_sat <= kMaxSelect，或同步提高 kMaxSelect 并更新 scratch 预算
```

已有 native device helper 可以继续复用：

```text
block_mlp2
block_mlp2_items
block_multi_query_attention / block_attention
block_competition_layer
gamma_sample_mt
block_dirichlet_log_prob_fast
```

### 9.14 旧代码清理标准

实现完成后，以下搜索在新主路径中不应再命中 BW actor 逻辑：

```text
rg "candidate_indices|users_obs_max|support_mask|support_members|priority_head|loc_head|alpha_head|log_scale_head" sagin_marl
rg "ego_uav_after_sat|sat_edges|user_edges|user_nodes|user_mask" sagin_marl
```

允许命中的情况：

```text
1. 旧实验兼容代码，文件名或函数名带 legacy。
2. 文档或 migration script。
3. critic/其它 actor 自己的字段，且不被 BW actor 读取。
```

## 10. 迁移顺序

按以下顺序改代码，减少 Python/native 不一致：

```text
1. 新增 structured_bw_actor_schema.py。
2. 修改 LocalBwState 和 BwPolicyOutput。
3. 写 Python builder reference，并用单元测试锁定 shape 和字段值。
4. 修改环境 access 干扰为 beta-continuous，并测试 rate/interference reference。
5. 重写 Python BwPolicy，先跑 deterministic/stochastic action 合法性测试。
6. 修改 buffer、MAPPO、eval 调用链，确保训练能用 Python actor 跑通。
7. 修改 native obs view、history buffer、CUDA local obs builder。
8. 修改 native actor ABI 和 actor_bw_live_kernel。
9. 加 native parity 和 native live-path smoke 测试，确认没有 Python actor fallback。
10. 同步 accel actor 和 critic 中所有 last interference 字段。
11. 删除或隔离旧 BW candidate/support 代码。
```

## 11. 测试

新增：

```text
tests/test_structured_bw_actor_schema.py
tests/test_structured_bw_actor_native_parity.py
tests/test_structured_bw_interference_model.py
```

必须覆盖：

```text
1. LocalBwState shape 与 schema dim 一致。
2. bw_valid_mask 与 association 完全一致。
3. access_rate_full_bw_ref_steps 与 numpy reference 一致。
4. cross_interference_mean/max 与 access_gain_matrix reference 一致。
5. selected_sat_tokens 只包含 capacity/workload/cost，不包含 raw geometry/doppler/backhaul_se_ref。
6. actor deterministic action invalid beta == 0。
7. actor deterministic action valid beta sum == 1。
8. stochastic action invalid beta == 0。
9. stochastic action valid beta sum == 1。
10. valid_count == 0 时 action/logprob/entropy 全为 0。
11. valid_count == 1 时唯一 valid GU beta = 1。
12. Python actor deterministic 与 native actor deterministic allclose。
13. env 执行合法 actor action 时不改变 beta。
14. beta-continuous 干扰：
    beta_h=0 无干扰贡献；
    beta_h=0.01 的干扰小于 beta_h=1.0；
    Python/native 结果一致。
15. accel actor last interference features 使用 beta-weighted 口径。
16. critic world last_access_interference 字段使用 beta-weighted 口径。
17. native live rollout 使用 fused actor_bw_live_kernel，不能回退到 Python actor 或 per-row LocalBwState list。
18. native BW action/logprob/entropy/diagnostics 写入 contiguous full-G buffer。
19. old_logprob buffer 是 per-ego per-latent objective scalar，并可按 env 求和；raw-logprob/raw-entropy diagnostics 是 per-ego scalar，不再是 per-GU/per-slot 张量。
```

## 12. 最终链路

对单台 ego UAV：

```text
BW prefix:
  association
  bw_valid_mask
  selected SAT
  selected SAT capacity/workload
  queues / EMA / last history
  access gain snapshot

-> LocalBwState:
  ego relay features
  selected SAT capacity/workload tokens
  full-G GU queue/link/cost/interference tokens
  full-G bw_valid_mask

-> Shared BW actor:
  encode ego
  encode selected SAT downstream
  encode GU tokens
  run competition only over valid GU
  compute per-valid-GU score
  masked mean-concentration Dirichlet simplex

-> Output:
  beta_by_gu for this ego UAV only

-> Environment:
  execute beta directly
  access rate and interference use beta-continuous model
```

一句话：

```text
BW actor 是单 ego UAV 的参数共享带宽分配器。
它只在当前连接有效 GU 上分配 full-G masked simplex beta；
下游 backhaul 用 selected SAT capacity/workload 表达；
access interference 必须随 beta 连续变化，并在 env / critic / accel actor / native 路径中统一。
```
