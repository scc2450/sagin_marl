# 模型结构说明

本文档描述当前正式课程配置实际启用的模型结构与动作语义，覆盖以下 3 个配置：

- `configs/phase1_actions_curriculum_stage1_accel.yaml`
- `configs/phase1_actions_curriculum_stage2_bw.yaml`
- `configs/phase1_actions_curriculum_stage3_sat.yaml`

相关实现：

- `sagin_marl/env/config.py`
- `sagin_marl/env/sagin_env.py`
- `sagin_marl/rl/policy.py`
- `sagin_marl/rl/distributions.py`
- `sagin_marl/rl/critic.py`
- `sagin_marl/rl/buffer.py`
- `sagin_marl/rl/mappo.py`
- `sagin_marl/rl/action_assembler.py`
- `sagin_marl/rl/baselines.py`

## 1. 结论

这 3 个正式配置使用的是同一套 MAPPO 框架：

- actor 都是 `ActorNet`
- critic 都是 `CriticNet`
- actor 编码器都为 `actor_encoder_type: set_pool`
- actor 隐层宽度都为 `actor_hidden: 256`
- set encoder 嵌入维度都为 `actor_set_embed_dim: 64`
- critic 隐层宽度都为 `critic_hidden: 256`
- 输入归一化都开启了 `input_norm_enabled: true`

但它们不是“完全相同的网络实例”。

更准确地说：

- actor 是否构建 `bw` / `sat` 头，由 `enable_bw_action` 与 `fixed_satellite_strategy` 决定
- 哪些参数可训练，由 `train_accel` / `train_bw` / `train_sat` / `train_shared_backbone` / `train_fusion_last_layer` 决定

当前 3 个阶段的实际语义如下：

| 阶段 | actor 可用动作头 | 动作向量维度 | 执行动作来源 | 当前可训练模块 |
| --- | --- | ---: | --- | --- |
| Stage 1 | `accel` | 2 | `accel` 来自当前策略；`bw=0`；`sat=0` | backbone + accel head |
| Stage 2 | `accel + bw` | 22 | `accel`、`bw` 来自当前策略；`sat=0` | bw head |
| Stage 3 | `accel + bw + sat` | 28 | `accel`、`bw`、`sat` 都来自当前策略 | sat head |

和旧版本相比，当前实现的关键变化是：

- `bw` 头输出的是最终 `bw_alloc`，不再输出 residual logit
- `sat` 头在策略内部完成 masked sequential categorical 采样，不再由环境对连续 logit 二次抽样
- PPO 重算 `log_prob` 时同时使用 `env_action` 和 `sat_indices`

### 1.1 最终标准化参数表

当前“最后定版”的标准化口径不是所有阶段共享同一个到达率和队列参数，而是：

- 资源参数 4 个阶段统一
- 负载和队列参数按 `fixed_satellite_strategy` 分成两组

公共资源参数如下：

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| `num_sat` | `144` | 4 个正式阶段统一使用 144 星 |
| `uav_tx_gain` | `63.1` | 约 `18 dBi` |
| `sat_rx_gain` | `63.1` | 约 `18 dBi` |
| `b_acc` | `1.0e7` | `10 MHz` |
| `b_backhaul_per_sat` | `1.5e7` | 每颗卫星可分享的总 backhaul 带宽池为 `15 MHz` |
| `sat_state_max` | `9` | critic 全局状态最多保留 9 颗卫星 |

`Stage 1 / Stage 2` 使用 fixed-sat 口径：

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| `task_arrival_rate` | `9.0e5` | 每 GU 每 slot 平均到达量 |
| `queue_ref_gu_per_step` | `1.80e7` | GU 层总入流参考 |
| `queue_ref_uav_per_step` | `1.80e7` | UAV 层总入流参考 |
| `queue_ref_sat_per_step` | `1.72e7` | SAT 层总入流参考，来自 rollout 实测取整 |
| `queue_ref_sat_active_count` | `1.0` | fixed-sat 下按 1 颗活跃卫星折算 SAT 队列 |
| `queue_max_gu_steps` | `50` | GU 最大缓冲时长 |
| `queue_max_uav_steps` | `100` | UAV 最大缓冲时长 |
| `queue_max_sat_steps` | `150` | SAT 最大缓冲时长 |
| `queue_max_gu` | `4.5e7` | 单 GU 队列上限 |
| `queue_max_uav` | `6.0e8` | 单 UAV 队列上限 |
| `queue_max_sat` | `2.58e9` | 单活跃 SAT 队列上限 |
| `queue_init_gu_steps` | `4` | GU 初始预加载 |
| `queue_init_uav_steps` | `8` | UAV 初始预加载 |
| `queue_init_sat_steps` | `12` | SAT 初始预加载 |

`Stage 3a / Stage 3` 使用开放选星口径：

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| `task_arrival_rate` | `1.2e6` | 每 GU 每 slot 平均到达量 |
| `queue_ref_gu_per_step` | `2.40e7` | GU 层总入流参考 |
| `queue_ref_uav_per_step` | `2.40e7` | UAV 层总入流参考 |
| `queue_ref_sat_per_step` | `2.36e7` | SAT 层总入流参考，来自 rollout 实测取整 |
| `queue_ref_sat_active_count` | `3.0` | 开放选星下按 3 颗活跃卫星折算 SAT 队列 |
| `queue_max_gu_steps` | `50` | GU 最大缓冲时长 |
| `queue_max_uav_steps` | `100` | UAV 最大缓冲时长 |
| `queue_max_sat_steps` | `150` | SAT 最大缓冲时长 |
| `queue_max_gu` | `6.0e7` | 单 GU 队列上限 |
| `queue_max_uav` | `8.0e8` | 单 UAV 队列上限 |
| `queue_max_sat` | `1.18e9` | 单参考活跃 SAT 队列上限 |
| `queue_init_gu_steps` | `4` | GU 初始预加载 |
| `queue_init_uav_steps` | `8` | UAV 初始预加载 |
| `queue_init_sat_steps` | `12` | SAT 初始预加载 |

### 1.2 到达率如何确定

当前到达率不是直接从静态链路预算反推出来的，而是按下面的顺序确定：

1. 先固定资源参数：
   `num_sat = 144`、`uav_tx_gain = 63.1`、`sat_rx_gain = 63.1`、`b_acc = 1.0e7`、`b_backhaul_per_sat = 1.5e7`
2. 使用 `cluster_center_queue_aware` baseline，在“大队列上限 + 零预加载”条件下做 rollout 网格，测真实分层流量：
   `arrival_per_step`、`gu_outflow_per_step`、`sat_incoming_per_step`
3. 同时统计每步实际活跃卫星数，用于决定 `queue_ref_sat_active_count`

在当前最终配置里：

- `Stage 1 / Stage 2` 的 fixed-sat 基线平均活跃卫星数约为 `1.0`
- `Stage 3a / Stage 3` 的开放选星基线平均活跃卫星数约为 `2.8`，最终配置取整为 `3.0`

`queue_ref_gu_per_step` 的确定方式是：

```text
queue_ref_gu_per_step
= task_arrival_rate * num_gu * tau0 * traffic_level_ratio
```

当前正式配置里 `num_gu = 20`、`tau0 = 1`、`traffic_level = 2`、`traffic_level_hard_ratio = 1.0`，因此：

```text
Stage 1 / Stage 2: 9.0e5 * 20 = 1.80e7
Stage 3a / Stage 3: 1.2e6 * 20 = 2.40e7
```

`queue_ref_uav_per_step` 取和 GU 层总入流同量级的 rollout 参考值。

`queue_ref_sat_per_step` 不直接用静态公式，而是用 rollout 中 `sat_incoming_per_step` 的实测均值取整：

- `Stage 1 / Stage 2` 取 `1.72e7`
- `Stage 3a / Stage 3` 取 `2.36e7`

最终 `queue_max_*` 的换算规则为：

```text
queue_max_layer
= queue_max_layer_steps * queue_ref_layer_per_step / ref_entity_count
```

其中：

- GU 层 `ref_entity_count = num_gu`
- UAV 层 `ref_entity_count = num_uav`
- SAT 层 `ref_entity_count = queue_ref_sat_active_count`

也就是说，SAT 队列上限不再按 `num_sat = 144` 均摊，而是按“参考活跃卫星数”折算。

## 2. 整体训练结构

当前训练仍然是标准的集中式 critic、分散式 actor：

- actor 输入是每个 UAV 的局部观测 `obs`
- critic 输入是环境全局状态 `env.get_global_state()` 与同一步的多智能体局部观测 `obs_step`
- 所有 UAV 共用同一个 actor 网络参数
- critic 内部按 UAV 编码局部观测后再聚合
- critic 输出单个标量团队价值 `V_team`

也就是说：

- actor 解决“每个 UAV 该怎么动、怎么分带宽、怎么选卫星”
- critic 解决“当前这一整步团队状态值多少钱”

## 3. Actor 输入定义

### 3.1 输入来源

actor 的输入来自 `SaginParallelEnv._get_obs()`，然后在 `policy.py` 中通过 `flatten_obs()` 展平成一维向量。

当前正式配置下，展平顺序固定为：

1. `own`
2. `danger_nbr`
3. `users`
4. `users_mask`
5. `bw_valid_mask`
6. `sats`
7. `sats_mask`
8. `sat_valid_mask`
9. `nbrs`
10. `nbrs_mask`

当前正式配置都启用了：

- `danger_nbr_enabled: true`
- `users_obs_max: 20`
- `sats_obs_max: 6`
- `nbrs_obs_max: 4`
- `append_action_masks_to_obs: true`

因此当前正式配置的 actor 输入维度固定为：

```text
obs_dim
= own(10)
+ danger_nbr(5)
+ users(20 * 5)
+ users_mask(20)
+ bw_valid_mask(20)
+ sats(6 * 12)
+ sats_mask(6)
+ sat_valid_mask(6)
+ nbrs(4 * 4)
+ nbrs_mask(4)
= 259
```

### 3.2 `own`，形状 `(10,)`

当前 `own` 特征为：

1. `uav_pos_x / map_size`
2. `uav_pos_y / map_size`
3. `uav_vel_x / v_max`
4. `uav_vel_y / v_max`
5. `uav_energy / uav_energy_init`
6. `uav_queue / queue_max_uav`
7. `t / T_steps`
8. `assoc_count / num_gu`
9. `assoc_centroid_rel_x`
10. `assoc_centroid_rel_y`

相比旧文档，这里最重要的变化是：

- `own` 现在不是 7 维，而是 10 维
- 新增了关联用户数量归一化和关联质心相对位置

虽然当前正式配置里 `energy_enabled: false`，能量特征仍然保留在观测中。

### 3.3 `danger_nbr`，形状 `(5,)`

这是一个“最危险邻机摘要”，不是邻机集合本身。

特征定义：

1. 邻机距离 `dist / map_size`
2. 接近速度 `closing_speed / v_max`
3. 相对方向 `dir_x`
4. 相对方向 `dir_y`
5. 有效标记 `1.0 / 0.0`

### 3.4 `users`，形状 `(20, 5)`，配合 `users_mask` 与 `bw_valid_mask`

每个 UAV 最多观察 `20` 个候选地面用户。当前正式配置中：

- `candidate_mode: nearest`
- `candidate_k: 20`
- `users_obs_max: 20`

单个用户特征定义：

1. `rel_x / map_size`
2. `rel_y / map_size`
3. `gu_queue / queue_max_gu`
4. 接入链路频谱效率 `eta`
5. 上一步是否关联到当前 UAV 的标记 `prev_association == u`

说明：

- `users_mask` 表示该槽位是否有真实候选用户
- `bw_valid_mask` 进一步表示该槽位当前是否允许参与带宽分配

### 3.5 `sats`，形状 `(6, 12)`，配合 `sats_mask` 与 `sat_valid_mask`

每个 UAV 最多观察 `6` 个候选卫星。

当前正式配置都满足：

- `visible_sats_max: 6`
- `sats_obs_max: 6`

单个卫星特征定义：

1. `rel_pos_x / (r_earth + sat_height)`
2. `rel_pos_y / (r_earth + sat_height)`
3. `rel_pos_z / (r_earth + sat_height)`
4. `rel_vel_x / (r_earth + sat_height)`
5. `rel_vel_y / (r_earth + sat_height)`
6. `rel_vel_z / (r_earth + sat_height)`
7. `nu / nu_max`
8. `spectral_efficiency`
9. `sat_queue / queue_max_sat`
10. `load_count / num_uav`
11. `1 / projected_count`
12. `stay_flag`

说明：

- `sats_mask` 表示该候选槽位是否有真实卫星
- `sat_valid_mask` 表示该槽位当前是否允许参与卫星选择

### 3.6 `nbrs`，形状 `(4, 4)`，配合 `nbrs_mask`

每个 UAV 最多观察 `4` 个邻近 UAV，按距离排序截断。

单个邻机特征定义：

1. `rel_pos_x / map_size`
2. `rel_pos_y / map_size`
3. `rel_vel_x / v_max`
4. `rel_vel_y / v_max`

### 3.7 关于队列归一化

观测里的 `uav_queue / queue_max_uav`、`gu_queue / queue_max_gu`、`sat_queue / queue_max_sat` 使用的是加载配置后的最终 `queue_max_*`。

当前代码里这些值可能来自两种来源：

- 直接在 yaml 中写定 `queue_max_*`
- 通过 `queue_max_*_steps` 与 `queue_ref_*_per_step` 在 `config.py` 中换算得到

此外，SAT 的 `queue_max_sat` 可以通过 `queue_ref_sat_active_count` 按“参考活跃卫星数”折算，而不是简单按 `num_sat` 均摊。

## 4. Actor 骨干结构

### 4.1 当前启用的是 `set_pool` 编码器

当前 3 个正式配置都没有使用旧的 `flat_mlp`，实际启用的是 `set_pool`。

其核心思路是：

- 对标量或小向量输入单独编码
- 对可变长集合输入先逐元素编码
- 再通过带 mask 的 mean pooling + max pooling 聚合
- 最后把所有分支特征拼接后送入融合 MLP

### 4.2 各子编码器

当前参数：

- `actor_set_embed_dim = 64`
- `actor_hidden = 256`
- `input_norm_enabled = true`

`_make_encoder(in_dim, hidden_dim)` 的结构统一为：

```text
LayerNorm(in_dim)    # 仅在 input_norm_enabled=true 时启用
Linear(in_dim -> hidden_dim)
ReLU
Linear(hidden_dim -> hidden_dim)
ReLU
```

因此：

- `own_encoder` 输入 10 维，输出 64 维
- `danger_nbr_encoder` 输入 5 维，输出 64 维
- `users_encoder` 输入 5 维，输出 64 维
- `sats_encoder` 输入 12 维，输出 64 维
- `nbrs_encoder` 输入 4 维，输出 64 维

其中 `users`、`sats`、`nbrs` 三个集合分支都会经过 masked mean pooling + masked max pooling，因此各自产生 `128` 维聚合特征。

### 4.3 特征融合层

当前正式配置启用了 `danger_nbr`，因此最终拼接维度为：

```text
fusion_in_dim
= own_feat(64)
+ danger_nbr_feat(64)
+ users_pool(128)
+ sats_pool(128)
+ nbrs_pool(128)
= 512
```

融合 MLP 为：

```text
Linear(512 -> 256)
ReLU
Linear(256 -> 256)
ReLU
```

其输出就是所有动作头共享的上下文表示 `ctx`。

## 5. Actor 输出头

### 5.1 `accel`

`accel` 头在所有阶段都存在：

```text
mu_head: Linear(256 -> 2)
log_std: trainable parameter, shape=(2,)
```

动作分布：

- 每一维使用高斯分布 `Normal(mu, std)`
- `log_std` 会 clamp 到 `[-5, 2]`
- 训练时使用 `rsample()`
- 最终经过 `tanh` squash 到 `[-1, 1]`

环境执行前再乘上 `a_max`，并可能叠加安全修正。

### 5.2 `bw`

当前 `bw` 头不再直接输出 20 个 residual logit，而是改为“共享上下文 + 逐用户打分”：

```text
user_emb[k] = BWUserEncoder(user_k)
score[k]    = BWScorer([ctx, user_emb[k], raw_user_k])
alpha[k]    = softplus(score[k]) + bw_alpha_floor
```

对应实现：

- `bw_user_encoder`
- `bw_scorer`
- `MaskedDirichlet`

采样语义：

- `n_valid = 0`：全 0，`log_prob = 0`
- `n_valid = 1`：唯一有效槽位置 1，`log_prob = 0`
- `n_valid >= 2`：只在有效槽位上构造 Dirichlet

输出的是最终 `bw_alloc`，不是 residual logit。

### 5.3 `sat`

当前 `sat` 头采用“共享上下文 + 逐卫星打分”：

```text
sat_emb[m] = SatActionEncoder(sat_m)
logit[m]   = SatScorer([ctx, sat_emb[m], raw_sat_m])
```

对应实现：

- `sat_action_encoder`
- `sat_scorer`
- `MaskedSequentialCategorical`

当前正式配置：

- `N_RF = 2`
- `sat_num_select = 2`
- 顺序无放回采样
- 输出 `sat_indices` 与 `sat_select_mask`

## 6. Hybrid distribution 与动作语义

当前策略分布由三部分组成：

- `accel_dist`：squashed Gaussian
- `bw_dist`：`MaskedDirichlet`
- `sat_dist`：`MaskedSequentialCategorical`

统一封装在 `HybridActionDist` 中，提供：

- `sample()`
- `mode()`
- `log_prob()`
- `entropy()`

`sample()` 的关键输出包括：

- `env_action`
- `sat_indices`
- `sat_select_mask`

其中：

- `env_action` 是送进环境的连续动作向量
- `sat_indices` 是 PPO 重算 `log_prob` 时需要保留的离散采样结果
- `sat_select_mask` 是环境执行用的卫星选择 mask

总 `log_prob` 与总 `entropy` 为三头求和，但训练日志会分头记录：

- `entropy_accel`
- `entropy_bw`
- `entropy_sat`

当前 actor 内部动作顺序固定为：

1. `accel(2)`
2. `bw(20)`，若启用
3. `sat(6)`，若启用

因此动作维度分别为：

- Stage 1：`2`
- Stage 2：`2 + 20 = 22`
- Stage 3：`2 + 20 + 6 = 28`

当前环境语义是：

- `bw` 段表示最终 `bw_alloc`
- `sat` 段表示最终 `sat_select_mask`

`action_assembler.py` 现在只做动作打包，不再负责旧版 residual 或 env-side sat 采样语义。

## 7. Critic 输入定义

critic 同时使用两类输入：

- `global_state`：`SaginParallelEnv.get_global_state()` 输出的一维全局状态
- `obs_step`：与同一步 `global_state` 对齐的多智能体局部观测，形状为 `[B, N, D]`

其中当前 global state 的拼接顺序为：

1. `uav_pos.flatten() / map_size`
2. `uav_vel.flatten() / v_max`
3. `uav_queue / queue_max_uav`
4. `uav_energy / uav_energy_init`
5. `gu_pos.flatten() / map_size`
6. `gu_queue / queue_max_gu`
7. `sat_pos.flatten() / (r_earth + sat_height)`
8. `sat_vel.flatten() / (r_earth + sat_height)`
9. `sat_queue / queue_max_sat`
10. `t / T_steps`

当前 3 个正式配置虽然都使用：

- `num_sat = 144`

但同时都设置了：

- `sat_state_max = 9`

因此 global branch 侧实际只截取 9 颗卫星状态，`state_dim` 仍保持为：

```text
state_dim
= uav_pos(3 * 2 = 6)
+ uav_vel(3 * 2 = 6)
+ uav_queue(3)
+ uav_energy(3)
+ gu_pos(20 * 2 = 40)
+ gu_queue(20)
+ sat_pos(9 * 3 = 27)
+ sat_vel(9 * 3 = 27)
+ sat_queue(9)
+ time(1)
= 142
```

`obs_step` 不引入新的环境专用字段，而是直接复用 actor 当前使用的展平局部观测定义。

在当前正式配置下：

- `N = num_uav = 3`
- `D = obs_dim = 259`

## 8. Critic 结构

critic 当前不再是“只吃 global state 的两层 MLP”，而是：

```text
global branch:
  LayerNorm(142)
  Linear(142 -> 256)
  ReLU
  Linear(256 -> 256)
  ReLU

local branch (per UAV):
  own / danger_nbr 单独编码
  users 做 visible pool + valid pool
  sats 做 visible pool + valid pool
  nbrs 做 masked pool
  再拼接 derived stats

value head:
  [g_ctx, local_ctx_i, d_i] -> v_i
  mean_i(v_i) -> V_team
```

其中：

- `users` 的两路 pooling 分别对应 `users_mask` 与 `bw_valid_mask`
- `sats` 的两路 pooling 分别对应 `sats_mask` 与 `sat_valid_mask`
- derived stats 直接从现有局部观测中计算，不额外引入新的 env 字段

最终输出仍然是单个标量团队价值，不改现有 shared-reward 的 GAE / PPO 接口。

### 8.1 Derived Stats（25 维）

当前 `d_i` 由 critic 在 `forward()` 内部从 `obs_step` 现算，组成如下：

- own / safety（5 维）
  - `assoc_count`
  - `assoc_centroid_rel_x`
  - `assoc_centroid_rel_y`
  - `danger_dist`
  - `danger_closing`
- users（8 维）
  - `visible_user_ratio`
  - `valid_user_ratio`
  - `user_eta_mean`
  - `user_eta_max`
  - `user_eta_top1_gap`
  - `user_queue_mean`
  - `user_queue_max`
  - `user_prev_assoc_ratio`
- sats（12 维）
  - `visible_sat_ratio`
  - `valid_sat_ratio`
  - `sat_se_mean`
  - `sat_se_max`
  - `sat_se_top1_gap`
  - `sat_queue_mean`
  - `sat_queue_min`
  - `sat_load_mean`
  - `sat_load_max`
  - `sat_projected_mean`
  - `sat_projected_min`
  - `sat_stay_ratio`

这些量都不是额外的环境输入，而是从 `own`、`danger_nbr`、`users`、`sats` 及其 masks 内部派生出来的摘要特征。

## 9. 各阶段训练语义

### 9.1 Stage 1

配置特征：

- `enable_bw_action: false`
- `fixed_satellite_strategy: true`
- `train_accel: true`
- `train_bw: false`
- `train_sat: false`
- `train_shared_backbone: true`

含义：

- actor 只有 accel 头
- backbone 与 accel 头一起训练
- 不学习带宽分配
- 不学习卫星选择

### 9.2 Stage 2

配置特征：

- `enable_bw_action: true`
- `fixed_satellite_strategy: true`
- `train_accel: false`
- `train_bw: true`
- `train_sat: false`
- `train_shared_backbone: false`
- `train_fusion_last_layer: false`
- `exec_accel_source: policy`
- `exec_bw_source: policy`

含义：

- actor 可用头为 `accel + bw`
- accel 头继续参与执行，但不参与训练
- 共享 backbone 全部冻结
- bw head 可训练

因此 Stage 2 现在就是“完全冻结 backbone，仅训练 bw 头”。

### 9.3 Stage 3

配置特征：

- `enable_bw_action: true`
- `fixed_satellite_strategy: false`
- `train_accel: false`
- `train_bw: false`
- `train_sat: true`
- `train_shared_backbone: false`
- `exec_accel_source: policy`
- `exec_bw_source: policy`
- `exec_sat_source: policy`

含义：

- 直接继承 Stage 2 的 actor / critic 初始化
- backbone 冻结
- accel / bw 不训练
- 只训练 sat 头
- 执行时三个动作都来自当前策略

## 10. PPO 重算与执行一致性

当前版本专门修复了“策略优化的动作”和“环境真实执行的动作”不一致的问题：

- `action_assembler.py` 只做动作打包
- `buffer.py` 保存 `sat_indices`
- `mappo.py` 在 update 时用 `env_action + sat_indices` 重算 hybrid `log_prob`
- `sagin_env.py` 缓存 step 前的有效用户与卫星槽位，并按同一时刻的数据执行动作

这也是当前动作分布重构里最核心的一点。

## 11. 文档范围说明

本文档只覆盖当前这 3 个正式课程配置实际启用的模型结构。

如果后续：

- 切回 `flat_mlp`
- 修改 `users_obs_max` / `sats_obs_max` / `nbrs_obs_max`
- 关闭 `danger_nbr_enabled`
- 关闭 `append_action_masks_to_obs`
- 调整 `sat_state_max`
- 改变 `queue_max_*_steps` / `queue_ref_*_per_step` / `queue_ref_sat_active_count` 的语义

那么本文中的维度数字和归一化说明都需要同步更新。
