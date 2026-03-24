# 模型结构说明

本文档只描述当前正式三阶段配置实际启用的模型结构，不再保留已经过时的 residual-bw / env-side sat 采样语义。

对应配置文件：
- `configs/phase1_actions_curriculum_stage1_accel.yaml`
- `configs/phase1_actions_curriculum_stage2_bw.yaml`
- `configs/phase1_actions_curriculum_stage3_sat.yaml`

相关实现：
- `sagin_marl/rl/policy.py`
- `sagin_marl/rl/distributions.py`
- `sagin_marl/rl/critic.py`
- `sagin_marl/rl/mappo.py`
- `sagin_marl/rl/action_assembler.py`
- `sagin_marl/rl/baselines.py`
- `sagin_marl/env/sagin_env.py`

## 1. 结论

这三个正式配置使用的是同一套 MAPPO 框架：
- actor 都是 `ActorNet`
- critic 都是 `CriticNet`
- actor 编码器类型都为 `actor_encoder_type: set_pool`
- actor 隐层宽度都为 `actor_hidden: 256`
- set encoder 嵌入维度都为 `actor_set_embed_dim: 64`
- critic 隐层宽度都为 `critic_hidden: 256`
- 输入归一化都开启了 `input_norm_enabled: true`

但它们不是“完全相同的网络实例”。

更准确地说，三阶段共享同一套骨干设计，但按阶段打开了不同动作头，并且训练状态不同：

| 阶段 | actor 内部启用的头 | 动作向量维度 | 真正执行的动作来源 | 训练状态 |
| --- | --- | ---: | --- | --- |
| Stage 1 | `accel` | 2 | `accel` 来自当前策略；`bw=0`；`sat=0` | 训练 backbone + accel head |
| Stage 2 | `accel + bw` | 22 | `accel`、`bw` 都来自当前策略；`sat=0` | 冻结 backbone 与 accel，仅训练 bw head |
| Stage 3 | `accel + bw + sat` | 28 | `accel`、`bw`、`sat` 都来自当前策略 | 冻结 backbone，联合训练三个动作头 |

当前版本和旧实现的核心差异：
- `bw` 头已经改为最终 `bw_alloc`，不再输出 residual logit
- `sat` 头已经改为策略内 masked categorical，不再由环境对连续 logit 二次抽样
- PPO 重算 `log_prob` 时会同时使用 `env_action` 和 `sat_indices`

## 2. 整体训练结构

当前训练仍然是标准的集中式 critic、分散式 actor：
- actor 输入是每个 UAV 的局部观测 `obs`
- critic 输入是环境全局状态 `env.get_global_state()`
- 每个 UAV 共用同一个 actor 网络参数
- critic 输出单个标量状态价值 `V(s)`

也就是说：
- actor 解决“每个 UAV 该怎么动、怎么分带宽、怎么选卫星”
- critic 解决“当前全局状态值多少钱”

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

因为当前正式配置都启用了：
- `danger_nbr_enabled: true`
- `users_obs_max: 20`
- `sats_obs_max: 6`
- `nbrs_obs_max: 4`
- `append_action_masks_to_obs: true`

所以当前正式配置的 actor 输入维度固定为：

```text
obs_dim
= own(7)
+ danger_nbr(5)
+ users(20 * 5)
+ users_mask(20)
+ bw_valid_mask(20)
+ sats(6 * 12)
+ sats_mask(6)
+ sat_valid_mask(6)
+ nbrs(4 * 4)
+ nbrs_mask(4)
= 256
```

### 3.2 `own`，形状 `(7,)`

当前 UAV 自身状态：
1. `uav_pos_x / map_size`
2. `uav_pos_y / map_size`
3. `uav_vel_x / v_max`
4. `uav_vel_y / v_max`
5. `uav_energy / uav_energy_init`
6. `uav_queue / queue_max_uav`
7. `t / T_steps`

虽然当前正式配置里 `energy_enabled: false`，但能量特征仍然保留在观测中。

### 3.3 `danger_nbr`，形状 `(5,)`

这是一个“最危险邻机摘要”，不是邻机集合本身。

特征定义：
1. 邻机距离 `dist / map_size`
2. 接近速度 `closing_speed / v_max`
3. 相对方向 `dir_x`
4. 相对方向 `dir_y`
5. 有效标记 `1.0/0.0`

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
1. 相对位置 `rel_pos_x / (r_earth + sat_height)`
2. 相对位置 `rel_pos_y / (r_earth + sat_height)`
3. 相对位置 `rel_pos_z / (r_earth + sat_height)`
4. 相对速度 `rel_vel_x / (r_earth + sat_height)`
5. 相对速度 `rel_vel_y / (r_earth + sat_height)`
6. 相对速度 `rel_vel_z / (r_earth + sat_height)`
7. 多普勒归一化 `nu / nu_max`
8. 当前链路频谱效率 `spectral_efficiency`
9. 卫星队列归一化 `sat_queue / queue_max_sat`
10. 当前负载计数归一化 `load_count / num_uav`
11. 预计接入后负载倒数 `1 / projected_count`
12. 是否是当前已连接卫星 `stay_flag`

说明：
- `sats_mask` 表示该候选槽位是否有真实卫星
- `sat_valid_mask` 进一步表示当前是否允许参与卫星选择

### 3.6 `nbrs`，形状 `(4, 4)`，配合 `nbrs_mask`

每个 UAV 最多观察 `4` 个邻近 UAV，按距离排序截断。

单个邻机特征定义：
1. `rel_pos_x / map_size`
2. `rel_pos_y / map_size`
3. `rel_vel_x / v_max`
4. `rel_vel_y / v_max`

## 4. Actor 骨干结构

### 4.1 当前启用的是 `set_pool` 编码器

当前三阶段都没有使用旧的 `flat_mlp`，实际启用的是 `set_pool`。

其核心思路是：
- 对标量/小向量输入单独编码
- 对可变长集合输入先逐元素编码
- 再通过带 mask 的 mean pooling + max pooling 聚合
- 最后把所有分支特征拼接后送入融合 MLP

### 4.2 各子编码器

当前参数：
- `actor_set_embed_dim = 64`
- `actor_hidden = 256`
- `input_norm_enabled = true`

`own_encoder`：
```text
LayerNorm(7)
Linear(7 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

`danger_nbr_encoder`：
```text
LayerNorm(5)
Linear(5 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

`users_encoder`：
```text
LayerNorm(5)
Linear(5 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```
之后用 `users_mask` 做 masked mean pooling + masked max pooling，得到 `128` 维用户集合特征。

`sats_encoder`：
```text
LayerNorm(12)
Linear(12 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```
之后用 `sats_mask` 做 masked mean pooling + masked max pooling，得到 `128` 维卫星集合特征。

`nbrs_encoder`：
```text
LayerNorm(4)
Linear(4 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```
之后用 `nbrs_mask` 做 masked mean pooling + masked max pooling，得到 `128` 维邻机集合特征。

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

`accel` 头在三个阶段都存在：

```text
mu_head: Linear(256 -> 2)
log_std: trainable parameter, shape=(2,)
```

动作分布：
- 对每一维使用高斯分布 `Normal(mu, std)`
- `log_std` 会 clamp 到 `[-5, 2]`
- 训练时使用 `rsample()`
- 最终经过 `tanh` squash 到 `[-1, 1]`

环境执行前再乘上 `a_max`，并可能叠加安全修正。

### 5.2 `bw`

当前 `bw` 头不再直接从 pooled summary 输出 `20` 个 logit，而是改为“共享上下文 + 逐用户打分”：

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

当前 `sat` 头改为“共享上下文 + 逐卫星打分”：

```text
sat_emb[m] = SatActionEncoder(sat_m)
logit[m]   = SatScorer([ctx, sat_emb[m], raw_sat_m])
```

对应实现：
- `sat_action_encoder`
- `sat_scorer`
- `MaskedSequentialCategorical`

当前正式配置：
- `sat_num_select = 2`
- 顺序无放回采样
- 输出 `sat_indices` 与 `sat_select_mask`

## 6. Hybrid distribution

当前策略分布由三部分组成：
- `accel_dist`：squashed Gaussian
- `bw_dist`：`MaskedDirichlet`
- `sat_dist`：`MaskedSequentialCategorical`

统一封装在 `HybridActionDist` 中，提供：
- `sample()`
- `mode()`
- `log_prob()`
- `entropy()`

总 `log_prob` 与总 `entropy` 为三头求和，但训练日志会分头记录：
- `entropy_accel`
- `entropy_bw`
- `entropy_sat`

## 7. 动作向量拼接与环境语义

当前 actor 内部动作顺序固定为：
1. `accel(2)`
2. `bw(20)`，若启用
3. `sat(6)`，若启用

因此三阶段动作维度分别是：
- Stage 1：`2`
- Stage 2：`2 + 20 = 22`
- Stage 3：`2 + 20 + 6 = 28`

当前环境语义是：
- `bw` 段表示最终 `bw_alloc`
- `sat` 段表示最终 `sat_select_mask`

环境只做安全过滤与 fallback，不再替代策略做主决策。

## 8. Critic 输入定义

critic 使用 `SaginParallelEnv.get_global_state()` 输出的一维全局状态。

当前 global state 的拼接顺序仍然是：

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

当前三个正式配置虽然 `num_sat` 不同：
- Stage 1 / 2：`num_sat = 72`
- Stage 3：`num_sat = 144`

但都设置了：
- `sat_state_max = 9`

因此三个正式配置的 `state_dim` 仍保持一致：

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

## 9. Critic 结构

critic 当前保持全局状态 MLP，不参与这次动作分布重构：

```text
LayerNorm(142)
Linear(142 -> 256)
ReLU
Linear(256 -> 256)
ReLU
Linear(256 -> 1)
```

输出标量状态价值 `V(s)`。

## 10. 三阶段各自训练什么

### 10.1 Stage 1

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

### 10.2 Stage 2

配置特征：
- `enable_bw_action: true`
- `fixed_satellite_strategy: true`
- `train_accel: false`
- `train_bw: true`
- `train_sat: false`
- `train_shared_backbone: false`
- `exec_accel_source: policy`
- `exec_bw_source: policy`

含义：
- 继承 Stage 1 的 backbone 与 accel 头
- 冻结 backbone 与 accel
- 只训练 bw 头
- 执行时 accel 与 bw 都来自当前策略

### 10.3 Stage 3

配置特征：
- `enable_bw_action: true`
- `fixed_satellite_strategy: false`
- `train_accel: true`
- `train_bw: true`
- `train_sat: true`
- `train_shared_backbone: false`

含义：
- 继承 Stage 2 的 backbone、accel、bw
- 冻结 backbone
- 联合训练 accel、bw、sat 三个动作头
- 执行时三个动作都来自当前策略

## 11. PPO 重算与执行一致性

当前版本专门修复了“策略优化的动作”和“环境真实执行的动作”不一致的问题：
- `action_assembler.py` 现在只做打包
- `buffer.py` 新增 `sat_indices`
- `mappo.py` 在 update 时用 `env_action + sat_indices` 重算 hybrid `log_prob`
- `sagin_env.py` 缓存 step 前的有效用户/卫星槽位，并按同一时刻的数据执行动作

这也是本轮结构调整最核心的变化。

## 12. 文档范围说明

本文档只覆盖当前三个正式配置实际启用的模型结构。

如果后续：

- 切回 `flat_mlp`
- 修改 `users_obs_max` / `sats_obs_max` / `nbrs_obs_max`
- 关闭 `danger_nbr_enabled`
- 关闭 `append_action_masks_to_obs`

那么本文中的维度数字需要同步更新。
