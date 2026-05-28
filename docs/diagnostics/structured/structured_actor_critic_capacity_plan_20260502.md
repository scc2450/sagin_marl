# Structured Actor/Critic 容量排查规划

日期：2026-05-02

## 1. 背景

这次排查发现一个明确配置失效：`config.py` 和当前 YAML 里设置了
`actor_set_embed_dim: 128`，但 `train_structured.py` 之前把 CLI 默认
`--embed_dim 64` 传进 `build_structured_modules_from_config()`，导致 structured
actor 实际按 64 维构建。

修复后：

- `actor_hidden` 默认从 `cfg.actor_hidden` 读取。
- `actor_set_embed_dim` 默认从 `cfg.actor_set_embed_dim` 读取。
- `train/eval/render` 入口的 `--hidden_dim/--embed_dim` 默认改成 `None`，避免 CLI 默认值覆盖配置。
- native actor embed 上限从 128 扩到 256。

短训显示 `actor_embed=128` 相比过去实际的 64 会明显改变训练动态。因此，容量问题不能简单排除；但也不能只说“加宽/加层”，需要把 actor、critic、宽度、交互深度、MLP 深度分开。

## 2. 当前实际规模

当前 3UAV/20GU 配置下，修复后 `actor_set_embed_dim=128`：

```text
actor total  = 2.61M
  accel      = 1.05M
  sat        = 0.71M
  bw         = 0.85M

critic total = 4.06M
  relational_blocks = 3.15M
```

输入维度：

```text
accel actor:
  ego 28, cell 18, GU token 27, peer token 28, SAT token 32

sat actor:
  ego 13, demand 8, role 1, SAT token 26

bw actor:
  ego 11, SAT token 9, GU token 13

critic:
  UAV node 19, GU node 15, SAT node 18
  UAV-GU edge 9, UAV-SAT edge 18, UAV-UAV edge 11
  global scalar 27
```

所以当前模型不是“小到明显不够”的状态。容量可能影响训练动力学，但不能只用“参数不够”解释训练失败。

## 3. 容量维度分类

### 3.1 已经可调的宽度

当前代码已经支持这些配置：

```yaml
actor_hidden: 256
actor_set_embed_dim: 128
accel_hidden: 0
accel_embed_dim: 0
sat_hidden: 0
sat_embed_dim: 0
bw_hidden: 0
bw_embed_dim: 0

critic_hidden: 256
critic_embed_dim: 128
critic_edge_embed_dim: 128
critic_global_embed_dim: 128
critic_system_token_dim: 128
critic_value_head_hidden: 256
```

注意：

- `actor_hidden` / `actor_set_embed_dim` 是三个 actor 的共享默认宽度。
- `accel_hidden` / `accel_embed_dim`、`sat_hidden` / `sat_embed_dim`、`bw_hidden` / `bw_embed_dim` 可以分别覆盖三个 actor；值为 `0` 时继承共享默认值。
- `critic_embed_dim`、`critic_edge_embed_dim`、`critic_global_embed_dim`、`critic_system_token_dim` 当前实现要求一致，应一起改。
- native actor 现在支持 `actor_set_embed_dim <= 256`。

### 3.2 已经可调的交互层数

当前已有配置：

```yaml
sat_competition_layers: 2
bw_competition_layers: 2
critic_message_layers: 2
```

含义：

- `sat_competition_layers`：SAT token self-attention block 层数。
- `bw_competition_layers`：BW GU token competition/self-attention block 层数。
- `critic_message_layers`：critic typed relational message passing block 层数。

这些是“token/graph 交互深度”，不是普通 MLP 深度。

### 3.3 代码固定为两层、但应该纳入容量排查的 MLP 深度

现在大量 MLP 固定为：

```text
Linear -> activation -> Linear
```

包括：

- accel actor 的 ego/cell/GU/peer/SAT encoder。
- accel actor 的 fusion MLP。
- sat actor 的 ego/demand/role/SAT encoder。
- sat actor 的 context fusion / item head / count head。
- bw actor 的 ego/SAT/GU encoder。
- bw actor 的 context fusion / score head / tau head / kappa head。
- critic 的 node/edge/global encoder。
- critic relational block 内部的 message/update MLP。
- critic value heads。

这些目前没有配置项，但不代表“不需要调”。如果要完整排查容量，应该新增类似：

```yaml
actor_encoder_mlp_layers
actor_fusion_mlp_layers
actor_head_mlp_layers
critic_encoder_mlp_layers
critic_message_mlp_layers
critic_value_head_layers
```

注意：actor 的 MLP 深度如果改，native CUDA actor 也必须同步改。目前 CUDA kernel 里的 `block_mlp2` / `block_mlp2_items` 假设两层 MLP。

### 3.4 accel actor 的结构深度缺口

accel actor 当前不是 sat/bw 那种 token self-attention 堆叠结构。它大致是：

```text
token encoder
ego/cell -> query
query attention over GU / peer UAV / SAT
mean/max pooling
fusion MLP
mu_head
```

它没有显式的多轮 token-to-token interaction block。因此，“给 accel 加层数”不能只理解成加 MLP 层。更合理的新增结构是：

```yaml
accel_interaction_layers
accel_attention_heads
accel_gu_query_count
accel_peer_query_count
accel_sat_query_count
```

也就是让 ego/GU/peer/SAT token 在输出动作前进行若干轮交互，并允许分别调 GU/peer/SAT 三类 attention pooling 的 query 数。

## 4. 实验矩阵规划

不要把 actor、critic、宽度、层数一次混在一起。建议按单阶段训练分别做。

### 4.1 Accel-only

基础配置：

```yaml
actor_hidden: 256
actor_set_embed_dim: 128
critic_hidden: 256
critic_embed_dim: 128
critic_edge_embed_dim: 128
critic_global_embed_dim: 128
critic_system_token_dim: 128
critic_message_layers: 2
```

单变量：

```text
A0 baseline: actor_hidden=256, actor_embed=128, critic_hidden=256, critic_embed=128, critic_msg=2
A1 actor hidden: actor_hidden=512, actor_embed=128, critic 不变
A2 actor embed:  actor_hidden=256, actor_embed=256, critic 不变
A3 critic hidden: critic_hidden=512, critic_embed=128, actor 不变
A4 critic embed:  critic_hidden=256, critic_embed=edge/global/system=256, actor 不变
A5 critic message depth: critic_message_layers=3, 其他不变
```

后续新增结构后再测：

```text
A6 accel interaction depth: accel_interaction_layers=1/2
```

### 4.2 Sat-only

```text
S0 baseline
S1 actor_hidden=512
S2 actor_embed=256
S3 sat_competition_layers=3
S4 critic_hidden=512
S5 critic_embed=edge/global/system=256
S6 critic_message_layers=3
```

### 4.3 BW-only

```text
B0 baseline
B1 actor_hidden=512
B2 actor_embed=256
B3 bw_competition_layers=3
B4 critic_hidden=512
B5 critic_embed=edge/global/system=256
B6 critic_message_layers=3
```

### 4.4 Joint

Joint 不应该先做全组合。只有单阶段发现某类容量改动有稳定收益后，才组合到 joint：

```text
J0 baseline
J1 actor 侧最佳单变量
J2 critic 侧最佳单变量
J3 actor + critic 最佳组合
```

### 4.5 Critic-only 第一轮小矩阵

这一轮只检查 critic 容量，不同时改 actor、reward、训练口径或环境参数。先用 accel-only 短训做快速判断：只训练 accel，sat/bw 用 queue-aware 执行，critic 仍按三阶段 head 正常训练。

基础口径固定为：

```yaml
critic_value_mode: relational
structured_critic_input_norm_enabled: false
critic_popart_enabled: false
critic_loss_target_standardize: false
critic_loss_running_standardize: false
critic_hidden: 256
critic_embed_dim: 128
critic_edge_embed_dim: 128
critic_global_embed_dim: 128
critic_system_token_dim: 128
critic_message_layers: 2
critic_encoder_mlp_layers: 2
critic_message_mlp_layers: 2
critic_value_head_hidden: 256
critic_value_head_layers: 2
```

先跑下面这些单变量，不跑 `C1`，也不先跑 `C9-C15` 组合项：

| ID | 目的 | 改动 |
| --- | --- | --- |
| C0 | baseline | 不改 critic 容量 |
| C2 | token/system 表示加宽 | `critic_embed_dim=256`, `critic_edge_embed_dim=256`, `critic_global_embed_dim=256`, `critic_system_token_dim=256` |
| C3 | critic 内部 MLP 加宽 | `critic_hidden=512` |
| C4 | value readout 加宽 | `critic_value_head_hidden=512` |
| C5 | node/edge/global encoder 加深 | `critic_encoder_mlp_layers=3` |
| C6 | relational block 内部 message MLP 加深 | `critic_message_mlp_layers=3` |
| C7 | typed relational message passing 加深 | `critic_message_layers=3` |
| C8 | value readout 加深 | `critic_value_head_layers=3` |

每个实验先跑 10 updates。主要看：

- `explained_variance_accel` 是否比 C0 更早、更稳定上升。
- `value_loss_accel` 是否同步下降，而不是 EV 偶然跳动。
- value 输出的 mean/std 是否接近 return target 的尺度。
- `approx_kl_accel`、`clip_frac_accel` 是否恢复到 actor 实际有更新。
- reward 和 `episode_length_mean` 是否至少不明显变坏。

如果 10 updates 看不出区别，只把有希望的项延长到 30 updates；不要把所有组合都延长。

#### 4.5.1 本轮 10-update 结果

运行口径：

```text
base config: configs/tmp/structured_accel_sanity_3uav_20gu_t250_currentenv_ppo_positive_nowarmup.yaml
updates: 10
rollout_env_steps: 250
num_envs: 8
device: cuda
train: accel only
exec sat/bw: queue_aware
```

实际构建出的 critic 维度已经核对：

| ID | embed | edge | global | system | hidden | message layers | encoder MLP | message MLP | value head hidden | value head layers | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 128 | 128 | 128 | 128 | 256 | 2 | 2 | 2 | 256 | 2 | 4.06M |
| C2 | 256 | 256 | 256 | 256 | 256 | 2 | 2 | 2 | 256 | 2 | 7.87M |
| C3 | 128 | 128 | 128 | 128 | 512 | 2 | 2 | 2 | 256 | 2 | 8.41M |
| C4 | 128 | 128 | 128 | 128 | 256 | 2 | 2 | 2 | 512 | 2 | 4.16M |
| C5 | 128 | 128 | 128 | 128 | 256 | 2 | 3 | 2 | 256 | 2 | 4.85M |
| C6 | 128 | 128 | 128 | 128 | 256 | 2 | 2 | 3 | 256 | 2 | 5.77M |
| C7 | 128 | 128 | 128 | 128 | 256 | 3 | 2 | 2 | 256 | 2 | 5.64M |
| C8 | 128 | 128 | 128 | 128 | 256 | 2 | 2 | 2 | 256 | 3 | 4.26M |

10 updates 结果：

| ID | last reward | last5 reward | last ep_len | last5 ep_len | last EV_accel | last5 EV_accel | last value_loss_accel | last KL | last clip | last danger active | avg env/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 baseline | 0.1344 | 0.1352 | 220.6 | 221.4 | 0.475 | 0.424 | 4.383 | 0.00212 | 0.0032 | 0.0180 | 157.9 |
| C2 embed256 | 0.1389 | 0.1386 | 226.0 | 223.8 | 0.421 | 0.288 | 4.861 | -0.00018 | 0.0007 | 0.0060 | 77.7 |
| C3 hidden512 | 0.1358 | 0.1358 | 218.7 | 224.5 | 0.407 | 0.255 | 5.085 | 0.00448 | 0.0385 | 0.0118 | 97.9 |
| C4 valuehead512 | 0.1349 | 0.1360 | 236.9 | 236.9 | 0.655 | 0.582 | 5.548 | 0.00522 | 0.0375 | 0.0140 | 166.4 |
| C5 encoder3 | 0.1359 | 0.1366 | 216.3 | 218.6 | 0.312 | 0.199 | 5.577 | 0.00060 | 0.0085 | 0.0160 | 154.7 |
| C6 message_mlp3 | 0.1365 | 0.1372 | 221.3 | 221.8 | 0.343 | 0.247 | 5.835 | 0.00052 | 0.0148 | 0.0083 | 34.6 |
| C7 message_layers3 | 0.1360 | 0.1392 | 217.6 | 226.2 | 0.511 | 0.308 | 3.149 | 0.00317 | 0.0125 | 0.0112 | 140.2 |
| C8 value_head_layers3 | 0.1342 | 0.1357 | 214.0 | 216.3 | 0.334 | 0.283 | 5.683 | 0.00232 | 0.0167 | 0.0113 | 164.9 |

短结论：

- C4 是第一轮最值得延长的项：几乎不降速，`last5 EV_accel=0.582` 明显高于 C0 的 `0.424`，episode length 也更长。
- C7 可以作为第二候选：`value_loss_accel` 最低，但 EV 和速度收益不如 C4 稳。
- C2/C3 单纯加大 token/hidden 参数量没有带来同等收益，而且速度明显下降。
- C6 代价过高，第一轮不继续。
- C5/C8 没有明显收益，第一轮不继续。

#### 4.5.2 C4+C7 组合结果

追加测试：

```yaml
critic_value_head_hidden: 512
critic_message_layers: 3
```

实际构建结果：

```text
embed=128, edge=128, global=128, system=128
hidden=256
message_layers=3
encoder_mlp_layers=2
message_mlp_layers=2
value_head_hidden=512
value_head_layers=2
params=5.736M
```

和 C0/C4/C7 对比：

| ID | last reward | last5 reward | last ep_len | last5 ep_len | last EV_accel | last5 EV_accel | last value_loss_accel | last KL | last clip | danger active | avg env/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 baseline | 0.1344 | 0.1352 | 220.6 | 221.4 | 0.475 | 0.424 | 4.383 | 0.00212 | 0.0032 | 0.0180 | 157.9 |
| C4 valuehead512 | 0.1349 | 0.1360 | 236.9 | 236.9 | 0.655 | 0.582 | 5.548 | 0.00522 | 0.0375 | 0.0140 | 166.4 |
| C7 message_layers3 | 0.1360 | 0.1392 | 217.6 | 226.2 | 0.511 | 0.308 | 3.149 | 0.00317 | 0.0125 | 0.0112 | 140.2 |
| C4+C7 | 0.1369 | 0.1380 | 224.3 | 230.0 | 0.363 | 0.379 | 6.229 | 0.00333 | 0.0286 | 0.0115 | 135.4 |

结论：C4+C7 没有叠加收益。它比 C4 单独 EV 更低、value loss 更高、速度更慢。因此下一步如果延长，只延长 C4，不优先延长 C4+C7。

#### 4.5.3 C4b value head 1024

追加测试：

```yaml
critic_value_head_hidden: 1024
```

实际构建结果：

```text
embed=128, edge=128, global=128, system=128
hidden=256
message_layers=2
encoder_mlp_layers=2
message_mlp_layers=2
value_head_hidden=1024
value_head_layers=2
params=4.359M
```

和 C0/C4 对比：

| ID | last reward | last5 reward | last ep_len | last5 ep_len | last EV_accel | last5 EV_accel | last value_loss_accel | last KL | last clip | danger active | avg env/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 baseline | 0.1344 | 0.1352 | 220.6 | 221.4 | 0.475 | 0.424 | 4.383 | 0.00212 | 0.0032 | 0.0180 | 157.9 |
| C4 valuehead512 | 0.1349 | 0.1360 | 236.9 | 236.9 | 0.655 | 0.582 | 5.548 | 0.00522 | 0.0375 | 0.0140 | 166.4 |
| C4b valuehead1024 | 0.1323 | 0.1361 | 226.1 | 224.9 | 0.482 | 0.386 | 4.827 | 0.00089 | 0.0101 | 0.0213 | 160.5 |

结论：1024 可以运行，但不如 512。当前现象更像 value readout hidden=512 是甜点，继续加宽会让 actor 更新幅度和 EV 回落。

### 4.6 Accel Actor-only 第一轮小矩阵

这一轮固定 critic 使用 C4：

```yaml
critic_value_mode: relational
critic_value_head_hidden: 512
structured_critic_input_norm_enabled: false
critic_popart_enabled: false
critic_loss_target_standardize: false
critic_loss_running_standardize: false
```

训练口径固定：

```yaml
reward_mode: positive_weighted_workload_level
avoidance_enabled: true
danger_imitation_enabled: true
danger_imitation_coef: 0.1
train_accel: true
train_sat: false
train_bw: false
exec_sat_source: queue_aware
exec_bw_source: queue_aware
```

只做下面这些单变量，序号重新编排：

| ID | 改动 | 目的 |
| --- | --- | --- |
| A0 | baseline | C4 critic + 当前 accel actor |
| A1 | `accel_interaction_layers=1` | 检查显式 token-token 交互是否有用 |
| A2 | `accel_head_mlp_layers=2` | 检查 action mean readout 是否太弱 |
| A3 | `accel_embed_dim=256` | 检查 accel token 表示维度是否不足 |
| A4 | `accel_hidden=512` | 检查 accel 局部 MLP 宽度是否不足 |
| A5 | `accel_encoder_mlp_layers=3` | 检查 raw feature 到 token 的编码是否太浅 |
| A6 | `accel_context_mlp_layers=3` | 检查 fusion/context 映射是否太浅 |
| A7 | `accel_gu_query_count=6` | 检查 GU attention pooling 容量是否不足 |
| A8 | `accel_peer_query_count=3` | 检查 peer UAV attention pooling 容量是否不足 |
| A9 | `accel_sat_query_count=4` | 检查 SAT attention pooling 容量是否不足 |
| A10 | `accel_interaction_layers=1`, `accel_attention_heads=8` | 条件项：只有 A1 比 A0 有明显收益时才测 |

每个实验先跑 10 updates。主要看：

- `env_reward_mean` / `episode_reward` 是否比 A0 上升。
- `episode_length_mean` 是否下降；下降通常说明碰撞或提前 reset 变多。
- `approx_kl_accel` / `clip_frac_accel` 是否恢复到 actor 实际有更新。
- `entropy_accel` / `grad_norm_accel` 是否异常塌缩或爆。
- `danger_imitation_active_rate` 是否变化太大，避免把安全样本触发差异误判成结构收益。

判断规则：

- 如果 A1 不优于 A0，就不跑 A10。
- 如果 A3/A4 有效，说明 accel actor 更可能受宽度限制。
- 如果 A5/A6 有效，说明问题更可能在 MLP 局部映射深度。
- 如果 A7/A8/A9 有效，说明对应 token set 的 pooling/query 容量是瓶颈。

#### 4.6.1 本轮 10-update 结果

实际构建结果已经核对：

| ID | hidden | embed | encoder | context | head | interaction | heads | q_gu | q_peer | q_sat | critic value head | actor params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 256 | 128 | 2 | 2 | 1 | 0 | 4 | 4 | 2 | 2 | 512 | 2.612M |
| A1 | 256 | 128 | 2 | 2 | 1 | 1 | 4 | 4 | 2 | 2 | 512 | 2.744M |
| A2 | 256 | 128 | 2 | 2 | 2 | 0 | 4 | 4 | 2 | 2 | 512 | 2.678M |
| A3 | 256 | 256 | 2 | 2 | 1 | 0 | 4 | 4 | 2 | 2 | 512 | 4.088M |
| A4 | 512 | 128 | 2 | 2 | 1 | 0 | 4 | 4 | 2 | 2 | 512 | 3.533M |
| A5 | 256 | 128 | 3 | 2 | 1 | 0 | 4 | 4 | 2 | 2 | 512 | 2.941M |
| A6 | 256 | 128 | 2 | 3 | 1 | 0 | 4 | 4 | 2 | 2 | 512 | 2.678M |
| A7 | 256 | 128 | 2 | 2 | 1 | 0 | 4 | 6 | 2 | 2 | 512 | 2.743M |
| A8 | 256 | 128 | 2 | 2 | 1 | 0 | 4 | 4 | 3 | 2 | 512 | 2.678M |
| A9 | 256 | 128 | 2 | 2 | 1 | 0 | 4 | 4 | 2 | 4 | 512 | 2.743M |
| A10 | 256 | 128 | 2 | 2 | 1 | 1 | 8 | 4 | 2 | 2 | 512 | 2.744M |

10 updates 结果：

| ID | last reward | last5 reward | last ep_len | last5 ep_len | last EV_accel | last5 EV_accel | last KL | last clip | danger active | avg env/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 baseline | 0.1349 | 0.1360 | 236.9 | 236.9 | 0.655 | 0.582 | 0.00522 | 0.0375 | 0.0140 | 158.7 |
| A1 interaction1 | 0.1348 | 0.1373 | 229.6 | 229.3 | 0.492 | 0.334 | 0.00687 | 0.0464 | 0.0108 | 137.6 |
| A2 head2 | 0.1385 | 0.1382 | 221.3 | 220.2 | 0.486 | 0.308 | 0.00113 | 0.0084 | 0.0077 | 163.2 |
| A3 embed256 | 0.1351 | 0.1378 | 222.6 | 220.6 | 0.376 | 0.335 | 0.00018 | 0.0026 | 0.0090 | 147.0 |
| A4 hidden512 | 0.1375 | 0.1374 | 219.3 | 215.1 | 0.339 | 0.285 | 0.00943 | 0.1157 | 0.0033 | 155.1 |
| A5 encoder3 | 0.1316 | 0.1349 | 208.2 | 207.1 | 0.222 | 0.167 | 0.00316 | 0.0335 | 0.0130 | 154.7 |
| A6 context3 | 0.1385 | 0.1382 | 221.3 | 220.2 | 0.486 | 0.308 | 0.00113 | 0.0084 | 0.0077 | 164.5 |
| A7 gu_query6 | 0.1364 | 0.1346 | 208.5 | 217.1 | 0.297 | 0.201 | 0.00688 | 0.0717 | 0.0302 | 160.6 |
| A8 peer_query3 | 0.1357 | 0.1370 | 201.6 | 207.8 | 0.281 | 0.246 | 0.00064 | 0.0056 | 0.0118 | 162.5 |
| A9 sat_query4 | 0.1333 | 0.1367 | 215.5 | 211.6 | 0.486 | 0.297 | 0.00160 | 0.0039 | 0.0103 | 162.4 |

A10 未运行。原因：A1 相比 A0 没有明显收益。A1 的 last5 reward 略高，但 last5 episode length、EV 和速度都明显低于 A0，因此不满足“interaction 有收益再测 heads=8”的条件。

短结论：

- 这轮没有一个 accel actor 容量改动同时改善 reward、episode length 和 EV。
- A2/A6 的 last5 reward 最高，但 episode length 和 EV 都比 A0 差，不适合作为明确正收益。
- A0 仍然是最稳的短训点：episode length 和 EV 最高。
- A5/A7/A8 对 episode length 伤害较明显，第一轮不继续。
- A1 未触发 A10。

## 5. 如何解释结果

### 5.1 actor_hidden 有效

说明局部 MLP 非线性容量有帮助。可能是 encoder/fusion/head 中某些局部映射不够。

### 5.2 actor_embed 有效

说明 token/context 表示维度是重要瓶颈。它不只影响最终表达能力，也会影响优化动力学、attention/query 有效秩、logprob 曲面和梯度传播。

### 5.3 sat/bw competition layers 有效

说明对应 stage 的 item-item 比较或竞争关系需要更深的 token 交互。

### 5.4 critic_hidden 有效

说明 critic 内部 MLP 的局部非线性容量不足。

### 5.5 critic_embed 有效

说明 critic 的 token/system 表示维度不足。

### 5.6 critic_message_layers 有效

说明问题主要在多跳关系传播，而不是单个 token 表示或 value head。

### 5.7 宽度和层数都无效

如果这些单变量都无效，不应该继续堆网络。应回到：

- actor/critic 输入是否仍有语义错位；
- reward/return/advantage 口径是否一致；
- native rollout 与 PyTorch update 是否一致；
- 当前 PPO 采样动作的 credit 是否被轨迹难度淹没；
- stage gating / critic head 训练口径是否合理。

## 6. 实现注意事项

### 6.1 native actor embed=256

已放宽：

```text
NATIVE_ACTOR_CUDA_MAX_EMBED = 256
kMaxEmbed = 256
```

scratch width 已按 `cfg.actor_set_embed_dim` 计算，理论上会随 256 放大。

### 6.2 actor MLP depth

如果新增 `actor_encoder_mlp_layers` 等配置，必须同时改：

- Python actor module。
- native actor weight ABI。
- CUDA actor kernel 的 MLP 执行逻辑。
- native weight packing / sync。

不能只改 Python，否则 rollout 和 update 会走不同 actor。

### 6.3 critic MLP depth

critic 只在 PyTorch update 中用，改 MLP depth 不涉及 native actor kernel。但如果 world/buffer 构造仍在 native 路径，输入 schema 必须保持不变。

### 6.4 checkpoint 兼容

宽度或层数改变会导致 checkpoint shape 不兼容。容量实验应从 scratch 开始，或明确写 adapter/init 规则，不要隐式 load 旧 checkpoint。

### 6.5 2026-05-03 native CUDA 同步状态

本轮已经把层数配置从“actor 共用一组”拆成“三个 actor 可分别调”：

```yaml
accel_encoder_mlp_layers
accel_context_mlp_layers
accel_head_mlp_layers
accel_interaction_layers
accel_attention_heads

sat_encoder_mlp_layers
sat_context_mlp_layers
sat_head_mlp_layers
sat_competition_layers
sat_attention_heads

bw_encoder_mlp_layers
bw_context_mlp_layers
bw_head_mlp_layers
bw_competition_layers
bw_attention_heads

critic_encoder_mlp_layers
critic_message_mlp_layers
critic_message_layers
critic_value_head_layers
```

本轮也把 actor 宽度从“全部共享”拆成“三个 actor 可分别调”：

```yaml
accel_hidden
accel_embed_dim
sat_hidden
sat_embed_dim
bw_hidden
bw_embed_dim
```

这些字段为 `0` 时继承 `actor_hidden / actor_set_embed_dim`。因此旧 YAML 不写这些字段时语义不变。

accel pooling query 数也已经可调：

```yaml
accel_gu_query_count
accel_peer_query_count
accel_sat_query_count
```

这些字段为 `0` 时继承旧 schema 默认 `4/2/2`。

`actor_encoder_mlp_layers / actor_context_mlp_layers / actor_head_mlp_layers` 仍保留为默认值；如果某个 stage 专属字段为 `0` 或未写，就回退到 actor 共享默认值。

native CUDA 已同步：

- `actor_kernels.cu` 的 `kMaxEmbed=256`、`kMaxHidden=512`。
- native actor weight ABI 支持 actor encoder/context/head MLP 深度 `2..4`。
- native actor int params 追加了 accel/sat/bw 各自的 encoder/context/head layer count。
- native actor int params 追加了 accel/sat/bw 各自的 hidden/embed width，以及 accel 三类 query count。
- `accel_interaction_layers` 使用 masked self-attention + FFN + LayerNorm block，CUDA 中对 GU/peer/SAT 三类 token set 分别执行同一组 interaction block 权重。
- `sat_attention_heads`、`bw_attention_heads` 已经在 factory 和 CUDA kernel 中生效。

Accel `mu_head` 也已经同步：

- `accel_head_mlp_layers=1` 时，native CUDA 使用旧的线性 `mu_head.weight/bias`。
- `accel_head_mlp_layers=2..4` 时，native CUDA 使用独立连续的 `mu_head` MLP 权重段：`.0/.2/.4/.6`。
- 深层 `mu_head` 不混用通用 extra MLP 槽，也不依赖 enum 相邻假设。

已验证：

```text
accel_hidden=384, accel_embed_dim=192, accel_gu/peer/sat_query_count=6/3/4
sat_hidden=256, sat_embed_dim=128
bw_hidden=320, bw_embed_dim=160
```

上述配置可以完成三阶段 policy native rollout smoke。

当前限制：

- actor MLP 深度超过 `4` 暂不支持 native CUDA parity。
