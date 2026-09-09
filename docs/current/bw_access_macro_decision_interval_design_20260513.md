# BW / Access Macro Decision Interval 设计

日期：2026-05-13

## 1. 目标

当前 BW/access 每个 primitive step 都重新决策：

```text
每 step:
  按 GU-UAV 位置关系重新计算接入
  根据接入结果构造 BW valid GU
  BW actor 采样一次带宽比例
  环境推进 1 step
```

这会让一次 BW 决策只影响一拍，BW 动作带来的回报差异很容易被后续随机 policy、链路随机性、队列历史和状态难度淹没。

本方案改成：

```text
每 K 个 primitive step 才重新确定一次接入和 BW 比例。
这 K 步期间：
  GU-UAV association 固定
  BW valid set 固定
  BW action 固定
  但环境物理状态、信道、队列、service、drop、reward 仍然每 step 真实更新
```

第一版建议：

```yaml
access_bw_decision_interval: 5
```

## 2. 核心修正

本设计不采用旧的 semi-MDP macro reward 口径。

不要这样做：

```text
R_macro(t) = r_t + gamma r_{t+1} + ... + gamma^(d-1) r_{t+d-1}
然后把它当成一条 macro transition reward
再用 gamma^d / (gamma lambda)^d 做 GAE
```

原因是：这会把 reward/return 口径从原来的 primitive-step 链改成 macro 链。遇到中途 reset、rollout tail、episode 重新开始后的不满 K 尾段时，很容易把 credit 算偏。

正确口径是：

```text
K 只影响“哪些 BW rows 是真正决策点”。
K 不改变 reward/return 的数学口径。
```

也就是：

```text
reward/return 仍然按 primitive step 一步一步算。
BW actor/critic 只在 macro-start rows 上取对应 target/advantage。
continuation rows 不产生新的 BW actor loss。
```

## 3. 执行语义

### 3.1 Primitive Step

primitive step 仍是环境最小时间步，时间长度仍是 `tau0`。

每个 primitive step 都要真实更新：

```text
UAV 位置/速度
信道与干扰
GU/UAV/SAT 队列
service / drop
reward
terminal / truncated
```

### 3.2 BW Macro Start

每个 env 独立维护 BW macro 状态。

在以下情况开启新的 BW macro decision：

```text
1. episode reset 后的第一个 primitive step
2. 当前没有有效 macro action
3. 上一个 macro 已经持有 K 个 primitive steps
```

在 macro start：

```text
按当前位置关系计算 GU-UAV association
根据 association 得到 BW valid mask
构造 BW local state
调用 BW actor 采样 action
保存 action / old_logprob / ref_action / tau / kappa / valid_count 等训练字段
```

### 3.3 Macro Continuation

在 macro continuation step：

```text
不重新计算 GU-UAV association
不重新构造新的 BW valid set
不重新采样 BW action
不产生新的 BW old_logprob
```

直接复用 macro start 保存的：

```text
association
bw_valid mask
BW action
BW diagnostic fields
```

注意：复用的是“归属关系”和“带宽比例”，不是冻结链路质量。access rate 仍然要用当前 primitive step 的距离、信道、干扰重新计算。

### 3.4 Episode Reset

如果某个 primitive step 发生 `terminated/truncated`，该 step 的 reward 仍属于当前 macro-start action 的后果。

reset 后的新 episode 第一个 primitive step 必须开启新的 macro decision，不能跨 episode 复用上一条 macro action。

## 4. Return / Advantage 语义

### 4.1 Primitive Return Chain

回报仍按 primitive step 递推：

```text
G_t = r_t + gamma * G_{t+1}
```

terminal 时自然断开；time-limit / rollout tail 是否 bootstrap，沿用当前 PPO/MC return 的既有规则。

### 4.2 BW 只取 Macro Start

假设 `K=5`：

```text
t=0   BW macro start，采样 action a0
t=1   continuation，继续用 a0
t=2   continuation，继续用 a0
t=3   continuation，继续用 a0
t=4   continuation，继续用 a0
t=5   下一次 BW macro start，采样 action a5
```

reward 每步都有：

```text
r0, r1, r2, r3, r4, r5, ...
```

return 也是 primitive step 口径：

```text
G0 = r0 + gamma r1 + gamma^2 r2 + ...
G1 = r1 + gamma r2 + ...
...
```

但 BW actor 只用 macro-start row：

```text
state      = BW state at t=0
action     = a0
old_logprob= log pi_old(a0 | state_t0)
target     = G0
advantage  = A0
```

`t=1..4` 的 reward 没丢，它们已经包含在 `G0/A0` 里。只是 `t=1..4` 没有新的 BW decision，所以不应该产生 BW actor loss。

### 4.3 GAE

如果使用 MC target：

```text
先按 primitive chain 算全局 returns[t]
BW stage target = returns[macro_start_t]
```

如果使用 GAE：

```text
先按 primitive chain / 当前 stage 口径算 primitive advantage
BW stage advantage = advantage[macro_start_t]
```

第一版 joint MC-GAE 训练建议继续使用 MC target 训练 critic，然后用训练后的 critic 计算 actor advantage；但无论 target 具体是哪种，BW stage 都只 gather macro-start 对应位置，不能先构造 macro_reward。

## 5. Buffer / Training View

### 5.1 Training Rows

K=1：

```text
每个 primitive step 都是 macro start
BW rows = primitive BW rows
```

K>1：

```text
BW rows = macro-start rows
continuation rows 不进入 BW training stage batch
```

### 5.2 需要保存的字段

BW training row 保存 macro start 的：

```text
world_batch
local_batch
action
old_logprob
old_logprob_per_agent
ref_action / det_mean
tau
kappa
valid_count
latent_count
transition_index
env_index
```

reward/return target 不在这里折段累计，而是从 primitive return 结果按 `transition_index` gather。

### 5.3 Continuation History

runtime/history 中仍可保存 continuation primitive step 的执行结果，用于：

```text
环境推进
primitive reward/return 计算
诊断
episode length / collision / queue 统计
```

但 continuation step 不保存为新的 BW actor training row。

## 6. Chunk / Compile 处理

K>1 后 BW macro-start row 数可能小于 primitive step 数，且会随 episode reset 位置略变。

这不是语义错误。训练 batch 的真实样本数可以变。

问题在于 `torch.compile(fullgraph)` 希望同一个 compiled function 看到固定 shape。处理方式：

```text
每个 actor/critic chunk 使用固定 chunk_size。
最后不足 chunk_size 的 chunk 用最后一条真实 row padding。
valid_i = 1 表示真实 row。
valid_i = 0 表示 padding row。
loss / entropy / KL / clip_frac 只对 valid_i=1 的 row 做 mean。
padding row 不参与梯度和指标。
```

这保持原始 batch mean 语义：

```text
masked_mean(x, valid) = sum(valid * x) / sum(valid)
```

它只解决 fixed-shape compile 问题，不改变 PPO 数学目标。

## 7. 必须通过的检查

### 7.1 K=1 Parity

`K=1` 必须走同一套 macro-start gather 路径，并与原 primitive 行为一致：

```text
BW row count = rollout_steps * num_envs
duration/diagnostic 全为 1
old_logprob parity 通过
训练 target 与原 primitive target 一致
```

### 7.2 Continuation Restore

K>1 continuation step 必须复用 macro start 的：

```text
association
bw_valid mask
BW action
old_logprob / per-agent logprob
ref_action
tau / kappa / valid_count
```

但 access rate 必须用当前 step 的链路重新计算。

### 7.3 Return Gather

对 K>1：

```text
primitive returns 先完整计算
BW targets = primitive_returns[macro_start_transition_indices]
```

不得使用：

```text
discounted_macro_segments(...)
gamma^duration GAE
macro_reward
```

### 7.4 Padding Parity

同一批样本上比较：

```text
未 padding eager loss/grad
padding + valid_i compiled loss/grad
```

二者应一致。

### 7.5 Shape Stability

K=5 训练时：

```text
critic compiled chunk 不应 recompile
actor compiled chunk 不应 recompile
最后一个变短 chunk 应由 padding + valid_i 处理
```

## 8. 当前实现状态

已实现并检查过的部分：

```text
1. native runtime 可以在 continuation step 恢复 macro start 的 BW action/history 字段。
2. K=1/K=3 的 raw history continuation restore 检查通过。
3. fused BW actor partial forward 已实现：K>1 时只对 macro-start rows 做 BW actor forward，continuation rows 从 history 恢复。
4. K=1 的 BW old-logprob parity 通过。
```

需要撤掉或重做的部分：

```text
1. actor compiled loss padding 的 valid_i parity 还需要专门测试。
```

下一步实现顺序：

```text
1. 做 eager/compiled padding parity。
2. 跑 K=1 1-update parity。
3. 跑 K=5 1-update shapecheck。
4. 再跑 K=5 50 updates。
```

当前代码状态：

```text
1. structured_buffer.py 已停用并删除 discounted macro reward helper。
2. return_view.stage_batches[2] 保持 primitive full rows，compute_gae 继续按 primitive chain 计算。
3. training_view.stage_batches[2] 只保留 macro-start BW rows。
4. train_joint_mcgae.py 从 primitive returns 按 macro-start transition_indices gather BW target。
5. _stage_gae_from_mc_targets(...) 在 BW macro K>1 时直接使用 macro-start MC residual：A = MC_return_at_macro_start - V(macro_start)。
6. actor compiled chunk 已接入 padding + valid_i；仍需更严格的 loss/grad parity 测试。
```
