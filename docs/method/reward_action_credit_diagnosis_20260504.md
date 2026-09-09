# Reward 好不好怎么判断

目标不是判断 reward 的绝对值高不高，而是判断它能不能给 actor 提供可学习的动作 credit。

对 PPO 来说，关键不是“某个动作比 ref 好多少”，而是：

```text
当前 policy 自己会采到的动作里，return 差异能不能和 grad log pi(a|s) 形成稳定协方差。
```

## 1. 三层判断

### 第一层：好动作是否存在

固定同一个状态 `s`、同一份未来随机性、同一个 follow policy，只换 first action：

```text
G(s, a) = 从 s 执行 first action a 后，到 episode 结束的 return
```

用 rule / wide random / branch search / policy samples 都可以。

要看：

```text
best_minus_ref = max_a G(s,a) - G(s,a_ref)
pairwise_gap   = 同一状态内不同动作的 return 差异
```

如果这一层都很小，说明 reward 或 action timescale 本身看不见动作影响。

### 第二层：当前 PPO 能不能读出方向

只看当前 policy stochastic samples：

```text
a_ik ~ pi(.|s_i)
G_ik = G(s_i, a_ik)
A_oracle_ik = G_ik - mean_k(G_ik)
```

`A_oracle` 表示：如果 critic 能完美扣掉同一状态难度，当前动作还剩多少 credit。

然后看：

```text
g_oracle = mean[A_oracle * grad log pi(a|s)]
```

核心指标：

```text
oracle_grad_snr = ||mean sample gradient|| / gradient sampling noise
```

判断：

```text
oracle_grad_snr < 1
  当前 PPO 一阶信号大概率被采样噪声淹没，单纯延长训练通常没用。

1 <= oracle_grad_snr < 3
  有信号但偏弱，训练可能慢、波动大，需要更多样本或更好的初始化/探索。

oracle_grad_snr >= 3
  当前 reward + policy 分布下，一阶 PPO 信号比较可用。
```

还要看：

```text
cancellation_ratio = ||mean_i g_i|| / mean_i ||g_i||
```

如果很小，说明每个状态里有信号，但 batch 平均后互相抵消。

### 第三层：真实 PPO advantage 有没有接住信号

第二层用的是 oracle-centered advantage。真实训练用的是 GAE / critic advantage。

要比较：

```text
cosine(g_ppo, g_oracle)
corr(A_ppo, A_oracle)
sign_agreement(A_ppo, A_oracle)
```

结论：

```text
如果 g_oracle 弱：
  reward/action distribution 本身不给 PPO 可用梯度。

如果 g_oracle 强但 g_ppo 不对齐：
  critic / GAE / advantage 口径没有把 reward 里的动作 credit 传给 actor。

如果两者都强但训练不升：
  再查 logprob、optimizer、更新实现、rollout/state 漂移。
```

## 2. 状态和动作怎么采

状态必须来自正常 PPO rollout buffer，不要手挑。

建议分层采：

```text
early / mid / late
low / medium / high workload
剩余 horizon 不能太短
多个 seed
```

动作必须包含当前 policy samples，因为 PPO 第一轮只能从这些动作里学：

```text
主要：K 个 current policy stochastic samples
辅助：deterministic ref、rule action、wide random、branch-search best
```

`ref` 和 `best branch` 只能回答“好动作是否存在”，不能直接回答 PPO 能不能学。

## 3. 好 reward 的标准

一个适合 PPO actor 训练的 reward，至少应该满足：

```text
1. 同状态换动作后 return 有可见差异。
2. 当前 policy samples 里，A_oracle 与 grad log pi 有稳定协方差。
3. oracle_grad_snr 不长期低于 1，最好接近或超过 3。
4. cancellation_ratio 不接近 0。
5. 真实 PPO advantage 与 A_oracle / g_oracle 大体同向。
```

如果只满足第 1 条，不代表 PPO 能学。那只能说明“好动作存在”，不能说明“当前 policy 能采到并读出方向”。
