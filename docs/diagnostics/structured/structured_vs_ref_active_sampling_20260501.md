# structured vs_ref 主动采样方案

本文只讨论 `sample-vs-ref` 训练里“抽哪些 row 去做昂贵 branch replay”。目标是把当前随机抽样改成一个确定、可实现、可对照的采样方案，不改变 `A_cf = G(sample) - G(ref)` 的定义，也不改变 actor/loss 结构。

## 1. 当前问题

现在 `vs_ref_rows_per_update=8` 是每个 stage 随机抽 8 个 `(rollout_step, env_id, target_uav)`。这个做法有三个问题：

1. 便宜但浪费。很多 row 已经接近无事可做、动作影响很小，branch replay 后 `A_cf` 接近 0。
2. 不稳定。少量 row 可能来自同一时间段、同一 UAV、同一类状态，导致更新方向覆盖很差。
3. 成本不可控。早期 row 的 `episode_remaining` 很长，branch replay 非常贵；随机抽到太多早期 row 会拖慢 update。

所以采样策略应该回答的问题不是“哪个 row reward 最大”，而是：

> 哪些 row 最值得花 branch replay 成本去估计 action contrast？

## 2. 采样单位

统一采样单位是：

```text
candidate row = (stage_id, sample_idx, target_uav)
```

其中：

- `stage_id=0` 是 accel。
- `stage_id=1` 是 sat。
- `stage_id=2` 是 bw。
- `sample_idx` 是该 stage batch 里的 state row。
- `target_uav` 表示本次 branch 只替换这个 UAV 在当前 stage 的 sampled action，其他 UAV 用 ref action。

每个 stage 独立采样，第一版仍保持：

```yaml
vs_ref_rows_per_update: 8
vs_ref_samples_per_row: 1
```

含义仍是每个 stage 8 个 candidate row，不是三 stage 总共 8 个。

## 3. 设计原则

第一版只做 cheap active sampling，不引入新的学习器。

必须满足：

1. 不提前做 branch replay。采样分数只能用 rollout buffer/history 里已经有的 state、action、mask、reward part、logprob/entropy 等便宜量。
2. 不用真实 `A_cf` 当作当前 row 的采样依据。`A_cf` 是采样后才知道的结果。
3. 不只按 top-k。必须保留随机性，避免被异常 row 或单一模式垄断。
4. 不把所有 stage 混在一起竞争预算。accel/sat/bw 分别抽自己的 quota，避免某个 stage 把预算吃光。
5. 不直接用 TD error 或 critic error。当前 `vs_ref` 是 critic-free，采样目标是 action contrast，不是 value fitting。

## 4. 第一版采样模式

新增配置：

```yaml
vs_ref_sampling_mode: active_mixture   # uniform | active_mixture
vs_ref_sampling_alpha: 0.6             # priority stochastic sampling 温度
vs_ref_sampling_random_frac: 0.25
vs_ref_sampling_time_frac: 0.25
vs_ref_sampling_leverage_frac: 0.25
vs_ref_sampling_uncertainty_frac: 0.25
vs_ref_sampling_cost_power: 0.5
```

`uniform` 保持现在行为，作为对照。

`active_mixture` 对每个 stage 的 8 个 row 分成四份：

```text
2 random
2 time-stratified
2 leverage-priority
2 uncertainty-priority
```

如果 `vs_ref_rows_per_update` 不是 8，则按比例四舍五入，并保证总数等于预算。

## 5. 四类采样

### 5.1 random

从所有合法 candidate 里均匀随机抽。

作用：

- 保底探索。
- 防止 active score 写错后完全偏掉。
- 为后续估计采样偏差保留背景样本。

### 5.2 time-stratified

把 rollout step 分成 4 个时间桶：

```text
[0, 25%), [25%, 50%), [50%, 75%), [75%, 100%)
```

按桶轮流随机抽，直到达到 quota。

作用：

- 防止全部抽早期 row 导致 update 极慢。
- 防止全部抽后期 row 导致只学“快结束状态”。
- 能看出不同时间段的 `A_cf` 是否有信号。

### 5.3 leverage-priority

leverage 表示“这个状态还有没有动作影响空间”。第一版只用每个 stage 都容易拿到的通用量：

```text
horizon_remaining
current_env_reward
done_or_truncated_near
```

具体分数：

```text
reward_gap = clamp(1 - env_reward, 0, 1)
priority = reward_gap / max(horizon_remaining, 1) ^ vs_ref_sampling_cost_power
```

默认 `cost_power=0.5`，即除以 `sqrt(horizon)`。这样不会完全排斥早期 row，但会压住超长 branch 的成本。

解释：

- `positive_weighted_workload_level` 越接近 1，说明 workload 越小，动作通常越没空间改善，所以 `reward_gap` 越小。
- `env_reward` 越低，说明当前 workload 越重，还有改善空间，所以 `reward_gap` 越大。
- `horizon_remaining` 越长，branch replay 越贵，所以 priority 要降权。
- 这里不用 queue/workload 复杂字段，先避免口径混乱。reward 已经是 workload 的压缩表达。

### 5.4 uncertainty-priority

uncertainty 表示“当前策略对这个 row 的动作还不确定，或者 sampled action 和 deterministic ref 差异足够大”。按 stage 分开定义。

accel：

```text
uncertainty = ||sample_action_u - ref_action_u||_2
```

sat：

```text
uncertainty = 1 if sampled_subset_u != ref_subset_u else 0
```

bw：

```text
uncertainty = L1(sample_bw_u - ref_bw_u) over valid slots
```

成本修正同 leverage：

```text
priority = uncertainty / max(horizon_remaining, 1)^cost_power
```

第一版不使用 logit margin、entropy、valid user count、队列 variance 等更复杂特征。原因是这些特征各 stage 口径不同，容易先把方案搞乱。先用“sample 是否真的偏离 ref”这个最直接的动作空间信号。

## 6. stochastic priority 抽样

leverage 和 uncertainty 都不用 top-k，而用 stochastic priority：

```text
p_i = (priority_i + eps)^alpha
p_i = p_i / sum_j p_j
```

默认：

```yaml
vs_ref_sampling_alpha: 0.6
```

含义：

- `alpha=0` 退化成 uniform。
- `alpha=1` 完全按 priority 比例。
- `0.6` 是温和优先，避免异常分数支配。

抽样时不放回；如果可选数量不足，就退回 uniform 补齐。

## 7. importance weight 第一版不启用

第一版不对 policy loss 加 importance correction。

原因：

1. 当前目标是提高 branch replay 的有效样本率，不是严格估计 uniform-row objective。
2. full correction 会抵消主动采样带来的收益。
3. 现在每 stage 只有 8 row，采样概率估计本身噪声大。

但实现时要记录每个 row 的 `sample_prob`，后续如果需要，可以加：

```text
w_i = clip(((1/N) / p_i)^beta, 0.25, 2.0)
normalize mean(w)=1
```

第一版只记录，不乘 loss。

## 8. 需要记录的指标

每个 stage 记录：

```text
vs_ref_sampling_mode_stage
vs_ref_sampling_random_count_stage
vs_ref_sampling_time_count_stage
vs_ref_sampling_leverage_count_stage
vs_ref_sampling_uncertainty_count_stage
vs_ref_sampling_horizon_mean_stage
vs_ref_sampling_horizon_min_stage
vs_ref_sampling_horizon_max_stage
vs_ref_sampling_priority_mean_stage
vs_ref_sampling_priority_max_stage
```

训练已有指标继续保留：

```text
vs_ref_adv_mean_stage
vs_ref_adv_std_stage
vs_ref_positive_frac_stage
grad_norm_stage
```

判断采样有没有帮助，主要看：

1. 同样 `rows_per_update=8` 下，`vs_ref_adv_std` 是否不再长期接近 0。
2. `vs_ref_positive_frac` 是否不再长期卡在接近随机但无改善的区域。
3. `update_total_time_sec` 是否没有明显变差。
4. 训练 reward 提升是否不再伴随 episode length 快速缩短。

## 9. 实现位置

主要改 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py)。

当前随机采样在 `_update_vs_ref_native(...)` 里，大致是：

```python
flat_total = num_samples * num_agents
row_budget = min(self.vs_ref_rows_per_update, flat_total)
perm = torch.randperm(flat_total)[:row_budget]
sample_idx_t = perm // num_agents
target_uav_t = perm % num_agents
```

应替换成：

```python
sample_idx_t, target_uav_t, sampling_stats = self._sample_vs_ref_rows(
    stage_id=stage_id,
    stage_batch=stage_batch,
    history_rows_np=history_rows_np,
    source_num_envs=source_num_envs,
    budget=self.vs_ref_rows_per_update,
)
```

新增 helper：

```python
def _sample_vs_ref_rows(...):
    if self.vs_ref_sampling_mode == "uniform":
        return _sample_vs_ref_rows_uniform(...)
    if self.vs_ref_sampling_mode == "active_mixture":
        return _sample_vs_ref_rows_active_mixture(...)
```

其中 `_sample_vs_ref_rows_active_mixture` 负责：

1. 构造所有 candidate flat ids。
2. 计算 `sample_idx`、`target_uav`、`horizon_remaining`。
3. 用 actor no-grad 计算 ref/sample action。
4. 计算 uncertainty score。
5. 从 stage batch reward 或 transition 对应 reward 计算 leverage score。
6. 按四类 quota 抽样。
7. 返回去重后的 selected candidates。

注意：当前 `_update_vs_ref_native` 后面还需要 ref/sample action。为了避免重复 actor forward，第一版可以先接受重复计算；如果速度明显受影响，再把 ref/sample action 一起从 sampler 返回。

## 10. 不做的事情

第一版明确不做：

1. 不做 embedding core-set/k-center。
2. 不用 critic TD error。
3. 不做 learned sampler。
4. 不做 full importance correction。
5. 不跨 stage 抢预算。
6. 不引入 queue/workload/链路的一大堆手工特征。

这些不是永远不做，而是等第一版证明“active row selection 确实提高有效 `A_cf` 样本率”之后再加。

## 11. 推荐实验

先做两个完全同 budget 对照：

```yaml
# A: 当前随机
vs_ref_sampling_mode: uniform
vs_ref_rows_per_update: 8

# B: 主动混合
vs_ref_sampling_mode: active_mixture
vs_ref_rows_per_update: 8
```

每个跑 20 updates，比较：

```text
env_reward_mean
episode_length_mean
completed_episode_count
vs_ref_adv_std_accel/sat/bw
vs_ref_positive_frac_accel/sat/bw
update_total_time_sec
```

如果 B 的 `A_cf` 有效样本率更高，但 episode 仍然快速变短，说明采样不是主因，安全/终止奖励还要继续处理。

如果 B 的 `A_cf` 更强且 episode length 不再快速塌缩，再考虑跑 100 updates。
