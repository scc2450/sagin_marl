# PPO Credit Alignment Audit

目的不是看总 reward 或 access，而是检查 PPO 训练信号本身是否把“同状态下更好的动作”推上去。

## 检查口径

对某个 stage 的 rollout row：

1. 从 rollout buffer 取真实 sampled action 和 PPO 使用的 advantage。
2. 在同一个 history snapshot 上，用相同外生随机性做 branch replay。
3. 估计同状态 oracle advantage：

```text
A_oracle_policy = G(s, a_rollout_sample) - mean_a G(s, a_policy_sample)
```

这里的 baseline 是当前 stochastic policy 的动作均值，不是 deterministic ref。另保留：

```text
A_oracle_ref = G(s, a_rollout_sample) - G(s, a_deterministic_ref)
```

只作为参考。

然后检查三件事：

```text
1. A_ppo 是否和 A_oracle_policy 对齐
2. 官方整批 PPO update 后，delta_logprob 是否提高 A_oracle_policy > 0 的动作
3. selected-only / per-row PPO update 是否正常
```

如果 per-row 正常、selected-only 或整批不正常，说明 logprob/loss 符号不是主问题，更像共享参数上的 batch 梯度抵消。

## 脚本

```text
scripts/audit_stage_ppo_credit_alignment.py
```

关键参数：

```text
--stage accel|sat|bw
--reward_mode <mode>
--sample_rows
--policy_action_samples
```

## SAT: sat_relay_processed 结果

命令：

```powershell
.\.venv\Scripts\python.exe scripts\audit_stage_ppo_credit_alignment.py `
  --stage sat `
  --config configs\tmp\structured_single_sat_3uav_20gu_t250_ppo.yaml `
  --reward_mode sat_relay_processed `
  --out runs\diagnostics\ppo_credit_alignment\sat_relay_processed_rows8_k4_perrow.json `
  --device cuda `
  --num_envs 8 `
  --rollout_env_steps 250 `
  --sample_rows 8 `
  --policy_action_samples 4 `
  --min_horizon 20 `
  --seed 45210 `
  --torch_threads 1
```

核心结果：

```text
corr(A_ppo_norm, A_oracle_policy)       = 0.696
sign_agree(A_ppo_norm, A_oracle_policy) = 0.875

official update:
  corr(A_oracle_policy, delta_logprob)  = -0.310
  mean delta_logprob when oracle > 0    = -0.568
  mean delta_logprob when oracle < 0    = -0.042

selected-only batch update:
  corr(A_ppo_norm, delta_logprob)       = 0.031
  corr(A_oracle_policy, delta_logprob)  = -0.283

per-row isolated update:
  corr(A_ppo_norm, delta_logprob)       = 0.483
  corr(A_oracle_policy, delta_logprob)  = 0.557
  sign(delta_logprob) follows A_ppo     = 1.000
```

## 当前判断

`sat_relay_processed` 不是完全没有 credit 信号。PPO advantage 和同状态 oracle credit 有明显相关。

但 SAT actor 是共享参数网络。单个 row 单独更新时，PPO loss 会按 advantage 正确改变该动作 logprob；多个 row 合起来时，更新方向基本被抵消，整批官方 update 甚至对 oracle-positive 样本更大幅降低 logprob。

所以 SAT 这条线上，下一步不应该只看 `x_rel/processed/drop/overlap` 是否变好。更直接的问题是：

```text
共享 SAT actor 在一批状态上的 policy-gradient 方向互相冲突，
导致实际 batch update 没有把 oracle-positive sampled actions 推上去。
```

这和“reward parts 是否有物理意义”是两层问题。当前这个审计说明：至少在 `sat_relay_processed` 下，reward 有一定动作区分度，但 PPO batch aggregation 没把它稳定转成策略改进。

## 多 seed / rows16 复查

为了确认不是 `rows=8` 的偶然现象，又跑了 3 个 seed、每个 seed 采 16 rows，并开启 per-row gradient conflict analysis：

```text
seed = 45210, 45211, 45212
sample_rows = 16
policy_action_samples = 4
reward_mode = sat_relay_processed
```

结果：

| 指标 | seed45210 | seed45211 | seed45212 | 判断 |
| --- | ---: | ---: | ---: | --- |
| corr(A_ppo_norm, A_oracle_policy) | -0.168 | 0.176 | 0.446 | 不稳定 |
| sign agree(A_ppo_norm, A_oracle_policy) | 0.375 | 0.438 | 0.688 | 不稳定 |
| corr(A_oracle_policy, official delta_logprob) | -0.023 | 0.270 | -0.481 | 不稳定 |
| corr(A_ppo_norm, selected-only delta_logprob) | 0.376 | 0.608 | 0.241 | 有正相关但不强 |
| corr(A_ppo_norm, per-row delta_logprob) | 0.612 | 0.734 | 0.823 | 稳定为正 |
| per-row sign(delta_logprob) follows A_ppo | 1.000 | 1.000 | 1.000 | 稳定正确 |

这说明两件事：

1. 单 row 的 PPO loss / logprob 符号没有写反；每个 row 单独更新时都会按 PPO advantage 的符号推。
2. 当前 `A_ppo` 和同状态 oracle action credit 的对齐不稳定，且整批 update 后对 oracle-positive action 的处理也不稳定。

## 梯度冲突分组

对每个 sampled row 单独算 policy-gradient 向量，再看 pairwise cosine。三组 seed 的整体冲突如下：

| 指标 | seed45210 | seed45211 | seed45212 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| PPO-adv gradient cosine mean | -0.048 | 0.026 | -0.013 | -0.012 |
| PPO-adv gradient cosine p10 | -0.937 | -0.549 | -0.894 | -0.793 |
| PPO-adv negative-cosine fraction | 0.533 | 0.533 | 0.483 | 0.517 |
| oracle-adv negative-cosine fraction | 0.533 | 0.525 | 0.492 | 0.517 |

也就是说，约一半 row pair 的梯度方向互相顶。这个现象在 PPO advantage 加权和 oracle advantage 加权下都存在，所以它不只是 critic/GAE 的问题。

检查过的分组：

```text
visible SAT set
selected/ref SAT set
sample/ref SAT overlap
hotspot/arrival proxy
SAT load
SAT queue
step interval
advantage sign
oracle sign
```

比较重要的分组现象：

| 分组 | 现象 | 结论 |
| --- | --- | --- |
| step interval | early 往往 PPO advantage 更负，mid 往往更正 | step 能解释一部分 baseline/advantage 偏置 |
| visible SAT set | `6|7|27|28` 常比 `7|8|27|28|29` 的 advantage 更高 | visible set 是 regime 信号 |
| ref selected set | 例如 ref=`6` 常偏正，ref=`8|29` 常偏负 | ref selection 也是 regime 信号 |
| sample/ref overlap | low-overlap 往往 oracle 更正，high-overlap 往往更负 | “sample 离 ref 多远”影响 credit |
| arrival/load | 有影响但不稳定 | 不是单独主因 |

但是这些分组都没有把冲突真正消掉。典型情况是：某个分组能把 advantage 均值分开，但组内 cosine 仍然大量为负。例如 seed45212 的 `visible_sat_set` 分组里，within conflict fraction 仍约 0.485，cross conflict fraction 约 0.483，几乎没改善。

## 当前结论

这轮结果更像两个问题叠在一起：

1. `A_ppo = GAE/return - V` 没有稳定扣掉 state/regime 难度，所以它和同状态 action credit `A_oracle_policy` 有时正相关、有时接近无关、有时反向。
2. 即使用 oracle advantage 给 per-row gradient 加权，SAT actor 的 row-to-row 梯度仍然约一半互相冲突；简单按 visible set、selected set、arrival/load、step 分桶不足以解决。

但这里还不能排除一个更普通的解释：critic 还没学出来。当前结果只能说明这个 checkpoint / 这批 rollout 里的 `A_ppo` 不稳定，不能直接推出 reward 一定不适合 PPO。要排除 critic 因素，需要冻结 actor，在固定 rollout bank 上做 critic-only 训练，然后重新计算 `G - V(s)` 和 `A_oracle_policy` 的相关。

## Regime baseline 后验检查

用 48 个 rows 做了一个不训练的 sanity check：把 `A_ppo_norm` 按不同 regime 扣掉组内均值，再看它和 `A_oracle_policy` 的相关是否变好。

结果：

| baseline 分组 | corr(residual A_ppo, A_oracle_policy) | 备注 |
| --- | ---: | --- |
| 不扣分组，仅原始 `A_ppo_norm` | 0.113 | 全 48 rows |
| step_bin | 0.037 | 变差 |
| visible_sat_set | 0.005 | 变差 |
| ref_selected_set | -0.026 | 变差，且组太碎 |
| sample/ref overlap | -0.058 | 变差 |
| step + visible_sat_set | 0.028 | 变差 |
| step + visible + ref | -0.043 | 变差，组太碎 |
| step + arrival_bin + load_bin | 0.217 | 有弱改善 |

这个检查不代表 learned baseline 没用，但说明“按 visible/ref set 简单分桶再做 advantage normalization”目前证据不足；它没有直接把 PPO advantage 变成更像同状态 action credit 的量。

因此下一步如果要修，不应该只做一个“随机分组 minibatch”，也不应该只按 visible/ref set 硬分桶。更合理的优先级是：

1. 先做 critic-only 复查：冻结 actor，在固定 rollout bank 上训练 critic，观察 heldout EV、`Var(G - V) / Var(G)`、`corr(G - V, A_oracle_policy)` 是否改善。
2. 如果 critic-only 后相关明显改善，说明前面的不稳定主要是 critic 还没学出来，应优先解决 critic 训练速度/尺度/结构。
3. 如果 critic-only 后相关仍不改善，再做 learned regime baseline / control variate，候选输入至少包括 `step/t_frac + arrival pressure + sat load + visible/ref selection summary`。
4. conflict-aware aggregation 仍然可以测，但不要随机分组；如果做 PCGrad/CAGrad，也应该先比较 `oracle_adv_gradient` 下的组内冲突是否能被 learned regime 表征降低。
5. 如果 critic / learned baseline 后相关仍不改善，问题才更可能在 SAT actor 表征/动作参数化：同一套共享参数无法把不同 SAT regime 的“该增大哪个 slot”映射成一致梯度。

## Critic-Only 复查：训练时 GAE 口径

为了排除“只是 critic 还没学出来”的解释，新增脚本：

```text
scripts/audit_stage_critic_only_fit.py
```

这次不用 MC return，而是按训练时口径固定 `train_gae` target：

```text
target = train_gae
train_rollouts = 4
train_samples = 8000
heldout_samples = 2000
critic_epochs = 40
reward_mode = sat_relay_processed
stage = sat
```

也就是说：冻结 actor，收多个 rollout bank，只训练 critic，然后在 heldout rollout 上看 critic 能否拟合训练实际使用的 GAE return target，并重新检查 `G_train_gae - V(s)` 和同状态 oracle credit 的相关。

结果：

| seed | initial heldout EV | final heldout EV | final heldout MSE | corr(G_train_gae - V, A_oracle_policy) | sign agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| 45210 | 0.026 | 0.281 | 0.000477 | -0.273 | 0.562 |
| 45211 | -0.029 | 0.312 | 0.000441 | 0.099 | 0.750 |
| 45212 | 0.004 | 0.347 | 0.000476 | -0.225 | 0.562 |

注意这里 MSE 很小不等于 critic 已经很好。heldout GAE target 的标准差只有约 `0.025`，critic 预测标准差约 `0.012~0.019`，所以绝对误差容易显得小；EV 仍只有 `0.28~0.35`。

当前更谨慎的结论：

1. critic 不是完全学不动：固定 policy / 多 rollout bank / critic-only 后，heldout EV 能从接近 0 提到约 `0.31`。
2. 但这还不足以说明 critic 已经把 state 难度扣干净；`G_train_gae - V(s)` 和 oracle action credit 仍然没有稳定正相关。
3. 所以目前仍不能把问题直接归因到 reward 本身，也不能说 critic 已经排除。更准确是：按训练时 GAE target，critic 学到了一部分 state 结构，但 residual 仍不像干净的 action credit。

下一步如果继续查 critic，应做 bank-size / epoch 曲线，而不是只看单个小 bank：

```text
train_rollouts = 4 / 8 / 16
critic_epochs = 40 / 80
看 heldout EV 是否继续上升，以及 corr(G - V, A_oracle_policy) 是否变成稳定正相关。
```
