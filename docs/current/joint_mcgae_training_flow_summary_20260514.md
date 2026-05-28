# Joint MC-GAE 训练流程梳理

本文记录这段时间为什么修改训练流程、具体改了什么、这些改动分别解决什么问题，以及当前结果说明了什么。

## 1. 一开始的问题

原始训练流程接近普通 PPO：

```text
rollout 一批数据
用当前 critic 计算 GAE
用 GAE return 训练 critic
用 GAE advantage 更新 actor
```

公式是：

```text
delta_t = r_t + gamma * V_old(s_{t+1}) - V_old(s_t)

A_t^GAE = sum_l (gamma * lambda)^l * delta_{t+l}

R_t^GAE = A_t^GAE + V_old(s_t)
```

critic 训练：

```text
min_theta (V_theta(s_t) - R_t^GAE)^2
```

actor 更新：

```text
rho_t = pi_theta(a_t | s_t) / pi_old(a_t | s_t)

L_actor = E[min(rho_t * A_t^GAE,
                clip(rho_t, 1-eps, 1+eps) * A_t^GAE)]
```

这个流程在标准 PPO 里是正常的，但当前任务里出现了一个实际问题：

```text
critic 没能稳定扣掉 state 难度。
actor 拿到的 A_t^GAE 很吵。
SAT/BW 的当前动作影响被 state 难度、后续随机动作、外生随机过程淹没。
```

更具体地说，当前 reward 是系统级目标，包含很多不是单个 stage 当前动作直接决定的量。对某个 stage 的 actor 来说，真正需要的是：

```text
A_stage(s_stage, a_stage)
  = Q_stage(s_stage, a_stage) - V_stage(s_stage)
```

其中：

```text
V_stage(s_stage)
  = E_{后续固定 policy, 外生随机}[return | s_stage]
```

但如果 critic 没学好，实际用到的 advantage 会更像：

```text
return 的单条样本波动 - 一个不够准的 baseline
```

这会让 actor 更新方向不稳定。

## 2. 为什么普通 train-GAE target 不够

普通 PPO 里 critic 学的是：

```text
R_t^GAE = A_t^GAE + V_old(s_t)
```

这个 target 是 bootstrapped target，依赖 `V_old`。

早期 `V_old` 很差时，`R_t^GAE` 不一定接近 full-horizon 的：

```text
V_pi(s_t) = E[sum_k gamma^k r_{t+k} | s_t]
```

我们做过 audit，看到 train-GAE target 和 full-horizon `V_hat_pi(s)` 的尺度明显不一致。也就是说问题不只是网络容量，而是：

```text
critic 一开始拿到的训练标签本身更像短 horizon / bootstrapped label，
不稳定地对应 full finite-horizon value。
```

如果 critic 没先学到正确尺度，actor 第一批更新就可能被错误 advantage 带偏。

当时的 SAT critic 审计给出了很直接的数值证据：

```text
heldout train-GAE target mean ≈ -0.146
heldout full-horizon V_hat_pi mean ≈ -2.012
```

也就是一次性 train-GAE target 和 full-horizon value 的尺度差了一个数量级。

继续让 critic 反复学 train-GAE target，也没有把它推到 full-horizon value：

| target | 训练方式 | value pred mean | EV(V_theta, V_hat_pi) |
| --- | --- | ---: | ---: |
| train-GAE | one-shot, 8 rollouts, 30 epochs | -0.151 | -0.006 |
| train-GAE | fitted, 3 rounds | -0.198 | 0.008 |
| train-GAE | fitted, 8 rounds | -0.193 | 0.003 |
| MC return | 8 rollouts, 30 epochs | -1.548 | 0.997 |

这说明：

```text
问题不是 critic 完全表达不了 V_pi。
而是 train-GAE target 由于包含 V 自举，在早期 V 很差时传播很慢、尺度也不对。
MC return 能把 critic 直接拉到接近 full-horizon V_pi 的尺度。
```

## 3. 为什么改成 MC return 训练 critic

现在 critic target 改成 finite-horizon MC return：

```text
R_t^MC = sum_{k=0}^{T-t-1} gamma^k r_{t+k}
```

critic 训练：

```text
min_theta (V_theta(s_t) - R_t^MC)^2
```

这样做的理论依据是固定 policy 下：

```text
E[R_t^MC | s_t] = V_pi(s_t)
```

也就是说，单条 MC return 是 noisy label，但它的条件均值就是 critic 应该学的 value。

所以这里不是让 actor 直接用 MC advantage 更新，而是：

```text
MC return 负责给 critic 正确的 full-horizon value 尺度。
critic 学完后，再重新计算 actor 用的 GAE advantage。
```

## 4. 为什么还要重新算 A_gae(V)

如果直接用：

```text
A_t^MC = R_t^MC - V_theta(s_t)
```

actor advantage 的方差会很大，因为单条 MC return 包含后续随机动作和外生随机过程的完整噪声。

所以现在流程是：

```text
先用 MC return 训练 critic。
再用训练后的 V_theta 重新计算 GAE。
```

重新计算：

```text
delta_t = r_t + gamma * V_theta(s_{t+1}) - V_theta(s_t)

A_t^GAE(V_theta)
  = sum_l (gamma * lambda)^l * delta_{t+l}
```

actor 仍然走 PPO ratio：

```text
L_actor = E[min(rho_t * A_t^GAE(V_theta),
                clip(rho_t, 1-eps, 1+eps) * A_t^GAE(V_theta))]
```

因此新流程的核心不是“放弃 PPO”，而是：

```text
critic target 从 train-time GAE return 换成 MC return；
actor advantage 仍然用 GAE；
actor objective 仍然是 PPO clipped objective。
```

## 5. 训练批次如何变化

原来更接近：

```text
num_envs = 8
rollout_env_steps = 250
每轮约 8 * 250 = 2000 primitive transitions
```

现在 joint MC-GAE 使用：

```text
num_envs = 64
rollout_env_steps = 250
每轮约 64 * 250 = 16000 primitive transitions
```

批量变大的原因是：

```text
当前任务 advantage 方差高；
stage 动作影响容易被状态差异和随机 continuation 淹没；
更大 batch 能降低 policy-gradient estimator 的采样噪声。
```

这不是改变 objective，而是降低估计噪声。

## 6. Critic cold fit

原来没有第一轮 critic cold fit。

旧流程第一轮是：

```text
rollout
critic 还很差
直接算 GAE
actor 也直接更新
```

现在第一轮改成：

```text
rollout
计算 MC return
critic cold fit 较多 epoch
用训练后的 critic 重新算 A_gae(V)
再更新 actor
```

当前使用的典型设置是：

```text
第 1 轮 critic:
  lr = 1e-3
  epochs = 20
  EV 不够时额外补训

后续 critic tracking:
  lr = 3e-4
  epochs = 5
  EV 不够时额外补训
```

这么做的目的：

```text
避免第一轮 actor update 使用几乎未训练 critic 给出的坏 advantage。
```

在当前 300-update joint run 里，第 1 轮实际触发了 EV gate 补训：

| stage | critic epochs | initial EV after | final EV after |
| --- | ---: | ---: | ---: |
| accel | 26 | 0.915 | 0.934 |
| sat | 26 | 0.947 | 0.922 |
| bw | 26 | 0.931 | 0.939 |

第 1 轮没有达到 soft target `0.96`，但都高于 hard floor `0.90`，所以标记为 low-confidence，但没有跳过 actor update。

后续 tracking 的例子，update 300：

| stage | critic epochs | initial EV after | final EV after |
| --- | ---: | ---: | ---: |
| accel | 11 | 0.949 | 0.963 |
| sat | 5 | 0.971 | 0.971 |
| bw | 5 | 0.977 | 0.977 |

这说明：

```text
第一轮确实需要更多 critic 训练来把 value 拉到可用区间。
后续 actor/policy 变化较小后，5 epoch 通常能跟上；
个别 stage EV 不够时再额外补训。
```

## 7. Critic EV gating

原来 actor update 不检查 critic 是否已经学到当前 batch。

旧流程：

```text
critic 训固定 epoch
无论 critic fit 怎么样
都更新 actor
```

现在增加了 critic quality gate：

```text
critic 先按默认 epoch 训练
计算 EV / fit quality

如果 EV 低于阈值:
  额外训练 critic 若干 epoch
  重新计算 value / advantage

如果仍然不达标:
  可以跳过该 stage actor update
```

这里 EV 的意义不是“完美评价 critic”，而是避免最明显的错误：

```text
在 critic 完全没扣掉 state 难度时，仍然强行更新 actor。
```

## 8. 三个 stage 如何训练

当前 joint 训练里，三个 stage 都使用同一套 MC-GAE 思路。

对每个 stage：

```text
stage batch = 该 stage 的 state/action/logprob/reward/value 序列
target = 该 stage 对应时间点的 finite-horizon MC return
critic 学 target
重新算 A_gae(V)
actor PPO 更新
```

也就是：

```text
accel critic 学 accel stage value
sat critic 学 sat stage value
bw critic 学 bw stage value
```

actor 更新是分 stage 的：

```text
accel actor optimizer step
sat actor optimizer step
bw actor optimizer step
```

不是把三个 actor loss 混在一起做一次 step。

## 9. BW macro K=5

BW 原来每个 primitive step 都重新决策。

问题是 BW action 对系统 return 的影响太碎：

```text
一步 BW allocation 的影响可能很快被下一步重新分配覆盖。
PPO 很难稳定把当前 BW action 和后续 workload 改善联系起来。
```

所以改成：

```text
access_bw_decision_interval = 5
```

语义是：

```text
每 5 步重新确定一次 access/BW valid set 和 BW action。
中间 4 步复用 macro-start 的 BW action。
```

训练上：

```text
环境仍然 primitive step 一步一步执行。
reward / value / return 仍然逐步计算。
BW actor 只在 macro-start row 上产生 action/logprob。
BW actor 只用 macro-start row 的 advantage 更新。
```

注意：

```text
不是把 5 步 reward 合成一个新 reward。
不是让 critic 只看 5-step chunk。
只是降低 BW action 决策频率，让一次 BW action 的后果持续更久。
```

## 10. KL early stopping

PPO 的 clipped objective 只能限制 surrogate，不保证真实 policy step 不过大。

如果一轮 actor update 里：

```text
approx_kl 很大
clip_frac 很大
```

说明这轮更新已经把 policy 推出 trust region。

所以现在 actor update 内部加入 KL early stopping：

```text
actor epoch 循环中持续监控 approx_kl / clip_frac

如果当前 stage 的 KL 超过阈值:
  提前停止该 stage 当前轮 actor epochs
```

目的：

```text
防止一次 update 内部连续多个 epoch 把策略推太远。
```

这尤其重要，因为现在每轮 batch 更大、critic 更强，advantage 可能更有力；如果 actor step 不控，可能会出现单轮更新过激。

## 11. LR decay

除了单轮 early stopping，还加入了跨 update 的动态 LR decay。

逻辑是：

```text
如果某个 stage 的 KL / clip_frac 的 EMA 持续过大:
  降低该 stage actor lr
```

当前策略是：

```text
允许 decay
不允许 grow
```

也就是：

```text
stage_actor_dynamic_lr_enabled = true
stage_actor_lr_grow_factor = 1.0
```

为什么关闭 grow：

```text
之前允许 grow 时，SAT/BW 后期可能被推得过激。
尤其 SAT lr 后期涨高后，joint final 不稳定。
```

所以现在只保留安全方向：

```text
KL/clip 太大 -> 降 lr
KL/clip 太小 -> 不自动升 lr
```

这让训练更保守，但这轮结果显示稳定性更好。

## 12. 现在完整训练流程

当前流程可以写成：

```text
for update in 1..N:
    1. native rollout 64 env * 250 step

    2. 构造 accel/sat/bw 三个 stage batch

    3. 对每个 stage:
        3.1 计算 finite-horizon MC return
        3.2 训练对应 critic
            - update 1: cold fit, epochs 多
            - update >1: tracking fit, epochs 少
        3.3 如果 EV 不够:
            - 额外训练 critic
            - 重新算 value / advantage
            - 仍不够则可跳过 actor update
        3.4 用训练后的 critic 重新计算 A_gae(V)

    4. 对每个 stage:
        4.1 用 PPO clipped objective 更新 actor
        4.2 actor epoch 内监控 KL / clip
        4.3 KL 超阈值则 early stop
        4.4 跨 update 根据 KL/clip EMA 做 lr decay

    5. 同步 policy，用于下一轮 rollout
```

BW 额外有：

```text
access_bw_decision_interval = 5
只在 macro-start row 更新 BW actor。
```

SAT 当前仍然：

```text
sat_decision_interval = 1
```

## 13. 和原流程的主要差异

| 项目 | 原流程 | 当前流程 |
| --- | --- | --- |
| rollout batch | 约 8 env * 250 | 64 env * 250 |
| critic target | train-time GAE return | finite-horizon MC return |
| actor advantage | 当前 critic 直接算 GAE | critic 训完后重新算 GAE(V) |
| 第一轮 critic | 无特殊 cold fit | cold fit 多 epoch |
| critic 不达标 | 仍更新 actor | 补训或跳过 stage actor |
| BW 决策 | 每步决策 | K=5 macro decision |
| actor KL 控制 | 普通 PPO clip 为主 | clip + KL early stop |
| actor lr | 固定或可 grow/decay | 只允许 decay，不允许 grow |

## 14. 当前结果说明什么

当前最干净的一轮：

```text
runs/diagnostics/joint_mcgae_macro_k5_nogrow_best_u300_20260514
```

核心设置：

```text
reward_mode = positive_weighted_workload_level
num_envs = 64
updates = 300
BW K = 5
SAT K = 1
LR grow off
```

native deterministic eval：

```text
rule baseline = 45.2764
final ASB     = 58.3363
best ASB      = 59.3289
```

8 种组合显示：

Final checkpoint：

```text
A only: +0.3519
S only: +0.1084
B only: +10.5967
ASB:    +13.0599
```

Best heads：

```text
A only: +0.2911
S only: +0.1292
B only: +12.4455
ASB:    +14.0525
```

因此当前结论是：

```text
只改 MC-GAE critic 后，收益仍更像主要来自 accel；
SAT/BW 并没有立刻变成主要收益来源。

进一步加入 BW K=5 后，
主要收益来源才明显转成 BW。

SAT 单独收益小，但和 BW 组合后有叠加。

Accel 单独收益小，目前不是主要收益来源。
```

这说明训练流程修改确实改变了 credit 的可学习性：

```text
BW 从过去很难稳定超过规则，
变成当前 joint 里最主要的 learned gain 来源。
```

但也说明还有未解决的问题：

```text
Accel 现在没有成为主要收益来源。
SAT 单独提升仍小。
best ASB 有 1/64 collision，需要继续确认安全性。
```

## 15. 给导师汇报用的 PPT 简版文字

### 第 1 页：问题背景

原始训练使用普通 PPO / GAE：

```text
A_t^GAE = sum_l (gamma * lambda)^l [r_t + gamma V(s_{t+1}) - V(s_t)]
```

问题：

```text
早期 V(s) 很差，GAE target 依赖 V 自举，critic 学 full-horizon value 很慢。
actor 拿到的 advantage 不能稳定反映当前 stage 动作好坏。
SAT / BW 动作 credit 容易被系统状态差异和后续随机过程淹没。
```

关键审计数：

```text
train-GAE target mean ≈ -0.146
full-horizon V_hat_pi mean ≈ -2.012

train-GAE fitted 8 rounds 后:
  EV(V_theta, V_hat_pi) ≈ 0.003

MC return target:
  EV(V_theta, V_hat_pi) ≈ 0.997
```

结论：

```text
critic 不是完全学不了 V_pi；
主要问题是 train-GAE target 由于 V 自举，早期尺度不对、传播太慢。
```

### 第 2 页：训练流程修改

旧流程：

```text
rollout
用当前 critic 算 GAE
用 GAE return 训练 critic
用 GAE advantage 更新 actor
```

新流程：

```text
1. rollout: 64 env * 250 step
2. 用 finite-horizon MC return 训练 critic
3. critic 训好后重新计算 A_gae(V)
4. actor 仍用 PPO clipped objective 更新
```

公式：

```text
R_t^MC = sum_k gamma^k r_{t+k}
E[R_t^MC | s_t] = V_pi(s_t)
```

含义：

```text
MC return 用来给 critic 正确的 value 尺度；
重新计算的 GAE(V) 用来给 actor 较低方差的 advantage。
```

### 第 3 页：critic 训练和稳定化

第一轮 critic cold fit：

```text
lr = 1e-3
基础 20 epochs
EV 不够则额外补训
```

后续 critic tracking：

```text
lr = 3e-4
基础 5 epochs
EV 不够则额外补训
```

当前 joint run 的第 1 轮 EV：

| stage | epochs | final EV |
| --- | ---: | ---: |
| accel | 26 | 0.934 |
| sat | 26 | 0.922 |
| bw | 26 | 0.939 |

update 300 的 tracking EV：

| stage | epochs | final EV |
| --- | ---: | ---: |
| accel | 11 | 0.963 |
| sat | 5 | 0.971 |
| bw | 5 | 0.977 |

说明：

```text
第一轮确实需要多训 critic；
后续 policy 变化较小后，较少 epochs 基本能跟上。
```

PPO 稳定化：

```text
KL early stopping:
  单轮 actor update 内 KL 过大就提前停止。

LR decay:
  KL / clip_frac 持续过大则降低对应 stage actor lr。

LR grow:
  当前关闭，避免后期学习率涨得过激。
```

### 第 4 页：BW macro K=5

只改 MC-GAE critic 后：

```text
收益仍更像主要来自 accel；
SAT / BW 还没有成为主要收益来源。
```

后来对 BW 加 macro decision：

```text
access_bw_decision_interval = 5
```

语义：

```text
每 5 步重新确定一次 BW action；
中间 4 步复用同一 BW action；
只在 macro-start row 更新 BW actor。
```

目的：

```text
让一次 BW action 的影响持续多步，
避免一步 BW 决策很快被下一步覆盖，
增强 BW action credit。
```

### 第 5 页：最终结果

实验设置：

```text
3 UAV / 20 GU
T = 250
64 env
300 updates
reward = positive_weighted_workload_level
SAT K=1
BW K=5
```

Native deterministic eval，64 episodes：

```text
rule baseline reward = 45.28
final policy reward  = 58.34
best-head policy     = 59.33
```

相对规则策略：

```text
final:     +13.06
best-head: +14.05
```

### 第 6 页：各 stage 贡献

8 种组合，`A=accel, S=sat, B=bw`。

Final checkpoint 相对 rule：

```text
A only: +0.35
S only: +0.11
B only: +10.60
ASB:    +13.06
```

Best heads 相对 rule：

```text
A only: +0.29
S only: +0.13
B only: +12.45
ASB:    +14.05
```

结论：

```text
BW 是当前主要收益来源。
SAT 单独收益小，但和 BW 组合后有叠加。
Accel 单独收益小，目前不是主要收益来源。
```

更准确的历史对比：

```text
只改 MC-GAE critic 时，收益仍主要像 accel。
加入 BW K=5 后，BW 才变成主要可学习收益来源。
```
