# Reward 看起来可用但训练不升：当前整理

这份文档只回答一个问题：

```text
为什么有些 reward 在诊断里看起来有动作信号，实际 PPO / vs_ref 训练仍然不稳定或不上升？
```

结论先放前面：

```text
之前的 reward 诊断证明的是“动作影响存在”或“局部一阶信号存在”。
它没有证明 PPO 实际使用的 GAE advantage 能稳定把这个信号变成 actor 更新。
```

所以现在不能简单说“reward 已经好”，也不能简单说“reward 完全坏”。更准确是：

```text
action -> return 这一层有信号；
return/credit -> PPO actor update 这一层没有稳定打通。
```

## 1. Reward 诊断到底证明了什么

已有 reward probe 看的是当前 policy samples 附近的 true autograd policy-gradient SNR：

```text
A_oracle = G(s, a_sample) - mean_a G(s, a)
g = mean[A_oracle * grad log pi(a_sample | s)]
```

这比只看 best branch 更接近 PPO，但仍然是一个局部诊断，不等于完整训练保证。

已有结果：

| stage | 最好候选 | true-grad SNR | 解释 |
| --- | --- | ---: | --- |
| accel | relative_weighted_workload_delta | 1.395 | 有弱信号，但没到好训区 |
| accel | positive_weighted_workload_level | 1.168 | 更弱 |
| sat | sat_relay_processed | 3.427 | 单 seed / rows8 下很强 |
| bw | accel_growth_safe | 1.609 | 弱到中等 |
| bw | positive/level 类 | 约 0.9-1.1 | 偏弱 |

因此当时说 `sat_relay_processed` “有希望”，含义应该只是：

```text
它在这个局部 probe 里比其他 SAT reward 更能产生一阶动作信号。
```

不是说：

```text
它一定能让 PPO 训练曲线稳定上升。
```

对应文档：

```text
docs/reward_candidate_stage_probe_20260504.md
```

## 2. 为什么 SAT reward 看起来强，短训仍不升

SAT-only 20 update 短训里：

```text
sat_relay_processed:
  env_reward_mean first5 -> last5: 0.775 -> 0.672
  explained_variance_sat: around 0

positive_weighted_workload_level:
  env_reward_mean first5 -> last5: 0.175 -> 0.159
  explained_variance_sat: around 0 to 0.16
```

这说明：

```text
reward 局部 probe 强
不等于真实训练 advantage / update 链路强。
```

后面做了 PPO credit alignment：

| 指标 | seed45210 | seed45211 | seed45212 | 判断 |
| --- | ---: | ---: | ---: | --- |
| corr(A_ppo_norm, A_oracle_policy) | -0.168 | 0.176 | 0.446 | 不稳定 |
| corr(A_oracle_policy, official delta_logprob) | -0.023 | 0.270 | -0.481 | 不稳定 |
| per-row sign(delta_logprob) follows A_ppo | 1.000 | 1.000 | 1.000 | 单 row loss 符号没反 |
| PPO-adv negative-cosine fraction | 0.533 | 0.533 | 0.483 | batch 内冲突很多 |

这几个数合起来说明：

```text
单 row 的 PPO loss 方向是按 advantage 符号走的；
但实际 A_ppo 和同状态 action credit 不稳定对齐；
整批 update 后不保证提高 oracle-positive action。
```

这里的“梯度冲突”不能单独当根因。单样本冲突在 policy gradient 里本来就会有。真正问题是：

```text
这些冲突在当前 batch / 当前 advantage 下没有稳定平均成好方向。
```

对应文档：

```text
docs/ppo_credit_alignment_audit_20260504.md
```

## 3. Critic 现在是什么状态

critic 要做两件事：

```text
1. 训练 target 要接近当前 policy 下的平均回报目标。
2. 网络要能把这个 target 拟合好。
```

我们按训练时 GAE 口径做过 critic-only 复查：

```text
target = train_gae
train_rollouts = 4
train_samples = 8000
heldout_samples = 2000
critic_epochs = 40
stage = sat
reward_mode = sat_relay_processed
```

结果：

| seed | initial heldout EV | final heldout EV | corr(G_train_gae - V, A_oracle_policy) |
| --- | ---: | ---: | ---: |
| 45210 | 0.026 | 0.281 | -0.273 |
| 45211 | -0.029 | 0.312 | 0.099 |
| 45212 | 0.004 | 0.347 | -0.225 |

解释：

```text
critic 不是完全学不动；
但 EV 只有约 0.28~0.35，还不能说 V 已经把 state 难度扣干净。
```

同时：

```text
G_train_gae - V(s)
```

仍然没有稳定变成同状态 action credit。

所以 critic 现在更像：

```text
学到了一部分 GAE target，
但还没学到足够好；
而且即便学到的 residual，也还不像干净 advantage。
```

## 4. 现在不该怎么解释

不要再用下面这些单句解释：

```text
reward 没信号
critic 完全坏
梯度冲突就是根因
actor 一定不行
PPO 理论不适合
```

这些都太粗。

当前证据支持的是更窄的判断：

```text
动作信号存在，但训练接口没有稳定读出来。
```

具体断点可能有两个：

```text
1. A_ppo = GAE - V 还不是可靠 action advantage。
2. 即使用局部 oracle credit，batch mean policy-gradient 也没有稳定转成好更新。
```

第 1 点和 critic/GAE 有关。

第 2 点不一定是 critic 问题，也不一定是 actor 结构问题；它可能是因为当前 oracle 还不是足够准确的 `Qπ(s,a)` 期望，也可能是 actor 参数空间确实把局部 credit 平均坏了。

## 5. Reward “合适”的标准要改严

以后不能只用：

```text
best branch 比 ref 好
```

或者单 seed rows8 的 true-grad SNR 来判断 reward 好。

更严格的标准应该是：

```text
1. 多 seed / 多 rows 下，当前 policy samples 附近 true-grad SNR 稳定。
2. A_ppo 和局部 oracle credit 大体同向。
3. critic-only 后 heldout EV 足够高，并且 G - V 与 action credit 不反向。
4. batch mean gradient 随 rows 增大后方向稳定，而不是 seed-to-seed 乱跳。
5. 短训至少能让 stage 直接物理指标改善，而不只看总 reward。
```

所以之前的 `sat_relay_processed` 应该降级为：

```text
局部一阶信号最强的 SAT 候选 reward。
```

而不是：

```text
已经证明能训练好的 reward。
```

## 6. 当前最简判断

按现在已有结果，我会把问题压成一句：

```text
reward 里确实有动作信息，但这个信息没有稳定穿过 GAE/critic 和 batch policy-gradient，变成可持续的 actor 参数更新。
```

这也是为什么会出现看起来矛盾的现象：

```text
branch/best candidate 有提升；
true-grad probe 某些 reward 看起来不错；
critic EV 又不够高；
per-row PPO loss 没写反；
但实际短训 reward 不升。
```

这些不是互相否定，而是在说明同一条链路的不同位置：

```text
动作影响存在
  -> reward 局部可分
  -> 但 GAE/critic 没稳定形成 clean advantage
  -> batch update 没稳定形成好方向
  -> 训练曲线不上升
```

## 7. 修正：Qπ 和单条 rollout 噪声不能混着讲

前面有一个容易误导的说法：

```text
后续 stochastic continuation 的波动盖住了当前 action credit。
```

这句话如果不拆开，会像是在说 PPO 理论本身不成立，这是不对的。

严格说：

```text
Qπ(s,a) = E[从 s 做 a 后，后续按 π 继续采样得到的总回报]
```

所以 `Qπ` 本身已经把后续随机 policy continuation 积分掉了。它是一个期望函数，不等于某条 rollout。

真正有噪声的是：

```text
G(s,a,future_sample) = Qπ(s,a) + epsilon
```

其中 `epsilon` 来自后续动作采样、外生随机、未来状态演化。PPO 用单条或少量 rollout 来估这个期望，理论上靠样本平均消掉 `epsilon`。

因此现在要查的不是一句“方差大所以不行”，而是链路中哪一步先断：

```text
state s
  -> sampled action a
  -> Qπ(s,a) 均值 credit
  -> A_ppo = GAE - V 是否读出这个 credit
  -> actor loss 是否按 advantage 改 logprob
  -> 参数小步后固定 state/action bank 的 objective 是否改善
```

新增脚本：

```text
scripts/audit_stage_credit_chain.py
```

它固定同一批 state/action，分三段输出：

```text
1. step1_qpi_credit:
   多 continuation 估 Qπ(s,a) 均值，看同 state 内 action gap 是否稳定。

2. step2_advantage_readout:
   比 A_ppo 和 A_qpi_rollout，定位 critic/GAE 是否把 credit 洗掉。

3. step3_actor_loss_update:
   分别用 A_ppo、A_qpi_rollout、A_qpi_all_candidates 做一次 probe update，
   看高 Qπ 候选动作的 logprob 是否被提高。
```

这比只看 `g_batch` 稳不稳定更有定位能力。

## 8. 旧 Qπ 数值应该怎么解释

已有 `scripts/audit_stage_qpi_action_credit.py` 结果仍然有用，但解释要改。

SAT + `sat_relay_processed`：

| run | rows | q action std | single-rollout continuation std | corr(A_ppo_norm, rollout_Q_adv) | split-half grad cosine | cancellation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stochastic seed45210 | 8 | 0.858 | 1.683 | 0.201 | -0.923 | 0.160 |
| stochastic seed45211 | 8 | 0.602 | 1.350 | -0.724 | -0.739 | 0.396 |
| stochastic seed45212 | 8 | 0.926 | 2.094 | -0.285 | 0.978 | 0.690 |
| stochastic seed45210 | 16 | 0.978 | 2.081 | -0.089 | 0.368 | 0.133 |
| deterministic seed45210 | 8 | 9.733 | 0.898 | -0.133 | 0.942 | 0.957 |

这里不能说：

```text
Qπ 本身被后续随机盖住。
```

更准确是：

```text
deterministic/fixed-follow 分支下，局部 action gap 很大；
stochastic follow 下，有限 continuation 估出来的 Qπ gap 小很多；
同时 A_ppo 和 rollout action 的 Qπ advantage 对齐很不稳定。
```

所以它指出的不是“PPO 错”，而是：

```text
现在需要定位 A_ppo、actor loss/update、还是 Qπ credit 本身哪一步先断。
```

`audit_stage_credit_chain.py` 就是为了做这个定位。

## 9. 断点判断标准

跑完 `audit_stage_credit_chain.py` 后按这个顺序判断：

| 现象 | 说明 |
| --- | --- |
| `q_gap_gt_2se_frac` 很低 | 多 continuation 后同 state 内 action 排序本身不稳，reward/action credit 仍不够干净 |
| `corr_ppo_norm_adv_vs_qpi_rollout_adv` 很低或反号 | Qπ credit 有，但 GAE/critic 没读出来 |
| `ppo_update_q_shift <= 0`，但 `qpi_selected_update_q_shift > 0` | actor/loss 能用好 advantage，主要断在 A_ppo |
| `qpi_selected_update_q_shift <= 0`，但 `qpi_all_update_q_shift > 0` | 只用 rollout sampled action 太稀疏，candidate/listwise credit 才能推得动 |
| `qpi_all_update_q_shift <= 0` | actor 参数化、action transform、optimizer/update 尺度本身有问题 |

这才是后续要用的定位表。

当前已有结论先保持保守：

```text
动作信号存在；
但还没有证明它能稳定通过当前 GAE/critic/PPO actor update。
```

## 10. SAT `sat_relay_processed` 链路审计结果

命令口径：

```text
stage = sat
reward_mode = sat_relay_processed
follow = stochastic
policy_action_samples = 4
continuations = 4
full horizon
```

结果：

| run | q_gap_gt_2se_frac | corr(A_ppo_norm, A_qpi_rollout) | PPO update q-shift | Qπ-selected update q-shift | Qπ-all-candidate update q-shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed45210 rows8 | 0.750 | -0.115 | -0.0006 | 0.0216 | 0.0554 |
| seed45211 rows8 | 0.875 | 0.135 | 0.0294 | -0.0168 | 0.0673 |
| seed45212 rows8 | 0.625 | -0.104 | 0.0550 | 0.0559 | 0.0590 |
| seed45210 rows16 | 0.625 | 0.230 | 0.0629 | 0.0630 | 0.0634 |

字段含义：

```text
q_gap_gt_2se_frac:
  同 state 内候选 action 的 Qπ gap 是否大于估计标准误。
  这里 0.625~0.875，说明 Qπ action credit 不是完全分不出来。

corr(A_ppo_norm, A_qpi_rollout):
  PPO 实际用的 advantage 是否和同 state rollout action 的 Qπ advantage 对齐。
  这里很低且 seed 间不稳定。

PPO update q-shift:
  用实际 A_ppo 做一次 probe update 后，高 Qπ 候选动作的 logprob 是否提高。

Qπ-selected update q-shift:
  只用 rollout sampled action 的 Qπ advantage 做一次 probe update。

Qπ-all-candidate update q-shift:
  用同 state 多候选 action 的 Qπ advantage 做一次 probe update。
```

这组结果目前更支持：

```text
1. SAT reward/action 本身不是完全没 credit。
2. A_ppo = GAE - V 没稳定读出这个 credit。
3. actor/loss 不是简单写反，因为给 candidate-wise Qπ credit 时能稳定正向推。
4. 单个 rollout sampled action 的 Qπ credit 仍偏稀疏，rows8 下 seed45211 会反；candidate-wise credit 稳定得多。
```

所以当前 SAT 的断点排序更像：

```text
主要断点：A_ppo / GAE / critic 对同状态 action credit 的读出。
次要断点：只用 rollout sampled action 做一阶更新太稀疏。
暂不支持：reward 完全没信号、actor loss 写反、actor 完全不能表达。
```

### Per-state 检查

为了确认 `Qπ-all-candidate update q-shift > 0` 不是只靠少数 state 拉高，又补了 per-state 指标：

```text
per_state_pos:
  每个 state 内 sum_a A_qpi(s,a) * delta_logprob(s,a) > 0 的比例。

best_up:
  每个 state 的 best-Qπ candidate logprob 是否上升。

margin_up:
  每个 state 的 best-Qπ candidate 相对 worst-Qπ candidate 的 logprob margin 是否变大。
```

结果：

| run | update | q-shift | per_state_pos | best_up | margin_up |
| --- | --- | ---: | ---: | ---: | ---: |
| seed45210 | PPO A | -0.0006 | 0.500 | 0.375 | 0.625 |
| seed45210 | Qπ selected | 0.0216 | 0.875 | 0.500 | 0.750 |
| seed45210 | Qπ all-candidate | 0.0554 | 0.875 | 0.500 | 0.625 |
| seed45211 | PPO A | 0.0294 | 0.875 | 0.750 | 0.875 |
| seed45211 | Qπ selected | -0.0168 | 0.250 | 0.375 | 0.375 |
| seed45211 | Qπ all-candidate | 0.0673 | 0.875 | 0.750 | 0.875 |
| seed45212 | PPO A | 0.0551 | 0.625 | 0.375 | 0.750 |
| seed45212 | Qπ selected | 0.0560 | 0.625 | 0.375 | 0.625 |
| seed45212 | Qπ all-candidate | 0.0590 | 0.625 | 0.375 | 0.750 |

解释：

```text
Qπ-all-candidate 不是保证每个 state 的 best action 都上升；
但它在 3 个 seed 上都让多数 state 的 A_qpi 加权 logprob shift 为正。
```

所以更准确的说法是：

```text
candidate-wise Qπ credit 在 batch 平均和多数 state 上能推对方向；
rollout sampled action 单点 credit 不稳定；
A_ppo 是否推对方向更不稳定。
```
