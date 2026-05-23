# BW `branch_delta_h3` Teacher-Student Delta Critic Follow-up (2026-04-13)

这份文档整理的是一条完整的排查链：

1. 为什么简单环境里 `T=1/2/5/10` 随 horizon 增大后，`BW-only PPO` 很快学不出来。
2. 为什么问题主要落在 `critic/value -> advantage -> actor update` 这条链上，而不是 PPO actor 本身完全失效。
3. 为什么短 horizon `V_h(s)` 只能部分缓解，而短 horizon `delta` 更对题。
4. 这次新做的 `teacher 更新 actor + student 并行蒸馏 + 按相关性逐步切到 student` 方案，具体怎么实现、怎么测、结果怎样。

关联文档：

- `docs/ppt_materials_20260409.md`
- `docs/bw_sanity_critic_value_followup_20260410.md`
- `docs/bw_env_modification_followup_20260411.md`
- `docs/bw_critic_path_feedback_experiments_20260411.md`
- `docs/bw_after_feedback_followup_20260411.md`
- `docs/structured_bw_two_gu_investigation_20260412.md`


## 1. 要解决的问题

目标不是“再调一点 PPO 超参”，而是解决这个更具体的问题：

- `BW` actor 需要的是“当前这一步带宽动作相对另一个动作好多少”的 credit。
- 旧路径给它的是 `state-value critic -> return/bootstrap -> raw_advantage -> PPO`。
- 在多步 queue 闭环下，这条链会把一个很小的动作差值，埋进几个大数的相减里。

最直接的坏例子在 `docs/structured_bw_two_gu_investigation_20260412.md` 对应的 probe 数据里已经看到：

- `u0004 sample 7`
- `true_adv_mc = +3.845`
- `raw_advantage = -198.046`

也就是一个真实上更好的动作，被旧的 value/return 链直接翻成了巨负 advantage。


## 2. 最早的现象：`T=1/2` 能学，`T=5` 开始掉队，`T=10` 明显不行

最早的简单环境 sweep 已经把这个分界点钉住了，见 `docs/ppt_materials_20260409.md`。

同口径 `u0030` checkpoint eval：

| T | learned reward | heuristic reward | reward差值 |
| --- | ---: | ---: | ---: |
| 1 | 0.682 | 0.576 | +0.106 |
| 2 | 1.133 | 1.048 | +0.085 |
| 5 | 2.212 | 2.249 | -0.036 |
| 10 | 3.344 | 3.939 | -0.595 |

最后一轮训练指标也支持这个判断：

| T | env_reward_mean | approx_kl_bw | clip_frac_bw | bw_kappa_mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6388 | 0.0176 | 0.317 | 49.56 |
| 2 | 0.5441 | 0.0670 | 0.365 | 51.00 |
| 5 | 0.4546 | 0.0474 | 0.360 | 40.52 |
| 10 | 0.3252 | 0.00387 | 0.015 | 19.45 |

结论可以压缩成一句话：

- `T=1,2` 能学。
- `T=5` 开始掉队。
- `T=10` 时 PPO 自己的更新强度已经明显塌了。

这说明问题不是“一步 BW 学不会”，而是随着多步闭环性增强，当前 `state-value PPO` 接口开始失效。


## 3. 为什么旧路径会失效

### 3.1 动作差值很小，但 critic/value 链的误差很大

旧 BW 路径真正送进 actor 的，不是“动作差值”本身，而是沿着 return 链构造出来的：

```text
delta_t^bw = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t^bw = sum_l (gamma * lambda)^l * delta_{t+l}^bw
return_target_t = V(s_t) + A_t^bw
raw_advantage_t = return_target_t - V(s_t)
```

也就是说，`raw_advantage` 一开始就不是“某个动作比策略平均值好多少”的直接读数，而是：

- 当前状态 baseline `V(s_t)` 的误差
- 后续 bootstrap `V(s_{t+1}), V(s_{t+2}), ...` 的误差
- 以及真正关心的动作差值

三者混在一起的结果。

如果把它和 probe 里更接近真实动作级信用的量对齐，式子可以直接拆成：

```text
raw_advantage
= return_target - value_pred
= true_adv_mc
+ (policy_q_mc - value_pred)
+ (return_target - sampled_q_mc)
```

其中：

- `true_adv_mc = sampled_q_mc - policy_q_mc`
  - 这是当前动作相对当前策略平均动作的近似真实 advantage
- `policy_q_mc - value_pred`
  - 这是当前状态 baseline 误差
- `return_target - sampled_q_mc`
  - 这是 return/bootstrap 链对 sampled-action Q 的误差

最典型的是旧 baseline `u0004 sample 7`：

- `true_adv_mc = +3.845`
- `value_pred = -71.910`
- `policy_q_mc = -253.569`
- `sampled_q_mc = -249.724`
- `return_target = -269.957`

代回去就是：

```text
raw_advantage
= +3.845
+ (-253.569 - (-71.910))
+ (-269.957 - (-249.724))
= +3.845 - 181.658 - 20.233
= -198.046
```

也就是：

- 真实动作差值其实只有 `+3.845`
- 但仅仅 baseline 误差就有 `-181.658`
- return 链额外又带来 `-20.233`

最后 actor 看到的不是“这是个略好的动作”，而是“这是个巨负 advantage”。

`docs/bw_sanity_critic_value_followup_20260410.md` 里的 heldout 量级对比，其实和这个分解是同一个结论：

- `T1` heldout value RMSE 约 `0.028`
- `T10 hotspot/hetero` heldout value RMSE 约 `0.508`
- 真正有用的 action advantage 通常只有 `0.02 ~ 0.04`

也就是到 `T10` 时，哪怕单看 heldout RMSE，`V` 的误差也已经和动作级 signal 不在一个量级；而一旦再经过 `return/bootstrap -> raw_advantage` 这条链，误差只会被进一步放大。

### 3.2 critic 的更新也确实会“学歪普通样本”

我后来直接把 baseline run 中某一批固定样本拿出来，看 critic 的 16 个 step 内到底怎么学，结果在：

- `runs/tmp/critic_timeline_summary_any_batch.json`
- `runs/tmp/critic_timeline_u4_like_seed3042.json`

最典型的 early case 是 `state_after_update=3`：

- 训练前：
  - `pred_mean = -68.93`
  - `pred_std = 1.51`
  - `target_mean = -94.12`
  - `target_std = 38.54`
- 训练后：
  - `pred_mean = -108.78`
  - `pred_std = 5.69`
  - `MSE: 2042 -> 1368`
  - `EV: 0.053 -> 0.224`
  - 但 `MAE: 27.26 -> 33.45`
  - `65%` 的样本绝对误差反而变大

更具体地看 quartile：

| quartile | target_mean | mae_pre | mae_post | improved_frac |
| --- | ---: | ---: | ---: | ---: |
| q1 最负 | -153.30 | 82.75 | 37.58 | 1.00 |
| q2 | -91.40 | 22.04 | 19.81 | 0.40 |
| q3 | -67.16 | 1.14 | 37.54 | 0.00 |
| q4 最不负 | -64.61 | 3.10 | 38.88 | 0.00 |

这说明 critic 并不是“完全不学”，而是：

- 会优先迁就最极端的负 target。
- 主要做的是整批输出整体下移。
- 一大批原来已经比较准的普通状态，反而会被学坏。

这也是为什么旧的 `raw_advantage = return_target - V` 很容易错：它并不是一个稳定的小残差。


## 4. 两条尝试路线

### 4.1 路线 A：尽量修 critic / value 路径

这条线已经做过不少尝试，见 `docs/bw_critic_path_feedback_experiments_20260411.md`：

- critic warmup 再更新 actor
- warmup 后重算 advantage
- `lambda = 1`
- `bw_episode_mc`
- critic replay bank
- target standardization / running standardization / PopArt
- 更大 critic
- global features
- flow proxy / reward proxy

这条线的结论不是“完全没帮助”，而是：

- 可以把 heldout RMSE 从 baseline 的 `3.624` 压到 `0.6` 左右
- 但这仍然比局部动作信号大一个数量级以上
- actor eval 并没有稳定恢复正常

最关键的一句其实是：

- 修 `V(s)` 能缓解数值问题
- 但不保证 actor 就拿到了正确的动作差值

### 4.2 路线 B：增强 BW leverage，让动作影响更明显

这条线见 `docs/bw_env_modification_followup_20260411.md`，主要做了两类事：

1. 补 observation proxy
- `arrival_rate`
- `recent_arrival`
- `recent_service`
- `queue_headroom`

2. 增强 leverage 本身
- 更强更久的 hotspot
- 更紧的 queue 上限
- 小幅 relay / urgency / service-gap 风险

这条线的结论也比较清楚：

- observation proxy 本身不改 leverage，只改可见性
- 真正让 leverage 变强的是 `hotspot / queue / relay`
- 但即使 leverage 变强，plain `state-value PPO` 仍然不稳

所以最终不是“环境里根本没有 BW leverage”，而是：

- leverage 有
- 但对旧的 `V(s)` credit 路径来说仍然太弱、太局部


## 5. 动作影响到底持续几步：queue / reward impulse 测试

为了回答“当前步 BW 动作的后效到底有多长”，后来又做了一个分支冲击实验，结果在：

- `runs/tmp/queue_reward_impulse_u4_like_seed3042.json`

做法是：

- 从同一个 BW snapshot 出发
- 分支 A 执行 `sampled_action`
- 分支 B 执行当前 actor 的 deterministic 动作
- 从第 2 步开始，两条分支都跟随同一个 deterministic future policy
- 记录每一步的 GU queue 差和 reward 差

100 个 BW 样本的汇总：

| k | mean_queue_l1 | median_queue_l1 | frac(queue_l1 > 1e5) | mean_abs_reward_delta | mean_abs_discounted_cum_delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 156356 | 67453 | 0.410 | 0.197 | 0.197 |
| 2 | 124848 | 45458 | 0.356 | 0.202 | 0.384 |
| 3 | 96754 | 0 | 0.275 | 0.199 | 0.566 |
| 4 | 88788 | 0 | 0.271 | 0.232 | 0.827 |
| 5 | 89493 | 0 | 0.267 | 0.243 | 1.093 |

这里判断“动作影响持续几步”，看的不是一句泛泛的“感觉还在影响”，而是三件具体的量：

1. `median_queue_l1`
   - 看典型样本里，分支 A/B 的 queue 扰动什么时候基本被后续策略纠正掉。
2. `frac(queue_l1 > 1e5)`
   - 看尾部样本里，还有多少状态在第 `k` 步仍然存在明显 queue 扰动。
3. `mean_abs_discounted_cum_delta`
   - 看如果 horizon 只截到第 `k` 步，累计 reward 差到底吃到了多少。

从这张表里可以直接读出：

- 到 `k=3` 时，`median_queue_l1 = 0`
  - 说明对“典型样本”来说，这一步 BW 动作带来的 queue 扰动大多在 `2~3` 步内就被纠正掉了。
- 但到 `k=3/4/5` 时，`frac(queue_l1 > 1e5)` 仍然是 `0.275 / 0.271 / 0.267`
  - 说明还有大约四分之一的尾部样本，动作影响会继续拖到 `3~5` 步，甚至更久。
- `mean_abs_discounted_cum_delta` 从 `0.197 -> 0.384 -> 0.566 -> 0.827 -> 1.093`
  - 说明如果只看 `h=1` 的当前步收益，只吃到了很小一部分后效。
  - 到 `h=3` 时已经能覆盖典型样本的大部分后效，但到 `h=5` 仍然还能继续吃到尾部样本的延迟影响。

代表性样本：

- `rel_idx=2`
  - `queue_l1_by_k = [0, 0, 0, 0, 0]`
  - 说明 sampled 和 det 在这个状态下几乎等价
- `rel_idx=92`
  - `queue_l1_by_k = [562429, 405934, 0, 0, 0]`
  - 典型的 2 步内消退
- `rel_idx=40`
  - `queue_l1_by_k = [455377, 455624, 455955, 456291, 456627]`
  - 属于少数会持续 5 步以上的尾部样本

所以“动作影响到底持续几步”最后不是拍脑袋定的，而是从数据里读出来的：

- `h=1` 明显太短
  - 因为 reward 后效到 `k=3/5` 还在继续累计
- 也没必要默认走很长 horizon
  - 因为典型样本的 queue 扰动在 `2~3` 步就已经归零
- 对这个 BW 问题，更准确的说法是：
  - “典型样本的有效影响大约 `2~3` 步，尾部样本能拖到 `3~5` 步”

这也是后来选择短 horizon delta，而不是继续追长 horizon `V(s)` 的直接依据。


## 6. 为什么短 horizon `V_h(s)` 还是不如短 horizon `delta`

这一步我直接做了两个实验：

- 短 horizon `V_h(s)`：`bw_nstep(h=3)`，actor 继续用 `raw_adv = G_3 - V_h`
- 短 horizon `delta`：直接用 `branch_delta_h3` 覆盖 actor advantage

对应 run：

- `runs/structured_short/two_gu_t10_probe_u10_vh3_rawadv_fg`
- `runs/structured_short/two_gu_t10_probe_u10_deltah3_rawadv_fg`

probe 汇总：

| run | updates | net panel reward delta | mean delta_actor_reward_mean | positive update frac | mean trueAdv->dlogp |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline GAE | 10 | -6.93 | -0.693 | 0.30 | 0.034 |
| `V_h3` | 10 | +13.97 | +1.397 | 0.70 | 0.077 |
| direct `branch_delta_h3` | 10 | +29.22 | +2.922 | 0.90 | 0.479 |

结论是：

- `V_h3` 确实比长 horizon `V` 好
- 但它仍然是 `state value`
- actor 仍然要从 `sample return - baseline` 里读动作差值
- `branch_delta_h3` 直接学的就是“当前动作相对当前默认动作的增量价值”

所以它更对题。


## 7. 这次选择的方案

### 7.1 目标

这次不是直接上“student 全接管”，而是先做一个更稳的过渡版本：

- `teacher` 继续提供正确的短 horizon 动作差值
- `student` 并行蒸馏 teacher
- actor 随着 student 和 teacher 的相关性上升，逐步从 teacher 切到 student

### 7.2 teacher 定义

这次 teacher 选的是：

```text
branch_delta_h3
= Q_3(s_t, a_sampled ; follow_det)
- Q_3(s_t, a_det ; follow_det)
```

其中：

- `a_sampled` 是 rollout 里真实采到的 BW 动作
- `a_det` 是当前 actor 的 deterministic BW 动作
- `follow_det` 表示从第 2 步开始，两条分支都跟随同一个 deterministic future policy
- reward 用的是当前环境 reward，也就是 `weighted_workload_level`

### 7.3 student 定义

student 是一个新的 learned delta critic，学的是：

```text
pred_delta(s_local, a_sampled, a_ref) ~= branch_delta_h3
```

它不是旧的 `V_bw(s)` 头，而是新加的 `bw_delta` 头。

输入：

- `BW local state`
- `sampled_action`
- `ref_action`

输出：

- 一个标量 `pred_delta`

### 7.4 teacher-student actor advantage

actor 用的 advantage 不再是 `compute_returns_and_advantages()` 的旧值，而是：

```text
A_actor = (1 - alpha) * A_teacher + alpha * A_student
```

这里：

- `A_teacher = branch_delta_h3`
- `A_student = pred_delta`
- `alpha` 由当前 batch 上 `corr(student, teacher)` 决定

这次固定的阈值方案是：

- `corr_low = 0.6`
- `corr_high = 0.9`
- `mix_power = 1.0`

也就是：

- `corr <= 0.6`：actor 只吃 teacher
- `corr >= 0.9`：actor 完全吃 student
- 中间线性插值

### 7.5 训练时序

每个 update 的顺序是：

1. 对当前 batch 现算 `branch_delta_h3` teacher 标签
2. 先做 `32` 轮 delta critic warmup
3. 用 warmup 后的 `student` 预测值和 `teacher` 混合，覆盖 actor advantage
4. 再做 PPO actor update

### 7.6 这次还额外做的一个关键改动

在 `BW-only + delta_critic / delta_teacher_student` 这条线上，已经把旧 BW value 路径完全绕开：

- 不再给 BW actor/BW critic 跑 `compute_returns_and_advantages`
- 不再为 BW 训练 `V_bw`
- 不再让 BW 依赖 bootstrap value

也就是说，这条线现在对 BW 来说是真正的 `delta-only`。

对应代码主要在：

- `sagin_marl/rl/structured_critic.py`
- `sagin_marl/rl/structured_bw_update_direction.py`
- `sagin_marl/rl/structured_mappo.py`
- `sagin_marl/rl/structured_train.py`
- `scripts/train_structured.py`

### 7.7 一个实现 bug 也在这次被顺手修掉了

最开始的 `delta-only` 快路径里，BW actor update 被我漏掉了，导致：

- `policy_loss ≈ 0`
- `approx_kl_bw ≈ 0`
- `branch->dlogp ≈ 0`

后来修掉以后，learned delta 的首个 update 才真正开始动起来。


## 8. 怎么测试的

### 8.1 smoke

先跑 1 update smoke：

```powershell
.\.venv\Scripts\python.exe scripts\train_structured.py `
  --config configs/tmp/structured_bw_sanity_1uav_2gu_t10_stronggap_probe_ppo_trueadv_rawadv.yaml `
  --run_dir runs/tmp/smoke_delta_teacher_student_w32_u1 `
  --updates 1 --num_envs 1 --vec_backend sync --device cpu `
  --bw_actor_advantage_override_mode delta_teacher_student `
  --bw_delta_critic_horizon 3 `
  --bw_delta_critic_samples 1 `
  --bw_delta_critic_ref_mode deterministic `
  --bw_delta_critic_follow_policy_mode deterministic `
  --bw_delta_critic_warmup_epochs 32 `
  --bw_delta_teacher_student_corr_low 0.6 `
  --bw_delta_teacher_student_corr_high 0.9 `
  --bw_delta_teacher_student_mix_power 1.0
```

结果：

- `bw_delta_student_teacher_corr = 0.880`
- `bw_delta_actor_mix_alpha = 0.933`
- `branch->dlogp = 0.899`
- `true->dlogp = 0.852`
- `delta_actor_reward_mean = +3.765`

这一步已经说明：

- student 32 轮 warmup 后，首个 update 就能接近 teacher
- actor 也真的顺着这个 signal 动了

### 8.2 长训

再跑 20 updates：

```powershell
.\.venv\Scripts\python.exe scripts\train_structured.py `
  --config configs/tmp/structured_bw_sanity_1uav_2gu_t10_stronggap_probe_ppo_trueadv_rawadv.yaml `
  --run_dir runs/structured_short/two_gu_t10_probe_u20_teacherstudent_h3_w32_fg `
  --updates 20 --num_envs 1 --vec_backend sync --device cuda `
  --bw_actor_advantage_override_mode delta_teacher_student `
  --bw_delta_critic_horizon 3 `
  --bw_delta_critic_samples 1 `
  --bw_delta_critic_ref_mode deterministic `
  --bw_delta_critic_follow_policy_mode deterministic `
  --bw_delta_critic_warmup_epochs 32 `
  --bw_delta_teacher_student_corr_low 0.6 `
  --bw_delta_teacher_student_corr_high 0.9 `
  --bw_delta_teacher_student_mix_power 1.0
```

输出目录：

- `runs/structured_short/two_gu_t10_probe_u20_teacherstudent_h3_w32_fg/metrics.csv`
- `runs/structured_short/two_gu_t10_probe_u20_teacherstudent_h3_w32_fg/update_direction_probe.csv`


## 9. 结果

### 9.1 student 很快就追上 teacher

来自 `metrics.csv`：

- `student-teacher corr` 平均 `0.954`
- 最小值 `0.861`
- 最大值 `0.997`
- `mix alpha` 平均 `0.991`
- 第一次 `alpha = 1.0` 出现在 `update 3`
- 从 `update 3` 起基本一直是 `1.0`

也就是：

- `32` 轮 warmup 没有白加
- student 很快进入“可跟踪 teacher”的区间

### 9.2 前 10 个 updates，teacher-student 基本追平 direct teacher

把 `teacher-student` 的前 10 个 update 和之前 direct `branch_delta_h3` 结果对齐：

| run | updates | net panel reward delta | mean delta_actor_reward_mean | positive update frac | mean trueAdv->dlogp | neg trueAdv->dlogp updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline GAE | 10 | -6.93 | -0.693 | 0.30 | 0.034 | 5 |
| direct `branch_delta_h3` | 10 | +29.22 | +2.922 | 0.90 | 0.479 | 1 |
| `teacher-student` 前10 | 10 | +28.99 | +2.899 | 0.80 | 0.496 | 0 |

这一步很关键，因为它说明：

- 修完 actor-update bug 以后
- 只要 student 足够接近 teacher
- `teacher-student` 的学习性质就能接近 direct teacher

### 9.3 20 个 updates 的整体结果

完整 20-update 汇总：

| run | updates | net panel reward delta | mean delta_actor_reward_mean | positive update frac | mean trueAdv->dlogp | neg trueAdv->dlogp updates | mean branch->dlogp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline GAE | 10 | -6.93 | -0.693 | 0.30 | 0.034 | 5 | -0.0003 |
| `V_h3` | 10 | +13.97 | +1.397 | 0.70 | 0.077 | 4 | 0.115 |
| direct `branch_delta_h3` | 10 | +29.22 | +2.922 | 0.90 | 0.479 | 1 | 0.557 |
| direct `branch_delta_h3` | 20 | +28.88 | +1.444 | 0.60 | 0.390 | 1 | 0.474 |
| `true_adv_mc` override | 10 | +28.29 | +2.829 | 0.90 | 0.538 | 1 | 0.593 |
| `teacher-student` | 20 | +28.74 | +1.437 | 0.65 | 0.326 | 3 | 0.400 |

也就是：

- direct teacher 和 `teacher-student` 在完整 `20 updates` 下最终 fixed-panel 效果几乎一样
  - direct `20`: `+28.88`
  - teacher-student `20`: `+28.74`
- `teacher-student` 最终固定 panel reward 从 `-70.45` 提到 `-41.71`
- 净提升 `+28.74`
- 明显好于 baseline
- 也明显好于短 horizon `V_h3`

但同时也能看到：

- direct teacher 和 `teacher-student` 两条线在 20 个 update 后半段都开始重新出现波动
- 不是每一轮都继续变好
- direct teacher 的 `mean trueAdv->dlogp` 也从 10-update 的 `0.479` 掉到 20-update 的 `0.390`
- `teacher-student` 的 `mean trueAdv->dlogp` 从前 10 的 `0.496` 掉到 20-update 平均的 `0.326`

这说明：

- 现在 student 拟合 teacher 这件事，已经不是主要瓶颈
- 因为哪怕完全 direct teacher 的 `20-update` 版，后半段也会开始波动
- 所以后半段残余问题更像是 PPO surrogate / batch update / closed-loop mismatch

### 9.4 正式 checkpoint eval 口径

为了避免只停留在 fixed panel probe，这次又补做了一轮正式 eval：

- `episodes = 32`
- `policy_mode = deterministic`
- `episode_seed_base = 42000`
- `fixed_policy = queue_aware_bw`

结果写在：

- `runs/structured_short/two_gu_t10_probe_u20_teacherstudent_h3_w32_fg/checkpoint_eval.csv`
- `runs/structured_short/two_gu_t10_probe_u20_deltah3_rawadv_fg/checkpoint_eval.csv`

同时也导出了逐 episode 明细：

- `runs/structured_short/two_gu_t10_probe_u20_teacherstudent_h3_w32_fg/eval_trained_final.csv`
- `runs/structured_short/two_gu_t10_probe_u20_deltah3_rawadv_fg/eval_trained_final.csv`

汇总如下：

| run | actor reward | fixed reward | actor weighted_level | fixed weighted_level | actor processed | fixed processed | actor backlog | fixed backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher-student` | `-87.281` | `-91.524` | `-90.494` | `-94.681` | `1.3793` | `1.3543` | `1.9450` | `2.0626` |
| direct `branch_delta_h3` | `-87.313` | `-91.524` | `-90.505` | `-94.681` | `1.3802` | `1.3543` | `1.9439` | `2.0626` |

结论：

- 在正式 `checkpoint_eval` 口径下，两条线都已经稳定超过 `queue_aware_bw`
- `teacher-student` 和 direct teacher 在正式 eval 上也几乎一样
- 因此“超过 heuristic”这件事，不只是 fixed panel probe 成立，在完整 episode 评测口径下也成立


## 10. 当前判断

### 10.1 这次方案已经解决了什么

已经坐实的结论有三条：

1. 旧问题的主故障点确实在 `critic/value -> advantage -> actor` 这条链上。
2. 对这个 BW 问题，短 horizon `delta` 明显比短 horizon `V_h(s)` 更对题。
3. `teacher 更新 actor + student 并行蒸馏 + 按相关性渐进切 student` 在这个简单环境里是成立的。

### 10.2 这次方案还没有解决什么

这次还没有解决的有两条：

1. 现在这版仍然是 dense teacher。
   - 每个 update 还是会对当前 batch 全量计算 `branch_delta_h3`
   - 所以这还不是“节省 rollout 成本”的最终版本
   - 这次的重点是先证明 student 可以逐步接 teacher，而不是先优化算力

2. 后半段 update 仍有残余不稳。
   - 由于 student-teacher corr 一直很高，这已经不太像是“student 没学会”
   - 更像是 teacher / PPO surrogate / closed-loop policy improvement 之间仍然有剩余不匹配

### 10.3 为什么当前还是选择这个方案

在现在这个阶段，我仍然认为这是最合理的下一步主线，因为它同时满足：

- 不再依赖旧的 `V_bw`
- 对准了动作级 credit，而不是 state value
- 已经在简单环境里证明能学
- 可以自然往后扩展成：
  - 稀疏 teacher
  - 下采样 teacher
  - 每 `K` 轮刷新 teacher
  - 最终 student 主导

相比之下：

- 继续修 `V(s)` 仍然可能改善数值，但没有直接对准动作差值
- 继续只做环境 leverage 增强，也不能保证 plain `state-value PPO` 就会自然恢复


## 11. 最后一句结论

这次 follow-up 的核心结论是：

- `BW` 这条线的问题，不只是“critic 误差大”，更是“actor 需要的是短 horizon 动作差值，而旧链路给的是 state value”
- `branch_delta_h3` 作为 teacher 是对的
- `learned delta critic + teacher-student gradual switch` 在简单环境里已经跑通，而且前 10 个 update 基本追平 direct teacher
- 现在剩下的主问题，已经从“student 学不会 teacher”转成了“当 teacher/student 都已经对时，PPO 后半段为什么还会出现 closed-loop 波动”


## 12. 后续降噪实验：更大 on-policy batch、更多 branch samples、batch gate

这一步的目的，是验证前面的判断：

- 后期会漂，不是因为 horizon 选错了，而更像是“真实改进信号在变小，但采样噪声 / surrogate 误差没有同步变小”
- 所以更值得优先试的，不是继续调 `h=1/2/3`，而是直接降噪

### 12.1 实验设置

三条 run 都使用：

- 简单环境：`1 UAV / 2 GU / T=10 / 只训 BW`
- actor credit：direct `branch_delta_h3`
- `ppo_epochs=1`
- `num_mini_batch=1`
- 关闭每 update 的 `DirProbe`
- 开启周期性 `checkpoint_eval`
- `checkpoint_eval_interval_updates=5`
- `checkpoint_eval_episodes=32`
- `checkpoint_eval_fixed_policy=queue_aware_bw`
- `checkpoint_eval_save_best_models=true`
- `checkpoint_eval_use_best_as_final=true`

共同改动：

- `num_envs=8`
- `vec_backend=subproc`
- `device=cuda`

对照矩阵：

| run | num_envs | branch_samples | batch gate | updates |
| --- | ---: | ---: | --- | ---: |
| `env8_bs1` | `8` | `1` | 无 | `30` |
| `env8_bs4` | `8` | `4` | 无 | `20` |
| `env8_bs1_gate039` | `8` | `1` | `branch_snr < 0.39` 时跳过 actor update | `30` |

其中 batch gate 用的指标是：

```text
branch_snr = branch_abs_mean / branch_std
```

这里：

- `branch_abs_mean = mean(|branch_delta_h3|)`
- `branch_std = std(branch_delta_h3)`

当 `branch_snr` 太低时，说明这轮 batch 上 teacher 的有效信号已经接近它自己的噪声尺度，于是只跳过这次 BW actor update，不跳过 rollout / teacher 计算 / critic 更新。

### 12.2 结果总表

| run | best eval update | best reward | last eval reward | heuristic reward | 前 10 次 snr 均值 | 后 5 次 snr 均值 | gate 触发次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `env8_bs1` | `25` | `-87.359` | `-87.364` | `-91.524` | `0.538` | `0.376` | `0` |
| `env8_bs4` | `20` | `-87.515` | `-87.515` | `-91.524` | `0.541` | `0.390` | `0` |
| `env8_bs1_gate039` | `20` | `-87.369` | `-87.645` | `-91.524` | `0.538` | `0.383` | `5` |

对应文件：

- `runs/structured_short/two_gu_t10_env8_bs1_1x1_ckpteval_fg/metrics.csv`
- `runs/structured_short/two_gu_t10_env8_bs1_1x1_ckpteval_fg/checkpoint_eval.csv`
- `runs/structured_short/two_gu_t10_env8_bs4_1x1_ckpteval_fg/metrics.csv`
- `runs/structured_short/two_gu_t10_env8_bs4_1x1_ckpteval_fg/checkpoint_eval.csv`
- `runs/structured_short/two_gu_t10_env8_bs1_1x1_gate039_ckpteval_fg/metrics.csv`
- `runs/structured_short/two_gu_t10_env8_bs1_1x1_gate039_ckpteval_fg/checkpoint_eval.csv`

### 12.3 如何解读

先看 `num_envs=8` 本身：

- `env8_bs1` 的 best eval 到了 `-87.359`
- 明显好于 heuristic 的 `-91.524`
- 也和之前 `num_envs=1` 下 direct teacher 的最好结果已经很接近

这说明把 on-policy batch 从大约 `10` 个 episode 提到大约 `80` 个 episode 之后，训练确实更稳，至少不会立刻比原来差。

但再看 `branch_samples=4`：

- `env8_bs4` 的 best eval 是 `-87.515`
- 比 `env8_bs1` 略差
- 训练耗时却接近翻倍

所以在这个简单环境里：

- 增加 `branch_samples` 并没有明显压掉 late-stage 漂移
- teacher 的主要噪声来源并不全在单次 branch MC 上
- 更像是 batch 本身已经足够大了，而 closed-loop / surrogate 误差开始成为主导

最后看 `batch gate`：

- `env8_bs1_gate039` 在 `u19/u21/u26/u27/u30` 触发了 `5` 次 skip
- 这些被跳过的 update 的 `policy_loss` 都变成了 `0.0`
- 最好的 checkpoint 出现在 `u20`
- 其 best eval `-87.369` 和 no-gate 的 best `-87.359` 几乎一样

也就是说：

- `branch_snr < 0.39` 这个 gate 确实已经能识别一部分 late-stage 低置信 batch
- 它也确实会真的阻止 actor 更新
- 但它还没有把最好结果再往上推
- 更像是把“最好点”提前锁在 `u20` 附近，而不是创造更高的上限

### 12.4 当前结论

这组 follow-up 给出的更具体判断是：

1. **更大 on-policy batch 是值得的。**
   - `num_envs=8 + subproc + cuda` 这条线是成立的
   - 至少在这个简单环境里，不会伤性能

2. **增加 `branch_samples` 的收益很小。**
   - `1 -> 4` 基本没带来更好的 eval
   - 但明显增加了 teacher rollout 成本

3. **batch gate 已经能工作，但还不是决定性改进。**
   - 它能识别低 `branch_snr` batch
   - 能真的跳过 actor update
   - 但目前更像“防训过头的保险丝”，还不是把最优性能继续抬高的主因

4. **现在最像真的结论，是“更大 batch 有帮助，但 late-stage 的剩余问题已经不主要是 teacher MC 噪声”。**
   - 因为 `num_envs=8` 后结果已经不错
   - `branch_samples=4` 没有继续明显提升
   - 所以剩下的瓶颈更像 PPO surrogate / closed-loop policy improvement 本身
