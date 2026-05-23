# Joint MC-GAE BW Macro K=10 300 Updates 评估记录

## 1. 训练设置

```text
run_dir = runs/diagnostics/joint_mcgae_macro_k10_nogrow_u300_20260514
config = configs/tmp/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
env = 3 UAV / 20 GU / T=250
num_envs = 64
updates = 300
reward_mode = positive_weighted_workload_level
access_bw_decision_interval = 10
sat_decision_interval = 1
stage_actor_lr_grow_factor = 1.0
```

含义：

```text
只把 BW macro K 从 5 改成 10。
SAT 保持 K=1。
KL early stop 和 LR decay 保持开启。
LR grow 仍关闭。
```

## 2. 训练期摘要

| stage | first mc_return | u100 | u200 | u250 | final | best | best update | last10 | last50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accel | 11.1934 | 11.9339 | 16.5316 | 17.1655 | 16.0674 | 17.6078 | 298 | 16.9601 | 16.9602 |
| sat | 11.1934 | 11.9339 | 16.5316 | 17.1655 | 16.0674 | 17.6078 | 298 | 16.9601 | 16.9602 |
| bw | 11.5417 | 12.3072 | 17.0012 | 17.6543 | 16.5331 | 18.0993 | 298 | 17.4436 | 17.4451 |

新增 episode reward 统计已经生效：

| metric | final | best | best update | last10 | last50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rollout_episode_reward_mean | 45.8497 | 49.5109 | 288 | 47.9018 | 47.9569 |
| rollout_completed_episode_reward_mean | 45.8497 | 49.5109 | 288 | 47.9148 | 47.9635 |

安全：

```text
rollout_collision_rate last50 = 0
rollout_episode_length_mean last10 ≈ 249.62
```

速度：

```text
last10 iteration_sec ≈ 46.04 s/update
```

## 3. Native Deterministic Eval

评估口径：

```text
episodes = 64
num_envs = 64
episode_seed_base = 900000
policy_mode = deterministic
access_bw_decision_interval = 10
sat_decision_interval = 1
```

规则 baseline：

```text
accel = cluster_center_queue_aware
sat   = queue_aware
bw    = queue_aware
```

## 4. Final Checkpoint 8 组合

| combo | reward | vs rule | processed | drop | pre_backlog | sat_overlap | collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rrr | 43.2102 | +0.0000 | 0.8749 | 0.0911 | 7.6579 | 0.7596 | 0.0000 |
| Arr | 43.3181 | +0.1079 | 0.8725 | 0.0890 | 7.9166 | 0.7714 | 0.0000 |
| rSr | 42.9567 | -0.2535 | 0.8686 | 0.0911 | 8.7074 | 0.9977 | 0.0000 |
| rrB | 51.0523 | +7.8421 | 0.8808 | 0.0947 | 6.1613 | 0.7596 | 0.0000 |
| ASr | 43.2674 | +0.0572 | 0.8737 | 0.0832 | 8.7738 | 0.9975 | 0.0000 |
| ArB | 51.2480 | +8.0378 | 0.8850 | 0.0860 | 6.2684 | 0.7828 | 0.0156 |
| rSB | 49.7723 | +6.5621 | 0.8756 | 0.0947 | 7.1821 | 0.9976 | 0.0000 |
| ASB | 50.1218 | +6.9116 | 0.8803 | 0.0865 | 7.1995 | 0.9976 | 0.0156 |

## 5. Best Stage Heads 8 组合

这里的 best 是在 final checkpoint 上按 stage 替换：

```text
A = best_stage_heads/best_accel.pt
S = best_stage_heads/best_sat.pt
B = best_stage_heads/best_bw.pt
```

| combo | reward | vs rule | processed | drop | pre_backlog | sat_overlap | collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rrr | 43.2102 | +0.0000 | 0.8749 | 0.0911 | 7.6579 | 0.7596 | 0.0000 |
| Arr | 43.4294 | +0.2192 | 0.8756 | 0.0863 | 7.8448 | 0.7754 | 0.0000 |
| rSr | 42.9370 | -0.2732 | 0.8681 | 0.0911 | 8.7845 | 0.9979 | 0.0000 |
| rrB | 51.3832 | +8.1730 | 0.8821 | 0.0934 | 6.1277 | 0.7596 | 0.0000 |
| ASr | 43.2517 | +0.0415 | 0.8733 | 0.0832 | 8.8668 | 0.9979 | 0.0000 |
| ArB | 52.1273 | +8.9171 | 0.8835 | 0.0875 | 6.3183 | 0.7790 | 0.0000 |
| rSB | 49.9144 | +6.7042 | 0.8761 | 0.0934 | 7.2421 | 0.9978 | 0.0000 |
| ASB | 50.7968 | +7.5866 | 0.8823 | 0.0843 | 7.3124 | 0.9979 | 0.0000 |

## 6. 与当前代码重跑 K=5 对比

K=5 对照来自：

```text
run_dir = runs/diagnostics/joint_mcgae_macro_k5_nogrow_u300_rerun_20260514
doc = docs/joint_mcgae_macro_k5_nogrow_u300_rerun_eval_20260514.md
```

训练期：

| K | bw best mc_return | bw best update | bw last50 mc_return | episode reward best | episode reward last50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 18.8038 | 290 | 17.6845 | 51.9637 | 49.2186 |
| 10 | 18.0993 | 298 | 17.4451 | 49.5109 | 47.9569 |

Native deterministic eval：

| K | rule reward | best rrB | best ArB | best rSB | best ASB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 45.2764 | 57.6790 | 58.3895 | 58.0850 | 59.2088 |
| 10 | 43.2102 | 51.3832 | 52.1273 | 49.9144 | 50.7968 |

当前结论：

```text
K=10 可以训练，且 BW 仍是主要收益来源。
但在当前代码重跑的同口径对照下，K=10 明显弱于 K=5。
K=5 的 BW-only 已经达到 57.6790，K=10 的 BW-only 只有 51.3832。
K=5 中 SAT / Accel 在 BW 基础上有叠加；K=10 的叠加弱很多。
因此当前更推荐继续使用 BW macro K=5。
```
