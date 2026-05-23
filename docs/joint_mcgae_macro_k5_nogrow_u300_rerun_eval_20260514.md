# Joint MC-GAE BW Macro K=5 300 Updates 重跑评估记录

## 1. 训练设置

```text
run_dir = runs/diagnostics/joint_mcgae_macro_k5_nogrow_u300_rerun_20260514
config = configs/tmp/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
env = 3 UAV / 20 GU / T=250
num_envs = 64
updates = 300
reward_mode = positive_weighted_workload_level
access_bw_decision_interval = 5
sat_decision_interval = 1
stage_actor_lr_grow_factor = 1.0
```

含义：

```text
只用当前代码和当前记录字段重跑 BW macro K=5。
SAT 保持 K=1。
KL early stop 和 LR decay 保持开启。
LR grow 关闭。
```

## 2. 训练期摘要

| stage | first mc_return | u100 | u150 | u200 | u250 | final | best | best update | last10 | last50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accel | 11.3203 | 11.8354 | 14.1100 | 16.2146 | 17.6955 | 16.1493 | 18.5717 | 290 | 17.2407 | 17.4635 |
| sat | 11.3203 | 11.8354 | 14.1100 | 16.2146 | 17.6955 | 16.1493 | 18.5717 | 290 | 17.2407 | 17.4635 |
| bw | 11.4774 | 11.9957 | 14.2979 | 16.4218 | 17.9183 | 16.3576 | 18.8038 | 290 | 17.4598 | 17.6845 |

新增 episode reward 统计已经生效：

| metric | first | u100 | u200 | u250 | final | best | best update | last10 | last50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rollout_episode_reward_mean | 33.5310 | 34.5018 | 46.0302 | 49.7854 | 46.0699 | 51.9637 | 290 | 48.6650 | 49.2186 |
| rollout_completed_episode_reward_mean | 33.7550 | 34.9392 | 46.0302 | 49.7854 | 46.0699 | 51.9637 | 290 | 48.7706 | 49.2467 |

安全：

```text
rollout_collision_rate last50 = 0.0000075
rollout_episode_length_mean last50 ≈ 249.54
```

速度：

```text
last10 iteration_sec ≈ 47.24 s/update
last50 iteration_sec ≈ 49.77 s/update
```

## 3. Native Deterministic Eval

评估口径：

```text
episodes = 64
num_envs = 64
episode_seed_base = 900000
policy_mode = deterministic
access_bw_decision_interval = 5
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
| rrr | 45.2764 | +0.0000 | 0.8860 | 0.0847 | 6.8239 | 0.7596 | 0.0000 |
| Arr | 46.5334 | +1.2570 | 0.9056 | 0.0641 | 6.4963 | 0.7710 | 0.0000 |
| rSr | 45.4156 | +0.1391 | 0.8908 | 0.0847 | 6.4520 | 0.4602 | 0.0000 |
| rrB | 52.6378 | +7.3614 | 0.8805 | 0.0954 | 5.9896 | 0.7596 | 0.0000 |
| ASr | 46.3178 | +1.0414 | 0.9000 | 0.0736 | 6.2895 | 0.4540 | 0.0000 |
| ArB | 55.9214 | +10.6450 | 0.8948 | 0.0801 | 5.7988 | 0.7630 | 0.0000 |
| rSB | 52.5977 | +7.3213 | 0.8850 | 0.0955 | 5.6995 | 0.4560 | 0.0000 |
| ASB | 55.1812 | +9.9047 | 0.8913 | 0.0871 | 5.6132 | 0.4513 | 0.0000 |

## 5. Best Stage Heads 8 组合

这里的 best 是在 final checkpoint 上按 stage 替换：

```text
A = best_stage_heads/best_accel.pt
S = best_stage_heads/best_sat.pt
B = best_stage_heads/best_bw.pt
```

| combo | reward | vs rule | processed | drop | pre_backlog | sat_overlap | collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rrr | 45.2764 | +0.0000 | 0.8860 | 0.0847 | 6.8239 | 0.7596 | 0.0000 |
| Arr | 45.2827 | +0.0063 | 0.8850 | 0.0782 | 6.8628 | 0.7790 | 0.0156 |
| rSr | 45.4388 | +0.1623 | 0.8916 | 0.0847 | 6.4064 | 0.5133 | 0.0000 |
| rrB | 57.6790 | +12.4026 | 0.8877 | 0.0887 | 5.7936 | 0.7596 | 0.0000 |
| ASr | 45.2804 | +0.0039 | 0.8910 | 0.0769 | 6.5416 | 0.5035 | 0.0156 |
| ArB | 58.3895 | +13.1130 | 0.8868 | 0.0812 | 5.9188 | 0.7739 | 0.0156 |
| rSB | 58.0850 | +12.8086 | 0.8932 | 0.0887 | 5.4039 | 0.5010 | 0.0000 |
| ASB | 59.2088 | +13.9324 | 0.8897 | 0.0831 | 5.6727 | 0.4953 | 0.0156 |

## 6. 结论

```text
K=5 在当前代码和当前记录字段下复现稳定。
BW 仍是主要收益来源：best rrB 比 rule 高 +12.40。
SAT / Accel 单独收益都小，但二者在 BW 基础上都有叠加：
  best rSB = 58.0850
  best ArB = 58.3895
  best ASB = 59.2088
K=5 明显强于 K=10。
```
