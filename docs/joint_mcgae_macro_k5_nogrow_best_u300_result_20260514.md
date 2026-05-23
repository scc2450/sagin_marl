# Joint MC-GAE Macro K=5 No-Grow 300 Updates 结果记录

## 1. Run

目录：

```text
runs/diagnostics/joint_mcgae_macro_k5_nogrow_best_u300_20260514
```

核心设置：

```text
env = 3 UAV / 20 GU / T=250
num_envs = 64
updates = 300
reward_mode = positive_weighted_workload_level
access_bw_decision_interval = 5
sat_decision_interval = 1
stage_actor_dynamic_lr_enabled = true
stage_actor_lr_grow_factor = 1.0
```

含义：

```text
仍允许 KL/clip 过大时 decay。
不再允许 LR grow。
BW 使用 macro K=5。
SAT 保持 K=1。
```

启动命令：

```powershell
$env:TEMP='D:\sagin_marl_tmp'
$env:TMP='D:\sagin_marl_tmp'
$env:TORCHINDUCTOR_CACHE_DIR='D:\sagin_cache\torchinductor'
$env:TRITON_CACHE_DIR='D:\sagin_cache\triton'
$env:SAGIN_MARL_NATIVE_CUDA_CACHE='D:\sagin_marl_native_cuda_cache'
.\.venv\Scripts\python.exe scripts\train_joint_mcgae.py `
  --config configs\tmp\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
  --run_dir runs\diagnostics\joint_mcgae_macro_k5_nogrow_best_u300_20260514 `
  --updates 300 `
  --rollout_env_steps 250 `
  --num_envs 64 `
  --device cuda `
  --torch_threads 1 `
  --access_bw_decision_interval 5 `
  --sat_decision_interval 1 `
  --save_every 50
```

## 2. 训练摘要

训练正常完成 300 updates，没有再出现 BW parity fail。

| stage | first mc_return | last mc_return | last10 mean | best | best update | lr first | lr last | max KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accel | 11.3203 | 16.9765 | 18.0604 | 19.3346 | 270 | 1e-4 | 5e-5 | 0.0545 |
| sat | 11.3203 | 16.9765 | 18.0604 | 19.3346 | 270 | 3e-4 | 3e-4 | 0.0346 |
| bw | 11.4774 | 17.1928 | 18.2867 | 19.5760 | 270 | 3e-4 | 1.5e-4 | 0.0540 |

保存的 per-stage best head：

```text
best_stage_heads/best_accel.pt  update=270  accel_mc_return_mean=19.3346
best_stage_heads/best_sat.pt    update=270  sat_mc_return_mean=19.3346
best_stage_heads/best_bw.pt     update=270  bw_mc_return_mean=19.5760
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

结果：

| policy | reward | vs final | vs rule | processed | drop | pre_backlog | sat_overlap | collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_rule_native | 45.2764 | -13.0595 | 0.0000 | 0.8860 | 0.0847 | 6.8239 | 0.7596 | 0.0000 |
| final_u300 | 58.3360 | 0.0000 | +13.0595 | 0.8970 | 0.0860 | 5.0909 | 0.9063 | 0.0000 |
| final + best_accel_only | 57.8431 | -0.4928 | +12.5667 | 0.8940 | 0.0880 | 5.3602 | 0.9071 | 0.0000 |
| final + best_sat_only | 58.3836 | +0.0476 | +13.1071 | 0.8978 | 0.0849 | 5.0323 | 0.8574 | 0.0000 |
| final + best_bw_only | 59.1939 | +0.8579 | +13.9175 | 0.9056 | 0.0770 | 5.2517 | 0.9065 | 0.0000 |
| final + all_best_heads | 59.3289 | +0.9929 | +14.0525 | 0.9030 | 0.0761 | 5.3709 | 0.8602 | 0.0156 |
| ckpt_u250 | 54.5906 | -3.7454 | +9.3141 | 0.8814 | 0.0978 | 6.2445 | 0.6489 | 0.0000 |

## 4. 单 Stage 效果

这轮单独替换 best head 的效果不是三个 stage 都同向变好。

```text
best_accel_only:
  reward 比 final 低 0.4928。
  说明按 accel_mc_return_mean 选出的 best accel head，在最终联合策略上下文里不一定更好。

best_sat_only:
  reward 比 final 高 0.0476。
  单独 SAT best head 有轻微正增益，但幅度很小。

best_bw_only:
  reward 比 final 高 0.8579。
  这是单 stage 替换里最明确的正增益，而且 collision=0。

all_best_heads:
  reward 比 final 高 0.9929。
  但 collision_episode_fraction=0.015625，也就是 64 个 episode 中有 1 个碰撞。
```

## 5. 当前结论

1. 关闭 LR grow 后，训练稳定性明显好于之前 grow-on 的 joint run。
2. 最终策略 `final_u300` 已经明显超过全规则 baseline：`58.3360` vs `45.2764`。
3. 单 stage 替换里，`best_bw_only` 是最有价值的，reward 提升最大且无碰撞。
4. `all_best_heads` reward 最高，但有轻微安全回退，不能直接说它比 `best_bw_only` 更稳。
5. `best_accel_only` 反而降低 final reward，说明 per-stage best metric 不能无条件当成 joint 最优选择。

当前更稳妥的候选：

```text
final_u300:
  稳定，无碰撞，reward=58.3360。

final + best_bw_only:
  更高 reward，无碰撞，reward=59.1939。

final + all_best_heads:
  最高 reward，reward=59.3289，但有 1/64 碰撞，需要继续确认安全性。
```

## 6. 8 种 Stage 组合对比

这里把三个 stage 分别记为：

```text
A = accel policy
S = sat policy
B = bw policy
r = rule baseline
```

规则 baseline 是：

```text
accel = cluster_center_queue_aware
sat   = queue_aware
bw    = queue_aware
```

评估口径同上：

```text
native deterministic eval
64 episodes
episode_seed_base = 900000
access_bw_decision_interval = 5
sat_decision_interval = 1
```

### 6.1 Final checkpoint 的 8 种组合

| combo | reward | vs rule | processed | drop | pre_backlog | sat_overlap | collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rrr | 45.2764 | +0.0000 | 0.8860 | 0.0847 | 6.8239 | 0.7596 | 0.0000 |
| Arr | 45.6283 | +0.3519 | 0.9012 | 0.0650 | 6.4558 | 0.7786 | 0.0156 |
| rSr | 45.3849 | +0.1084 | 0.8934 | 0.0847 | 6.5389 | 0.9156 | 0.0000 |
| rrB | 55.8731 | +10.5967 | 0.8823 | 0.0951 | 5.4985 | 0.7596 | 0.0000 |
| ASr | 45.7376 | +0.4612 | 0.9075 | 0.0658 | 6.1558 | 0.9190 | 0.0156 |
| ArB | 57.6140 | +12.3376 | 0.8917 | 0.0846 | 5.3591 | 0.7830 | 0.0000 |
| rSB | 56.6001 | +11.3237 | 0.8897 | 0.0951 | 5.2088 | 0.9058 | 0.0000 |
| ASB | 58.3363 | +13.0599 | 0.8970 | 0.0860 | 5.0909 | 0.9063 | 0.0000 |

### 6.2 Best stage heads 的 8 种组合

这里的 `best` 不是重新训练出的整套 checkpoint，而是在 `final_u300` base 上按 stage 替换：

```text
A 用 best_accel.pt
S 用 best_sat.pt
B 用 best_bw.pt
未启用 policy 的 stage 仍走 rule baseline
```

| combo | reward | vs rule | processed | drop | pre_backlog | sat_overlap | collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rrr | 45.2764 | +0.0000 | 0.8860 | 0.0847 | 6.8239 | 0.7596 | 0.0000 |
| Arr | 45.5675 | +0.2911 | 0.8868 | 0.0817 | 6.9512 | 0.7781 | 0.0000 |
| rSr | 45.4056 | +0.1292 | 0.8926 | 0.0847 | 6.4588 | 0.8649 | 0.0000 |
| rrB | 57.7219 | +12.4455 | 0.8904 | 0.0860 | 5.6830 | 0.7596 | 0.0000 |
| ASr | 45.8478 | +0.5714 | 0.8978 | 0.0768 | 6.4941 | 0.8641 | 0.0000 |
| ArB | 58.6875 | +13.4110 | 0.8986 | 0.0774 | 5.7569 | 0.7821 | 0.0000 |
| rSB | 58.8310 | +13.5545 | 0.8971 | 0.0860 | 5.3203 | 0.8596 | 0.0000 |
| ASB | 59.3289 | +14.0525 | 0.9030 | 0.0761 | 5.3709 | 0.8602 | 0.0156 |

### 6.3 从 8 种组合看单 stage 贡献

Final checkpoint：

```text
A only: +0.3519，但有 1/64 碰撞
S only: +0.1084
B only: +10.5967
```

Best stage heads：

```text
A only: +0.2911
S only: +0.1292
B only: +12.4455
```

所以这轮最明确的结论是：

```text
BW policy 是主要收益来源。
SAT policy 单独看收益很小，但和 BW 组合后有一定叠加。
Accel policy 单独收益小，final accel 还带来轻微碰撞风险；best accel 没碰撞，但收益仍小。
```

组合层面：

```text
best rrB = 57.7219
best ArB = 58.6875
best rSB = 58.8310
best ASB = 59.3289
```

这说明在 best-head 口径下：

```text
BW 是底座收益。
SAT + BW 比 BW only 再高 +1.1091。
Accel + BW 比 BW only 再高 +0.9656。
三者全开最高，但出现 1/64 碰撞。
```
