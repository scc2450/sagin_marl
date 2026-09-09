# Joint MC-GAE K=5 Danger Imitation 消融记录

## 1. 对照设置

共同设置：

```text
config = configs/tmp/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
env = 3 UAV / 20 GU / T=250
num_envs = 64
updates = 300
reward_mode = positive_weighted_workload_level
access_bw_decision_interval = 5
sat_decision_interval = 1
stage_actor_lr_grow_factor = 1.0
native safety shield = enabled
```

两组唯一训练语义差异：

```text
danger-on:
  run_dir = runs/diagnostics/joint_mcgae_macro_k5_nogrow_u300_rerun_20260514
  danger_imitation_enabled = true
  danger_imitation_trigger_mode = intervention_any

danger-off:
  run_dir = runs/diagnostics/joint_mcgae_macro_k5_nodanger_u300_20260515
  danger_imitation_enabled = false
  danger_imitation_coef = 0
```

实现备注：

```text
train_joint_mcgae.py 默认会为 accel 训练打开 danger imitation。
因此本次新增 --disable_danger_imitation 显式消融开关，避免只改 YAML 但实际没有关闭。
```

## 2. 训练期摘要

| run | bw first | bw u100 | bw u200 | bw u250 | bw final | bw best | best update | bw last10 | bw last50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| danger-on | 11.4774 | 11.9957 | 16.4218 | 17.9183 | 16.3576 | 18.8038 | 290 | 17.4598 | 17.6845 |
| danger-off | 11.4774 | 11.7436 | 14.8633 | 17.6382 | 16.7007 | 18.8890 | 298 | 17.9218 | 17.5749 |

Episode reward：

| run | first | u100 | u200 | u250 | final | best | best update | last10 | last50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| danger-on | 33.5310 | 34.5018 | 46.0302 | 49.7854 | 46.0699 | 51.9637 | 290 | 48.6650 | 49.2186 |
| danger-off | 33.5310 | 33.7701 | 42.2140 | 49.1915 | 47.0371 | 52.0765 | 297 | 49.9312 | 48.9866 |

安全相关：

| run | collision last50 | episode length last50 | danger active last50 |
| --- | ---: | ---: | ---: |
| danger-on | 0.0000075 | 249.54 | 0.0331 |
| danger-off | 0.0000025 | 249.85 | 0.0000 |

训练期观察：

```text
关掉 danger imitation 后，没有出现明显碰撞恶化。
danger-off 的训练期 best episode reward 略高，但 last50 与 danger-on 基本同量级。
danger-off 在 u200 左右明显低于 danger-on，说明早中期学习更慢或波动更大。
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

## 4. Final Checkpoint 8 组合

| combo | danger-on reward | danger-off reward | diff off-on | collision off |
| --- | ---: | ---: | ---: | ---: |
| rrr | 45.2764 | 45.2764 | +0.0000 | 0.0000 |
| Arr | 46.5334 | 46.4319 | -0.1015 | 0.0000 |
| rSr | 45.4156 | 44.5028 | -0.9127 | 0.0000 |
| rrB | 52.6378 | 53.1643 | +0.5264 | 0.0000 |
| ASr | 46.3178 | 45.5281 | -0.7897 | 0.0000 |
| ArB | 55.9214 | 56.4604 | +0.5389 | 0.0000 |
| rSB | 52.5977 | 49.8155 | -2.7821 | 0.0000 |
| ASB | 55.1812 | 52.5741 | -2.6070 | 0.0000 |

## 5. Best Stage Heads 8 组合

| combo | danger-on reward | danger-off reward | diff off-on | collision off |
| --- | ---: | ---: | ---: | ---: |
| rrr | 45.2764 | 45.2764 | +0.0000 | 0.0000 |
| Arr | 45.2827 | 46.4741 | +1.1914 | 0.0000 |
| rSr | 45.4388 | 44.6793 | -0.7594 | 0.0000 |
| rrB | 57.6790 | 54.1289 | -3.5502 | 0.0000 |
| ASr | 45.2804 | 45.8251 | +0.5447 | 0.0000 |
| ArB | 58.3895 | 58.2316 | -0.1579 | 0.0000 |
| rSB | 58.0850 | 50.7584 | -7.3266 | 0.0000 |
| ASB | 59.2088 | 54.5454 | -4.6634 | 0.0000 |

## 6. 训练时间窗口对比

只看全程均值会掩盖早期差异。关掉 danger imitation 后，前 50 个 update 明显更慢；后期速度接近，甚至 danger-off 略快，所以全程均值看起来差不多。

| window | danger-on iteration_sec | danger-off iteration_sec | off-on |
| --- | ---: | ---: | ---: |
| u1-10 | 60.12 | 70.17 | +10.05 |
| u1-15 | 60.92 | 71.87 | +10.95 |
| u1-30 | 55.38 | 65.70 | +10.32 |
| u1-50 | 52.30 | 59.87 | +7.57 |
| u101-150 | 39.84 | 39.87 | +0.03 |
| u151-200 | 38.97 | 34.84 | -4.13 |
| u251-300 | 49.77 | 46.71 | -3.06 |
| all | 46.10 | 46.33 | +0.23 |

分段耗时也说明早期变慢不是单一来源：

| window | collect off-on | critic off-on | actor off-on |
| --- | ---: | ---: | ---: |
| u1-10 | +0.85 | +6.15 | +3.05 |
| u1-30 | +2.95 | +6.52 | +0.85 |
| u1-50 | +2.09 | +4.86 | +0.61 |
| u251-300 | +0.02 | -3.19 | +0.12 |

安全介入压力：

| window | danger-on intervention | danger-off intervention | off-on |
| --- | ---: | ---: | ---: |
| all | 0.1053 | 0.2033 | +0.0980 |
| u101-150 | 0.1542 | 0.4414 | +0.2872 |
| u151-200 | 0.0330 | 0.3235 | +0.2905 |
| u251-300 | 0.0331 | 0.0351 | +0.0020 |

注意：

```text
当前日志记录的是 intervention_rate，不是 native shield 内部迭代次数。
因此可以说：
  关 danger 后安全介入压力更高，早期训练更慢；
但不能直接说：
  早期变慢完全来自安全层迭代次数增加。
如果论文里要展示“安全层迭代次数”，还需要额外记录 native shield solver iteration。
```

## 7. 结论

```text
1. danger imitation 不是当前低碰撞的唯一原因。
   在 native safety shield 开启时，danger-off 也基本没有碰撞。

2. danger imitation 对最终全策略组合仍有正面作用。
   danger-on best ASB = 59.2088；
   danger-off best ASB = 54.5454。

3. danger-off 的 BW-only 变弱。
   danger-on best rrB = 57.6790；
   danger-off best rrB = 54.1289。

4. danger-off 的 Accel+BW 组合接近 danger-on。
   danger-on best ArB = 58.3895；
   danger-off best ArB = 58.2316。
   这说明关 danger 后不是所有组合都崩，但 SAT/BW/全组合协同明显变差。

5. 当前建议：
   保留 danger imitation。
   它不只是防碰撞辅助，在 joint 训练里还可能稳定 accel 分布，从而帮助 SAT/BW 的组合效果。
```
