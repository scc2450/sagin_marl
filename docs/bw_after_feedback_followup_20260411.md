# BW-only T10 after-doc follow-up experiments (2026-04-11)

这份文档续接：

- `docs/bw_critic_path_feedback_experiments_20260411.md`

这里只整理**上一份文档之后**继续做的尝试。重点不再是继续扫 critic 超参，而是回答更具体的两个问题：

1. `branch_delta` 这种更接近局部长期收益的 teacher signal，PPO actor 通路到底能不能吃下？
2. 如果 teacher 本身是强的，那么问题是不是已经不在 `critic -> advantage -> PPO` 这条链上，而在于当前 improvement operator 和最终 stationary closed-loop policy 不匹配？

基础口径仍然是简化的 BW-only T10 case：

- `num_uav = 1`
- `num_gu = 5`
- `T_steps = 10`
- `train_accel = false`
- `train_sat = false`
- `train_bw = true`
- `exec_accel_source = zero`
- `exec_sat_source = zero`
- `exec_bw_source = policy`
- fixed eval baseline = `queue_aware_bw`


## 1. 尝试 A：`branch_delta` 直接喂 actor 的 clean PPO 控制实验

### 1.1 要解决的问题

前面已经怀疑：

- 标准 critic advantage 在 T10 下经常给错方向；
- 但还没有把“critic 给错方向”和“PPO actor/logprob 通路本身坏掉”完全分开。

这个实验的目的就是做一个更干净的正控：

- 不再让 BW actor 用 critic advantage；
- 直接把 branch rollout 得到的 `branch_delta` 覆盖成 stage-2 actor 的 advantage；
- 再看 PPO 这一步是否至少会顺着这个 signal 走。

### 1.2 具体实现

改动文件：

- `sagin_marl/rl/structured_mappo.py`
- `sagin_marl/rl/structured_bw_update_direction.py`
- `scripts/train_structured.py`

新增能力：

- 训练前对当前 rollout 的每个 BW transition 计算 `branch_delta`
- 允许用 `branch_delta` 覆盖 BW actor advantage
- update probe 继续记录 `branch_delta -> dlogp`、`true_adv_mc -> dlogp`

这里的 `branch_delta` 定义为：

```text
branch_delta = Q_h(s, a_sampled) - Q_h(s, a_ref)
```

其中：

- `a_sampled` 是 rollout 里实际采样的 BW action
- `a_ref` 是 deterministic / mean reference action
- `Q_h` 用 simulator rollout `h=10` 步得到
- 之后 future follow 当前 policy

clean PPO 口径：

- `ppo_epochs = 1`
- `num_mini_batch = 1`
- `entropy_coef = 0`

代表 run：

- baseline: `runs/structured/structured_bw_t10_cleanppo_baseline_u10_20260411`
- branch advantage override: `runs/structured/structured_bw_t10_cleanppo_branchadv_u10_20260411`

### 1.3 结果数据

最后一轮 probe：

| run | `adv->branch` | `adv->true` | `branch->dlogp` | `true->dlogp` | `delta_actor_reward_mean` |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `-0.374` | `-0.433` | `-0.082` | `-0.268` | `+0.000247` |
| branch override | `+0.911` | `+0.865` | `+0.104` | `+0.064` | `+0.000776` |

训练末轮 metrics：

| run | `policy_loss` | `approx_kl_bw` | `clip_frac_bw` | `entropy_bw` |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.0` | `-7.25e-07` | `0.0` | `-4.567` |
| branch override | `0.00582` | `+5.72e-07` | `0.0` | `-4.570` |

u10 checkpoint eval：

| run | reward | heuristic | gap | processed | backlog |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `3.308` | `3.839` | `-0.530` | `0.939` | `4.833` |
| branch override | `3.319` | `3.839` | `-0.520` | `0.941` | `4.835` |

结果文件：

- `runs/structured/structured_bw_t10_cleanppo_baseline_u10_20260411/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_cleanppo_branchadv_u10_20260411/update_direction_probe.csv`
- `runs/structured/structured_bw_t10_cleanppo_baseline_u10_20260411/metrics.csv`
- `runs/structured/structured_bw_t10_cleanppo_branchadv_u10_20260411/metrics.csv`
- `runs/structured/structured_bw_t10_cleanppo_baseline_u10_20260411/checkpoint_eval.csv`
- `runs/structured/structured_bw_t10_cleanppo_branchadv_u10_20260411/checkpoint_eval.csv`

### 1.4 结论

这一步把一个关键疑点排掉了：

- **PPO actor/logprob 通路不是根本坏掉。**
- 当 advantage 换成更接近局部长期 payoff 的 `branch_delta` 后，更新方向会明显变正常。

但同时也暴露出第二个问题：

- clean PPO 这一步的 **步长极小**；
- `approx_kl_bw` 和 `clip_frac_bw` 基本是 0；
- 所以方向虽然变对了，最终 eval 提升仍然极小。


## 2. 尝试 B：fixed rollout + fixed branch labels 的 actor line search

### 2.1 要解决的问题

在尝试 A 之后，还需要继续回答：

- 是不是因为 clean PPO 步子太小，所以 teacher signal 没法真正转化成策略收益？
- 如果固定同一批 rollout 和同一批 `branch_delta` labels，只扫 actor step size，actor 会不会沿着 teacher 稳定改进？

### 2.2 具体实现

新增/扩展脚本：

- `scripts/diagnose_structured_bw_branch_linesearch.py`

做法：

1. 固定一份 rollout
2. 固定这份 rollout 上算好的 `branch_delta`
3. 只更新 BW actor 一步
4. sweep `actor_lr` scale
5. 记录
   - `mean(branch_delta * delta_logprob)`
   - `mean(trueAdv * delta_logprob)`
   - full-batch achieved KL
   - panel reward delta

代表分析 run：

- `runs/analysis/bw_branch_actor_linesearch_cleanbaseline_full_20260411/summary.json`
- `runs/analysis/bw_branch_actor_linesearch_branchprobe_full_20260411/summary.json`

### 2.3 结果数据

对 baseline fixed batch：

- sweep scale: `0.25, 0.5, 1, 2, 4, 8, 16, 32`
- `mean(branch_delta * dlogp)`：`0.00087 -> 0.18174`
- `mean(trueAdv * dlogp)`：`0.00099 -> 0.22621`
- full-batch KL：`0.00258 -> 2.9916`
- `delta_actor_reward_mean`：**始终为负**，区间 `-0.00045 -> -0.07231`

对 branchprobe teacher fixed batch：

- sweep scale 同上
- `mean(branch_delta * dlogp)`：`0.00097 -> 0.01521`
- `mean(trueAdv * dlogp)`：`0.00112 -> 0.01298`
- full-batch KL：`0.00353 -> 0.05804`
- `delta_actor_reward_mean`：**始终为负**，区间 `-0.00073 -> -0.01918`

### 2.4 结论

这一步说明：

- actor 能顺着 teacher 走，而且能走出非零 KL；
- 但 **旧 batch 上 sampled-action surrogate 变好，不等于 full closed-loop panel 变好**。

也就是说，问题又往前切了一层：

- 现在已经不再是“actor 不跟 signal”
- 而更像是“teacher / surrogate / 真正 closed-loop policy improvement 之间不一致”


## 3. 尝试 C：`Δ_local` probe + mean/concentration 冻结对照

### 3.1 要解决的问题

在尝试 B 之后，还有一个很自然的怀疑：

- `A * dlogp` 之所以看起来为正，会不会主要是因为 policy 在改 concentration / entropy，而不是 mean action 本身在变好？

所以这一步想分清：

- mean-only 更新会怎样？
- concentration-only 更新会怎样？
- `Δ_local > 0` 时，为何 full-policy deterministic/stochastic panel 仍然是负的？

### 3.2 具体实现

继续扩展：

- `scripts/diagnose_structured_bw_branch_linesearch.py`

新增能力：

- `freeze_mode = none | mean_only | concentration_only`
- `local_eval.delta_local_mean`
- `local_eval.delta_local_det_mean`
- deterministic / stochastic panel delta
- `det_mean_l1_valid_mean`
- `kappa_abs_delta_mean`
- `entropy_delta_mean`

代表 run：

- `runs/analysis/bw_branch_actor_linesearch_v2_none_20260411/summary.json`
- `runs/analysis/bw_branch_actor_linesearch_v2_meanonly_20260411/summary.json`
- `runs/analysis/bw_branch_actor_linesearch_v2_conconly_20260411/summary.json`

这里只跑了两个 scale：`1` 和 `8`。

### 3.3 结果数据

#### `freeze_mode = none`

| scale | `KL` | `Δ_local` | `Δ_local_det` | stochastic panel | deterministic panel | `det_mean_l1` | `kappa_abs_delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0121` | `+0.0351` | `+0.00356` | `-0.00284` | `-0.00351` | `0.00648` | `0.0511` |
| 8 | `0.0580` | `+0.0120` | `+0.0132` | `-0.0156` | `-0.0180` | `0.0227` | `0.3651` |

#### `freeze_mode = mean_only`

| scale | `KL` | `Δ_local` | `Δ_local_det` | stochastic panel | deterministic panel | `det_mean_l1` | `kappa_abs_delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.00256` | `+0.0325` | `+0.000739` | `-0.000770` | `-0.000994` | `0.00182` | `0.0` |
| 8 | `0.02297` | `+0.00426` | `+0.00537` | `-0.00667` | `-0.00807` | `0.01299` | `0.0` |

#### `freeze_mode = concentration_only`

| scale | `KL` | `Δ_local` | `Δ_local_det` | stochastic panel | deterministic panel | `det_mean_l1` | `kappa_abs_delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `6.47e-05` | `+0.0318` | `0.0` | `+8.5e-06` | `0.0` | `0.0` | `0.0512` |
| 8 | `7.22e-04` | `-0.00102` | `0.0` | `+7.4e-05` | `0.0` | `0.0` | `0.4464` |

### 3.4 结论

这一步有三个结论：

1. **单改 concentration 基本没有用。**
   - `concentration_only` 几乎不改变 deterministic action
   - deterministic panel 也几乎不动

2. **只改 mean 时，局部 teacher 指标是正的，但 closed-loop panel 仍然是负的。**
   - 所以问题不只是“kappa/entropy 在捣乱”

3. **mean + kappa 一起改会更差。**
   - `none` 比 `mean_only` 的 panel 掉得更明显
   - 说明 concentration 变化会放大坏结果，但不是唯一根因

因此，这一步支持的结论是：

- **当前问题已经更像 improvement operator mismatch**
- 而不是简单的“actor 坏了”或“只是 concentration 在骗人”


## 4. 尝试 D：online planner pilot

### 4.1 要解决的问题

在尝试 A/B/C 之后，一个更直接的问题变成：

- 如果直接把 search teacher 当在线 controller 用，它自己有没有 closed-loop 价值？

这一步非常关键，因为它能把问题分岔成：

- teacher 本身不强
- 或者 teacher 强，但 PPO / shared policy 吸收方式不对

### 4.2 具体实现

修改脚本：

- `scripts/evaluate_structured_bw_select.py`

主要改动：

- 支持和当前训练口径一致的 `exec_accel_source` / `exec_sat_source`
- 当前实测口径为 `zero accel / zero sat`
- 支持 `follow_bw_source = queue_aware | policy`
- 支持导出 `select_bank.pt`

planner 形式：

- 每个 BW step 构 candidate set：
  - `heuristic`
  - `latent_det`
  - `simplex_det`
  - `6` 个 policy sample
- 用 short horizon rollout 选 best first action
- 下一步重新搜索

代表 run：

- baseline compare: `runs/analysis/bw_planner_pilot_t10_20260411/baselines.json`
- `k=1`: `runs/analysis/bw_planner_pilot_t10_20260411/k1_tailqueue/summary.json`
- `k=2, tail=queue_aware`: `runs/analysis/bw_planner_pilot_t10_20260411/k2_tailqueue/summary.json`
- `k=2, tail=policy`: `runs/analysis/bw_planner_pilot_t10_20260411/k2_tailpolicy/summary.json`
- `k=3, tail=queue_aware`: `runs/analysis/bw_planner_pilot_t10_20260411/k3_tailqueue/summary.json`

### 4.3 结果数据

同一组 seeds、`12` episodes：

| setting | reward | processed | backlog |
| --- | ---: | ---: | ---: |
| learned policy | `3.251` | `0.930` | `4.967` |
| heuristic | `3.964` | `1.052` | `4.407` |
| planner `k=1`, tail=`queue_aware` | `3.879` | `1.039` | `4.448` |
| planner `k=2`, tail=`queue_aware` | `4.018` | `1.062` | `4.362` |
| planner `k=2`, tail=`policy` | `4.011` | `1.061` | `4.362` |
| planner `k=3`, tail=`queue_aware` | `4.007` | `1.061` | `4.375` |

selection 统计：

| setting | `selected_beats_latent_frac` | `selected_beats_heuristic_frac` | selection source 主体 |
| --- | ---: | ---: | --- |
| `k=1` | `0.992` | `0.517` | heuristic `58/120`，其余多为 samples |
| `k=2, tail=queue_aware` | `1.000` | `0.367` | heuristic `76/120`，其余为 samples |
| `k=2, tail=policy` | `1.000` | `0.342` | heuristic `79/120`，其余为 samples |
| `k=3` | `0.983` | `0.417` | heuristic `70/120`，其余为 samples |

### 4.4 结论

这一步的结论非常硬：

- **teacher / planner 本身是强的**
- `k=2/3` 的 online search 已经能稳定超过 heuristic
- learned PPO policy 还明显落后

同时还看到两个补充现象：

- `k=1` 还不够，`k=2` 开始明显变好，说明多步 planning 确实有价值
- `tail=queue_aware` 和 `tail=policy` 都能赢 heuristic，说明“旧 tail mismatch”不是唯一原因

所以从这一步开始，主问题已经更像：

- **teacher 强，但当前 learner / improvement operator 吸收 teacher 的方式不对**


## 5. 尝试 E：planner -> selector student 的监督蒸馏

### 5.1 要解决的问题

在尝试 D 之后，新的问题变成：

- planner 的 closed-loop 价值能不能被一个监督 student 吃下来？
- 如果能，说明“换外层算法”这条线不只是理论上合理，而是已经在这个简化 case 里可行。

### 5.2 具体实现

使用/修改脚本：

- bank 导出仍来自 `scripts/evaluate_structured_bw_select.py`
- student 训练与 live eval 脚本为 `scripts/distill_bw_select_v1.py`

这次没有直接蒸馏成独立 actor，而是先学一个**selector student**：

- 输入：`local_state + policy_det_action + heuristic_action + candidate features`
- 输出：在候选动作集合里选哪个动作
- loss：
  - soft target KL
  - 加少量 hard CE
- live eval 时仍然在线生成 candidate set，再由 selector 选动作

为了和 planner pilot 对齐，这次也把 live eval 改成了：

- `exec_accel_source = zero`
- `exec_sat_source = zero`

训练数据：

- bank = `runs/analysis/bw_planner_pilot_t10_20260411/k2_tailqueue_bank/select_bank.pt`
- `120` 个 states
- train/holdout = `100 / 20`

代表 run：

- `runs/analysis/bw_planner_pilot_t10_20260411/selector_distill_k2_tailqueue/summary.json`

### 5.3 结果数据

offline：

| split | top1 acc | capture ratio | `pred_gain_vs_base` mean | `pred_gain_vs_heuristic` mean |
| --- | ---: | ---: | ---: | ---: |
| train | `1.00` | `1.00` | `+0.0742` | `+0.00944` |
| holdout | `0.45` | `0.746` | `+0.0277` | `+0.000817` |

live eval：

| policy | reward | processed | backlog |
| --- | ---: | ---: | ---: |
| learned PPO policy | `3.251` | `0.930` | `4.967` |
| heuristic | `3.964` | `1.052` | `4.407` |
| online planner `k=2` | `4.018` | `1.062` | `4.362` |
| selector student | `3.980` | `1.055` | `4.384` |

live selection source：

- heuristic `83`
- sample_0 `11`
- sample_1 `5`
- sample_2 `6`
- sample_3 `5`
- sample_4 `2`
- sample_5 `8`

### 5.4 结论

这一步说明：

- planner 的价值**可以被监督 student 吃下很大一块**
- selector student 已经：
  - 明显超过原 learned PPO policy
  - 略高于 heuristic
  - 离 online planner 只差约 `0.038`

但这一步也有边界：

- 这还不是独立的单步 BW actor
- 它仍然依赖 runtime candidate set
- 所以它更像“轻量 planner imitation / action selector”，不是最终部署形态


## 6. 总结：这批 follow-up 把问题推进到了哪里

按时间顺序，这批尝试把判断链条推进成了下面这样：

1. **`branch_delta` 直喂 actor 的 clean PPO 控制实验**
   - 解决的问题：actor/logprob 通路是不是根本坏了
   - 结论：**不是**

2. **fixed rollout + fixed labels 的 actor line search**
   - 解决的问题：只要 teacher 更对、步长更大，PPO 会不会直接涨 panel reward
   - 结论：旧 batch surrogate 可以变好，但 closed-loop panel 仍可能变差

3. **`Δ_local` + mean/concentration 冻结对照**
   - 解决的问题：是不是主要是 concentration / entropy 假阳性
   - 结论：concentration 不是主解；mean-only 也仍然会遇到 closed-loop mismatch

4. **online planner pilot**
   - 解决的问题：teacher 本身到底强不强
   - 结论：**强，而且 `k=2` 就能赢 heuristic**

5. **planner -> selector student**
   - 解决的问题：teacher 的 closed-loop 价值能不能被监督 learner 吸收
   - 结论：**能**

因此，这一批 follow-up 给出的主判断是：

- 继续修 `critic -> advantage -> PPO` 已经不是最高优先级
- 当前更值得走的是：
  - `planner / search teacher`
  - `supervised / conservative distillation`
  - 再往后才是“怎么把 selector/planner 压缩成更便宜的 mean-only residual policy”


## 7. 相关脚本与结果文件

实现相关：

- `sagin_marl/rl/structured_mappo.py`
- `sagin_marl/rl/structured_bw_update_direction.py`
- `scripts/train_structured.py`
- `scripts/diagnose_structured_bw_branch_linesearch.py`
- `scripts/evaluate_structured_bw_select.py`
- `scripts/distill_bw_select_v1.py`

结果相关：

- `runs/structured/structured_bw_t10_cleanppo_baseline_u10_20260411`
- `runs/structured/structured_bw_t10_cleanppo_branchadv_u10_20260411`
- `runs/analysis/bw_branch_actor_linesearch_cleanbaseline_full_20260411`
- `runs/analysis/bw_branch_actor_linesearch_branchprobe_full_20260411`
- `runs/analysis/bw_branch_actor_linesearch_v2_none_20260411`
- `runs/analysis/bw_branch_actor_linesearch_v2_meanonly_20260411`
- `runs/analysis/bw_branch_actor_linesearch_v2_conconly_20260411`
- `runs/analysis/bw_planner_pilot_t10_20260411`
