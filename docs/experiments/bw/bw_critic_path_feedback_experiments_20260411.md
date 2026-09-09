# BW-only T10 critic/value path feedback experiments (2026-04-11)

## 1. 背景和问题

这份文档整理的是收到外部意见之后做的一组 follow-up 实验。

目标不是重新证明 `T=10` 表现差，而是回答更具体的问题：

- 标准 PPO 的 `state-value critic -> GAE/advantage -> actor update` 这条路径到底哪里不够好；
- 改善 critic 精度、target scaling、样本覆盖、critic 结构之后，是否能让 `T=10` 的 PPO 更新恢复正常；
- 如果 state-value critic 仍然不够精，是否能通过 simulator branch rollout 看到更接近局部长期收益的方向。

基础设定仍然是简化的 BW-only case：

- `num_uav = 1`
- `num_gu = 5`
- `T_steps = 10`
- `train_accel = false`
- `train_sat = false`
- `train_bw = true`
- `exec_accel_source = zero`
- `exec_sat_source = zero`
- `exec_bw_source = policy`
- checkpoint eval 参考策略为 `queue_aware_bw`

主配置：

- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_gae_vbw_chainprobe_u10.yaml`

主要短训口径：

- `updates = 10`
- `vec_backend = sync`
- `torch_threads = 1`
- `save_interval = 10`
- update-direction probe 通常只在最后一次 update 运行：`update_direction_probe_interval_updates = 10`、`update_direction_probe_start_update = 10`

少量 `u30` run 用于检查短期有效的改动是否随训练长度保持稳定。

## 2. 收到意见后的实验方向

收到的意见大体建议优先攻 critic/value path，而不是马上全面切换到 action-dependent / counterfactual credit。后续实验按这个思路做了几类低风险或中等风险尝试。

### 2.1 Critic-first / PPG-style 简化相位更新

这不是完整 PPG，而是一个简化控制实验：

1. rollout；
2. 先单独训练 critic 若干轮；
3. 再更新 actor；
4. 可选在 actor update 前重算 actor 用的 advantage。

实现开关：

- `critic_warmup_before_actor_epochs`
- `critic_warmup_recompute_advantages`
- `critic_warmup_recompute_mode = advantage_only | gae`

两种 recompute 语义：

- `advantage_only`：固定 rollout 时算出的 return target，只用 warmup 后的新 critic 重算 `A = target - V_new`；
- `gae`：warmup 后刷新 buffer 里的 critic value，然后重新跑 `compute_returns_and_advantages`。

代表短训：

```powershell
.\.venv\Scripts\python.exe scripts\train_structured.py `
  --config configs\structured_bw_sanity_1uav_static_gap_debug_t10_gae_vbw_chainprobe_u10.yaml `
  --updates 10 --vec_backend sync --torch_threads 1 `
  --update_direction_probe_interval_updates 10 --update_direction_probe_start_update 10 `
  --save_interval 10 `
  --critic_warmup_before_actor_epochs 16 `
  --critic_warmup_recompute_advantages `
  --critic_warmup_recompute_mode advantage_only
```

### 2.2 降低 bootstrap/GAE 敏感性

尝试了：

- `gae_lambda = 1.0`
- `bw_return_mode = bw_episode_mc`

在当前 `T=10` BW-only 设置里，`bw_episode_mc` 和 `lambda=1` 的主要效果非常接近；它不是解决项，只是用来确认 lambda-return/GAE bootstrap 是否是主要误差源。

### 2.3 Critic replay bank

实现了一个很轻量的 critic replay bank：

- 存 `(world_state, return_target)`；
- warmup critic 时用当前 batch + bank 里的旧 target 一起训练；
- 这是 old-target replay，不是严格的 off-policy relabel，也不是 Retrace/V-trace。

开关：

- `critic_replay_bank_enabled`
- `critic_replay_bank_capacity`

典型设置：

- `critic_replay_bank_capacity = 800`

### 2.4 Critic target scaling / normalization

尝试了三种：

- `critic_loss_target_standardize`：每个当前 batch 内标准化 value loss；
- `critic_loss_running_standardize`：用 running mean/std 标准化 value loss；
- `critic_popart_enabled`：PopArt 风格地维护 per-stage target 均值/方差，并重参数化 value head，使 critic 输出仍保持原始 V 尺度。

注意：

- 日志里的 `value_loss_bw` 仍然记录原始尺度 MSE；
- PopArt 只改变 critic loss 的训练尺度和 head 参数化，不直接改 actor objective。

### 2.5 Critic 表达和结构

尝试了：

- 更大 critic：`critic_hidden_dim = 512`、`critic_embed_dim = 128`；
- global feature critic：在 relational critic 的 team context 上加入 18 维手工全局特征，包括 queue 统计、candidate/valid/eta 统计、time fraction、visible/sat valid 等。

global feature 这一版只是快速验证，不是最终结构化 critic 设计。

### 2.6 Reward decomposition / proxy 方向的简化尝试

没有直接实现完整 multi-head reward decomposition，也没有把 reward redistribution 改成主目标。

只做了两个低成本 proxy：

- `bw_flow_proxy_aux_enabled`：用已有 flow proxy signal 加辅助 loss；
- `bw_train_target_mode = gu_service_queue`：把 BW 训练 target 暂时替成一个 GU service/queue proxy。

结果表明，proxy 可能改善某些 value 指标，但没有稳定改善 actor eval；`gu_service_queue` 作为替换主 target 明显不适合当前 env reward critic。

### 2.7 Branch-rollout update probe

后续新增了一个诊断项，而不是训练项。

它比较同一个 BW state 下：

- sampled BW action；
- deterministic / policy-mean reference action；
- 之后 `h = 2/5/10` 步 follow 当前 policy；
- 两条分支共享同一个环境 RNG snapshot，并共享后续 stochastic actor sample seed。

新增 CSV 指标包括：

- `corr_raw_advantage_vs_branch_delta`
- `corr_branch_delta_vs_delta_logprob`
- `branch_delta_abs_mean`
- `corr_raw_advantage_vs_branch_delta_h2/h5/h10`
- `corr_branch_delta_vs_delta_logprob_h2/h5/h10`

这里的 `branch_delta` 是：

```text
Q_h(s, sampled_bw_action) - Q_h(s, deterministic_ref_action)
```

它比原来的 `true_adv_mc` 更接近“同一个 state 上这个 sampled action 比 ref action 在局部长期 rollout 里好不好”。

### 2.8 Stratified fixed-policy value probe

也扩展了 value probe，用来按 regime 看 critic heldout 误差。

新增可选模式：

- `--stratified_collection`
- `--stratified_pool_multiplier`

分桶特征：

- `time_bin`
- `queue_total_bin`
- `queue_imbalance_bin`
- `hotspot_active`
- `backlog_slope_bin`
- 组合 `bucket_key`

输出：

- `targets.csv`
- `holdout_by_feature.csv`
- `summary.json` 中的 `state_bank_distribution` 和 `holdout_by_feature`

## 3. T=10 短训结果总表

表中：

- `gap = learned reward - fixed heuristic reward`；
- `rawAdv->true` 是 `corr_raw_advantage_vs_true_adv_mc`；
- `true->dlogp` 是 `corr_true_adv_mc_vs_delta_logprob`；
- `adv->dlogp` 是 PPO 自己 normalized advantage 和 logprob 变化的相关；
- `dSelf` 是 probe panel 上 actor 更新后的自身随机策略期望 reward 变化；
- `raw->branch` / `branch->dlogp` 只在新增 branch probe run 中有。

| label | eval reward | gap | value_loss_bw | EV_bw | KL_bw | clip_bw | rawAdv->true | true->dlogp | adv->dlogp | dSelf | raw->branch | branch->dlogp | judgement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_u10 | 3.315 | -0.524 | 0.077 | 0.971 | -0.0020 | 0.000 | -0.382 | +0.037 | +0.187 | +0.0040 | | | critic_advantage_suspect |
| warmadv16 | 3.307 | -0.532 | 0.052 | 0.992 | +0.0029 | 0.017 | -0.029 | +0.219 | +0.103 | +0.0080 | | | critic_advantage_suspect |
| warmgae16 | 3.320 | -0.519 | 0.042 | 0.992 | +0.0102 | 0.130 | -0.016 | +0.280 | +0.257 | +0.0088 | | | critic_advantage_suspect |
| criticstd | 3.317 | -0.522 | 0.140 | 0.927 | -0.0013 | 0.000 | -0.402 | -0.204 | +0.168 | +0.0033 | | | critic_advantage_suspect |
| critic512 | 3.314 | -0.525 | 0.048 | 0.982 | -0.0003 | 0.012 | -0.228 | +0.236 | +0.175 | +0.0040 | | | critic_advantage_suspect |
| lambda1_warmadv16 | 3.304 | -0.535 | 0.043 | 0.989 | +0.0071 | 0.065 | +0.191 | +0.246 | +0.172 | +0.0058 | | | inconclusive_or_consistent |
| bank800_warmgae16 | 3.322 | -0.517 | 0.082 | 0.985 | +0.0030 | 0.007 | -0.069 | +0.224 | +0.252 | +0.0020 | | | critic_advantage_suspect |
| runvnorm_warmgae16 | 3.317 | -0.522 | 0.050 | 0.989 | +0.0188 | 0.207 | +0.038 | +0.262 | +0.248 | +0.0051 | | | critic_advantage_suspect |
| lambda1_bank800_warmadv16 | 3.357 | -0.482 | 0.079 | 0.984 | +0.0151 | 0.027 | +0.182 | +0.154 | +0.251 | +0.0081 | | | inconclusive_or_consistent |
| env8_roll50_lambda1 | 3.251 | -0.588 | 0.347 | 0.906 | +0.0270 | 0.197 | -0.149 | -0.024 | +0.253 | +0.0078 | | | critic_advantage_suspect |
| bwepisodemc_warmadv16 | 3.304 | -0.535 | 0.043 | 0.989 | +0.0071 | 0.065 | +0.191 | +0.246 | +0.172 | +0.0058 | | | inconclusive_or_consistent |
| lambda1_warmadv16_u30 | 3.221 | -0.618 | 0.100 | 0.993 | -0.0013 | 0.090 | -0.112 | -0.025 | +0.200 | +0.0032 | | | critic_advantage_suspect |
| lambda1_bank800_u30 | 3.301 | -0.538 | 0.041 | 0.976 | +0.0215 | 0.227 | +0.055 | +0.215 | +0.166 | -0.0040 | | | inconclusive_or_consistent |
| globalfeat_lambda1 | 3.323 | -0.516 | 0.110 | 0.982 | +0.0039 | 0.027 | -0.124 | +0.034 | +0.219 | +0.0066 | | | critic_advantage_suspect |
| flowproxy_lambda1 | 3.317 | -0.522 | 0.052 | 0.995 | +0.0074 | 0.140 | +0.184 | -0.046 | +0.182 | +0.0100 | | | inconclusive_or_consistent |
| guservice_lambda1 | 3.299 | -0.540 | 4.322 | 0.993 | +0.0189 | 0.085 | -0.096 | +0.189 | +0.334 | -0.0011 | | | critic_advantage_suspect |
| popart_bank | 3.325 | -0.514 | 0.099 | 0.988 | +0.0104 | 0.017 | +0.122 | +0.269 | +0.355 | +0.0010 | | | inconclusive_or_consistent |
| popart_no_bank | 3.291 | -0.548 | 0.063 | 0.988 | +0.0059 | 0.050 | +0.124 | +0.402 | +0.265 | +0.0007 | | | critic_advantage_suspect |
| bank_fullgae_lambda1 | 3.340 | -0.499 | 0.100 | 0.983 | +0.0143 | 0.047 | +0.044 | +0.354 | +0.356 | +0.0056 | | | critic_advantage_suspect |
| critic512_bank_lambda1 | 3.337 | -0.502 | 0.083 | 0.957 | +0.0042 | 0.007 | +0.235 | +0.065 | +0.326 | +0.0017 | | | inconclusive_or_consistent |
| env8_roll25_bank_lambda1 | 3.252 | -0.587 | 0.363 | 0.955 | +0.0040 | 0.010 | +0.262 | +0.032 | +0.283 | -0.0028 | | | inconclusive_or_consistent |
| criticlr1e3_bank_lambda1 | 3.311 | -0.527 | 0.057 | 0.991 | +0.0061 | 0.017 | +0.100 | +0.321 | +0.137 | +0.0008 | | | inconclusive_or_consistent |
| branchprobe_lambda1_bank | 3.337 | -0.502 | 0.089 | 0.984 | +0.0035 | 0.017 | +0.421 | +0.001 | +0.293 | +0.0040 | +0.357 | +0.078 | inconclusive_or_consistent |

主要观察：

- 多数改动能让 `adv->dlogp` 保持为正，说明 PPO 对自己手里的 advantage 并不是完全失效。
- 但 eval reward 基本仍在 `3.25~3.36`，没有接近 heuristic `3.839`。
- `lambda1 + bank800 + warmadv16` 是这批短训里相对最好的一条，但 `u30` 没有稳定继续变好。
- `PopArt`、`critic_lr=1e-3`、`critic512`、更多 env steps 都没有稳定改善 eval。
- `flowproxy` 可让某些 value 指标变好，但没有带来 actor eval 提升。
- `gu_service_queue` 作为替换主 target 明显不合适。

## 4. Fixed-policy value probe 结果

这些 probe 固定 actor checkpoint，用 MC/finite-horizon 估 `V^pi(s)`，再比较 online critic / 监督继续训练 critic / fresh critic。

表中：

- `online_holdout_rmse` 是训练中保存下来的 critic 对 heldout states 的 RMSE；
- `loaded_holdout_rmse` 是从 online critic 继续监督训练后的 heldout RMSE；
- `fresh_holdout_rmse` 是从随机初始化 critic 监督训练后的 heldout RMSE。

| label | online train RMSE | online holdout RMSE | online holdout corr | online holdout bias | loaded holdout RMSE | fresh holdout RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.001 | 3.624 | 0.883 | +3.561 | 1.390 | 1.822 |
| warmadv16 | 0.926 | 0.888 | 0.981 | +0.774 | 0.941 | 1.856 |
| warmgae16 | 1.074 | 1.594 | 0.972 | +1.525 | 0.838 | 1.866 |
| lambda1 | 0.976 | 0.904 | 0.979 | +0.822 | 1.013 | 1.726 |
| bank800_warmgae16 | 1.947 | 0.655 | 0.935 | -0.194 | 1.787 | 1.672 |
| runvnorm_warmgae16 | 1.073 | 1.701 | 0.971 | +1.637 | 0.778 | 1.833 |
| globalfeat_lambda1 | 0.756 | 1.089 | 0.730 | -0.512 | 1.427 | 1.595 |
| flowproxy_lambda1 | 0.739 | 0.631 | 0.984 | +0.423 | 0.782 | 1.888 |
| guservice_lambda1 | 59.549 | 37.612 | 0.257 | -36.868 | 1.607 | 1.844 |
| popart_bank | 0.874 | 0.578 | 0.979 | -0.136 | 1.179 | 1.459 |
| env8_roll25_bank | 0.803 | 0.751 | 0.975 | -0.101 | 1.465 | 1.662 |
| branchprobe_stratified | 0.750 | 0.621 | 0.930 | +0.490 | 0.579 | 0.428 |

主要观察：

- 有些方法能把 online heldout RMSE 从 baseline 的 `3.624` 压到 `0.6` 左右。
- 但新增 branch probe 显示 `branch_delta_abs_mean` 只有大约 `0.020`。
- 也就是说，即使 heldout RMSE 到 `0.4~0.6`，它仍然比 action-level 局部 advantage 大一个数量级以上。
- 这解释了为什么 `value_loss_bw` 或 explained variance 看起来不错时，actor update 仍可能无法稳定改善最终 reward。

## 5. Branch-rollout probe 结果

代表 run：

- `runs/structured/structured_bw_t10_branchprobe_lambda1_bank800_warmadv16_u10_20260411`

设置：

- `lambda = 1.0`
- `critic_warmup_before_actor_epochs = 16`
- `critic_warmup_recompute_mode = advantage_only`
- `critic_replay_bank_capacity = 800`
- `update_direction_probe_branch_enabled = true`
- `update_direction_probe_branch_horizons = 2,5,10`
- `update_direction_probe_branch_samples = 2`
- `update_direction_probe_branch_follow_policy_mode = stochastic`
- `update_direction_probe_bw_sample_limit = 12`

总体结果：

| horizon | rawAdv->branch_delta | branch_delta->dlogp | branch_delta_abs_mean | branch_delta_std | sign_agree_raw_branch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | +0.618 | -0.084 | 0.0205 | 0.0292 | 0.583 |
| 5 | +0.533 | +0.003 | 0.0233 | 0.0299 | 0.500 |
| 10 | +0.357 | +0.078 | 0.0203 | 0.0265 | 0.417 |

同一 run 的旧 MC 指标：

| metric | value |
| --- | ---: |
| `corr_raw_advantage_vs_true_adv_mc` | +0.421 |
| `corr_true_adv_mc_vs_delta_logprob` | +0.001 |
| `corr_advantage_vs_delta_logprob` | +0.293 |
| `delta_actor_reward_mean` | +0.0040 |
| eval reward gap | -0.502 |

解释：

- `rawAdv -> branch_delta` 是正的，说明 critic advantage 对局部 branch payoff 不是完全没有信息。
- 但是 `branch_delta -> dlogp` 很弱，尤其 h2/h5 几乎为 0 或为负。
- 这说明 actor 更新主要还是“跟随自己的 normalized advantage”，但这个跟随没有明显转化成“提高局部 branch payoff 的 sampled actions 的 logprob”。
- 换句话说，T10 的异常不再能简单归结为“critic 完全没有信号”；更像是信号太小、太噪、被归一化/多样本/minibatch/参数化后的更新稀释，最终只留下很弱的局部长期收益方向。

## 6. Stratified value probe 分桶结果

代表 probe：

- `runs/analysis/bw_value_gen_t10_branchprobe_stratified_lambda1_bank800_small_20260411`

设置：

- `train_states = 48`
- `holdout_states = 48`
- `stratified_collection = true`
- `stratified_pool_multiplier = 4`
- `mc_rollouts = 6`
- `horizon = 20`
- `epochs = 64`

整体：

| model | holdout RMSE | MAE | bias | corr | EV |
| --- | ---: | ---: | ---: | ---: | ---: |
| online critic | 0.621 | 0.536 | +0.490 | 0.930 | 0.861 |
| loaded critic supervised | 0.579 | 0.377 | +0.064 | 0.854 | 0.683 |
| fresh critic supervised | 0.428 | 0.324 | +0.049 | 0.916 | 0.827 |

online critic 按特征分桶：

| feature bucket | n | RMSE | bias | corr | target mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `time_bin=0` | 13 | 0.617 | +0.417 | 0.895 | 2.351 |
| `time_bin=1` | 19 | 0.683 | +0.535 | 0.848 | 1.289 |
| `time_bin=2` | 16 | 0.541 | +0.496 | 0.801 | 0.449 |
| `queue_total_bin=0` | 12 | 0.866 | +0.765 | 0.882 | 1.667 |
| `queue_total_bin=1` | 22 | 0.573 | +0.405 | 0.937 | 1.587 |
| `queue_total_bin=2` | 14 | 0.403 | +0.388 | 0.916 | 0.522 |
| `queue_imbalance_bin=0` | 10 | 0.409 | +0.379 | 0.993 | 1.439 |
| `queue_imbalance_bin=1` | 17 | 0.452 | +0.274 | 0.955 | 1.622 |
| `queue_imbalance_bin=2` | 17 | 0.758 | +0.667 | 0.920 | 1.039 |
| `queue_imbalance_bin=3` | 4 | 0.948 | +0.937 | 0.996 | 0.648 |
| `hotspot_active=0` | 21 | 0.503 | +0.390 | 0.950 | 1.235 |
| `hotspot_active=1` | 27 | 0.699 | +0.568 | 0.923 | 1.344 |

fresh supervised critic 分桶：

| feature bucket | n | RMSE | bias | corr |
| --- | ---: | ---: | ---: | ---: |
| `queue_imbalance_bin=0` | 10 | 0.398 | +0.320 | 0.985 |
| `queue_imbalance_bin=1` | 17 | 0.458 | -0.154 | 0.935 |
| `queue_imbalance_bin=2` | 17 | 0.435 | +0.036 | 0.858 |
| `queue_imbalance_bin=3` | 4 | 0.334 | +0.297 | 0.978 |
| `hotspot_active=0` | 21 | 0.502 | +0.047 | 0.906 |
| `hotspot_active=1` | 27 | 0.360 | +0.051 | 0.942 |
| `queue_total_bin=0` | 12 | 0.523 | -0.034 | 0.795 |
| `queue_total_bin=1` | 22 | 0.459 | +0.020 | 0.930 |
| `queue_total_bin=2` | 14 | 0.254 | +0.167 | 0.904 |

解释：

- online critic 明显整体高估 value，bias `+0.490`。
- 高 queue imbalance 的状态更难，online RMSE 从 `0.409/0.452` 上升到 `0.758/0.948`。
- hotspot active 时也更难，RMSE 从 `0.503` 上升到 `0.699`。
- fresh supervised critic 能显著降低部分分桶误差，说明数据/训练 protocol 确实还有改进空间。
- 但即使 fresh supervised critic，整体 holdout RMSE 仍为 `0.428`，远大于 branch_delta 的 `~0.02`。

## 7. 当前结论

这组实验后，结论比之前更细：

1. 单纯提高 critic loss 训练强度不够。

   `critic_lr=1e-3`、`warmup16`、`PopArt`、target standardization、running standardization、larger critic，都没有稳定改善 `T=10` eval。

2. 数据覆盖和泛化是问题，但不是一个小改能补上的问题。

   stratified value probe 说明 online critic 的 heldout 误差集中在 hotspot active、高 queue imbalance 等 regime；监督训练能改善，但离 action-level branch_delta 的尺度仍很远。

3. PPO 不是完全“不按 advantage 更新”。

   多数 run 的 `adv->dlogp` 为正，说明 logprob surrogate 对 normalized advantage 有响应。

4. 但这不等于“按长期收益方向更新”。

   branch probe 显示 `rawAdv->branch_delta` 可以为正，但 `branch_delta->dlogp` 很弱；也就是说，更新对局部长期收益方向的跟随不稳定、不干净。

5. 当前 T10 的核心难点更像是：

   - state-value baseline 需要分辨的 action-level advantage 只有约 `0.02`；
   - critic heldout V error 即使短训改善后仍在 `0.4~0.6`；
   - queue / hotspot / imbalance regime 又让 heldout 泛化更难；
   - 最终 actor update 虽然跟随 normalized advantage，但没有足够强地跟随真正的 branch payoff。

## 8. 下一步建议

最直接的下一步不是继续扫 critic 超参，而是做一个更硬的 actor-path 控制实验：

- 固定 rollout；
- 用 branch rollout 得到的 `branch_delta` 直接替换 actor advantage；
- 只做一小步 BW actor update；
- 观察 `branch_delta->dlogp`、`delta_actor_reward_mean`、checkpoint eval 是否明显转正。

判别逻辑：

- 如果 `branch_delta-as-advantage` 能让 actor reward 上升，说明 logprob/actor 接口基本可用，主要问题仍是 critic advantage 不够准；
- 如果这样也不能让 actor reward 上升，问题就更靠近 actor 参数化、PPO clipping/minibatch、masked-softmax/simplex logprob 接口。

同时可以补一个 `T=1` 的 branch-probe 对照，用“能学的设置”给这些过程指标定标。

## 9. 相关文件和 run

代码改动主要涉及：

- `sagin_marl/env/config.py`
- `sagin_marl/rl/structured_mappo.py`
- `sagin_marl/rl/structured_critic.py`
- `sagin_marl/rl/structured_factory.py`
- `sagin_marl/rl/structured_train.py`
- `sagin_marl/rl/structured_bw_update_direction.py`
- `scripts/train_structured.py`
- `scripts/diagnostics/probe/probe_structured_bw_value_generalization.py`

主要 run：

- `runs/structured/structured_bw_t10_probefinal_baseline_u10_20260410`
- `runs/structured/structured_bw_t10_probefinal_warmadv16_u10_20260410`
- `runs/structured/structured_bw_t10_probefinal_warmgae16_u10_20260410`
- `runs/structured/structured_bw_t10_probefinal_lambda1_fixedA_warm16_u10_20260410`
- `runs/structured/structured_bw_t10_probefinal_lambda1_bank800_warmadv16_u10_20260410`
- `runs/structured/structured_bw_t10_probefinal_lambda1_bank800_warmadv16_u30_20260411`
- `runs/structured/structured_bw_t10_probefinal_popart99_lambda1_bank800_warmadv16_u10_20260411`
- `runs/structured/structured_bw_t10_branchprobe_lambda1_bank800_warmadv16_u10_20260411`

主要 analysis：

- `runs/analysis/bw_value_gen_t10_probefinal_popart99_lambda1_bank800_small_20260411`
- `runs/analysis/bw_value_gen_t10_probefinal_env8roll25_lambda1_bank1600_small_20260411`
- `runs/analysis/bw_value_gen_t10_branchprobe_stratified_lambda1_bank800_small_20260411`

验证：

- `python -m py_compile` 覆盖了本次改动的主要文件；
- 相关 pytest：`15 passed, 22 deselected`。
