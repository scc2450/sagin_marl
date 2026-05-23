# `proposal + select + distill` 快线拆解（2026-04-08）

本文把 `bw_possibility_v2.md` 里提出的 `proposal + select + distill` 拆成可执行的两段 gate，并给出对应的止损标准。

目标不是“一口气做成最终方案”，而是：

- 先用最短链路确认这条路线到底有没有可兑现的上界；
- 再判断这个上界能不能被一个轻量 student/readout 吃下来；
- 只有前两步都过，才值得往 live integration 继续投时间。

## 1. 为什么现在先走这条线

截至 2026-04-08，已经拿到三类关键信号：

1. 当前 `BW` policy 的 stochastic proposal 里有更好的动作。
2. 当前 deterministic readout 没把这些动作读出来。
3. 当前 route-2 v0 的 proxy/critic estimator，在 within-state action 排序上不够好。

对应证据如下。

### 1.1 `best-of-N readout` 给出强正信号

见：

- [`runs/bw_bestofn_readout_u0120_20260408/summary.json`](./../runs/bw_bestofn_readout_u0120_20260408/summary.json)

同一份 `noaux u0120` policy，在固定 panel states 上：

- `latent_det.mean = 2.6030`
- `simplex_det.mean = 2.5402`
- `heuristic.mean = 2.6509`
- `sample_best_of_8.mean = 2.6367`
- `sample_best_of_32.mean = 2.6741`

关键 win-rate：

- `sample_best_of_32_beats_latent = 1.0000`
- `sample_best_of_32_beats_simplex = 1.0000`
- `sample_best_of_32_beats_heuristic = 0.6563`

这说明：

- 现有 stochastic policy 的支持集里确实有比当前 deterministic readout 更好的动作；
- 甚至已经开始超过 heuristic；
- 所以最快的路不是先改 estimator，而是先把这些动作选出来并读出来。

### 1.2 `old_log_prob` 账本目前没有发现污染主结论的错误

见：

- [`runs/bw_oldlogreplay_noaux_u0120_20260408/summary.json`](./../runs/bw_oldlogreplay_noaux_u0120_20260408/summary.json)
- [`runs/bw_oldlogreplay_bwcfv0_u20_20260408/summary.json`](./../runs/bw_oldlogreplay_bwcfv0_u20_20260408/summary.json)
- [`runs/bw_oldlogreplay_bwsupportv0_u10_20260408/summary.json`](./../runs/bw_oldlogreplay_bwsupportv0_u10_20260408/summary.json)

joint / per-agent `old_logprob` 回放误差都在 `1e-6 ~ 1e-5` 量级。

这说明当前最主要矛盾不是 rollout 账本错位。

### 1.3 `proxy / critic rank-corr` 更像负信号

见：

- [`runs/bw_credit_rankcorr_cf_v0_u20_20260408/summary.json`](./../runs/bw_credit_rankcorr_cf_v0_u20_20260408/summary.json)

within-state 平均相关性：

- `proxy_vs_true.spearman.mean = -0.0282`
- `proxy_vs_true.pairwise_acc.mean = 0.4823`
- `boot_vs_true.spearman.mean = 0.0596`
- `boot_vs_true.pairwise_acc.mean = 0.5212`

这说明 route-2 v0 的 estimator 排序质量不够好。

但这条证据更适合说明“当前 estimator 不值得继续”，不如 `best-of-N` 那样，已经直接给出“上界就在这里”的正信号。

## 2. 这条快线到底在解决什么

它不是在直接解决 “policy 为何没学好”。

它解决的是更窄的问题：

> 当前 stochastic BW policy 已经能产出一些更好的动作时，能不能用一个更短链路把这些动作变成稳定 deterministic 决策？

所以它本质上分成两个 gate：

1. `proposal + select`：确认“上界”是否真存在、是否足够大。
2. `distill`：确认这个上界能不能被一个轻量 student/readout 吃下来。

只有这两个 gate 都过了，才值得进一步讨论 online integration。

## 3. Gate A：`proposal + select` 上界

### 3.1 目标

回答一句话：

> 在同一个已训练好的 stochastic BW policy 里，如果允许每个 state 采样多个候选动作并短回滚选优，能不能明显优于当前 deterministic readout？

### 3.2 当前实现

现成脚本：

- [`scripts/diagnose_bw_best_of_n_readout.py`](./../scripts/diagnose_bw_best_of_n_readout.py)

当前候选集：

- `latent_mean_pushforward`
- `simplex_argmax_logprob`
- `heuristic`
- `N` 个 stochastic BW samples

当前 follow policy：

- 固定 accel/sat
- `k_steps = 2`
- 由同一 actor 继续 follow

### 3.3 当前结论

这一关已经过了。

因为 `best-of-32.mean = 2.6741 > heuristic.mean = 2.6509 > latent_det.mean = 2.6030`。

按快判标准，这已经足够说明：

- 这条线不是空想；
- 部署端确实有可以兑现的增益空间；
- 值得进入下一关 `distill`。

### 3.4 Gate A 的放弃标准

若后续换 checkpoint / 换候选集时重跑这一关，则统一用下面标准：

- 若 `best-of-32.mean <= current_det.mean + 0.02`，停止。
- 若 `best-of-32` 不能至少在 `60%` 的 state 上打过 current deterministic，停止。
- 若 `best-of-32.mean <= heuristic.mean` 且没有明确接近 heuristic，停止。

换句话说，只有当上界明显高于当前 deterministic，而且至少触到 heuristic 这条线时，才值得继续做 student。

## 4. Gate B：`distill` 最小版

### 4.1 目标

回答一句话：

> `proposal + select` 找到的 winner，能不能被一个轻量 offline student/readout 学下来，并在 deterministic 部署时复现大部分上界增益？

### 4.2 最小版不要做什么

这一关先不要做：

- online finetune
- PPO 混合训练
- critic / estimator 改造
- preference/pull 复杂组合
- 新的 environment curriculum

这里只做最短链路：

- 固定一个强 checkpoint；
- 固定 bank；
- 用明确 action label 做 supervised distill；
- 只看 student deterministic 是否能吃下上界。

### 4.3 数据集怎么来

基于 Gate A 的固定 bank，针对每个 state 存：

1. `local_state`
2. `snapshot_state`
3. `current_det_action`
4. `heuristic_action`
5. `N` 个 stochastic samples
6. 每个候选动作的 `k=2` score
7. `winner_action`
8. `winner_source`
9. `winner_score - current_det_score`
10. `winner_score - heuristic_score`

建议把 train / holdout split 固定下来，避免后续再混口径。

### 4.4 Student 先做成什么样

最小 student 不要做成独立 full policy。

建议从小到大按两版来：

#### `v0`: readout student

- 输入：当前 `BW local_state`
- 输出：一个完整 `bw action`
- 训练目标：直接拟合 `winner_action`

最小损失：

- `masked KL(winner || student)`

可选小正则：

- `lambda * masked KL(student || base_det)`

`v0` 的目的是先看“纯 supervised readout 能不能吃下 winner”。

#### `v1`: proposal-conditioned student

如果 `v0` 吃不下，再上这一版。

- 输入：`local_state + current_det_action + heuristic_action + sampled proposal summary`
- 输出：一个完整 `bw action`

这一版才开始利用候选集上下文，但仍然保持 offline supervised。

### 4.5 Gate B 的评估口径

评估必须分两层。

#### B1. Offline holdout

在固定 holdout states 上，比：

- `base_det`
- `heuristic`
- `winner_upper_bound`
- `student_det`

关键指标：

- `student_det.mean`
- `student_det - base_det`
- `student_det - heuristic`
- `student_det - winner_upper_bound`
- `student_beats_base_det` 比例
- `student_beats_heuristic` 比例

#### B2. Fixed-partner live eval

只在 B1 过关后才跑。

建议：

- 固定 accel/sat partner
- 只部署 student 的 deterministic BW
- `20` episodes 起步
- deterministic only

对照：

- base learned-BW deterministic
- heuristic BW deterministic
- student deterministic

### 4.6 Gate B 的放弃标准

#### Offline 放弃线

若 student 在 holdout 上达不到下面任一条，就停止：

- `student_det.mean < base_det.mean + 0.04`
- `student_det.mean < heuristic.mean`
- `student_det` 吃下的上界不到 `50%`

其中“吃下上界不到 `50%`”定义为：

`student_gain / upper_bound_gain < 0.5`

这里：

- `student_gain = student_det.mean - base_det.mean`
- `upper_bound_gain = winner_upper_bound.mean - base_det.mean`

#### Live 放弃线

若 offline 过了，但 fixed-partner live eval 不满足：

- `student_det_live > base_det_live`

则停止，不再往 online integration 上花时间。

如果 student 只能在离线 bank 上好看，落不到 live eval，就把它记为“bank mismatch / deployment mismatch”，而不是继续盲目加训练。

## 5. 为什么这里先不把 estimator 混进来

因为这两条线回答的是两个不同问题：

- `proposal + select + distill`：
  - 现有 stochastic BW policy 里已经有好动作时，能不能把它稳定变成 deterministic 决策？
- `estimator / per-agent advantage`：
  - 如何让训练过程更稳定地产生这些好动作？

当前证据里：

- `best-of-N` 给的是直接正信号；
- `rank-corr` 给的是诊断性负信号。

所以执行顺序上，先做 `proposal + select + distill` 更高效：

- 它的链路更短；
- 不依赖 critic / proxy 排序质量；
- 一旦过了，直接就是可部署改进；
- 一旦不过，也能更清楚地说明“问题不只是 readout，确实更深地在训练信号侧”。

## 6. 下一步具体实现顺序

### Step 1

固化 Gate A 数据集导出：

- 复用 [`scripts/diagnose_bw_best_of_n_readout.py`](./../scripts/diagnose_bw_best_of_n_readout.py)
- 增加 winner 数据导出，形成固定 `winner bank`

### Step 2

做 `distill v0`：

- 新增最小 offline student trainer
- 只做 `winner_action` 的 masked KL / CE
- 固定 train / holdout split

### Step 3

做 `offline holdout` gate：

- 若 student 吃不下至少 `50%` upper bound，停止

### Step 4

只有 Step 3 过关，才做 `fixed-partner live eval`

## 7. 当前建议

当前最合理的执行顺序是：

1. 不再继续 route-2 v0 proxy estimator。
2. 不再继续调当前 deterministic readout 本身。
3. 直接进入 `proposal + select + distill` 的 Gate B，也就是先把 `winner bank` 导出来，再做最小 `distill v0`。

如果 `distill v0` 都吃不下这条上界，那时再转向 `per-agent advantage / estimator`，因果会更干净。
