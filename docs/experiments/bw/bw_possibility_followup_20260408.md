# 针对 `bw_possibility.md` 的后续尝试整理（2026-04-08）

本文整理的是围绕 [bw_possibility.md](./bw_possibility.md) 四条路线，在 2026-04-08 前后实际做过的 follow-up 尝试。配套背景文档是 [bw_status_summary_20260407.md](./bw_status_summary_20260407.md)。

这里采用的原则不是“不能大改训练”，而是：

- 允许大改。
- 但每条路线都必须先设计成短程、强判别实验。
- 只要在对应 gate 上没有给出足够强、足够稳的正信号，就及时停止，不再靠后续微调硬救。

实际落地顺序是：第二条 -> 第一条 -> 第四条 -> 第三条。原因是第二条和第一条最容易做快判，第四条其次，第三条如果直接上 full autoregressive 改动面太大，所以先做了一个最小 decomposition 版本。

## 1. 一页结论

截至 2026-04-08，`bw_possibility.md` 提出的四条路线里，没有一条通过了“值得继续投入”的快判标准。

| 路线 | 实际落地版本 | 关键结果 | 结论 |
| --- | --- | --- | --- |
| 第一条：`bank -> dataset` | broad-to-local offline extraction / preference fitting | holdout 上所有 variant 的 deterministic `det_gap_to_best.mean` 都没有优于 baseline；best 也只是 `0.1481 > 0.1472` | 当前 formulation 失败，停止 |
| 第二条：BW-specific counterfactual credit | structured `bw_counterfactual_credit_enabled` v0 | mismatch 诊断证明 credit 确实脏，但训练后 `u20` 仍只有 `110.60`，远低于 checkpoint eval 固定参考 `162.73` | 当前 v0 失败，停止 |
| 第三条：decomposition / autoregressive | `support + epsilon tail + support 内 Dirichlet` v0 | 零步结构替换即回退；`u+10` 只回到 baseline 附近，没有打开局面 | 当前最小版失败，停止 |
| 第四条：换 PPO 更新族 | `AWR` + `V-MPO-lite` | `AWR v0` 只有弱正信号，`V-MPO-lite` 也不稳且 stochastic 变差 | operationally 可排除，停止 |

当前最稳的工程结论没有变：如果目标是交付结果，而不是继续做新的研究项目，那么主解仍然是 `accel=policy, sat=policy, bw=heuristic`。

## 2. 第一条：把 bank 从 curriculum 改成 dataset

### 2.1 原命题

`bw_possibility.md` 的第一条是：不要再把 snapshot/high-gap bank 当 curriculum，而是把它改成离线策略提取数据集，先做 weighted BC / IQL / AWAC 风格的 extraction，再考虑小步 online finetune。

### 2.2 实际落地版本

这次没有直接上完整 IQL/AWAC，而是先做了一个更便于快判的 offline gate：

- 从当前 `noaux u0120` learned BW 模型收集 panel states。
- 固定当前 deterministic readout。
- 用 panel 内偏好学习和 pull 正则，检查能否先把 holdout 上的 deterministic readout 朝“更接近 best local action”的方向拉动。

正式结果目录：

- `runs/bw_broad2local_offline_noaux_u0120_gate1_subproc_fixed/summary.json`

关键设置：

- `episodes=24`
- `panel_states=32`
- `holdout_count=8`
- `vec_backend=subproc`
- `device=cuda`

### 2.3 结果

holdout baseline：

- `pairwise_acc = 0.5189`
- `spearman = 0.0175`
- `det_gap_to_best.mean = 0.14724`

各 variant 的 holdout `det_gap_to_best.mean`：

- `panel_pref_only = 0.14928`
- `panel_pref_plus_pull = 0.14815`
- `panel_pref_topk_pairs = 0.17274`
- `panel_pref_topk_pairs_plus_pull = 0.17348`
- `panel_pref_topk_pairs_plus_softpull = 0.17306`
- `panel_pref_topk_pairs_plus_softpull_low = 0.17404`
- `panel_pref_topk_pairs_plus_softpull_delayed = 0.16534`

其中 `topk_pairs` 类 variant 的 ranking 指标确实有提升：

- baseline：`pairwise_acc = 0.5189`，`spearman = 0.0175`
- `panel_pref_topk_pairs`：`pairwise_acc = 0.6970`，`spearman = 0.4974`

但 deterministic holdout 指标没有改善，反而整体更差。

### 2.4 结论

这条路线在当前落地形式下失败。

更准确地说，不是“bank 完全没信息”，而是“当前这套 broad-to-local preference / pull extraction 还不能把 ranking 改善翻译成 deterministic BW 改善”。按预先约定的止损标准，这已经足够停止，不再继续往 live transfer 或更长训练上加码。

## 3. 第二条：给 BW 单独做 counterfactual credit

### 3.1 原命题

`bw_possibility.md` 的第二条是：不要再让 BW 吃 shared advantage，而是直接构造 BW-specific counterfactual credit。

### 3.2 先前已有的“近亲证据”

在这轮 follow-up 之前，仓库里其实已经有过一个较弱的近亲版本，不是真正的 counterfactual baseline，而是旧栈上的 BW credit proxy：

- `runs/bw_geom_credit_50u/b0_headwisesharedadv/eval_trained_n20_seed42000.csv`
- `runs/bw_geom_credit_50u/b1_headwisebwcredit/eval_trained_n20_seed42000.csv`
- 对照：
  - `runs/bw_geom_credit_50u/a0_joint/eval_trained_n20_seed42000.csv`
  - `runs/bw_geom_credit_50u/a1_bwgrad025/eval_trained_n20_seed42000.csv`

对应 deterministic reward：

- `b0_headwisesharedadv = -296.872`
- `b1_headwisebwcredit = -273.773`
- `a0_joint = 103.471`
- `a1_bwgrad025 = 106.580`

这个结果只能说明：旧栈上的“BW credit 补丁版”不好。它不能单独证明真正的 BW counterfactual credit 已经被证伪。

### 3.3 这轮新做的快判

先做机制诊断，再做训练版 v0。

诊断目录：

- `runs/_tmp_bw_credit_mismatch_u0120_clusterpartners/summary.json`

核心 mismatch 指标：

- `positive_joint_adv_fraction = 0.6641`
- `bad_local_reward_but_positive_adv_fraction = 0.5391`
- `bad_local_processed_but_positive_adv_fraction = 0.4922`
- `bad_local_backlog_but_positive_adv_fraction = 0.4844`

这一步给出的结论很明确：shared advantage 的确在搅脏 BW credit，第二条至少值得做一次训练版快判。

随后实现 structured `bw_counterfactual_credit_enabled`，并在固定 partner 的 `bwonly_clusterpartners` 口径上跑了最小训练版：

- `runs/structured_bw_cf_credit_v0_u20_20260407/checkpoint_eval.csv`
- `runs/structured_bw_cf_credit_v0_u20_20260407/eval_det_ep20.csv`
- `runs/structured_bw_cf_credit_v0_u20_20260407/eval_stoch_ep20.csv`

### 3.4 结果

checkpoint eval：

- `u10 reward_sum = 112.977`
- `u20 reward_sum = 110.599`
- 同一口径下的固定参考 `fixed_reward_sum = 162.727`

最终 20-episode eval：

- deterministic：`117.801`
- stochastic：`196.682`

### 3.5 结论

这条路线的现版快判失败。

更准确地说，第二条出现了一个“诊断为真、训练兑现为假”的结果：

- 机制诊断证明 credit 污染是真问题。
- 但当前 structured counterfactual credit v0 没能把它翻译成足够明显的 fixed-partner 改善。

因此，第二条不能说被数学上彻底证伪，但当前这条实现线已经不值得继续加时间。

## 4. 第四条：只要保留 online RL 主体，就把 BW 的更新族从 PPO clip 换掉

### 4.1 原命题

`bw_possibility.md` 的第四条是：如果还保留 online RL 主体，就不要再把希望押在 clipped ratio 上，而是转向 `REPS / MPO / V-MPO / MDPO` 一类的 KL / mirror-descent 风格 policy improvement。

### 4.2 实际落地版本

这条线做了两层快判：

1. 先用 `AWR` 做一个低成本代理实验，检查“只要把 BW update 从 PPO clip 换成 advantage-weighted update，会不会立刻明显变好”。
2. 如果 `AWR` 不够有说服力，再上一个更像原命题的 `V-MPO-lite` 最小版。

对应结果目录：

- `runs/structured_bw_awr_v0_u20_20260408`
- `runs/structured_bw_awr_v1_lowtemp_u20_20260408`
- `runs/structured_bw_awr_v2_kl_u20_20260408`
- `runs/structured_bw_vmpolite_v0_u20_20260408`

### 4.3 结果

共同 base：

- deterministic：`108.933`
- stochastic：`214.557`

`AWR` 结果：

- `v0`：det `116.750`，stoch `216.897`
- `v1 low-temp`：det `113.238`
- `v2 +KL`：det `104.420`

`AWR v0` checkpoint eval：

- `u10 reward_sum = 116.512`
- `u20 reward_sum = 116.360`
- 同口径固定参考 `fixed_reward_sum = 162.727`

`V-MPO-lite` 结果：

- deterministic：`111.737`
- stochastic：`206.326`
- `collision_episode_fraction = 0.05`

`V-MPO-lite` checkpoint eval：

- `u10 reward_sum = 117.165`
- `u20 reward_sum = 116.476`
- 同口径固定参考 `fixed_reward_sum = 162.727`

### 4.4 结论

第四条可以在工程上判为“停止”。

这里的措辞要精确一点：

- 它不能证明所有 `REPS / MPO / V-MPO / MDPO` 变体都不可能成功。
- 但它足够排除“第四条是当前阶段的高性价比主线”。

理由是：

- `AWR v0` 只有弱正信号，而且不稳。
- 一旦稍改超参，改善就不再稳健。
- 更接近原命题的 `V-MPO-lite` 也没有给出更强结果，反而 stochastic 更差，并出现 `0.05` 的碰撞占比。

按“短程、强判别”的标准，这已经足够停止 route 4。

## 5. 第三条：把 BW 改成 sequential / autoregressive / decomposition

### 5.1 原命题

`bw_possibility.md` 的第三条是：把 BW 从 one-shot simplex 头改成 sequential / autoregressive / decomposition，并把训练分布和部署读出拆开。

### 5.2 实际落地版本

这条线没有直接上 full autoregressive，而是先做了一个更适合快判的最小版：

- support 选择
- support 内做 Dirichlet 分配
- support 外保留小 tail，避免硬 top-k 直接 starvation

对应参数化是 `score_support_kappa_dirichlet`，结果目录：

- 初始化零步 eval：
  - `runs/_tmp_bwsupportv0_init_eval_det_ep20.csv`
  - `runs/_tmp_bwsupportv0_init_eval_stoch_ep20.csv`
- 短程训练：
  - `runs/structured_bw_support_v0_u10_20260408/checkpoint_eval.csv`
  - `runs/structured_bw_support_v0_u10_20260408/eval_det_ep20_seq.csv`
  - `runs/structured_bw_support_v0_u10_20260408/eval_stoch_ep20_seq.csv`

### 5.3 结果

对照 base：

- deterministic：`108.933`
- stochastic：`214.557`

零步结构替换后：

- deterministic：`100.568`
- stochastic：`188.844`

也就是说，旧 checkpoint 直接换到这套新读法后，det 和 stoch 都明显回退。

给了它一个 `u+10` 的短程恢复机会之后：

- checkpoint eval `u10 reward_sum = 112.138`
- 同口径固定参考 `fixed_reward_sum = 162.727`

最终 20-episode eval：

- deterministic：`109.199`
- stochastic：`212.197`

### 5.4 结论

第三条的这个最小版失败。

需要注意两层含义：

- 可以判死的是这条 `support + epsilon tail` 的轻量 decomposition v0。
- 不能因此数学上判死所有 full autoregressive / full sequential allocator。

但在当前项目节奏下，这已经足够说明：如果第三条还要继续做，就不再是“快判一个小变体”，而是要启动一个新的工程项目。

## 6. 最终判断

把四条路线放在一起看，到 2026-04-08 为止，可以得到下面这个更精确的结论：

1. 第一条失败在“offline ranking 改善没有翻译成 deterministic readout 改善”。
2. 第二条失败在“credit 污染诊断成立，但训练兑现不足”。
3. 第四条失败在“换更新族有弱信号，但不稳，且更 faithful 的版本也没打开局面”。
4. 第三条当前失败在“最小 decomposition 版本没有打开局面，full autoregressive 尚未进入实现阶段”。

因此，针对 `bw_possibility.md` 的 follow-up 可以收敛成一句话：

> 截至 2026-04-08，四条备选路线都没有通过“值得继续投入”的快判标准；当前最稳的工程结论仍然是 `accel=policy, sat=policy, bw=heuristic`。

如果后续还要继续做 learned BW，唯一还没有被真正试透、同时又和现有失败路线有明显归纳偏置差异的方向，只剩下“完整 autoregressive / sequential BW allocator”。但它已经不再像 `bw_possibility.md` 里这些可以快速判别的 follow-up，而更像一个新的研究项目。
