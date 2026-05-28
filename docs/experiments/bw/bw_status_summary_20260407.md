# BW 改动阶段总结（2026-04-07）

## 1. 这份文档回答什么

这份文档不是按单个 `config` 或单次聊天记录整理，而是按 **BW 这条线的修改脉络** 来收束：

1. 改了什么。
2. 每条修改线想解决什么问题。
3. 这条线对应的代表性结果数据是什么。
4. 现在应当如何判断它。

本文重点回答的核心问题是：

> **“global / broad leverage 不小，但 PPO 在当前 on-policy 邻域里拿到的 local learnable signal 太浅、而且方向还有点脏。”**

截至 **2026-04-07**，我的结论是：

- 这句话在**实践意义上依然成立**。
- 但我们已经把它收窄了很多：不是“BW 没 leverage”，而是“旧参数化、旧 PPO 几何和分布族又把这个问题进一步放大了”。
- 这些放大器里，有一部分已经被修掉了；然而 **pure on-policy PPO 的 learned BW 头仍然没有稳定打赢 heuristic BW**。

---

## 2. 一页结论

### 2.1 现在已经比较稳的结论

- **BW 不是没 leverage。**
  `bw_broad2local_v1` 这套诊断已经证明 broad leverage 存在，而且不小。

- **当前最有价值的算法修复线是：**
  `Dirichlet + per_simplex_dim + per-UAV surrogate + noaux`

- **如果线上执行接受 stochastic，当前最强 all-policy BW 主线就是上面这条。**
  它已经能做出可用的 stochastic policy，而且 5-seed 下没有碰撞崩坏。

- **但 learned BW 头本身还没有超过 heuristic BW。**
  这不是猜测，而是当前 `all8 hybrid` 的直接结果。

- **所以当前最强工程解不是 all-policy，而是 hybrid。**
  就现有证据看，最强可用配置更像：
  `accel=policy, sat=policy, bw=heuristic`

### 2.2 当前推荐的状态标签

- **保留为主线**
  - structured 责任拆解与 hybrid 评估工具链
  - `geometry-only / per-UAV surrogate / raw ratio`
  - `Dirichlet + per_simplex_dim`
  - `noaux`

- **保留为历史过渡版**
  - `alpha-only`

- **降级成 ablation**
  - `tau`
  - `bw_flow_proxy_aux`

- **不再作为主解释**
  - “环境 leverage 太小”
  - “只要继续微调主 reward / target 就能解”

---

## 3. 修改线 1：旧的 bootstrap / heuristic-prior 线

### 3.1 改了什么

这条线主要出现在旧训练栈里，代表性开关和配置包括：

- `bw_head_zero_init`
- `bw_log_std_init=-1.5` / 冻结 BW 方差
- `imitation_bw`
- `eta_bw_align`
- `heuristic_residual`

代表配置可以看：

- [`configs/phase1_actions_queuefix_v3_3_stage2_full.yaml`](./../configs/phase1_actions_queuefix_v3_3_stage2_full.yaml)
- [`configs/phase1_actions_curriculum_stage2_bw_setpool_throughput_only_reward_q002_drop6_from_stage1_accel_teacher_ckpteval50_start200_norewardnorm_klstop002_frozenbackbone_zeroinit_bwstdm15_ent0_gupen005_no_backhaul_bwim_qaware_piecewise.yaml`](./../configs/phase1_actions_curriculum_stage2_bw_setpool_throughput_only_reward_q002_drop6_from_stage1_accel_teacher_ckpteval50_start200_norewardnorm_klstop002_frozenbackbone_zeroinit_bwstdm15_ent0_gupen005_no_backhaul_bwim_qaware_piecewise.yaml)

### 3.2 想解决什么

这条线针对的是最早期的两个问题：

- BW 从零开始训练时，动作太乱，起步很差。
- heuristic 很强，想先把 learned BW 往一个“别太离谱”的区域拉。

### 3.3 代表性结果数据

这条线本身后来被 structured 诊断体系覆盖掉了，所以不适合再拿单个旧 run 做最终 headline。

比较稳的代表性证据是：**即便做了这类 bootstrap / prior 引导，后续固定 accel/sat 后，learned BW 仍然打不过 heuristic BW。**

来自 [`runs/bw_execution_swap_1ep_20260401/summary.json`](./../runs/bw_execution_swap_1ep_20260401/summary.json)：

| 固定 partner | BW 头 | reward | processed | drop | backlog |
| --- | --- | ---: | ---: | ---: | ---: |
| cluster accel + cluster sat | heuristic BW | 368.2015 | 1.0536 | 0.0015 | 3.5705 |
| cluster accel + cluster sat | bwonly BW | 327.3639 | 1.0211 | 0.0300 | 5.5179 |
| cluster accel + cluster sat | joint BW | 326.3477 | 1.0202 | 0.0308 | 5.5772 |

### 3.4 当前判断

- **有用，但只是 bootstrap，不是主解。**
- 它能解决“起步太差、太飘”的问题。
- 它没有解决“learned BW 超过 heuristic BW”这个核心问题。
- 还要补一句更现实的判断：
  **这条线不是“还没认真试”，而是确实试过了，而且没有成为成功主线。**
  再加上 heuristic BW 本身也远谈不上最优，所以无论是 `imitation_bw`、`eta_bw_align` 还是 `heuristic_residual`，都容易把 learned BW 锁进一个带上限的先验附近。

---

## 4. 修改线 2：环境 leverage / target 清洗线

### 4.1 改了什么

这一线包含：

- `throughput_only`
- `pure x_acc`
- `no_backhaul`
- `bw_focus_env_v1`
- `snapshot bank`

主文档：

- [`docs/bw_focus_env_v1.md`](./bw_focus_env_v1.md)
- [`docs/bw_snapshot_bank.md`](./bw_snapshot_bank.md)
- [`docs/bw_broad2local_v1.md`](./bw_broad2local_v1.md)

### 4.2 想解决什么

这条线的核心问题是：

- 到底是不是环境里 **BW leverage 太小**，导致 actor 学不动？
- 或者，高 leverage 状态是不是太少、太稀？

### 4.3 代表性结果数据

#### 证据 A：broad leverage 是存在的，而且不小

来自 [`docs/bw_broad2local_v1.md`](./bw_broad2local_v1.md) 和 [`docs/bw_readout_geometry_v2.md`](./bw_readout_geometry_v2.md) 的 policy-independent 审计：

- `K=2 reward gap` 的 `p50 ≈ 0.221`
- `K=2 reward gap` 的 `mean ≈ 0.242`
- `82.1%` 的状态满足 `gap >= 0.1`

这已经足够反驳“高 leverage 状态极稀少”。

#### 证据 B：只清洗 target / 环境，并不能直接把 learned BW 推过 heuristic

来自 [`docs/bw_snapshot_bank.md`](./bw_snapshot_bank.md) 的总结：

- 在 pure `x_acc`、`noaux`、只训 BW 的实验里：
  - deterministic `x_acc = 0.999850`
  - stochastic `x_acc = 1.002817`
  - heuristic `x_acc = 1.003734 / 1.003015`

这说明：

- signal 不是零；
- actor 也不是完全学不动；
- 但 target 清洗本身还不足以稳定赢 heuristic。

#### 证据 C：`snapshot / high-gap` 也做过短程 runtime-bank 训练，但只是弱正信号

这条线不是只停留在文档提案。仓库里确实跑过基于 runtime state bank 的短程对照：

- [`runs/structured_local_signal_highbank_u0110_20260406`](./../runs/structured_local_signal_highbank_u0110_20260406)
- [`runs/structured_local_signal_randombank_u0110_20260406`](./../runs/structured_local_signal_randombank_u0110_20260406)

它们分别绑定了：

- `high_local_opportunity_bank.pkl.gz`
- `random_bank.pkl.gz`

来自两个 run 的 [`eval_trained.csv`](./../runs/structured_local_signal_highbank_u0110_20260406/eval_trained.csv) 与 [`eval_trained.csv`](./../runs/structured_local_signal_randombank_u0110_20260406/eval_trained.csv)：

| bank | reward | processed | drop | backlog | sat_overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| highbank | 207.9310 | 1.0770 | 0.0000 | 10.5937 | 0.8593 |
| randombank | 202.6282 | 1.0565 | 0.0000 | 10.3391 | 0.8460 |

这个结果的含义是：

- highbank 相比 random bank 有一点正信号；
- 但幅度不大，还远远称不上“终于把 learned BW 打开了局面”；
- 所以它更像**有诊断价值的弱正信号**，而不是已经跑成的主解。

### 4.4 当前判断

- **“环境 leverage 太小”已经不是主解释。**
- 环境和 target 清洗帮助我们确认了：问题更多在 **leverage 到 update 的翻译**。
- `snapshot / high-gap` 也不该被当成“还没试”的备胎路线。
  更准确地说，它已经提供了有价值的 bank / state-distribution 证据，但到目前为止**没有跑成解决 learned BW 的成功主线**。

---

## 5. 修改线 3：structured joint 重构与责任拆解

### 5.1 改了什么

这条线做了几件关键事：

- partner-swap
- component-split
- `nobwtrain / nosattrain`
- structured 三阶段 driver / actor / buffer / learner / eval

主文档：

- [`docs/problem_analysis_20260402.md`](./problem_analysis_20260402.md)
- [`docs/structured_joint_redesign_20260402.md`](./structured_joint_redesign_20260402.md)

代表性代码：

- [`sagin_marl/rl/structured_actor.py`](./../sagin_marl/rl/structured_actor.py)
- [`sagin_marl/rl/structured_mappo.py`](./../sagin_marl/rl/structured_mappo.py)
- [`sagin_marl/env/structured_driver.py`](./../sagin_marl/env/structured_driver.py)

### 5.2 想解决什么

这条线不是直接“修 BW”，而是先回答：

- 到底哪个头更像坏结果的直接承载者？
- 是 BW 自己坏，还是上游 regime 把它一起带坏？
- shared reward / shared advantage 是否在把本来局部错误的动作头一起奖进去？

### 5.3 代表性结果数据

#### 证据 A：partner 一换，性能大幅坍塌

来自 [`runs/partner_swap_matrix_u0050_20260401/summary.json`](./../runs/partner_swap_matrix_u0050_20260401/summary.json)：

| BW | partner | reward | processed | drop | backlog |
| --- | --- | ---: | ---: | ---: | ---: |
| bwonly BW | cluster partner | 225.0113 | 0.9104 | 0.1262 | 11.0252 |
| bwonly BW | joint partner | 99.3930 | 0.7649 | 0.2345 | 22.7702 |
| joint BW | cluster partner | 223.6307 | 0.9089 | 0.1275 | 11.0948 |
| joint BW | joint partner | 103.4709 | 0.7699 | 0.2296 | 22.2594 |

关键对比：

- `bwonly_bw__cluster_partner - bwonly_bw__joint_partner`
  - `reward +125.6183`
  - `processed +0.1455`
  - `drop -0.1083`
  - `backlog -11.7450`

#### 证据 B：主要放大器是 accel，不是 sat

来自 [`runs/partner_component_split_jointbw_u0050_20260401/summary.json`](./../runs/partner_component_split_jointbw_u0050_20260401/summary.json)：

| partner 组合（固定 joint BW） | reward | processed | drop | backlog |
| --- | ---: | ---: | ---: | ---: |
| cluster accel + cluster sat | 223.6307 | 0.9089 | 0.1275 | 11.0948 |
| joint accel + cluster sat | 108.1574 | 0.7770 | 0.2259 | 21.6286 |
| cluster accel + joint sat | 224.4260 | 0.9096 | 0.1269 | 10.8231 |
| joint accel + joint sat | 103.4709 | 0.7699 | 0.2296 | 22.2594 |

关键 contrasts：

- `joint_accel_gain_over_cluster_given_cluster_sat`
  - `reward -115.4733`
  - `processed -0.1320`
  - `drop +0.0983`
  - `backlog +10.5338`

- `joint_sat_gain_over_cluster_given_cluster_accel`
  - `reward +0.7953`
  - `processed +0.0007`
  - `drop -0.0007`
  - `backlog -0.2717`

#### 证据 C：`nobwtrain` 有帮助，但不够；再加 `nosattrain` 才明显回升

来自：

- [`runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_u100_env12_subproc_20260401/checkpoint_eval.csv`](./../runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_u100_env12_subproc_20260401/checkpoint_eval.csv)
- [`runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_nosattrain_u100_env12_subproc_20260401/checkpoint_eval.csv`](./../runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_nosattrain_u100_env12_subproc_20260401/checkpoint_eval.csv)

| 配置 | update | reward | processed | drop | backlog | sat_overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nobwtrain | 100 | 84.0617 | 0.6095 | 0.2186 | 49.1778 | 0.8554 |
| nobwtrain + nosattrain | 100 | 149.3919 | 0.8169 | 0.2009 | 16.1294 | 0.6986 |

### 5.4 当前判断

- **这条线已经解决了“是谁在带坏系统”这个定位问题。**
- 结论很稳：
  - `accel` 是强 regime-shaper；
  - `sat` 也会学歪；
  - shared reward / shared advantage 确实会把 credit 搅脏。

---

## 6. 修改线 4：broad-to-local / readout 诊断线

### 6.1 改了什么

这一线做的不是训练改造，而是**把问题说清楚**的诊断工具：

- leverage audit
- local finite difference
- within-state full-episode probe
- offline broad2local panel audit

主文档：

- [`docs/bw_broad2local_v1.md`](./bw_broad2local_v1.md)
- [`docs/bw_readout_geometry_v2.md`](./bw_readout_geometry_v2.md)

### 6.2 想解决什么

这条线直接回答：

- leverage 到底有没有？
- 如果有，为什么 PPO 没把它翻译成好的 local update？

### 6.3 代表性结果数据

来自 [`docs/bw_broad2local_v1.md`](./bw_broad2local_v1.md)：

- `best_reward_gain p50 ≈ 0.0194`
- `top1_hit ≈ 0.229`
- `pairwise_acc ≈ 0.609`

同时，文档还指出：

- `loc` 与 `drop_gain` 的 Spearman 均值大约 `0.747`
- 明显高于它与 `reward_gain / processed_gain / backlog_gain` 的相关性（大约 `0.25 ~ 0.30`）

这说明：

- local neighborhood 里不是完全没有更好方向；
- 但当前 actor 在局部的排序很浅，也不够稳；
- 而且学到的方向偏向 `drop` 这类 proxy，而不是 broad reward ordering。

### 6.4 当前判断

这一线得出的核心句子，**到今天仍然成立**：

> **global / broad leverage 不小，但 PPO 在当前 on-policy 邻域里拿到的 local learnable signal 太浅、而且方向还有点脏。**

后续所有几何、参数化、Dirichlet 的工作，本质上都是在继续追问：

- 是什么机制把这个问题放大了？
- 哪些机制能把它修掉一部分？

---

## 7. 修改线 5：PPO 几何线

### 7.1 改了什么

这条线主要是：

- `geometry-only`
- `per-UAV surrogate`
- `raw ratio`
- Dirichlet 下的 `valid_count > 1` mask
- `per_simplex_dim` 熵归一

主文档：

- [`docs/bw_readout_geometry_v2.md`](./bw_readout_geometry_v2.md)
- [`docs/bw参数化.md`](./bw参数化.md)
- [`docs/bw_dirichlet.md`](./bw_dirichlet.md)

代表性配置：

- [`configs/bw_geom_live_phase1_legacy_from_u0140.yaml`](./../configs/bw_geom_live_phase1_legacy_from_u0140.yaml)
- [`configs/bw_geom_live_phase1_alpha_param_from_u0140.yaml`](./../configs/bw_geom_live_phase1_alpha_param_from_u0140.yaml)
- [`configs/bw_geom_live_phase1_dirichlet_simplexdim_from_u0140.yaml`](./../configs/bw_geom_live_phase1_dirichlet_simplexdim_from_u0140.yaml)

### 7.2 想解决什么

这条线针对的是：

- joint BW `log_prob` 过度耦合；
- `valid_count / latent_count` 变化时，BW `log_prob` 与 `entropy` 的尺度一起变形；
- PPO update 几何本身在放大“local signal 浅/脏”的问题。

### 7.3 代表性结果数据

严格说，`geometry-only` 更多是**底座修复**，不是单独冲 headline 的主角。

它更重要的价值是：后续的 `alpha-only / Dirichlet` 比较，终于建立在一个相对更干净的 PPO 几何之上。

同一条几何底座上的 live matched eval（`u0120`，固定从 `u0140` geometry run 续训）如下：

来自：

- [`runs/structured_bw_geom_live_phase1_legacy_u0120_resume_20260407/matched_eval_det_seed123450_ep20_parallel.csv`](./../runs/structured_bw_geom_live_phase1_legacy_u0120_resume_20260407/matched_eval_det_seed123450_ep20_parallel.csv)
- [`runs/structured_bw_geom_live_phase1_legacy_u0120_resume_20260407/matched_eval_stoch_seed123450_ep20_parallel.csv`](./../runs/structured_bw_geom_live_phase1_legacy_u0120_resume_20260407/matched_eval_stoch_seed123450_ep20_parallel.csv)
- [`runs/structured_bw_geom_live_phase1_alpha_u0120_resume_20260407/matched_eval_det_seed123450_ep20_parallel.csv`](./../runs/structured_bw_geom_live_phase1_alpha_u0120_resume_20260407/matched_eval_det_seed123450_ep20_parallel.csv)
- [`runs/structured_bw_geom_live_phase1_alpha_u0120_resume_20260407/matched_eval_stoch_seed123450_ep20_parallel.csv`](./../runs/structured_bw_geom_live_phase1_alpha_u0120_resume_20260407/matched_eval_stoch_seed123450_ep20_parallel.csv)
- [`runs/structured_bw_geom_live_phase1_dirichlet_simplexdim_u0120_resume_20260407/matched_eval_det_seed123450_ep20_parallel.csv`](./../runs/structured_bw_geom_live_phase1_dirichlet_simplexdim_u0120_resume_20260407/matched_eval_det_seed123450_ep20_parallel.csv)
- [`runs/structured_bw_geom_live_phase1_dirichlet_simplexdim_u0120_resume_20260407/matched_eval_stoch_seed123450_ep20_parallel.csv`](./../runs/structured_bw_geom_live_phase1_dirichlet_simplexdim_u0120_resume_20260407/matched_eval_stoch_seed123450_ep20_parallel.csv)

| 分支 | 模式 | reward | processed | drop | backlog |
| --- | --- | ---: | ---: | ---: | ---: |
| legacy | det | 178.7981 | 1.0231 | 0.0268 | 17.9028 |
| legacy | stoch | 226.3097 | 1.0878 | 0.0000 | 4.0396 |
| alpha-only | det | 165.8900 | 0.9808 | 0.0396 | 20.6586 |
| alpha-only | stoch | 233.0437 | 1.0878 | 0.0000 | 2.4605 |
| dirichlet + per_simplex_dim | det | 212.8125 | 1.0808 | 0.0000 | 9.5188 |
| dirichlet + per_simplex_dim | stoch | 232.7277 | 1.0860 | 0.0000 | 2.3881 |

### 7.4 当前判断

- **几何线是正方向，而且要保留。**
- 但它不是单独的最终解释。
- 更准确地说：它给后续参数化和分布改造提供了一个更干净的训练底座。

---

## 8. 修改线 6：参数化 / 分布线

### 8.1 改了什么

这一线经历了三步：

1. 旧 `ALR logistic-normal`
2. `alpha-only`
3. `score_alpha_kappa_dirichlet`

期间还尝试过 `tau`，但没有升成主线。

主文档：

- [`docs/bw参数化.md`](./bw参数化.md)
- [`docs/bw_dirichlet.md`](./bw_dirichlet.md)

代表性代码：

- [`sagin_marl/rl/structured_actor.py`](./../sagin_marl/rl/structured_actor.py)
- [`sagin_marl/rl/distributions.py`](./../sagin_marl/rl/distributions.py)
- [`sagin_marl/rl/structured_factory.py`](./../sagin_marl/rl/structured_factory.py)
- [`sagin_marl/rl/structured_types.py`](./../sagin_marl/rl/structured_types.py)

### 8.2 想解决什么

这条线针对的是更深一层的问题：

- 旧 BW 参数化把排序、sharpness、stochastic 形状、deterministic readout 绑得太死；
- 因此即使排序学好了，也未必会转成更好的 deterministic simplex 动作。

### 8.3 代表性结果数据

#### 证据 A：`alpha-only` 第一次证明“参数化层确实是病灶之一”

来自：

- [`runs/bw_broad2local_offline_audit_geom_u0140_20260407/summary.json`](./../runs/bw_broad2local_offline_audit_geom_u0140_20260407/summary.json)
- [`runs/bw_broad2local_offline_audit_geom_param_u0140_20260407/summary.json`](./../runs/bw_broad2local_offline_audit_geom_param_u0140_20260407/summary.json)

`panel_pref_topk_pairs` 对比：

| 分支 | pairwise_acc | spearman | det_score | det_gap_to_best | det_x_acc | det_pre_backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 0.6173 | 0.2439 | 1.8245 | 0.2871 | 2.0861 | 8.8974 |
| alpha-only | 0.6023 | 0.2325 | 1.8925 | 0.2141 | 2.1516 | 8.7554 |

这说明：

- 排序指标没更漂亮；
- 但 deterministic 质量第一次一起往对的方向走。

#### 证据 B：Dirichlet 进一步把“排序改善”和“deterministic 改善”拉到同一方向

来自 [`runs/bw_broad2local_offline_audit_geom_param_dirichlet_u0140_20260407/summary.json`](./../runs/bw_broad2local_offline_audit_geom_param_dirichlet_u0140_20260407/summary.json)：

`panel_pref_topk_pairs`：

| 分支 | pairwise_acc | spearman | det_score | det_gap_to_best | det_x_acc | det_pre_backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 0.6173 | 0.2439 | 1.8245 | 0.2871 | 2.0861 | 8.8974 |
| alpha-only | 0.6023 | 0.2325 | 1.8925 | 0.2141 | 2.1516 | 8.7554 |
| dirichlet | 0.7140 | 0.6040 | 1.9364 | 0.1703 | 2.1858 | 8.1507 |

`panel_pref_topk_pairs_plus_softpull`：

| 分支 | pairwise_acc | spearman | det_score | det_gap_to_best | det_x_acc | det_pre_backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alpha-only | 0.5966 | 0.2072 | 1.9042 | 0.2024 | 2.1626 | 8.7178 |
| dirichlet | 0.7652 | 0.6678 | 1.9493 | 0.1573 | 2.1981 | 8.1274 |

#### 证据 C：`tau` 只带来边际收益，没有升成主线

来自 [`runs/bw_broad2local_offline_audit_geom_param_tau_u0140_20260407/summary.json`](./../runs/bw_broad2local_offline_audit_geom_param_tau_u0140_20260407/summary.json)：

`panel_pref_topk_pairs_plus_softpull` 下，

| 分支 | pairwise_acc | spearman | det_score | det_gap_to_best | det_x_acc | det_pre_backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alpha-only | 0.5966 | 0.2072 | 1.9042 | 0.2024 | 2.1626 | 8.7178 |
| alpha + tau | 0.6231 | 0.2788 | 1.9050 | 0.2016 | 2.1635 | 8.7221 |

也就是说：

- `tau` 对 ordering 有一点帮助；
- 对 deterministic 指标几乎没有带来新的级别变化；
- 不值得单独升成主线。

### 8.4 当前判断

- **参数化线已经证明“旧参数化确实是病灶之一”。**
- `alpha-only` 是重要过渡版，但不是最后主线。
- **当前更值得保留的是 Dirichlet 这条线。**

---

## 9. 修改线 7：`bw_flow_proxy_aux`

### 9.1 改了什么

这条线在 PPO 更新里给 BW 加了 flow proxy 的 pairwise / regression 辅助信号。

代表配置：

- [`configs/structured_joint_dirichlet_simplexdim_step250_gamma0995.yaml`](./../configs/structured_joint_dirichlet_simplexdim_step250_gamma0995.yaml)
- [`configs/structured_joint_dirichlet_simplexdim_step250_gamma0995_noaux.yaml`](./../configs/structured_joint_dirichlet_simplexdim_step250_gamma0995_noaux.yaml)

### 9.2 想解决什么

它直接针对的是：

- local signal 太浅；
- 所以想给 BW 一个更密的 proxy 排序信号。

### 9.3 代表性结果数据

最终最有判别力的不是 full-policy joint，而是**固定 accel/sat，只比较 BW 头**。

来自：

- [`runs/structured_joint_dirichlet_simplexdim_aux_u0050_env8_20260407/hybrid_eval_bwonly_det_seed123450_ep20/summary.json`](./../runs/structured_joint_dirichlet_simplexdim_aux_u0050_env8_20260407/hybrid_eval_bwonly_det_seed123450_ep20/summary.json)
- [`runs/structured_joint_dirichlet_simplexdim_aux_u0050_env8_20260407/hybrid_eval_bwonly_stoch_seed123450_ep20/summary.json`](./../runs/structured_joint_dirichlet_simplexdim_aux_u0050_env8_20260407/hybrid_eval_bwonly_stoch_seed123450_ep20/summary.json)
- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0050_env8_20260407/hybrid_eval_bwonly_det_seed123450_ep20/summary.json`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0050_env8_20260407/hybrid_eval_bwonly_det_seed123450_ep20/summary.json)
- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0050_env8_20260407/hybrid_eval_bwonly_stoch_seed123450_ep20/summary.json`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0050_env8_20260407/hybrid_eval_bwonly_stoch_seed123450_ep20/summary.json)

| 分支 | 模式 | reward | processed | drop | backlog |
| --- | --- | ---: | ---: | ---: | ---: |
| aux | det | 148.3716 | 0.9504 | 0.1077 | 10.5084 |
| noaux | det | 156.9414 | 0.9658 | 0.0980 | 9.5099 |
| aux | stoch | 146.7995 | 0.9480 | 0.1089 | 10.7332 |
| noaux | stoch | 156.1247 | 0.9646 | 0.0985 | 9.6579 |

训练吞吐也更快：

- aux `u0050`：平均 `52.045 env/s`
- noaux `u0050`：平均 `61.968 env/s`

来自：

- [`runs/structured_joint_dirichlet_simplexdim_aux_u0050_env8_20260407/bootstrap.log`](./../runs/structured_joint_dirichlet_simplexdim_aux_u0050_env8_20260407/bootstrap.log)
- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0050_env8_20260407/bootstrap.log`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0050_env8_20260407/bootstrap.log)

### 9.4 当前判断

- **这条线没有被证实为默认增益项。**
- 在最干净的“固定 accel/sat，只比较 BW”口径下，`noaux` 更好。
- 所以它现在应当降级为 **ablation**，不再是默认主线。

---

## 10. 修改线 8：Dirichlet 训练接线与实现/提速修补

### 10.1 改了什么

这一部分不是新算法结论，而是让当前主线真正可跑、可复现：

- 新增 `MaskedMeanConcentrationDirichlet`
- 新增 `score_alpha_kappa_dirichlet`
- 在 training path 里接入 `per_simplex_dim`
- 增加 `bw_kappa_*` 训练日志
- 关闭默认的 BW grad diagnostics
- 让 trace 日志可关闭
- 减少 rollout 热路径里重复 CPU/GPU/NumPy 往返

代表代码：

- [`sagin_marl/rl/distributions.py`](./../sagin_marl/rl/distributions.py)
- [`sagin_marl/rl/structured_actor.py`](./../sagin_marl/rl/structured_actor.py)
- [`sagin_marl/rl/structured_mappo.py`](./../sagin_marl/rl/structured_mappo.py)
- [`sagin_marl/rl/structured_train.py`](./../sagin_marl/rl/structured_train.py)
- [`sagin_marl/env/config.py`](./../sagin_marl/env/config.py)
- [`scripts/train_structured.py`](./../scripts/train_structured.py)

### 10.2 代表性结果数据

- 相关回归测试通过：
  - `tests/test_config_parsing.py`
  - `tests/test_structured_mappo_rollout.py`
  - `tests/test_structured_multi_env_resume_eval.py`
  - 合计 `29 passed`

- smoke run 正常：
  - [`runs/_tmp_joint_dirichlet_speed_smoke_20260407`](./../runs/_tmp_joint_dirichlet_speed_smoke_20260407)

- 当前长程 noaux 主实验吞吐：
  - 平均 `60.934 env/s`
  - 最后一个 update `63.300 env/s`

来自 [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/bootstrap.log`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/bootstrap.log)

### 10.3 当前判断

- 这部分主要是**把实验系统搭稳**，不是单独的算法结论。
- 但它是现在能够持续做 BW 长程实验的必要前提。

---

## 11. 当前主线：`Dirichlet + per_simplex_dim + noaux`

### 11.1 当前主配置

当前推荐的 all-policy BW 主线配置是：

- [`configs/structured_joint_dirichlet_simplexdim_step250_gamma0995_noaux.yaml`](./../configs/structured_joint_dirichlet_simplexdim_step250_gamma0995_noaux.yaml)

主 run：

- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407)

### 11.2 5-seed stochastic 结果

来自：

- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed123450_ep20.csv`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed123450_ep20.csv)
- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed223450_ep20.csv`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed223450_ep20.csv)
- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed323450_ep20.csv`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed323450_ep20.csv)
- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed423450_ep20.csv`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed423450_ep20.csv)
- [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed523450_ep20.csv`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/stoch_eval_seed523450_ep20.csv)

| 指标 | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| reward | 218.1305 | 8.1715 | 208.1974 | 228.3716 |
| processed | 1.0675 | 0.0093 | 1.0561 | 1.0808 |
| drop | 0.0019 | 0.0022 | 0.0000 | 0.0058 |
| backlog | 4.6608 | 1.4904 | 2.9128 | 6.2858 |
| sat_overlap | 0.3790 | 0.0145 | 0.3593 | 0.4043 |
| collision | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 11.3 同 checkpoint 的 deterministic 参考

来自 [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/eval_det_seed123450_ep20.csv`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/eval_det_seed123450_ep20.csv)：

| 模式 | reward | processed | drop | backlog | sat_overlap | collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic | 114.9201 | 0.7518 | 0.0768 | 43.3200 | 0.9940 | 0.0000 |

### 11.4 `kappa` 是否饱和

来自 [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/metrics.csv`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/metrics.csv) 最后一行：

- `bw_kappa_mean = 15.9038`
- `bw_kappa_p90 = 18.2192`
- `bw_kappa_hi_frac = 0.0`

这说明：

- `kappa` 在长程里有上升；
- 但当前没有“明显打满上界”的证据。

### 11.5 当前判断

- **如果线上执行接受 stochastic，这是一条可用的 all-policy BW 主线。**
- 但 deterministic 仍然明显弱，这意味着：
  - 它还不能支持 deterministic 执行；
  - 也不能说明 deterministic readout 已经完全学好。

---

## 12. 与 heuristic 的当前关系：最关键的一页

### 12.1 `all8 hybrid` 结果

来自 [`runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/hybrid_eval_all8_stoch_seed123450_ep20/summary.json`](./../runs/structured_joint_dirichlet_simplexdim_noaux_u0120_env8_20260407/hybrid_eval_all8_stoch_seed123450_ep20/summary.json)：

| accel | sat | bw | reward | processed | drop | backlog |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| policy | policy | policy | 206.3913 | 1.0455 | 0.0226 | 5.9101 |
| policy | policy | heuristic | 217.2038 | 1.0652 | 0.0078 | 5.1370 |
| policy | heuristic | policy | 210.4880 | 1.0552 | 0.0147 | 6.1551 |
| policy | heuristic | heuristic | 216.8493 | 1.0661 | 0.0065 | 5.9606 |
| heuristic | policy | policy | 158.3105 | 0.9687 | 0.0942 | 9.6059 |
| heuristic | policy | heuristic | 165.8530 | 0.9823 | 0.0800 | 9.3875 |
| heuristic | heuristic | policy | 158.8577 | 0.9695 | 0.0938 | 9.4946 |
| heuristic | heuristic | heuristic | 166.3939 | 0.9840 | 0.0800 | 9.2622 |

### 12.2 这张表说明什么

#### 结论 A：整体系统已经能赢纯 heuristic

- `all-policy = 206.3913`
- `all-heuristic = 166.3939`

#### 结论 B：当前最强工程解其实是 hybrid

- `policy/policy/heuristic = 217.2038`
- 明显高于 `policy/policy/policy = 206.3913`

#### 结论 C：BW policy 头自己还没赢 heuristic BW

看固定 `accel=heuristic, sat=heuristic` 这一组：

- `heuristic/heuristic/policy = 158.8577`
- `heuristic/heuristic/heuristic = 166.3939`

只换 BW 时，policy 反而更差。

同一个 summary 里给出的 isolated delta 也支持这一点：

- `bw` isolated delta vs heuristic baseline
  - `processed -0.0145`
  - `drop +0.0138`
  - `backlog +0.2324`

### 12.3 当前判断

这是现在最重要的结论：

- **all-policy 系统整体可以赢 all-heuristic；**
- **但 BW 头本身还没有赢 heuristic BW。**

这也是为什么当前最合理的工程推荐不是 “all-policy everything”，而是：

- **`accel=policy, sat=policy, bw=heuristic`**

---

## 13. 对核心问题的直接回答

### 13.1 “global / broad leverage 不小” 这半句还成立吗？

**成立。**

它已经被 leverage 审计、snapshot 恢复一致性、broad panel gap 分布等证据反复支持。

### 13.2 “local learnable signal 太浅而且方向有点脏” 这半句还成立吗？

**也成立，而且到今天仍然是更接近真相的那半句。**

不过现在可以把它说得更精确：

- 旧 `ALR logistic-normal` 参数化会放大这个问题；
- 旧 BW PPO 几何也会放大这个问题；
- shared reward / shared advantage 会继续把它搅脏；
- 即便修掉这些放大器之后，**pure on-policy PPO 的 learned BW 头仍然没有稳定超过 heuristic BW**。

### 13.3 那是不是“解决不了了”？

我不会下这个结论，但我会下另一个更重要的结论：

> **继续沿“纯 on-policy PPO + 小幅参数化微调”这条线硬拧，已经不再像高性价比主线。**

更准确地说：

- 它不是被证明“不可能”；
- 但到目前为止，它还没有把 learned BW 拉过 heuristic BW；
- 所以如果还要继续做 learned BW，下一步更像是**换范式**，而不是继续做同类小修补。

---

## 14. 现在该怎么用这些结论

### 14.1 如果目标是工程可用

当前推荐：

- **主系统**：`accel=policy, sat=policy, bw=heuristic`
- **如果一定要 all-policy**：`Dirichlet + per_simplex_dim + noaux`
- **执行口径**：以 stochastic 为主，不以 deterministic 为主 gate

### 14.2 如果目标是继续研究 learned BW

我不建议再把主力放在“继续微调 PPO 参数化”上。

更重要的是，下面这些路线已经试过，或者至少已经试到足以不该再当默认主线：

- `heuristic + residual`
- `imitation_bw / eta_bw_align / distillation`
- `snapshot / high-gap state curriculum`
- `tau`
- `bw_flow_proxy_aux`

它们各自的问题是：

- `heuristic + residual / imitation`
  不是没试过，而是效果并没有把 learned BW 推成成功主线；同时 heuristic 本身也不是令人满意的 teacher。
- `snapshot / high-gap`
  bank 审计和短程 runtime-bank 对照是有信息量的，但没有变成 decisive live solution。
- `tau`
  只带来边际 ordering 变化，没有提供足够强的新收益。
- `bw_flow_proxy_aux`
  在纯 BW 口径下，`noaux` 反而更好。

所以如果还要继续研究 learned BW，更准确的说法不是“回头重开这些老路线”，而是：

> **需要一个仓库里还没有被跑成失败版本的、新方向。**

至少截至 2026-04-07，这个“新方向”不应再默认是 heuristic-informed，也不应再默认是 snapshot/high-gap 或同类 PPO 小修补的重复尝试。

---

## 15. 最后的收束

把所有修改线压成一句话：

> **我们已经证明：BW 的问题不是“完全没 leverage”，而是“pure PPO 在当前任务上拿到的 BW 学习信号和归纳偏置还不够强”；旧参数化、旧 PPO 几何和 shared credit 会把这个问题进一步放大。Dirichlet + per_simplex_dim + noaux 已经修掉了其中一部分真病灶，也给出了一个可用的 stochastic all-policy BW 主线；但截至 2026-04-07，learned BW 头仍然没有稳定打赢 heuristic BW，所以当前最强工程解仍是 hybrid。至于继续研究 learned BW，这已经不再像‘把老路线再多试一点’的问题，而更像需要一个真正新的方向。**
