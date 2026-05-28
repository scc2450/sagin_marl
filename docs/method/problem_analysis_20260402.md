# 2026-04-02 问题分析重整

## 1. 这份文档要回答什么

这份文档不是为了继续追问“到底是 `accel`、`bw`、`sat` 哪一个最坏”，而是要重新整理当前已经得到的证据，回答下面 4 个问题：

1. 现在到底已经证实了什么，哪些只是推断。
2. 当前 joint 失败，更像是哪一类问题。
3. 为什么之前很多实验会让人感觉“每条线都像有问题”。
4. 接下来如果要继续找根因，什么方向更符合现有证据。

这份整理的目标是减少混乱，不是为了给某一个单一真凶下结论。


## 2. 已经证实的事实

### 2.1 `bw` 有一个真实的头内优化病灶

这件事已经不需要再争论。

- `|log_ratio_bw|` 会随着 `valid_count` 增大而明显放大。
- 这个放大主要来自 Dirichlet `concentration`，不是 mean allocation 本身。
- 这说明当前 `bw` 的 PPO 几何确实不干净，不是“日志口径问题”。

主要证据：

- `runs/bw_decomp_pair_4buf_20260401/summary.json`
- `runs/privatebwtrunk_50u/validcount_logratio_execcheck_20260401/report.json`

这个结论的含义是：

- `bw` 存在一个独立于 joint 耦合之外的、真实的头内优化缺陷。
- 即使未来确认 joint 主问题更偏结构或 credit，这个 `bw` 病灶仍然需要单独处理。


### 2.2 `accel` 会强烈改变系统所处的 regime

partner-swap 和 component-split 的结果都支持这一点。

- 固定同一个 `bw`，把 partner 从 heuristic/cluster 换成 joint learned partner，性能会大幅下降。
- 再把 partner 拆开后，主要的放大器是 `accel`，不是 `sat`。
- learned `accel` 会把系统推入更差的几何/负载分布状态。

主要证据：

- `runs/partner_swap_matrix_u0050_20260401/summary.json`
- `runs/partner_component_split_jointbw_u0050_20260401/summary.json`
- `runs/partner_component_split_jointbw_regime_20260401/summary.json`

这类结果显示：

- `accel` 是最强的 regime-shaper。
- 它会先改变前端几何、关联负载、队列分布，再间接影响后续 `bw` 和 `sat` 的工作点。


### 2.3 `bw` 不只是训练期有问题，它在执行期也会直接影响表现

固定 `accel/sat`，只换 `bw` 时，heuristic `bw` 仍明显优于 learned `bw`。

主要证据：

- `runs/bw_execution_swap_1ep_20260401/summary.json`

这个实验只有 1 episode，不能拿来做精确定量结论，但它至少说明：

- `bw` 不是“只有训练时统计异常，执行时其实无关”。
- 在给定 regime 下，`bw` 执行质量本身也会拉开性能差距。


### 2.4 新 env 下的 joint 失败，不是到中后期才出现

新重跑的 joint timeline 表明，在最早可见的保存窗口里，异常已经存在。

主要证据：

- `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_u300_env12_subproc_20260401/checkpoint_eval.csv`
- `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_u300_env12_subproc_20260401/metrics.csv`

需要注意：

- 这里的最早可见点是 `u0050`，不是整个训练的真正起点。
- 因此可以说“到 `u0050` 时问题已经存在”，但不能说“问题就是从 `u0050` 开始的”。


### 2.5 `nobwtrain` 能缓解早期伤害，但不能单独解决 joint 崩坏

把 `bw` 从训练里拿掉后：

- `u0050` 会明显改善。
- `u0100` 仍会掉进坏 regime。

主要证据：

- 原 joint：
  `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_u300_env12_subproc_20260401/checkpoint_eval.csv`
- `nobwtrain`：
  `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_u100_env12_subproc_20260401/checkpoint_eval.csv`

这说明：

- `bw` 的训练病灶确实会伤害早期训练。
- 但它不是足以单独解释后续 joint 崩坏的唯一上游原因。


### 2.6 在去掉 `bw` 训练后，剩余的崩坏明显涉及 `sat/backhaul`

在 `nobwtrain` 的基础上再做 `nosattrain`，`u0100` 表现明显回升。

主要证据：

- `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_u100_env12_subproc_20260401/checkpoint_eval.csv`
- `runs/phase1_actions/joint_ka_vsat_puremappo_criticdecoupled_diag_timeline_nobwtrain_nosattrain_u100_env12_subproc_20260401/checkpoint_eval.csv`

这说明：

- 去掉 `bw` 后，剩余的明显崩坏还需要 `sat` 分支或 `accel × sat` 耦合参与。
- 这一步并不证明“只有 `sat` 有问题”，但至少说明 `sat` 不是无关项。


### 2.7 `sat` 的失败不是 overlap 惩罚造成的

相关配置里 overlap 惩罚是关着的。

主要证据：

- `configs/phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_nobwtrain.yaml`

所以当前这轮 `sat` 学歪，不该再优先归咎于 overlap reward。


### 2.8 `sat` 在 `nobwtrain u0100` 时已经学成了系统性错误的选星偏好

这部分是当前最直接、最完整的证据链。

主要现象：

- policy 选中的星明显比 heuristic 更远、仰角更低、频谱效率更差。
- 这不是 slot permutation 造成的假象。
- 也不是“还没学会，所以随机乱选”的状态，因为熵已经明显下降。
- 在同一步反事实里，policy 选到的组合几乎从来不是一步最优。
- 更关键的是，很多局部上更差的 `sat` 动作，仍然会拿到正的 joint advantage。

主要证据：

- `runs/joint_head_timeline_nobwtrain_u0100_5ep_20260401/summary.json`
- `runs/sat_selection_gap_nobwtrain_u0100_5ep_20260401/summary.json`
- `runs/sat_logit_feature_corr_nobwtrain_u0100_5ep_20260401/summary.json`
- `runs/sat_slot_permutation_nobwtrain_u0100_5ep_20260401/summary.json`
- `runs/sat_counterfactual_step_nobwtrain_u0100_1ep_stride4_20260401/summary.json`
- `runs/sat_credit_mismatch_nobwtrain_u0100_1ep_stride4_20260401/summary.json`

这部分可以稳稳地说明：

- `sat` 头当前的学习信号确实能把它推向局部更差的动作。
- 这不是单纯“训练时间不够”。


### 2.9 旧的“按动作头分开的 critic/baseline”并不等于局部 credit

这一点非常重要，因为之前已经试过一批效果差的 per-head critic 变体。

当前代码里，旧路径的本质是：

- `critic_multihead_value_enabled` 只是共享 trunk 后接多个 `V` 头。
- `_compute_per_head_gae_targets` 仍对多个头使用同一条全局 reward。
- `ppo_per_head_advantage_enabled` 产生的多个 advantage 最后又会回到 shared/joint actor 更新里。
- 它不是 `Q(s, a_component)`，也不是 counterfactual baseline。

主要代码位置：

- `sagin_marl/rl/critic.py`
- `sagin_marl/rl/mappo.py`

主要旧实验配置：

- `runs/phase1/joint_perheadbaseline_u0300_subproc12_t2_20260331/config_source.yaml`

这意味着：

- 旧 per-head critic 的失败，不足以反驳“更局部的 credit 可能有帮助”。
- 但它确实说明：简单地把 `V(s)` 分成多个头，并不能解决现在的问题。


## 3. 现在这个问题属于哪一类

按文献常见分类，当前问题不是单一类别，而是三层叠加。

### 3.1 第一层：结构化联合动作的参数化失配

当前系统的联合动作不是普通“几个弱耦合的并列动作”。

它更像：

- `accel` 决定几何、覆盖和负载分布。
- `bw` 在当前动态 valid set 上做 simplex 分配。
- `sat` 在候选卫星组合上做 `2-of-K` 的组合选择。

这类问题从建模上更接近：

- parameterized action
- multidimensional action
- structured/combinatorial action

而当前 actor 更接近：

`pi(a_accel, a_bw, a_sat | o) = pi_accel * pi_bw * pi_sat`

三头只共享一个 `ctx`，没有显式跨头条件化。

特别是 `sat` 头目前只是“对每个候选独立打分”，然后交给 `sat_num_select=2` 的 masked categorical/top-k 逻辑去采样，并没有显式建模“两颗星这个组合本身”的交互质量。

主要代码位置：

- `sagin_marl/rl/policy.py`

这说明：

- 当前参数化天然会吃亏。
- 即使 credit 完全干净，这种结构也不一定足以表达真正的联合动作偏好。


### 3.2 第二层：共享标量 advantage 的 credit 太粗

`V(s) + GAE` 在很多问题里够用，但它本质上只是给 joint action 一个共享 advantage 近似。

它解决的是时间 credit，不是自动解决“动作头之间谁该背锅、谁该领奖”的 credit。

在你这个系统里：

- `accel` 会强烈改状态分布，是 regime-shaper。
- `bw` 的质量依赖当前 valid set。
- `sat` 的局部作用会被全局 return 和别的头盖住。

所以 shared advantage 很容易出现：

- 某个头局部上选坏了，
- 但因为别的头选得好，整步 advantage 还是正的，
- 于是这个局部坏动作也被正向强化。

这件事在 `sat` 上已经有直接实验证据。

因此更准确的说法是：

- 不是 PPO/MAPPO 不能处理多动作头。
- 而是“并列多头 + 共享标量 advantage”在当前这个强耦合任务里显得过粗。


### 3.3 第三层：`bw` 还有一个独立的头内优化病灶

这一层与结构失配、粗 credit 不是同一个问题。

`bw` 现在有一个独立存在的优化缺陷：

- `valid_count` 尺度效应
- `concentration` 主导的 `log_ratio` 放大

这意味着：

- 即使未来 joint 结构和 credit 改好了，`bw` 仍然需要单独修。


## 4. 按文献看，什么情况下更适合走哪种改造

### 4.1 什么情况更适合优先做结构改造

适合结构改造的典型特征：

- 动作分量之间存在明显先后/条件关系。
- 某个动作分量的意义依赖于另一个动作分量先把状态改成什么样。
- 组合动作的质量不是各分量独立打分就能表达。
- 当前 factorized policy 无法表达联合动作中的交互项。

对应文献思路：

- parameterized action / hierarchical policy
- multidimensional action 的 autoregressive/sequence factorization
- combinatorial action 的结构化策略

当前系统符合的地方：

- `accel -> (bw, sat)` 的条件关系很强。
- `sat` 是 `2-of-K` 的组合选择，不是普通单分类。
- `bw` 是在动态 valid simplex 上分配，不是固定维度独立输出。

因此，从结构角度看，当前系统明显不属于“普通并列多头 PPO 足够”的那一类。


### 4.2 什么情况更适合优先做 credit 改造

适合 credit 改造的典型特征：

- 策略表达能力看起来大体够，但 credit 被共享 reward/shared advantage 混掉。
- 在同一状态下，能直接观察到局部坏动作经常得到正 advantage。
- 问题不完全在参数化，而是在 blame/credit 分配。

对应文献思路：

- COMA 式 counterfactual baseline
- reward/advantage decomposition
- coordinated/sequential update

当前系统符合的地方：

- `sat` 上已经直接测到“局部更差动作被正向强化”。
- `nobwtrain + nosattrain` 的对照显示，单靠 shared advantage 很难让系统稳定学好。

因此，credit 改造在当前系统里也是合理方向。


### 4.3 什么情况更适合优先做头内优化修复

适合头内优化修复的典型特征：

- 某个头已经有明确、稳定、可重复的训练病灶。
- 这个病灶不是结构或 credit 模糊推断出来的，而是直接能在训练统计里看到。

当前系统符合的地方：

- `bw` 正属于这一类。


## 5. 当前系统更像哪种情况

### 5.1 不是单头问题

现在的证据不支持：

- “只有 `accel` 有问题”
- “只有 `sat` 有问题”
- “只有 `bw` 有问题”

如果坚持找唯一坏头，反而会把证据越看越乱。


### 5.2 也不是简单的“所有头各自独立地都坏”

更准确地说：

- `bw` 的问题最像头内优化病灶。
- `sat` 的问题最像当前最清楚暴露出来的粗 credit 结果。
- `accel` 的问题最像 regime-shaping 的上游入口。

这三者不是同一种“坏”。


### 5.3 最像的总体诊断

当前系统最像的是：

> 一个强耦合、条件化、部分组合化的联合动作问题，
> 被当前实现拆成了共享 trunk 的并列多头策略，
> 再用共享标量 advantage 去训；
> 在这个系统上，`bw` 还额外带了一个独立的头内优化病灶。

换句话说：

- **系统级主问题**更接近“结构失配 + 粗 credit”。
- **局部独立问题**则明确存在一个 `bw` 优化病灶。

如果一定要问“更根上的那层是什么”，当前证据更支持：

- 根上的系统级问题是：**当前 factorized MAPPO 对这个结构化联合动作任务有失配**。
- 这不是一句“PPO 不行”，而是“当前这套动作拆法和 credit 口径，对这个任务不够合适”。


## 6. 为什么之前会感觉“到处都像有问题”

因为前面很多实验在回答不同层级的问题，但这些层级经常被混在一起解读。

### 6.1 `head-only` 实验回答的是“单独训时好不好学”

它不能直接回答：

- 哪个头在 joint 里先把系统拖坏
- 哪个头是上游原因


### 6.2 partner-swap 实验回答的是“哪个头更像当前坏结果的直接承载者”

它也不能直接回答：

- 这个头是不是因果上游


### 6.3 `nobwtrain / nosattrain` 这类实验回答的是“某条训练线是不是必要条件的一部分”

它们不是在证明某个头“单独就是根因”，而是在做排除。


### 6.4 `sat` 的局部反事实诊断回答的是“shared credit 是否在给错方向”

它说明 credit 问题真实存在，但它也不自动等于“结构一定没问题”。


因此，前面之所以会越来越乱，不是实验都没价值，而是：

- 有些实验在看执行侧症状，
- 有些在看必要条件，
- 有些在看局部 credit，
- 有些在看头内优化病灶，

如果把这些混成同一个“根因判断”，结论就会不断打架。


## 7. 当前最稳的结论边界

### 7.1 已经可以稳说的

- `bw` 有真实的头内优化病灶。
- 当前任务不是普通的弱耦合多头动作，更像结构化联合动作控制。
- plain factorized 多头 + shared scalar advantage 在这个任务里明显吃亏。
- `sat` 上已经直接暴露出 shared credit 的错配现象。
- `accel` 是最强的 regime-shaper，不是无关头。


### 7.2 还不能稳说的

- 不能说“只要修 credit 就够了”。
- 不能说“只要改结构就够了”。
- 不能说“最早的 first mover 已经被彻底找到了”。
- 不能说“旧 per-head critic 的失败已经否定了所有局部 credit 思路”。


## 8. 这份整理最终想表达什么

如果只用一句话概括当前分析：

> 现在遇到的不是“某一个动作头单独学坏”的简单问题，
> 而是当前 factorized MAPPO 在一个强耦合、条件化、带组合语义的联合动作任务上的失配；
> 这种失配表现为结构不够对、credit 不够细，
> 同时 `bw` 还带着一个独立存在的头内优化病灶。

这也是为什么：

- 单看 `accel`，会看到 regime 改坏；
- 单看 `sat`，会看到局部坏动作被错奖；
- 单看 `bw`，会看到 PPO 几何异常；

但这些都不该被简化成“唯一真凶是谁”。


## 9. 参考文献

- MAPPO:
  https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf
- COMA:
  https://www.cs.ox.ac.uk/publications/publication11394-abstract.html
- HAPPO / HATRPO:
  https://openreview.net/forum?id=EcGGFkNTxdJ
- PRD-MAPPO:
  https://openreview.net/forum?id=nfSlBFKFmq
- CoPPO:
  https://proceedings.neurips.cc/paper/2021/hash/de73998802680548b916f1947ffbad76-Abstract.html
- Policy Gradient for Multidimensional Action Spaces:
  https://openreview.net/forum?id=rk3b2qxCW
- Hybrid / Parameterized Action PPO:
  https://www.ijcai.org/Proceedings/2019/316
