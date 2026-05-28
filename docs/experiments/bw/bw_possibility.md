According to a document from 2026-04-07, 我的直接判断是：**有可能，但不是沿着你现在这条“pure on-policy PPO + 同类小修补”主线继续硬拧出来；要继续做，得换范式。** 你这份总结已经很清楚地说明了三件事：broad leverage 是有的；`Dirichlet + per_simplex_dim + noaux` 已经做出了**可用的 stochastic all-policy BW**；但 **learned BW 头在固定 partner 下还没赢 heuristic BW，而且 deterministic readout 明显弱**，所以工程上当前最强解仍是 `accel=policy, sat=policy, bw=heuristic`。

换句话说，如果“训练出 BW”指的是“有一个能跑的 stochastic BW policy”，你其实已经部分做到了；如果指的是“在 fixed accel/sat 下，learned BW 稳定超过 heuristic BW，并且 deterministic 可部署”，那截至这份文档还没有做到。文档里给出的 all8 hybrid 结果、deterministic/stochastic 对比和最终总结都支持这个判断。  

所以我对“还有没有可能”的回答是：**有，但只在“换问题表述 + 换学习范式”的前提下。** 我对“最终一定能超过 heuristic”没有把握；但我对另一点把握很高：**继续把主力放在 pure on-policy PPO、再做一轮参数化/aux/tau/snapshot/teacher 小修补，性价比已经很低了。** 这点其实就是你文档自己的结论。 

我会优先考虑下面四条线，而且只考虑这四条，不再回头重押旧线。

* **第一条：把 bank 从 curriculum 改成 dataset。**
  你文档里已经说明 high-gap / snapshot bank 作为 curriculum 只有弱正信号，没有成为 decisive live solution；但这不等于 bank 没价值。更像是：它不适合继续当“让 PPO 自己悟”的采样分布，适合改成**离线策略提取的数据集**。IQL 的核心就是**不直接给数据外动作做 Q 查询**，而是用 expectile value 再配合 advantage-weighted behavioral cloning 提策略；AWAC 和更近的 hybrid on-policy NPG 也都是“用离线/次优历史数据起步，再在线微调”。对你这里最自然的做法是：固定 accel/sat partner，收集 `heuristic BW + 当前 stochastic BW + 局部 simplex 搜索得到的 better actions`，然后做 weighted BC / IQL 式的 BW policy extraction，最后只做小步 online finetune。 ([offline-rl-neurips.github.io][1])

* **第二条：给 BW 单独做 counterfactual credit，而不是再吃 shared advantage。**
  你文档里最硬的一条系统证据，其实不是“BW 自己学不动”，而是 **accel 是强 regime-shaper，shared reward/shared advantage 在搅脏 credit**。这和 cooperative MARL 里经典的 credit assignment 问题是同型的；COMA 的做法就是固定其他 agent 的动作，只对当前 agent 做 counterfactual baseline。落到你这里，就是直接构造 `Q(s, a_acc, a_sat, a_bw)`，再用 `A_bw = Q(full) - E_{a'_bw}[Q(s, a_acc, a_sat, a'_bw)]` 更新 BW。顺手说一句：在我目前看到的 config 片段里，我看到了 `sat_counterfactual_credit_enabled`，但没有看到对称的 `bw_counterfactual_credit_enabled` 字段；至少从我手头这部分信息看，**BW 专属 counterfactual credit 还不像一个已经明确跑过的主线**。  ([arXiv][2])

* **第三条：把 BW 从 one-shot simplex 头改成 sequential / autoregressive / decomposition，并把训练分布与部署读出彻底拆开。**
  你现在最强的异常信号是：**stochastic 可用，但 deterministic 很差。** 这更像“policy distribution 学到了一些东西，但 deterministic readout 不该继续复用同一个头的默认读法”。而且你的离线审计脚本里已经有 `bw_deterministic_readout`、`simplex_argmax_logprob`、`opt_steps` 这些接口，说明系统里本来就已经承认“部署读出”可以独立于训练分布。针对 allocation / simplex 任务，外部文献其实也在往这个方向走：Dirichlet 本身在 simplex 上是合理底座，不是错路；但进一步要么走 **action-space decomposition**，要么走 **autoregressive / sequential allocation**，再配合 pruning/flooring，往往比直接在原始约束空间上学一个通用头更稳。PAKDD 2023 的 ADBO、NeurIPS 2024 的 PASPO，以及 2025 的 MARL-OD-DA 都是这条思路。 ([arXiv][3])

* **第四条：只要还保留 online RL 主体，就把 BW 的更新族从 PPO clip 换掉。**
  你文档已经把“旧 PPO 几何在放大问题”说得很清楚了。那我就不会再把希望押在 clipped ratio 上，而会直接看 **REPS / MPO / V-MPO / MDPO** 这一类基于 relative-entropy / KL / mirror descent 的 policy improvement。REPS 是从 relative-entropy 约束出发来避免信息丢失和 premature convergence；MPO/V-MPO 是同一路线的 practical policy improvement 版本，其中 V-MPO 是 on-policy；MDPO 则明确把 PPO/TRPO 放回 mirror-descent 视角，并给出与 PPO/TRPO 至少相当、常常更好的结果。对你这种“broad leverage 有，但 PPO 在 local neighborhood 里翻译不出来”的问题，这比继续扫 `tau` 或 `bw_flow_proxy_aux` 更对症。 ([信息系统与自动化研究所][4])

反过来说，我**不会**再把主力放回这些方向：`heuristic + residual`、`imitation_bw / eta_bw_align`、`snapshot/high-gap curriculum`、`tau`、`bw_flow_proxy_aux`。你的文档已经不是“还没认真试”，而是“已经试到足以不该再当默认主线”。

为了避免再陷入一轮很长的试错，我建议以后所有“新 BW 方向”都先过三个 gate，再决定要不要继续：

* **Gate 1：固定 accel/sat 后，BW 单头先赢 heuristic BW。** 这应该是第一关，不要再一上来就看 full joint。
* **Gate 2：deterministic 至少吃掉一半 stochastic gap。** 你现在 deterministic 太差，不解决这个问题，部署意义很弱。
* **Gate 3：在 all8 hybrid 里，`policy/policy/policy` 不能继续明显输给 `policy/policy/heuristic`。** 这才说明 learned BW 真开始替代 heuristic。

所以最后我给你的结论是：

**BW 不是“没可能”。**
**但“用这套 pure on-policy PPO 继续磨”基本已经不该是主线。**
**真正还值得做的，是：offline-to-online BW extraction、BW-specific counterfactual credit、autoregressive/decomposed BW、以及 KL / mirror-descent 风格的 policy improvement。**

要把这四条再收成更具体的“先改哪几个文件、先跑哪三组实验”，我接下来最需要三样系统信息：

1. BW 动作维度和 `valid_count` 分布；
2. 现在是否能固定 accel/sat，对单个 state 的 BW 做 batched full-episode re-eval；
3. critic 目前能不能比较自然地扩成 `Q(s, a_acc, a_sat, a_bw)`，而不只是 `V(s)`。

[1]: https://offline-rl-neurips.github.io/2021/pdf/24.pdf "https://offline-rl-neurips.github.io/2021/pdf/24.pdf"
[2]: https://arxiv.org/abs/1705.08926 "https://arxiv.org/abs/1705.08926"
[3]: https://arxiv.org/abs/2201.08445 "https://arxiv.org/abs/2201.08445"
[4]: https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JanPeters/Peters2010_REPS.pdf "https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JanPeters/Peters2010_REPS.pdf"
