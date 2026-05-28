According to documents from 2026-04-05 and 2026-04-06, **问题不只是“bw 动作对目标影响小”**，但这确实是第一根因。更准确地说，你现在的 `bw` 是一个**只在少数状态里才有明显杠杆的前端拥塞调节动作**：大多数状态里它对 `D_sys` 近乎间接、对 `x_acc` 也只是小幅边际；只有在“当前关联集内部真的出现前两层拥塞/即将溢出”的那部分状态里，它才有明显作用。代码上，GU 先按最小路径损耗硬关联到唯一 UAV，`candidate_mode=assoc` 时候选就是这个关联集，`bw_valid_mask` 也只放行当前已关联用户；随后 `bw_alloc` 只是在这些用户上归一化成 `betas`，最后 GU 真正出流又被 `min(q_before, access_rate * tau0)` 截断。与此同时，`x_acc = outflow_sum / arrival_ref`、`x_rel = backhaul_sum / arrival_ref`、`d_pre = (gu_drop + uav_drop)/arrival_ref`、`pre_backlog_steps_eval = q_total_active/arrival_ref`，而 `D_sys_report = q_total / sat_processed_sum`，所以 `bw` 直接控制的主要是前两层的出流、积压和掉包，不是整个流水线的最终“处理完成度”。    

但如果只说“杠杆小”，又不够。因为你自己的当前总结已经说明：在旧主线里，`K=1` 的 bw override 基本没用，`K=2` 才开始出现收益，这意味着 bw 更像**短前缀多步 credit**，不是单步即时 credit；同时 pure `x_acc`、no-aux、只训 bw 的实验里，deterministic `x_acc` 从 `0.987584` 提到 `0.999850`，stochastic 到 `1.002817`，虽还没稳定超过 heuristic `1.003734/1.003015`，但也已经证明“信号不是零，actor 也不是完全学不动”。所以现在更准确的判断应当是：**第一层是 bw 的直接控制权小；第二层是这些有杠杆的状态在整条训练分布里占比太低；第三层才是 actor 在这些少数高杠杆状态里还没把局部排序学稳。** 

你前面那次 `bw_focus_env_v1` 审计没过，我现在觉得原因也更清楚了：**不是“burst 没意义”，而是你那版 burst 的作用域没有跟当前 bw 的真实作用域对齐。** 当前代码里，`sticky_subset_hotspot` 的子集是在 reset 时用当时的 `_associate_users()` 构出来的，并存成固定的 `_hotspot_member_mask`；后续每步到达率都只按这个固定 mask 乘上 `rho_hot`。但 structured 控制里，`run_accel_stage()` 之后会基于**当前** post-motion 几何重新计算 association、candidate 和 `bw_valid_mask`，也就是说 bw 真正面对的是**运动后的当前关联集**。于是你之前那版环境很可能出现：hotspot 还在“reset 时那一簇人”上，但 bw 真正能分的已经是“运动后另一套关联集”。这会直接削弱你想放大的 leverage。  

所以我现在给的方案，不再是“继续改 reward”或者“继续拍脑袋改环境参数”，而是围绕**真实高杠杆状态的占比**来做。核心思想一句话：**先别改系统语义，也别再改 target；先把训练分布重心移到“当前系统里 bw 真的有用的那些状态”上。**

第一步是做一个新的诊断量，不是“平均 leverage”，而是 **leverage occupancy**。具体做法：固定你现在最稳的 accel/sat partner，在当前原始系统上跑 rollout；每到 bw-stage 状态，就用一个小动作面板做敏感性评估，比如 `{uniform, queue_aware_bw, current policy, current policy 随机采样若干次}`，只看 `K=1` 和 `K=2` 下的 `x_acc / d_pre / pre_backlog_steps_eval` 变化。这里不是在找“真 oracle”，只是要回答：**这个状态到底是不是 bw-sensitive state**。如果某状态在这组动作面板下 `K=2` 的 `d_pre` 或 `pre_backlog_steps_eval` gap 很大，它就是你该拿来训练 bw 的状态；如果 gap 很小，它就是对 bw 近似平坦的状态。你已经有 override 审计脚本，所以这一步不需要大改框架。这个指标比“平均全局 leverage”更贴问题本身，因为你现在的问题很可能不是“系统里根本没有杠杆”，而是“有杠杆的状态太稀、太晚、太少”。 

第二步是基于这些真实 bw-sensitive state 建一个 **snapshot bank**。注意，这不是我之前说的那种随便 near-good-start 的 curriculum，而是**从你当前真实系统里采出来的、经 leverage 审计筛过的状态库**。每个 snapshot 至少要存：

* post-accel / post-sat 的 world state
* 当前 assoc / candidates / bw_valid_mask
* 当前 GU/UAV/SAT queues
* 后续若干步的随机种子或可复现实验上下文
  这样它完全保留你当前系统的几何、association、sat 语义，不会把 accel 任务换掉，也不会改动作定义。

第三步是训练流程改成三段，但**不改 reward，不改主任务，不改 actor 语义**。
Phase A：`bw-only`，冻结 accel/sat，70% rollout 从 snapshot bank 里采样，30% 保留原始 reset，horizon 只跑 8–12 步。这里目的不是“偷改任务”，而是把原来整局里很稀少的高杠杆状态大幅增密。
Phase B：把 snapshot 占比降到 50%，另外 50% 用原始 episode，从而让 bw 开始适应完整分布。
Phase C：恢复 joint 训练，回到原始 reset，只把 Phase B 学到的 bw 当初始化。
这套流程解决的是你最初自己提到的第二个根因：episode 里 accel 主导太久，bw 长时间拿不到真正有分辨率的样本。这里不是抽象地说“做 curriculum”，而是**只对经过审计确认的高杠杆 bw-stage 状态做 curriculum**。

第四步，如果你还想改环境 leverage，我现在只建议改**association-conditioned burst**，不再用 reset 固定 subset。具体做法是：每步在 `run_accel_stage()` 之后、当前 association 已经重算完之后，再在**某一架 UAV 当前的关联集内部**选 2–4 个用户进入 sticky burst，持续 10–20 steps，并保持 mean-preserving。这样 burst 压力永远和当前 bw 的实际决策域对齐，而不会像 `bw_focus_env_v1` 那样随着 UAV 运动发生错位。这个改动仍然不改几何、不改 accel 任务、不改 hard association，只是把 arrival 非平稳性从“reset 固定子集”改成“post-motion 当前关联集条件化子集”。它比上一版更贴你系统本身。  

第五步是验收标准，我建议也按“问题本身”来改。不要再先看“平均环境 gap 放大 3 倍”这种粗口径，而要看两件事。
第一件是 **leverage occupancy**：在原始 full episode 分布里，高杠杆 bw-state 的比例是多少；在 snapshot curriculum 或 association-conditioned burst 下，这个比例提高了多少。
第二件是 **在高杠杆状态上的 deterministic 能力**：不是看平均 episode，而是看 snapshot eval 上，deterministic policy 相对 uniform / reset / heuristic 在 `d_pre`、`pre_backlog_steps_eval` 和 `x_acc` 上能关掉多少 gap。你之前最新那条 fullchain `u100` 结果里，deterministic `policy/policy/policy` 的 `x_acc_mean` 只有 `0.9874`，heuristic bw 是 `1.0037`，stochastic `policy/policy/policy` 是 `0.9986`，这说明问题集中在“deterministic 在关键状态上还不够好”，而不是“系统里完全没有可学 signal”。如果 snapshot eval 上 deterministic 还是和 reset 一个量级，那才说明不是状态分布问题。

这样一来，失败分流也会很清楚。
如果你做完 leverage occupancy 统计以后发现：原始系统里高杠杆状态本来就极少，而 association-conditioned burst 也提不上来，那么答案就是“**是，系统语义下 bw 对主目标的可控空间真的太小**”，这时再讨论改 action semantics 或 association 才有意义。
如果高杠杆状态其实不少，或者经过 snapshot/burst 后已经很多，但 policy 还是学不出来，那才是**算法侧**，而且这时矛头会很集中：不是 old factorized 的老问题，而是“在当前 structured 实现里，bw 对高杠杆局部排序仍然学不稳”。这时再回去看 actor/critic/分布，才不会浪费时间。

把上面压成一句话：**现在的根因不是单纯“bw 完全没用”，而是“bw 只在少数前端拥塞状态下有用，而当前训练分布没有把这类状态以足够高的密度喂给它”；因此最正确的下一步，不是继续改 reward，也不是继续乱改环境几何，而是先在你当前系统里显式统计并重采样这些 bw-sensitive states。**

这也是我现在最推荐的执行顺序：

1. 保持当前系统语义不变。
2. 做 bw-stage 的 leverage occupancy 统计。
3. 建真实 snapshot bank。
4. 先做 bw-only 的 snapshot curriculum。
5. 不够时再上 association-conditioned burst。
6. 只有在“高杠杆状态已经很多，policy 仍不学”时，才回算法侧。

如果你愿意，我下一条就直接把这个方案写成一份**可落地的实现清单**：要在 `structured_driver.py`、训练脚本和审计脚本里加哪些字段，snapshot bank 怎么存，leverage occupancy 怎么算。
