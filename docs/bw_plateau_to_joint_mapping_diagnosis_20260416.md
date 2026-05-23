# BW 平台问题到 “shared student 学不会联合映射” 的完整排查记录（2026-04-16）

## 1. 文档目的

这份文档整理的是这一轮从“`h=5` clean 主线训练到了平台”开始，到后来定位出：

- teacher target 本身大多是有用的；
- 但 batch 内部梯度更新彼此冲突；
- 更准确地说，**当前 shared student 学不会这批 target 的联合映射**；

这一整条排查过程。

这份文档不只写最终判断，也会写：

- 当时为什么会怀疑某个方向；
- 哪些结论后来被新证据推翻或修正；
- 现在哪些能说“已经证实”，哪些还只能说“更像”。

---

## 2. 最终结论先说

到目前为止，最稳妥的总结是：

1. **平台不是因为 teacher 没信号了。**
   很多 state 上，teacher 给的 `target` 仍然比当前 `ref` 动作好。

2. **平台也不主要像是“输入明显缺失”导致的。**
   目前没有找到大量“`LocalBwState` 很像、但 target 差很多”的强证据。

3. **更强的瓶颈在 shared student 这一侧。**
   具体说，是：
   - 这些 per-state 的好 target 放在一起时，
   - 当前共享参数 student 不能把它们联合拟合好，
   - 或者即使局部拟合了一部分，也会把别的状态映射带坏。

4. **标签层有 follow-policy 条件性，但不像是主因。**
   target 确实会随着 follow policy 改变，但它不是“一换 follow policy 就整体失效”的那种标签。

5. **当前最准确的问题定义不是“teacher 错了”，也不是“单一打分函数天然不行”，而是：**
   **当前这套 shared student 学不会这批 target 的联合映射。**

---

## 3. 最开始的问题：为什么训练到了平台

起点是这条主线：

- `h=5`
- exact gate
- trust-region
- `actor_lr=1e-3`

正式 60 updates 的结果在：

- [checkpoint_eval.csv](/d:/研三上/毕设/sagin_marl/runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/checkpoint_eval.csv:1)
- [metrics.csv](/d:/研三上/毕设/sagin_marl/runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/metrics.csv:1)

当时看到的是：

- fixed eval 一直在改善，但到后半段基本平台；
- 训练里的 `r` 仍然波动很大，看不出清晰上升趋势。

当时第一个问题是：

> 到底是 teacher 没信号了，还是 student 没把信号吃进去？

---

## 4. 先排除“实现噪声”和“测速误判”

在讨论平台本身之前，先做了一轮实现和性能排查，避免把工程噪声误当成学习问题。

### 4.1 Windows 子进程和清理噪声

修过：

- `BrokenPipe / EOF / Invalid argument <stdin>` 的 Windows 清理问题
- `subproc` worker 退出时的噪声

主要相关文件：

- [sagin_marl/env/structured_vec_env.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_vec_env.py:1)
- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:1)

### 4.2 clean teacher rollout 的性能

还做过：

- worker 端整段 clean rollout
- grouped / batched probe engine

核心结论是：

- 慢点主要在 exact counterfactual rollout 本身；
- 不是主环境 rollout；
- 也不是 teacher 信号太弱。

这一步的意义是：

> 后面看到平台，不会再把它误判成“因为 teacher 太慢/太弱，所以没学起来”。

---

## 5. 再判断：teacher signal 弱不弱

接着做了几类最早的诊断，想回答：

> clean 这条线是不是根本没有有效监督？

当时得到的几条关键事实是：

- `target` 相比 `ref` 的 exact return 大多数时候更好；
- `clean_target_beats_ref_frac` 长时间维持在高位；
- `mean_target_gap` 不是接近 0；
- `teacher` 不是退化成“几乎不动”的标签。

这一步的阶段性结论是：

**teacher 不是纯噪声，也不是后期完全没信号。**

但这一步还不能回答：

- 为什么有信号却平台；
- 为什么训练后期更新没有继续稳定带来收益。

---

## 6. `r` 为什么不往上走：先纠正一个误区

一开始很自然会怀疑：

> 是不是后半段 update 在回撤，所以 `r` 才不变好？

后来做了固定策略的 rollout 方差对照，脚本：

- [scripts/diagnose_structured_fixed_policy_rollout_variance.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_fixed_policy_rollout_variance.py:1)

结果文件：

- [fixed_policy_variance_h5_8env.csv](/d:/研三上/毕设/sagin_marl/runs/tmp/fixed_policy_variance_h5_8env.csv:1)

结论是：

- 即使策略完全不学，训练日志里的 `r=env_reward_mean` 自己也会大幅波动；
- 它本来就不是一个低噪声、适合看单调趋势的量。

所以这里修正了一个早期误区：

**训练里的 `r` 不上升，不能直接等于“update 一定在变坏”。**

真正更值得看的，是：

- fixed eval；
- 以及后面引入的 fixed-bank / fixed-panel 诊断。

---

## 7. 先看 horizon：平台是不是 teacher `h` 不够

我们还做过一轮不训练的 horizon sensitivity probe，避免在 `h` 上盲扫。

结果文件：

- [horizon_sensitivity_h5best_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/horizon_sensitivity_h5best_64.json:1)
- [horizon_sensitivity_h5best_20vs30_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/horizon_sensitivity_h5best_20vs30_64.json:1)

得到的结论是：

- `5 -> 10`：变化很小；
- `10 -> 20`：还有一些提升；
- `20 -> 30`：几乎饱和。

这一步很重要，因为它排除了一个常见误判：

**后期平台，不是因为“训练过程中 `h` 没继续增加，所以 naturally 平台”。**

`h` 影响的是 teacher 可见的局部 horizon 信息，
但训练平台本身不是由“`h` 没继续增大”直接解释的。

---

## 8. 第一次真正抓住问题：fixed panel / fixed bank

后面我们开始把“训练 update 到底带来了什么”拆开看。

### 8.1 fixed panel 的 pre/post

一开始用的是：

- 固定一批 panel states
- 做一次真实 online update
- 看同一批 panel 上 pre/post 变化

这个阶段有过一版 4/4 负值的 probe，曾经让我一度倾向于：

> 真实 online update 在平台附近系统性变差。

但后来又做过更大 GPU 版的 one-step probe，整网 `all-scope` 平均反而是正的。  
这说明：

- one-step probe 对口径、panel、样本数很敏感；
- **它能提供信号，但不能单独当最终判据。**

所以这一步后来的修正是：

**不要把某一版 one-step probe 结果直接写成“总规律”。**

### 8.2 fixed bank repeated fit

更关键的一步，是固定一批 snapshot bank 和 fixed teacher target，反复训练 student，看它到底能不能稳定把 actor 拉近这批 target。

重要脚本：

- [scripts/diagnose_structured_bw_fixed_teacher_fit.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_fixed_teacher_fit.py:1)

重要结果：

- [fixed_teacher_fit_h5_u60_panel32.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fixed_teacher_fit_h5_u60_panel32.json:1)

最关键的量是：

- `current_target_gap.mean`
  - step 0: `0.119`
  - step 1: `0.169`
  - step 5: `0.133`
  - step 10: `0.141`
  - step 20: `0.103`

这个结果第一次非常明确地暴露出：

**即使 target 固定不变，student 也不能稳定地把 actor 拉近这批 target。**

也就是说，问题不只是“online data 不稳”，而是：

> 在最有利的 fixed-bank 设定里，当前 student 也没有表现出稳定 target-fit。

这一步直接推翻了一个更早的模糊想法：

- 不是“student 只是 online 吃不进去”；
- 更准确地说，是“student 连 fixed bank 上已有的好 target 都吃不稳”。

---

## 9. first-step target 到底对不对：插值实验

为了区分：

- 是 first-step target 本身错；
- 还是后续 follow policy 变坏；

我们做了 first-step interpolation 诊断：

- 固定 `step10` 或 `step20` 的 actor 作为 follow policy
- 在同一批 snapshot 上
- 只插值第一拍动作，从 current 到 fixed target
- 看 return 是否随 “更接近 target” 单调变化

脚本：

- [scripts/diagnose_structured_bw_fixed_target_interpolation.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_fixed_target_interpolation.py:1)

结果：

- [user0_fixed_target_interpolation_step10_step20.json](/d:/研三上/毕设/sagin_marl/runs/tmp/user0_fixed_target_interpolation_step10_step20.json:1)

结论非常清楚：

- 在固定同一个 follow policy 时，
- 第一拍动作越接近 fixed target，
- return 越好。

所以这一步把一个关键问题坐实了：

**坏的不是 first-step target，坏的是后续 follow policy。**

这也修正了另一个常见误解：

- 不能说“更接近 target 但 return 更差，所以 target 有问题”
- 更准确的是：
  - `current_target_gap` 只看第一拍动作；
  - `panel_current_return` 看整段 rollout；
  - first-step 变好和整段变好不是同一件事。

---

## 10. 梯度冲突：第一次直接看到 batch 内部怎么打架

接下来开始查：

> 为什么这些看起来好的 per-state target，一起学就不稳？

我们做了 per-sample gradient conflict 诊断。

核心脚本：

- [scripts/diagnose_structured_bw_clean_gradient_conflict.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_clean_gradient_conflict.py:1)

早期关键结果之一：

- `loc_head` 上：
  - `batch_vs_panel_cos ≈ -0.9977`
  - 换 seed 仍然 `≈ -0.9928`
- `neg_cos_frac ≈ 0.50`

这说明：

1. batch 内部，不同样本的 clean 监督梯度确实在互相打架；
2. 更关键的是，平均出来的更新方向，在 `loc_head` 上和 held-out states 想要的方向几乎反着。

这一步之后，问题第一次被比较清楚地表述成：

**不是 teacher 没信号，而是很多 per-state target 一起学时，梯度在共享参数空间里互相拉扯。**

但这里也有一个后来的修正：

- “有梯度冲突”本身是正常现象；
- 真正不正常的，是**冲突平均后的方向对整体不利**。

所以不能把结论简单写成：

- “梯度冲突存在，所以网络不行”

更准确的是：

- **这些冲突叠加后，当前 student 没法把它们联合实现好。**

---

## 11. 输入缺失吗：`LocalBwState` 很像但 target 差很多吗

为了避免把问题错误归咎为“输入不够”，我们专门查过：

> 会不会存在很多 state：对 actor 来说看起来差不多，但 teacher 却要求完全不同的 target？

相关脚本：

- [scripts/diagnose_structured_bw_local_state_target_ambiguity.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_local_state_target_ambiguity.py:1)

相关结果：

- [local_state_target_ambiguity_h5_u60_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/local_state_target_ambiguity_h5_u60_64.json:1)
- [local_rep_target_ambiguity_h5_u60_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/local_rep_target_ambiguity_h5_u60_64.json:1)

主要结论：

- 没有找到大量“state 很近但 target 差很多”的强证据；
- 在 `user_0 / user_1 / fused / loc_src` 这些内部表示上，也没看到明显的 target 歧义灾难。

这一步的重要意义是：

**当前没有足够证据把问题直接定成“输入缺失”或“同态不同标”。**

所以后来当我们说“shared student 学不会”时，语气必须更精确：

- 不是说“actor 看不到关键信息，所以必然学不会”；
- 而是说“在当前输入和表示条件下，这个 shared student 没能把这批 target 联合拟合好”。

---

## 12. 结构实验：为什么不是“单一打分函数天然不行”

中间我们一度有过一个过强的说法：

> 单一共享打分头太受限，所以学不会这些 target。

这句话后来被明确修正了。原因是：

- heuristic baseline 本身就是同一套共享打分规则；
- 而且效果还比当前训练出来的 actor 更好。

也就是说：

**问题不是“共享单一打分函数天然不行”，而是“当前学出来的共享打分函数不稳定，且没有学成一个像 heuristic 那样稳定、任务对齐的规则”。**

这一步非常关键，因为它改变了后面结构分析的出发点。

### 做过的结构尝试

我们试过：

- `user0`
- `user0 + residual fused`
- `fused_moe2`
- `fused_ctx_dot`

其中：

- `user0` 的 fixed-target 优化幅度大，但端到端表现不如 `fused`
- `user0 + residual fused` 从头短训不如基线
- `fused_moe2` 在强偏置下基本等于旧 `fused`，弱偏置下 fixed-bank 有点信号，但还不足以说明端到端会更好
- `fused_ctx_dot` 在 fixed-bank target-fit 阶段就明显掉队

相关结果包括：

- [fused_vs_fused_moe2_fixed_bank_fit.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fused_vs_fused_moe2_fixed_bank_fit.json:1)
- [fused_vs_fused_moe2_fixed_bank_fit_lessbiased.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fused_vs_fused_moe2_fixed_bank_fit_lessbiased.json:1)
- [fused_vs_fused_ctx_dot_fixed_bank_fit.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fused_vs_fused_ctx_dot_fixed_bank_fit.json:1)
- [user0_fixed_bank_fit_on_fused_teacher_bank.json](/d:/研三上/毕设/sagin_marl/runs/tmp/user0_fixed_bank_fit_on_fused_teacher_bank.json:1)

这一步的阶段性结论是：

- 不是任何“更显式/更强”的头都会自然更好；
- 结构确实重要，但不能靠一两次 ad-hoc 改头就下总判断；
- 更大的问题仍然是 shared student 的联合拟合能力，而不是某个具体 head 单点 bug。

---

## 13. 两层验证：标签层 vs shared student 层

到这里，问题已经收缩成两个更标准的层次：

1. **标签层问题**
   - 这些 target 是不是强依赖 old follow policy？
   - 它们是不是根本拼不成一张新 policy？

2. **shared student 问题**
   - 就算标签单独都好，当前 shared student 能不能把它们联合拟合？

### 13.1 cross-follow-policy consistency

脚本：

- [scripts/diagnose_structured_bw_cross_follow_policy_consistency.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_cross_follow_policy_consistency.py:1)

结果：

- [cross_follow_policy_consistency_h5_u60_16.json](/d:/研三上/毕设/sagin_marl/runs/tmp/cross_follow_policy_consistency_h5_u60_16.json:1)

主要现象：

- `step10` 和 `step20` 生成的 target 差别不小：
  - `target_shift_l1.mean ≈ 0.495`
- 但 target 不是“一换 follow policy 就全失效”：
  - `cross_10_target_under_20` 仍然平均正增益，`beats_ref_frac = 0.75`
  - `cross_20_target_under_10` 平均接近 0，`beats_ref_frac = 0.5625`

这一步的结论是：

**标签层有条件性，但它不像是主因。**

也就是：

- target 的确不是完全静态真值；
- 但它也不是“完全不联合自洽”的那种标签。

### 13.2 lookup vs shared student

脚本：

- [scripts/diagnose_structured_bw_lookup_vs_shared_fit.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_lookup_vs_shared_fit.py:1)

结果：

- [lookup_vs_shared_fit_h5_u60_16.json](/d:/研三上/毕设/sagin_marl/runs/tmp/lookup_vs_shared_fit_h5_u60_16.json:1)

这个实验非常关键：

- fixed bank
- fixed target
- same frozen follow policy
- 比较：
  - 共享 student：当前 `fused` 单头，只训 `loc_head`
  - 非共享 lookup：每个样本自己一组 logits

主要结果：

- `shared student`
  - `current_target_gap`: `0.0748 -> 0.0848`
  - return 基本没改善

- `lookup student`
  - `current_target_gap`: `0.864 -> 0.0097`
  - final return `≈ -631.67`
  - fixed target return `≈ -631.64`

这一步把问题非常清楚地定住了：

**这批 fixed target 本身不是“不可能”。**

因为：

- 非共享 lookup 几乎可以把它们拟合到位；
- return 也几乎追上 fixed target。

所以真正更像的问题不是：

- “teacher 标签本身根本拼不起来”

而是：

- **当前 shared student 做不到这种联合拟合。**

---

## 14. 现在的最准确表述

走到这一步，问题的表述已经和最开始很不一样了。

### 最开始的模糊版本

- 是不是 teacher 变弱了？
- 是不是 `h` 不够？
- 是不是 loc_head 太黑盒？
- 是不是输入缺了？

### 现在更准确的版本

**当前平台的主因更像是：**

- teacher 给的 per-state target 大多仍然有用；
- 这些标签确实有一定 follow-policy 条件性，但没强到整体无效；
- 非共享 lookup 能把这批 target 几乎拟合到位；
- 但当前 shared student 学不会这批 target 的联合映射；
- 训练 update 因此会出现：学这批 state 时，把别的 state 一起带偏。

一句话总结就是：

**主问题不在 teacher，而在 current shared student。**

---

## 15. 哪些结论后来被推翻或修正了

为了避免以后回看时只剩结论，这里单独列一下被修正过的点。

### 被修正 1：`r` 不上升 = update 在变坏

修正后：

- `r` 本身噪声就很大，不能直接当主证据。

### 被修正 2：平台因为 `h` 不再增大

修正后：

- `h` 影响 teacher horizon 信息，但不能解释训练过程里的平台。

### 被修正 3：单一共享打分函数天然不行

修正后：

- heuristic 本身就是共享打分规则，而且更好；
- 问题是当前 learned 共享规则不稳定，不是“共享规则”这个形式天然不行。

### 被修正 4：问题主因就是 `loc_head` 太黑盒

修正后：

- `loc_head` 是问题最明显的暴露点；
- 但根因更准确地说是：
  - shared-parameter student 的联合拟合失败，
  - 而不是某一个 head 形式单独导致一切。

### 被修正 5：target 后期可能变坏了

修正后：

- 插值实验和 panel 检查都说明 first-step target 本身局部方向没坏；
- 坏的是后续 follow policy。

---

## 16. 涉及到的关键代码和脚本

### 训练 / actor / clean teacher 主逻辑

- [sagin_marl/rl/structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:1)
- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:1)
- [sagin_marl/rl/structured_factory.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_factory.py:1)
- [sagin_marl/env/structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:1)
- [sagin_marl/rl/structured_stage_builders.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_stage_builders.py:1)
- [sagin_marl/rl/baselines.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/baselines.py:516)

### 关键诊断脚本

- [scripts/diagnose_structured_fixed_policy_rollout_variance.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_fixed_policy_rollout_variance.py:1)
- [scripts/diagnose_structured_bw_fixed_teacher_fit.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_fixed_teacher_fit.py:1)
- [scripts/diagnose_structured_bw_fixed_target_interpolation.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_fixed_target_interpolation.py:1)
- [scripts/diagnose_structured_bw_clean_gradient_conflict.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_clean_gradient_conflict.py:1)
- [scripts/diagnose_structured_bw_local_state_target_ambiguity.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_local_state_target_ambiguity.py:1)
- [scripts/diagnose_structured_bw_cross_follow_policy_consistency.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_cross_follow_policy_consistency.py:1)
- [scripts/diagnose_structured_bw_lookup_vs_shared_fit.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_lookup_vs_shared_fit.py:1)

### 关键结果文件

- [runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/checkpoint_eval.csv](/d:/研三上/毕设/sagin_marl/runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/checkpoint_eval.csv:1)
- [runs/tmp/fixed_policy_variance_h5_8env.csv](/d:/研三上/毕设/sagin_marl/runs/tmp/fixed_policy_variance_h5_8env.csv:1)
- [runs/tmp/horizon_sensitivity_h5best_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/horizon_sensitivity_h5best_64.json:1)
- [runs/tmp/horizon_sensitivity_h5best_20vs30_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/horizon_sensitivity_h5best_20vs30_64.json:1)
- [runs/tmp/fixed_teacher_fit_h5_u60_panel32.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fixed_teacher_fit_h5_u60_panel32.json:1)
- [runs/tmp/user0_fixed_target_interpolation_step10_step20.json](/d:/研三上/毕设/sagin_marl/runs/tmp/user0_fixed_target_interpolation_step10_step20.json:1)
- [runs/tmp/local_state_target_ambiguity_h5_u60_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/local_state_target_ambiguity_h5_u60_64.json:1)
- [runs/tmp/local_rep_target_ambiguity_h5_u60_64.json](/d:/研三上/毕设/sagin_marl/runs/tmp/local_rep_target_ambiguity_h5_u60_64.json:1)
- [runs/tmp/cross_follow_policy_consistency_h5_u60_16.json](/d:/研三上/毕设/sagin_marl/runs/tmp/cross_follow_policy_consistency_h5_u60_16.json:1)
- [runs/tmp/lookup_vs_shared_fit_h5_u60_16.json](/d:/研三上/毕设/sagin_marl/runs/tmp/lookup_vs_shared_fit_h5_u60_16.json:1)

---

## 17. 当前阶段最值得记住的一句话

**这轮排查最后得到的不是“teacher 错了”，也不是“共享打分函数天然不行”，而是：当前 shared student 学不会这批 target 的联合映射。**

这就是为什么：

- per-state target 单独看常常是好的；
- 非共享 lookup 几乎能把它们吃进去；
- 但 shared student 一起学时会互相拉扯，最后平台。
