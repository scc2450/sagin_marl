# BW 共享规则读出头方案（2026-04-16）

## 1. 背景与关键纠正

当前 BW clean 线已经确认几件事：

1. `target` 对很多单个 state 来说是好的。
2. 但训练更新网络后，整段 rollout 的评估并不能持续变好。
3. 真正坏掉的不是 first-step target 本身，而是更新后的 follow policy。

这里有一个很重要的纠正：

- 不能把当前问题简单解释成“单一打分函数不够”。
- 因为 heuristic 在 `sagin_marl/rl/baselines.py` 里的 `queue_aware_bw_policy()` 本质上也是同一个共享打分规则：
  - `weights = q * (0.5 + eta)`
  - 可选乘 `1 + assoc_bonus * prev`
  - 再归一化成带宽分配
- heuristic 仍然比当前训练出来的 actor 更稳、更强。

所以当前更准确的问题定义不是：

- “共享单头天然不行”

而是：

- **当前通过深度网络学出来的共享打分规则，没有学成 heuristic 那样稳定、显式、任务对齐的规则；它在每次 clean update 后会漂，并且会把别的 state 一起带偏。**

---

## 2. 当前最稳妥的原因分析

### 2.1 已证实的部分

1. `target` 本身不是主要问题。
   - 插值实验已经说明：在固定 follow policy 时，first-step 动作越接近 fixed target，return 越好。
   - 说明 target 的局部方向是对的。

2. 真正的问题出在共享参数网络更新后，后续状态上的策略变差。
   - 也就是说：
     - 初始这批 state 上，朝 target 走往往是对的；
     - 但同一次网络更新，会把 downstream states 上的动作一起改掉；
     - 整段 return 最后被 downstream states 的变差拖垮。

3. 这不是 teacher 质量问题，也不只是采样噪声问题。
   - 现有证据更支持：
   - **共享参数网络在拟合当前 batch 的 target 时，会把别的状态上的映射一并改坏。**

### 2.2 现在真正缺的是什么

不是“更复杂的 teacher”，而是：

- 一个**稳定的共享打分规则**
- 并且这个规则要能随 state 变化，但不能像现在这样每次 update 后整体漂移

heuristic 的优势，本质上不是“它简单”，而是：

- 规则显式
- 规则稳定
- 规则与任务直接对齐

当前 learned actor 的劣势，本质上是：

- 规则不显式
- 规则藏在黑盒 `loc_head(embedding)` 里
- 更新时很容易为了当前 batch 改坏别的 state

---

## 3. 当前结构为什么容易漂

当前 neural BW actor 的关键路径在：

- `sagin_marl/rl/structured_actor.py`

主要流程是：

1. `user_0 = user_encoder(...)`
2. `user_1 = user_0 + user_refine(...)`
3. `fused = user_fusion(...)`
4. `score = loc_head(...)`
5. `action = masked_softmax(score)`

也就是现在更像在做：

\[
score_i = \text{MLP}(fused_i)
\]

问题在于：

- `fused_i` 里已经混了大量上下文
- `loc_head` 又是一个黑盒逐槽位 MLP
- “当前 state 下到底按什么规则给各 user 打分”并没有被显式表示出来

因此一次 update 后，改变的不是一个清晰的规则，而是：

- slot 表示
- 读出规则
- 以及它们之间的纠缠关系

于是就容易出现：

- 这批 state 更接近 target
- downstream states 的打分规则却一起漂了

---

## 4. 方案目标

本方案的目标不是把 heuristic 形式硬塞回网络，也不是搞 heuristic residual。

目标是：

- **让网络自己学一个“显式、共享、可条件化”的打分规则**
- 同时把“局部内容”和“全局规则”分开

换句话说，我们不是直接让网络输出：

\[
score_i = \text{black-box}(slot_i)
\]

而是让网络先学：

- 当前这个 state 下，打分标准是什么

再拿这个标准去给每个 slot 打分。

---

## 5. 具体结构方案

### 5.1 核心公式

方案的核心形式是：

\[
score_i = \langle w(c), z_i \rangle
\]

其中：

- `z_i`：第 `i` 个 user 槽位的局部内容表示
- `c`：当前 state 的全局上下文
- `w(c)`：当前这个 state 下的共享打分规则

也就是：

- 先从整个 state 里生成一条“当前该怎么打分”的规则
- 再让所有 user 都按这同一条规则来打分

### 5.2 `z_i` 用什么

建议：

\[
z_i = P(user0_i)
\]

也就是：

- 从 `user_0` 出发
- 过一个线性投影得到低维 slot content

原因：

- `user0_i` 更接近局部槽位内容
- 不把全局上下文提前混进 slot content
- 这样“内容”和“规则”分得最清楚

### 5.3 `c` 用什么

建议：

\[
c = \text{pool}(fused_1,\dots,fused_n)
\]

例如：

- 对 valid slots 做 masked mean pooling

原因：

- 前面实验已经说明纯 `user0` 信息不够
- 全局上下文仍然是需要的
- 但它更适合用来表达“当前局面下该按什么规则排序”

### 5.4 `w(c)` 怎么得到

建议：

\[
w(c) = \text{rule\_head}(c)
\]

其中 `rule_head` 是一个小 MLP，输出低维规则向量。

这样做的意思是：

- 不同 state 可以有不同规则
- 但规则变化被压缩在一个低维向量里
- 不会像现在的黑盒逐槽位 MLP 那样容易乱漂

### 5.5 为什么这样比当前结构更稳

因为现在学的是：

- “每个 slot 直接出多少分”

而新结构学的是：

- “当前 state 的共享打分规则是什么”

这会带来两个约束：

1. 所有 slot 在同一个 state 内必须共享同一条规则
2. 规则变化必须通过低维 `w(c)` 来表达

所以它天然更接近：

- heuristic 的“共享规则”优点

但又没有把 heuristic 的具体公式硬编码进去。

---

## 6. 不是 heuristic residual

这点需要明确：

- 本方案不依赖 heuristic 公式
- 不引入 heuristic 打分 residual
- 不做 `heuristic + learned residual`

本方案的归纳偏置只有一条：

- **把“规则”显式化**

让网络自己学：

- 当前 state 下该用什么共享规则打分

而不是继续用黑盒 MLP 直接逐槽位出分。

---

## 7. 代码层面的最小改法

### 7.1 新增一个 `loc_readout` 模式

在以下文件里新增枚举值，例如：

- `fused_ctx_dot`

涉及文件：

- `sagin_marl/env/config.py`
- `sagin_marl/rl/structured_actor.py`

### 7.2 在 `BwLocReadoutHead` 中新增分支

当前读出头在：

- `sagin_marl/rl/structured_actor.py`
  - `class BwLocReadoutHead`

最小结构建议增加：

- `slot_proj: Linear(embed_dim -> rule_dim)`
- `rule_head: MLP(embed_dim -> rule_dim)`
- 可选一个可学习温度 `rule_temp`

建议 `rule_dim` 先用比 `embed_dim` 更小的值，例如：

- `rule_dim = 32`

故意限制规则自由度，防止继续漂。

### 7.3 前向计算

在 `BwLocReadoutHead.forward()` 里新增类似逻辑：

1. 取 `user_0` 做 slot content：

\[
z = slot\_proj(user_0)
\]

2. 取 pooled `fused` 做 context：

\[
c = masked\_mean(fused)
\]

3. 由 context 生成规则：

\[
w = rule\_head(c)
\]

4. 打分：

\[
score_i = \langle w, z_i \rangle / \sqrt{d}
\]

然后继续走现有：

- masked softmax
- clean teacher
- exact gate
- trust region

其它部分第一版都不动。

---

## 8. 为什么选择 `user0` 做 `z_i`，`fused` 做 `c`

这不是拍脑袋，而是来自已有现象：

1. `user0` 更像局部内容
   - 作为 slot content 更自然
   - 也更容易让“规则”和“内容”分开

2. 纯 `user0` 端到端不够好
   - 说明全局信息仍然需要

3. `fused` 包含更多上下文
   - 适合做当前 state 的规则条件
   - 但不适合直接再作为 slot content 去走黑盒读出

所以这一版的分工是：

- `user0` 负责“这个 user 是什么”
- pooled `fused` 负责“当前该按什么标准排序”

---

## 9. 验证方法

### 9.1 第一阶段：fixed-bank target-fit

不要一上来跑长训练，先做辅助验证。

口径：

- 固定一批 snapshot
- 固定 teacher target
- 冻结 trunk
- 只训练读出头

对比：

- 现有 `fused` 单头
- 新的 `fused_ctx_dot`

主看：

1. `current_target_gap` 能否更稳定下降
2. `panel_current_return.mean` 是否不再像当前这样很快变坏

### 9.2 第二阶段：短训

只有第一阶段有正信号时，再跑 `u10` 短训。

主判据：

- `checkpoint_eval reward_sum`

不再主要看训练里的 `r`。

---

## 10. 本方案要解决的根问题

一句话概括：

**当前网络没有学成一个稳定的共享打分规则，而是在每次 clean update 后把整套规则拧来拧去。**

本方案的作用不是让网络“更强”这么抽象，而是：

1. 让共享规则显式化
2. 让局部内容与全局规则分离
3. 让规则变化更低维、更平滑
4. 减少“学这批 target 时把别的 state 一起带偏”

---

## 11. 相关代码文件

现有 heuristic 规则：

- `sagin_marl/rl/baselines.py`

现有 BW actor 结构：

- `sagin_marl/rl/structured_actor.py`

clean target 与更新逻辑：

- `sagin_marl/rl/structured_mappo.py`

BW stage / 输入状态构造：

- `sagin_marl/rl/structured_stage_builders.py`
- `sagin_marl/env/structured_driver.py`
- `sagin_marl/rl/structured_types.py`

当前相关辅助诊断脚本：

- `scripts/diagnostics/diagnose/diagnose_structured_bw_fused_head_compare.py`
- `scripts/diagnostics/diagnose/diagnose_structured_bw_fixed_target_interpolation.py`
- `scripts/diagnostics/diagnose/diagnose_structured_bw_online_update_direction.py`
- `scripts/diagnostics/diagnose/diagnose_structured_bw_fixed_teacher_fit.py`

---

## 12. 当前结论

当前最推荐的不是：

- 继续调 teacher horizon
- 继续换 loss
- 继续磨 PCGrad
- 继续做 heuristic residual

而是：

**把 BW 读出头从黑盒 `MLP(fused_i) -> score_i`，改成显式共享规则头：**

\[
score_i = \langle w(pool(fused)), P(user0_i) \rangle
\]

这是当前最贴合已有证据、又不依赖 heuristic 先验形式的结构方向。
