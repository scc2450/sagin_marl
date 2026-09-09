你提的 4 点里，第 3 点我接受；第 4 点我要先更正一下：**按你现在这套正式 stage3 配置，确实不是“只训练 sat 头”，而是 `train_accel: true / train_bw: true / train_sat: true`，三个动作头都在训练，只是 shared backbone 仍冻结。** 这一点在你上传的结构说明和审计代码里是明确写出来的。 

下面我先把你这次的疑问解释清楚，再给一份**完整、可直接照着改**的三阶段联合修改清单。

---

# 先回答你的 4 个疑问

## 1）`log1p(B_pre)` 的系数要不要按你现在的数据调？

**要。**
而且要看的是**reward 各项的量级**，不是看 `policy_loss / value_loss / approx_kl` 那些图来调。

你当前正式三阶段训练 reward 实际就是：

[
r = 0.5 x_{\text{acc}} + 0.5 x_{\text{rel}} - w_{\text{pre-drop}} d_{\text{pre}} - w_{\text{pre-growth}} \max(g_{\text{pre}}, 0)
]

这在你代码里就是 `reward_mode == "controllable_flow"` 的实现。

而 `B_pre` 对应的现成评估量其实已经在环境里有同口径变量：

[
\text{pre_backlog_steps_eval} = \frac{Q_{\text{gu}} + Q_{\text{uav}}}{A_{\text{ref}} + \varepsilon}
]

同一个代码块里也已经把 `x_acc / x_rel / g_pre / d_pre / pre_backlog_steps_eval` 都写进了 `last_reward_parts`。

所以这里我建议的不是盲调，而是按你现在的量级来定：

* 你给的 stage1 最终 eval：`pre_backlog ≈ 14.33`
* `cluster_center`：`pre_backlog ≈ 11.71`
* 此时
  [
  \log(1+14.33)\approx 2.73,\quad \log(1+11.71)\approx 2.54
  ]

如果 `w_pre_backlog = 0.08`，那么 backlog 项大约是：

* trained：`0.08 × 2.73 ≈ 0.218`
* cluster_center：`0.08 × 2.54 ≈ 0.203`

这个量级是合适的：
它**足够影响策略**，但还**不会压过**前面的 `0.5*x_acc + 0.5*x_rel` 主项。

所以我的建议是：

* **第一版固定用 `reward_w_pre_backlog = 0.08`**
* 不要一上来就 0.2
* 也不要低到 0.02

具体调节规则我给你定成：

* 如果训练 100~150 updates 后，`pre_backlog_steps_eval` 仍长期 `> 18`，且仍出现明显三机聚堆，就把它从 `0.08` 提到 `0.10`
* 如果训练后发现 `x_acc` 和 `x_rel` 明显掉得太多，`pre_backlog` 也没改善，再降到 `0.06`
* **初版不要超过 `0.12`**

也就是说，这个系数要结合你当前数据调，但**先从 0.08 起步**最稳。

---

## 2）Stage 1 里的“当前与 UAV 关联的 GU 质心”怎么定义？按队列还是按通信速率？

我的建议是：

**第一版用“未加权几何质心”，不要按队列加权，也不要按通信速率加权。**

定义如下。
在当前 step 里，UAV 先移动，然后环境按新的 UAV 位置做 GU 关联。你代码的 GU 关联规则就是“对每个 GU，算所有 UAV 的 pathloss，取最小者；若最优 pathloss 超过门限则不关联”。

所以对 UAV (u)，定义它本 step 的关联用户集合：

[
\mathcal{G}_u = {k \mid \text{assoc}[k] = u}
]

然后质心定义成：

[
c_u =
\begin{cases}
\frac{1}{|\mathcal{G}*u|}\sum*{k\in\mathcal{G}_u} p_k, & |\mathcal{G}_u| > 0[4pt]
\text{invalid}, & |\mathcal{G}_u| = 0
\end{cases}
]

其中 (p_k) 是 GU 的二维位置。

### 为什么我不建议第一版按队列加权

因为你 base reward 里已经有 backlog 项了。
如果质心再按 `gu_queue` 加权，会把“几何 shaping”和“队列 shaping”搅在一起，信号更复杂，不利于你先把 stage1 的主问题单独纠正。

### 为什么我不建议第一版按通信速率加权

因为通信速率本身就强依赖 UAV 位置。
如果你再用“由当前位置决定的速率”去给几何质心加权，会形成比较强的闭环，reward 更容易变成“哪里已经好就更推哪里”，不利于纠正“远离自己服务簇”的问题。

### 所以 Stage 1 第一版用什么

**就用未加权几何质心。**

如果你后面发现它还不够，可以再试第二版：

[
c_u = \frac{\sum_{k\in\mathcal{G}*u} \sqrt{q_k+\epsilon},p_k}{\sum*{k\in\mathcal{G}_u} \sqrt{q_k+\epsilon}}
]

注意是 **(\sqrt{q})**，不是 (q) 本身，避免大队列用户权重过大。

但这属于第二版，不是第一版。

---

## 3）Stage 2 你同意

这部分我在清单里就直接按你同意的版本写，不再绕。

---

## 4）Stage 3 为什么我之前说错了？

因为我前一版把“我建议的 sat-only 预热方案”和“你当前正式 stage3 的实际训练配置”混说了。

这里重新写清楚：

### 你当前正式 stage3 的事实

不是只训 sat，而是：

* `train_accel: true`
* `train_bw: true`
* `train_sat: true`
* `train_shared_backbone: false`

也就是：**三个头都训练，backbone 冻结。** 

### 我建议的“sat-only”是什么

那不是对你当前代码事实的描述，
而是我后面给你的一个**可选修改方案**：

* 先做一个短的 sat-only 预热段：只训 sat 头
* 再做主训练 `stage3b`：恢复 accel+bw+sat 三头联合

这个是为了降低 stage3 一上来三头一起动带来的扰动。
但它是**建议方案**，不是你现在正式配置的事实。

---

# 三阶段联合修改清单

---

## A. 先改公共 reward：三阶段统一 base objective

### A1. 当前正式 reward 先说明白

你当前正式三阶段训练 reward 都来自 `reward_mode == "controllable_flow"`，实际形式是：

[
r = 0.5 x_{\text{acc}} + 0.5 x_{\text{rel}} - 1.0 \cdot d_{\text{pre}} - 0.2 \cdot \max(g_{\text{pre}}, 0)
]

代码里就是 access、relay、pre-drop、pre-growth 四项。

这里的问题是：

* 你自己也认可：`d_pre` 大部分时间几乎不生效
* `g_pre` 大部分时间也不生效

于是训练几乎退化成：

[
r \approx 0.5 x_{\text{acc}} + 0.5 x_{\text{rel}}
]

这正是 stage1 容易学成“共同折中点”的重要原因。

---

### A2. 新的三阶段统一 base reward

三阶段统一改成：

[
R_{\text{base}}
===============

0.5,x_{\text{acc}}
+
0.5,x_{\text{rel}}
------------------

## 0.08,\log(1 + B_{\text{pre}})

1.0,d_{\text{pre}}
]

其中：

[
B_{\text{pre}} = \frac{Q_{\text{gu}} + Q_{\text{uav}}}{A_{\text{ref}} + \varepsilon}
]

这和你环境里现成的 `pre_backlog_steps_eval` 是同一口径，只是训练时按每 step 算。`x_acc / x_rel / d_pre` 也都是你环境里现成有的量。

### A3. 具体系数

统一固定成：

* `reward_w_access = 0.5`
* `reward_w_relay = 0.5`
* `reward_w_pre_backlog = 0.08`
* `reward_w_pre_drop = 1.0`

### A4. 不再用的旧项

把下面这个从训练 reward 里拿掉：

* `reward_w_pre_growth * max(g_pre, 0)`

也就是：

* `reward_w_pre_growth = 0.0`

### A5. 为什么这样改

因为你真正关心的是 **GU+UAV 前端 backlog 本身**，不是只关心“backlog 是否继续增长”。

`log(1+B_pre)` 的优点是：

* backlog 从小到大时一直有惩罚，不像 `max(g_pre,0)` 那样大量 step 失活
* 又比线性 `B_pre` 更温和，不容易让 backlog 项压过 throughput 项

---

## B. 先加公共日志，三阶段都要记

在环境 `last_reward_parts` 或训练日志里，三阶段都新增这几项：(我注：类似的项现在已经有了吧？不要弄得太乱)

1. `b_pre_steps = (q_gu + q_uav) / arrival_ref`
2. `term_pre_backlog = -0.08 * log1p(b_pre_steps)`
3. `term_pre_drop = -1.0 * d_pre`
4. `term_access = 0.5 * x_acc`
5. `term_relay = 0.5 * x_rel`

这样你后面不会再出现“我加了 reward 项，但根本不知道有没有生效”的情况。

---

## C. Stage 1 修改清单

### C1. 目标

Stage 1 的核心目标不是单纯让 `reward_sum` 更高，
而是让 accel 学到**更合理的空间分工**，不要三机聚到一个共同折中点。

你现在的 stage1 正式配置本来就是：

* `enable_bw_action: false`
* `fixed_satellite_strategy: true`
* `train_accel: true`
* `train_bw: false`
* `train_sat: false`
* `train_shared_backbone: true`

也就是训练 backbone + accel 头。

---

### C2. Stage 1 训练 reward

Stage 1 用：

[
R^{(1)}*{\text{train}} = R*{\text{base}} + \lambda_1(u),A_1
]

其中 Stage 1 的唯一辅助项定义为：（我注：这一项对不同的agent是不一样的吧？是不是要单独修改？）

[
A_1
===

*

\frac{1}{\max(N_{\text{valid}},1)}
\sum_{u:|\mathcal{G}_u|>0}
\frac{|p_u - c_u|}{\text{map_size}}
]

解释：

* (p_u)：当前 step、UAV 移动后的二维位置
* (c_u)：当前 step 关联到 UAV (u) 的 GU 的**未加权几何质心**
* (N_{\text{valid}})：当前 `assoc_count > 0` 的 UAV 数

### C3. 质心怎么计算

按当前 step 的关联结果算：

1. UAV 先移动
2. 环境重新做 GU 关联
3. 对 `assoc == u` 的所有 GU，取位置平均
4. 得到 `c_u`

**第一版不要按队列加权，不要按速率加权。**

### C4. Stage 1 辅助项系数

我建议：

* `lambda1_init = 0.15`

### C5. Stage 1 辅助项退火

设 Stage 1 总更新数为 `U1`。（我注：训练中有early stop 按总更新算合适吗？）
令 `r = update / U1`，辅助项系数用下面这个分段退火：

* `r <= 0.30`：`lambda1 = 0.15`
* `0.30 < r <= 0.70`：从 `0.15` 线性降到 `0.05`
* `0.70 < r <= 1.00`：从 `0.05` 线性降到 `0.00`

也就是说：

* 前 30%：强引导，先把几何拉正
* 中间 40%：逐步减弱
* 最后 30%：回到统一目标

### C6. 为什么系数是 0.15

因为你现在几何问题非常明显：

* trained 的“UAV 到自己服务簇质心距离”比 `cluster_center` 大很多
* 但 reward 差距并不大

所以这个项必须**足够明显地影响策略**。
`dist / map_size` 是 0~1 量纲，0.15 的初始系数不会过大，但足以把“离服务簇很远”这件事显式写进 reward。

### C7. Stage 1 观测改动

在 actor 的 `own` 或单独小分支里，额外加这 3 个特征：

1. `assoc_count_u / num_gu`
2. `(assoc_centroid_x - uav_x) / map_size`
3. `(assoc_centroid_y - uav_y) / map_size`

理由：

* 现在 actor 虽然能看到 20 个候选用户，但没有显式知道“我当前服务簇的摘要”
* 这 3 个量是系统上合理、执行时也可得的信息
* 它们能直接帮助 stage1 学空间分工

### C8. Stage 1 不要同时加的东西

第一版**不要**再同时加：

* UAV 之间重叠惩罚
* 连接用户数奖励
* 轨迹平滑项
* 额外能耗项

原因很简单：
你担心 reward 太杂，这个担心是对的。
Stage 1 第一版只加 **一个辅助项** 就够了。

### C9. Stage 1 handoff checkpoint 选法

Stage 1 结束后，不要直接拿最后一个 actor 进 Stage 2。
在 checkpoint-eval 里按这个规则选 handoff checkpoint：

1. `collision_episode_fraction == 0`
2. `pre_backlog_steps_eval` 最小
3. 如果 backlog 差距在 5% 以内，再选 `reward_sum` 更高者
4. 如果还接近，再选 `processed_ratio_eval` 更高者

---

## D. Stage 2 修改清单

### D1. 目标

Stage 2 的目标是：
在 Stage 1 已经修正过的几何布局上，学习真正有用的 bw 分配，而不是继续被“坏几何 + 冻结表征”卡住。

你当前正式 Stage 2 是：

* `train_accel: false`
* `train_bw: true`
* `train_sat: false`
* `train_shared_backbone: false`

也就是：**只训练 bw head，backbone 冻结。**

---

### D2. Stage 2 训练 reward

Stage 2 **只用统一 base reward**：

[
R^{(2)}*{\text{train}} = R*{\text{base}}
]

也就是：

[
R^{(2)}_{\text{train}}
======================

0.5,x_{\text{acc}}
+
0.5,x_{\text{rel}}
------------------

## 0.08,\log(1 + B_{\text{pre}})

1.0,d_{\text{pre}}
]

**Stage 2 不额外加新的辅助 reward 项。**

### D3. 为什么 Stage 2 不再加 reward 项

因为 Stage 2 当前更像是**表示瓶颈**，不是“reward 项不够多”。

你上传的审计里，Stage 2 的 learned bw 和 `queue_aware` 在同 observation 上已经非常接近；同时现在 `bw_valid_count` 也并不小，说明不是动作空间退化，而是“当前能学的东西没被拉开”。

---

### D4. Stage 2 需要加的新配置项：部分解冻 fusion

你当前代码只有 `train_shared_backbone` 这个总开关。
为了 Stage 2，需要新加一个更细的开关：

* `train_fusion_last_layer: true`

### D5. Stage 2 的参数冻结方式改成

保持：

* `train_accel: false`
* `train_bw: true`
* `train_sat: false`
* `train_shared_backbone: false`

再新增：

* `train_fusion_last_layer: true`

实际含义：

* `own/users/sats/nbrs` 各 set encoder 继续冻结
* fusion MLP 的最后一层 Linear 解冻
* `bw_user_encoder` 和 `bw_scorer` 继续训练

### D6. 如果实现上不好只解冻最后一层

那就退一步：

* 新增 `train_fusion: true`
* 解冻整个 fusion MLP
* 其他 backbone 模块仍冻结

### D7. 为什么 Stage 2 要先解冻 fusion

因为 Stage 2 现在的问题是：

* 只训练 bw 头
* 但 bw 打分时又依赖共享上下文 `ctx`
* 当前 `ctx` 是 Stage 1 为 accel 学出来的表征

所以 Stage 2 最该先做的是：
让共享上下文的最后一小部分能适配 bw 任务。
这比继续往 reward 里加新项更直接。

### D8. Stage 2 handoff checkpoint 选法

Stage 2 结束后，选给 Stage 3 的 checkpoint 规则：

1. `collision_episode_fraction == 0`
2. `pre_backlog_steps_eval` 最小
3. 若 backlog 差距在 5% 以内，再选 `reward_sum` 更高者
4. 再看 `processed_ratio_eval`

不要只看最终点，也不要只看 reward plateau 结束点。

---

## E. Stage 3 修改清单

### E1. 先明确现状

你当前正式 Stage 3 不是只训 sat，而是：

* `train_accel: true`
* `train_bw: true`
* `train_sat: true`
* `train_shared_backbone: false`

也就是：**三个动作头都训练，backbone 冻结。** 

### E2. Stage 3 的核心目标

你自己说得很明确：
Stage 3 不是只想“多送一点流量”，而是想让不同 UAV **不要挤占同一卫星的带宽**。

而你的 sat 观测里本来就有：

* `spectral_efficiency`
* `sat_queue`
* `load_count`
* `1 / projected_count`
* `stay_flag`

所以 Stage 3 的问题不是“看不到卫星负载”，而是“虽然看到了，仍然没有学会分流”。这在你当前 actor 观测结构里是明确的。

---

### E3. Stage 3 训练 reward

Stage 3 用：

[
R^{(3)}*{\text{train}} = R*{\text{base}} + \lambda_3(u),A_3
]

其中辅助项定义成**归一化的同星拥挤惩罚**：（这个也是对不同的agent不一样的吧？）

先定义每步每颗卫星的连接 UAV 数：

[
c_l = #{u \mid l \in \text{selected_sats}(u)}
]

对 UAV (u)，它的 overlap 定义为：

[
\text{overlap}_u
================

\frac{1}{|\mathcal{S}*u|}
\sum*{l \in \mathcal{S}*u}
\frac{c_l - 1}{N*{\text{uav}} - 1}
]

其中：

* (\mathcal{S}_u)：UAV (u) 本步选中的卫星集合
* (N_{\text{uav}}=3)

最后：

[
A_3 = - \frac{1}{N_{\text{uav}}}\sum_u \text{overlap}_u
]

这个量天然在 `[-1, 0]` 附近，很好控系数。

### E4. 为什么用这个形式

因为它直接表达你要的行为：

* 如果三架 UAV 老是连同一颗或同一组卫星，`overlap` 就高
* 如果它们更分散，`overlap` 就低

而且这个项是**按实际选择结果**算的，不是按某种 heuristic 去约束它。

---

### E5. Stage 3 辅助项系数

建议：

* `lambda3_init = 0.10`

### E6. Stage 3 辅助项退火

设 Stage 3 总更新数为 `U3`。（这里也有early stop不按总更新的问题）
令 `r = update / U3`，退火如下：

* `r <= 0.40`：`lambda3 = 0.10`
* `0.40 < r <= 0.80`：从 `0.10` 线性降到 `0.03`
* `0.80 < r <= 1.00`：从 `0.03` 线性降到 `0.00`

### E7. 为什么是 0.10

因为这个 overlap 项是 0~1 的归一化量。
`0.10` 的初始系数对应“很明显地提醒策略别挤同星”，但不会压过主 reward。

---

### E8. Stage 3 训练方式：给你两版（我注：我觉得改成sat-only好一些，是否合理？）

#### 版本 1：保守版，继续三头联合

保持：

* `train_accel: true`
* `train_bw: true`
* `train_sat: true`
* `train_shared_backbone: false`

只新增：

* `reward_stage3_overlap_weight_init = 0.10`
* `reward_stage3_overlap_decay = 上面那套`

这是最少改代码的版本。

#### 版本 2：更稳版，先 sat-only 预热，再三头联合

这是我更推荐的版本。

##### sat-only 预热段

从 Stage 2 best checkpoint 开始，先跑一个短阶段：

* `train_accel: false`
* `train_bw: false`
* `train_sat: true`
* `train_shared_backbone: false`

reward 用：

[
R_{\text{base}} + \lambda_3 A_3
]

建议时长：

* `80 ~ 120 updates`

##### Stage 3b：恢复正式三头联合

然后从这个预热段的 best checkpoint 开始：

* `train_accel: true`
* `train_bw: true`
* `train_sat: true`
* `train_shared_backbone: false`

reward 仍然用：

[
R_{\text{base}} + \lambda_3 A_3
]

这样做的目的是：

* 先让 sat 头学会“别挤同星”
* 再让 accel+bw+sat 联合微调

### E9. Stage 3 要不要像 Stage 2 一样也解冻 fusion？

第一版**先不必**。
原因不是“Stage 3 只训 sat”，而是：

* 当前 Stage 3 已经是三头联合训练
* 第一优先问题是“同星拥挤”这个行为失败模式
* 先用 `A_3` 把这个行为扳正，更直接

如果做完 `A_3` 之后，Stage 3 仍然严重同星拥挤，再做第二版：

* 也给 Stage 3 加上 `train_fusion_last_layer: true`

也就是：

* Stage 2 先部分解冻 fusion，是因为它**只有 bw 头在动**
* Stage 3 不先做这一步，是因为它**先要解决明显的行为失败模式**
* 但 Stage 3 **不是不能解冻 fusion**，只是放在第二顺位

### E10. Stage 3 checkpoint 选法

最终 Stage 3 选 checkpoint 时，按这个顺序：

1. `collision_episode_fraction == 0`
2. `pre_backlog_steps_eval` 最小
3. `processed_ratio_eval` 更高
4. 新增 `sat_overlap_eval` 更低
5. 再看 `reward_sum`

---

## F. 训练顺序总表

### 第一步：先改代码和公共 reward

三阶段统一加：

* `reward_w_access = 0.5`
* `reward_w_relay = 0.5`
* `reward_w_pre_backlog = 0.08`
* `reward_w_pre_drop = 1.0`
* `reward_w_pre_growth = 0.0`

并把 `b_pre_steps` 与各项 term 记进日志。

### 第二步：重训 Stage 1

Stage 1 用：

[
R_{\text{base}} + \lambda_1 A_1
]

同时给 actor 增加：

* `assoc_count_u / num_gu`
* `assoc_centroid_rel_x`
* `assoc_centroid_rel_y`

训练完按前面规则选 best checkpoint，作为 Stage 2 起点。

### 第三步：重训 Stage 2

Stage 2 用：

[
R_{\text{base}}
]

并新增：

* `train_fusion_last_layer: true`

训练完按前面规则选 best checkpoint，作为 Stage 3 起点。

### 第四步：重训 Stage 3

优先用两段式：

* sat-only 预热段：`80~120 updates`
* `stage3b` 三头联合微调

reward 用：

[
R_{\text{base}} + \lambda_3 A_3
]

---

## G. 你担心“reward 项多了会不会不收敛”，这里给你一个明确回答

你的担心是对的。
所以这份方案刻意做得很克制：

### 公共 base reward 只有 4 项

1. `0.5 * x_acc`
2. `0.5 * x_rel`
3. `-0.08 * log1p(B_pre)`
4. `-1.0 * d_pre`

### 每个阶段最多只加 1 个辅助项

* Stage 1：只加 `A_1`
* Stage 2：不加辅助项
* Stage 3：只加 `A_3`

这不算“项很多”，而是**少项、归一化、可退火**。

真正容易把训练弄坏的是：

* 奖励项很多，而且相互冲突
* 某些项量级过大
* 某些项长期失活
* 某些项没有日志，根本不知道在不在工作

而这份方案恰恰是在解决你现在已有的“失活项”问题。

---

## H. 这一版里最重要的 6 个具体数字

你要的“系数和退火怎么弄”，最关键的数字就在这里：

### 公共

* `reward_w_access = 0.5`
* `reward_w_relay = 0.5`
* `reward_w_pre_backlog = 0.08`
* `reward_w_pre_drop = 1.0`
* `reward_w_pre_growth = 0.0`

### Stage 1

* `lambda1_init = 0.15`
* 退火：`0.15 -> 0.05 -> 0.00`
* 分段：`30% / 40% / 30%`

### Stage 3

* `lambda3_init = 0.10`
* 退火：`0.10 -> 0.03 -> 0.00`
* 分段：`40% / 40% / 20%`

---

## 最后一句总结

这份联合方案的核心不是“每阶段换一套目标”，而是：

* **三阶段统一一个最终 base objective**
* Stage 1 只补“服务簇几何”这个辅助项
* Stage 2 不再堆 reward，先解决表示冻结
* Stage 3 只补“同星拥挤”这个辅助项

如果你要，我下一条可以直接把这份清单写成**配置字段 + 伪代码修改模板**，按 `sagin_env.py / mappo.py / policy.py` 三个文件分开列。
