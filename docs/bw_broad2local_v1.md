看完你这批新数据后，我会明确修正我的判断：

**主问题不是“bw 的 global leverage 太小”。**
更准确地说，是：

**global / broad leverage 不小，但 PPO 在当前 on-policy 邻域里拿到的 local learnable signal 太浅、而且方向还有点脏。**

这两个层面必须分开。

你给的 policy-independent 审计已经很清楚地说明，旧环境里 **broad leverage 是存在的，而且不算小**：one-step `reward` range 的 `p50≈0.313`，`K=2 reward gap` 的 `p50≈0.221`，而且 `82.1%` 的状态满足 `K=2 reward gap >= 0.1`。同时，高 gap bank 的均值/中位数都明显高于 random bank，而且 snapshot restore 是精确可复现的。也就是说，“高 leverage 状态极稀少”这个判断不成立。 

但你给的另外两组诊断又说明，**训练真正接触到的局部邻域很浅**：
local finite difference 里，`best_reward_gain` 的 `p50≈0.0194`，`top1_hit≈0.229`，`pairwise_acc≈0.609`；而 within-state full-episode 诊断里，flatter 与 sharper 的 `selected_return` 差只有大约 `-0.00165`，immediate reward 差也只有 `-0.00333` 量级，说明在当前 policy 周围，动作变化带来的可学习增益确实很小。    

还有一个很重要的细节：**当前 local signal 可能还在学“错的子目标”。**
在那份 broad panel 审计里，`d_pre` 基本是 0，没有形成主要可分来源；但在 local credit probe 里，`loc` 和 `drop_gain` 的 Spearman 均值大约 `0.747`，明显高于它和 `reward_gain / processed_gain / backlog_gain` 的相关性（大约 `0.25~0.30`）。这说明当前 bw 头更容易学到“局部掉包式的 proxy 方向”，而不是真正 broad reward ordering。 

所以我现在给你的完整方案会和之前很不一样：

# 新结论

**现在不该再把主线放在“改环境 leverage”上，也不该再把主线放在“继续洗 target / 改 reward surrogate”上。**
当前最该做的是：

## 把“已经被你审计证明存在的 broad leverage”，直接变成 bw 头可学习的、阶段化的训练信号。

换句话说：

* 不是再去造更强的环境
* 也不是再去赌 `-ΔW` 或 pure `x_acc`
* 而是把**当前系统里真实存在的 panel-level action gap**，变成 actor 真能吃到的监督/偏好信号

---

# 这版完整方案

我把它叫做：

## `bw_broad2local_v1`

这版方案的核心目标只有一个：

**把 broad action gap 翻译成 local policy update。**

---

## 一、先定死：哪些东西这次不动

这次先不要动这些：

* 不改环境几何
* 不改 hard association
* 不改 GU 总数 / active mask / hotspot burst
* 不改 `bw` 动作语义
* 不改 joint 三阶段控制顺序
* 不改主 reward
* 不改 `D_sys` 定义
* 不改 accel / sat 的任务
* 不再继续做 “pure x_acc / fullchain x_acc / -ΔW” 这类主线实验

原因很直接：

1. 你现在已经有证据说明 **broad leverage 不小**。
2. 你也已经有证据说明 **pure x_acc 并没有把 bw 直接救出来**。
3. 所以再动环境或主 reward，只会继续混因果。

---

## 二、问题重述成一个更精确的训练问题

你现在的问题，不是：

> “系统里有没有好的 bw 动作？”

而是：

> “当前 policy 附近的梯度/采样，能不能把 deterministic bw 推向那些已经存在的好动作？”

基于你给的数据，答案是：

* **系统里有好的 bw 动作**：broad panel gap 很大。
* **当前 policy 不太会往那里走**：top1 命中低、pairwise 一般、local best gain 小。
* **而且现在学到的局部方向可能偏向 drop-like proxy**，不是 reward ordering 本身。

所以方案要解决的，是：

### `broad-to-local translation`

---

## 三、主方案：用真实 panel 审计结果做 bw 的辅助训练信号

这一步是整套方案的中心。

### 3.1 建一个 `bw_panel_bank`

你已经有现成的 panel 审计能力，而且高 gap bank 已经被证明：

* 比 random bank 更有信息量
* snapshot restore 是精确一致的。

所以现在不要再把它只当诊断工具，要把它升格成**训练数据源**。

### 3.2 bank 里每个样本存什么

对每个被选中的 bw-stage state，存：

1. `BwStageSnapshot` 或足以重建它的 world state
2. 当前 `bw_valid_mask`
3. 当前 valid user 数
4. action panel 中每个候选动作的完整 `bw action vector`
5. 每个候选动作的 one-step score
6. 每个候选动作的 `K=2` score
7. `best_action_idx`
8. `worst_action_idx`
9. `gap = best_score - mean_score` 或 `best - second_best`
10. 当前 policy 在该状态下的 deterministic action
11. 当前 policy 的若干 stochastic samples（可选）

### 3.3 panel 里放哪些动作

先直接沿用你现在已经审计过的面板，不要再发明新的：

* `uniform`
* `heuristic`
* `queue_top1 / top2`
* `eta_top1 / top2`
* `qeta_top1 / top2`
* `qeta_prev_top1 / top2`
* `random_0 / random_1`

这很关键，因为你已经知道这套 panel 在当前系统里能拉出真实 broad gap。

### 3.4 用什么分数给动作排序

**主排序分数就用 `K=2 env-reward`。**

也就是：

[
S(a \mid s_t)=r_t(a)+\gamma_{\text{env}}, r_{t+1}(a,\pi_{\text{follow}})
]

其中 follow policy 固定成你审计时已经用过的那种稳定口径即可。
这一步不追求“真 oracle”，追求的是：

* 和当前主训练目标同语义
* 比 local finite difference 更强
* 比 flow proxy 更接近真实系统结果

不要再用 `d_pre` 单独做主标签。
因为你自己的 broad 审计已经说明，在那批真正有 reward leverage 的状态里，`d_pre` 本身几乎没在分东西。

---

## 四、训练时不用再靠 local proxy aux，改用 panel-based preference aux

这一条是对当前 flow-proxy aux 的替换，不是叠加。

### 4.1 为什么要替换

你现在已有的 flow proxy aux，本质上还是：

* local
* proxy
* 而且很可能更偏前端 drop/backlog 子目标

但你这次数据表明：

* broad reward leverage 很大
* local signal 很浅
* 当前 actor 又更贴 drop_gain，不够贴 reward_gain。

所以继续沿“local proxy”方向加码，不是主解。

### 4.2 新辅助损失：`bw_panel_pref_loss`

对每个 bank state，取一个正样本动作 `a+` 和一个负样本动作 `a-`：

* `a+`：panel 里 `K=2 score` 最好的动作
* `a-`：panel 里最差的动作，或当前 policy deterministic 动作，或 bottom-k 随机一个

然后直接用当前 actor 的 `evaluate_actions` 计算：

[
\log \pi_\theta(a^+|s), \quad \log \pi_\theta(a^-|s)
]

做一个 pairwise preference loss：

[
L_{\text{pref}}
===============

* w(s,a^+,a^-),
  \log \sigma!\Big(
  \log \pi_\theta(a^+|s)-\log \pi_\theta(a^-|s)
  \Big)
  ]

其中权重可以直接取 gap：

[
w = \text{clip}\Big(\frac{S(a^+)-S(a^-)}{c}, 0, w_{\max}\Big)
]

### 4.3 这为什么比 imitation 更适合做第一刀

因为你这里的 teacher 不是“真最优动作”，只是 **panel 中更好的动作**。
直接 imitation 到某个单一 heuristic，容易把问题变成“模仿那个 heuristic”。
而 preference loss 只要求：

> 好动作的概率要比差动作大

这和你现在的 broad leverage 诊断更匹配。

---

## 五、再加一个很小的 best-action imitation，只做 mode 拉拽

pairwise preference 是主力。
但考虑到你现在还有一个很明显的现象：

* stochastic 比 deterministic 更接近好动作
* deterministic mode 没被拉到位

所以我建议再加一个很小的 second loss：

### `bw_best_action_kl`

设 panel 里 best 动作为 `a*`，actor 当前 mode 为 `\mu_\theta(s)`（就是当前 deterministic bw）：

[
L_{\text{imit}} = \text{KL}(a^* ,|, \mu_\theta(s))
]

或者简单一点，直接在 simplex 上做 masked MSE / cross-entropy。

但这项**权重一定比 preference 小**。
它的作用不是“主导训练”，只是把 deterministic mode 往当前已经证明确实更好的方向轻轻拉。

---

## 六、主 PPO 目标这次怎么设

这次非常明确：

### 6.1 主 PPO 目标先回到当前 joint 主线 reward

也就是你现在的主 reward 语义，不再继续折腾 `pure x_acc` / `fullchain x_acc` / `-ΔW`。

原因：

* pure x_acc 你已经测过，不够。
* broad reward leverage 已经不小。
* 当前 first-order 矛盾不是 reward 没信号，而是 actor 没把 broad signal 变成 local policy update。

### 6.2 flow proxy aux 先关掉或压到极小

建议：

* `bw_flow_proxy_aux_enabled = false`

或者至少：

* 权重压到 panel aux 的 10% 以下

因为你现在最不该继续做的，就是让 bw 头继续主要吃“local proxy”。

---

## 七、完整训练流程

这次不要一上来就 joint。

### Phase A：离线 / 半离线 bw warm-start

目标：先把 deterministic bw 从“平庸局部平台”推向更好的 broad neighborhood。

做法：

* 冻结 accel / sat
* 冻结 critic 或只轻微更新 `V_bw`
* 只训 bw actor
* loss 以 `bw_panel_pref_loss` 为主
* `bw_best_action_kl` 为辅
* 不需要大量 live rollout

长度建议：

* 20~40 个 update 等价的 bank-only 训练

### Phase B：bw-only mixed training

目标：让 PPO 在更好的 neighborhood 里接管微调。

做法：

* 仍冻结 accel / sat
* 每个 update 同时做：

  * live PPO rollout
  * bank aux mini-batch
* 权重大致：

  * PPO : panel_pref : best_action_kl = 1 : 1 : 0.1
    这是起始量级，不是死值。

长度建议：

* 50~100 updates

### Phase C：joint fine-tune

目标：把已经学到的 bw 再放回 joint。

做法：

* 恢复 joint
* 但 panel aux 不立刻关掉，而是保留一个衰减尾巴
* 例如 30~50 updates 内从 1.0 线性衰到 0.1

---

## 八、bank 不是一次性做完，要滚动刷新

因为 policy 在变，state distribution 也会变。

建议：

### 8.1 双池策略

bank 分两部分：

* 50%：最新 policy 重新采样的 fresh bank
* 50%：历史高 gap bank

### 8.2 分层采样

采样时不要只挑最大 gap 状态。
要按这几个维度分层：

* leverage gap 分位数
* valid user count
* GU/UAV queue pressure
* 当前 policy 与 panel-best 的差距

这样不会把训练全压在一种“极端但少见”的状态上。

---

## 九、验证口径：这次该怎么判断方案有没有用

这次不要先看大而泛的 episode 平均。
先看三组更贴问题的指标。

### 9.1 Bank-state 上的局部排序指标

这是第一关，必须先过。

你现在的基线大约是：

* `top1_hit ≈ 0.229`
* `pairwise_acc ≈ 0.609`
* `best_reward_gain p50 ≈ 0.0194`。

我建议通过标准设成：

* `top1_hit >= 0.45`
* `pairwise_acc >= 0.75`
* `best_reward_gain p50 >= 0.04`

也就是先翻一倍左右。
如果这关都不过，就不要看 full episode。

### 9.2 Within-state deterministic 改善

你现在 flatter/sharper 的 return 差很小，大约 `1e-3`。

通过标准可以设成：

* deterministic mode 相对当前基线的 within-state best-minus-current gap
  至少提升到 `>= 0.01`

这表示 deterministic 已经被拉出原来的浅平台。

### 9.3 Live matched eval

最后才看你熟悉的 matched eval。

我建议看三件事：

1. deterministic 与 heuristic 的 gap 是否明显缩小
2. stochastic - deterministic 的差距是否缩小
3. `x_acc`、`reward_sum`、`pre_backlog_steps_eval` 是否一起改善

如果仍然出现：

* stochastic 明显比 deterministic 好很多

那说明 mode 还没被拉到位，应该继续加 best-action imitation 权重，而不是去改环境。

---

## 十、如果这版没效果，怎么分流

### 情况 A：bank-state 指标明显提升，但 live matched eval 不动

这说明：

* actor 学会了 bank 上的排序
* 但 on-policy 访问到的状态分布还没跟上

这时再启用我之前提过、但这次不作为第一刀的：

## `high-gap snapshot curriculum`

不是为了“制造 leverage”，而是为了**让 policy 真正在 rollout 里走进这些已经证明确实有 broad gap 的状态**。

你现在 summary 里已经证明：

* high-gap bank 比 random bank 更强
* restore consistency 是 0 差。

所以 snapshot curriculum 这次是合理的第二刀，不是第一刀。

做法：

* 20%~30% env 从 high-gap bank snapshot 开局
* horizon 只跑 8~12 步
* 其余 env 仍是原始 reset

### 情况 B：bank-state 指标都提不起来

这时才说明问题不是 state distribution，也不是 broad-to-local translation，
而是 **当前 actor/optimization chain 本身对这些 panel labels 也学不进去**。

到这一步，才值得重新回到算法本体，比如：

* bw 头输出几何
* `log_prob` 目标形态
* deterministic readout
* 甚至更强的离散结构化 bw 头

但必须是**在你已经证明 broad signal -> panel aux 这条路都走不通之后**再做。

### 情况 C：bank-state 和 live 都起来了，但 joint 又掉

这说明：

* bw 自身可学
* 问题转成了 joint conflict

这时才去看：

* accel / sat 是否吞掉 bw credit
* joint schedule
* 甚至之后再讨论 `-ΔW`

而不是现在就去改它。

---

## 十一、这版和我之前方案最大的变化

我这次明确收回三件事：

第一，**不再把主问题说成“高 leverage 状态很稀少”**。
你的 summary 已经反证了这一点。

第二，**不再把主线放在“继续改环境 leverage”上**。
因为 broad leverage 已经存在。

第三，**不再把主线放在“继续改 reward / target”上**。
因为 pure x_acc 也没把问题直接解掉，而当前 broad reward leverage 又已经足够大。

---

## 最后压成一句话

**现在最正确的主线，不是继续造更强环境，也不是继续洗更纯 target，而是把你已经审计出来的 broad panel gap，直接变成 bw 的训练信号。**
也就是：

**旧系统不动 → 建 high-gap / random 混合 bank → 用真实 `K=2 env-reward` 做 panel-based preference aux → 先把 deterministic bw 从浅局部平台推到更好的 neighborhood → 再让 PPO 接管。**

这版方案里，snapshot curriculum 不是主线，只是 bank 有效但 live 不动时的第二刀；`-ΔW` 不是主线，只是以后 joint 对齐 D_sys 时再讨论的事。

下一步最值得做的，就是把这版方案先落成两个文件级组件：
一个 `collect_bw_panel_bank.py`，一个 `bw_panel_pref_aux`。
