根据 **2026-04-06 的当前代码** 和你这轮 **readout-only / geometry-only / geometry+dense-ranking** 的结果，我现在同意你的收束，而且我会把结论再压得更具体一点：

## 现在最可信的诊断

**主问题已经不是环境 leverage，也不是 readout。主问题是：当前 BW 头的“输出参数化”把三件事绑错了。**

这三件事是：

1. **排序**：谁该大、谁该小
2. **尖锐度**：应该离 uniform 多远
3. **stochastic 分布形状**：log-prob / entropy 怎么变化

你现在这版代码里，这三件事没有被干净地分开，所以就出现了你观测到的那个非常典型的分裂现象：

* dense ranking 能把 **ordering** 学好；
* hard pull 能把 **deterministic mode** 硬拉好；
* 但两者在当前 BW 头里很难自然统一成“既会排、又真把 deterministic 动作推到更好位置”。

这和你这轮实验现象是完全一致的。

---

## 为什么我现在认为“参数化本身”是主因

### 1) 现在的 `panel_pref` 和 `hard pull`，本来就在优化两件不同的东西

离线 broad2local 审计代码里，`pref_loss` 用的是：

* `actor.evaluate_bw(...)`
* 比较的是 **best action** 和 **negative action** 的 `log_prob` 差。 

而 `pull_loss` 用的是：

* `actor.bw_policy.deterministic_action(...)`
* 再和 target action 做 masked simplex KL。 

也就是说：

* **dense ranking 学的是“分布更偏好哪类动作”**
* **hard pull 学的是“deterministic 输出长什么样”**

你现在的结果正是这两者分裂：
dense ranking 把排序修好了，hard pull 把 deterministic 拉好了，但它们合不起来。

这不是训练脚本偶然，而是**优化目标和参数化映射本来就分裂**。

---

### 2) 现在 `log_prob` 用到 `loc + log_scale`，但 deterministic 动作基本只靠 `loc`

`BwPolicy.evaluate_actions()` 里，`logprob_raw = dist.log_prob(action)`，这里 `dist` 是 `MaskedLogisticNormal(loc, log_scale, mask)`，所以 **ranking / preference 学习同时会改 `loc` 和 `log_scale`**。 

但 deterministic 路径里，不管是原来的 `dist.mode()`，还是你后来加的 `deterministic_action(...)` 默认 `latent_mean_pushforward`，本质上都主要是从 `loc` 出发回到 simplex；`log_scale` 不决定当前默认 deterministic action 的形状。`MaskedLogisticNormal.mode()` 也是直接把 latent 取成 `loc` 再映射回 simplex。  

这就给了一个非常直接的解释：

**dense ranking 完全可能主要通过改 `log_scale` 来提升 panel 上的 `log_prob` 排序，而没有把 deterministic mode 一起推过去。**

这正好解释了你现在看到的现象：

* dense ranking：`pairwise_acc / spearman` 明显变好
* 但 `det_score / x_acc / pre_backlog` 不同步变好

而 `single + hard pull` 之所以有效，是因为它绕过了这条路，直接对 deterministic mode 下手。

我现在认为，这是当前最关键的一条机制解释。

---

### 3) 现在的 logistic-normal 还有一个“参考分量”问题，会让动作几何更别扭

`MaskedLogisticNormal` 用的是 ALR（additive log-ratio）坐标，而且**最后一个 valid component 被固定成 reference component**。代码里 `_flat_ref_idx` 就是取最后一个 valid 位置，`_flat_latent_mask` 再把这个 reference 分量从 latent 里拿掉。

与此同时，`BwPolicy._params()` 仍然给**每个 user 都输出一份 `loc` 和 `log_scale`**，然后再通过 `_latent_mask()` 把 reference 分量盖掉。

这会带来两个副作用：

* 输出坐标是 **slot-order / reference-dependent** 的
* “离 uniform 多远”这件事，实际上被编码成了“非 reference 用户相对 reference 的偏移”

这会让“排序变好”很难自然转成“mode 也更有用”。
因为 mode 的几何不是直接建在“masked simplex 上的排序+尖锐度”上，而是建在“相对某个 reference 分量的 ALR 坐标”上。

我不会说这一定是唯一病根，但它肯定在放大你现在的分裂现象。

---

## 所以我现在的判断是

你这轮结论可以再压成一句：

**geometry 是正方向，但只是把 PPO 更新做得不那么别扭；真正更深的病根，已经落到“当前 BW 参数化没有把 ranking、sharpness 和 deterministic mode 对齐”上。**

这也解释了为什么：

* `readout-only` 没打中
* `geometry-only` 能小幅改善，但不是根治
* `geometry + dense ranking` 仍然不能把 deterministic 质量一起拉起来
* 只有 `single + hard pull` 这种直接碰 mode 的东西，才能把 `det_score` 真拉上去

---

# 我给你的更新方案

我会把主线改成：

## **保留 geometry 分支，停止继续折腾 readout；下一步专门重做 BW 参数化。**

不是大改环境，不是重写系统语义，也不是再改 reward。
就是重做 `structured_actor.py` 里 `BwPolicy` 的输出几何。

---

## 方案核心：把 BW 头拆成“排序”和“尖锐度”两部分

### 当前问题

现在 `BwPolicy` 直接输出：

* per-user `loc`
* per-user `log_scale`。

这太自由了，导致：

* `log_prob` 可以主要靠 `log_scale` 学
* deterministic mode 又不吃 `log_scale`
* 排序和最终动作被拆开了

### 新参数化

我建议直接改成下面这版。

#### A. 排序头：`score_head`

输出每个 valid user 的一个 **score**：

[
s_i
]

这只负责“谁该更大”。

#### B. 尖锐度头：`sharpness_head`

每个 UAV 再只输出 **1 个标量**，比如：

* `tau`：temperature
* `alpha`：离 uniform 的混合强度

推荐直接用两个标量：

[
\tau > 0,\qquad \alpha \in [0,1]
]

然后 deterministic mean 定义成：

[
p_{\text{det}}
==============

(1-\alpha),u
+
\alpha \cdot \text{softmax}(s/\tau)
]

其中 `u` 是 valid users 上的 uniform。

这一步的意义非常大：

* **排序** 由 `s` 决定
* **离 uniform 多远** 由 `alpha` 决定
* **尖不尖** 由 `tau` 决定

这样“更好的排序”终于能直接映射到“更有用的 deterministic 动作”。

---

## stochastic 分布怎么接

### 最小改法

先不推翻 `MaskedLogisticNormal`。

做法是：

1. 先算出上面的 `p_det`
2. 再把 `p_det` 转成 distribution 的 mean 坐标
3. `log_scale` 不再是 per-user head，而是**每个 UAV 一个 scalar**，然后 broadcast 到 latent dims

也就是：

* `loc` 由 `p_det` 反推得到
* `log_scale` 只有一个 state-level scalar

这样你还可以继续用现有的：

* `rsample`
* `log_prob`
* `entropy`

但把最致命的问题先去掉了：

**不再允许 ranking 学习主要躲在 per-user `log_scale` 里。**

---

## 这一步为什么比继续调 loss 更值得做

因为你现在已经验证了：

* dense ranking 作为 loss，能修 ordering
* hard pull 作为 loss，能修 mode
* 但两者合不起来

这说明问题不是“loss 还不够复杂”，而是：

**当前参数化不支持“同一个更新同时兼顾排序和 mode”。**

所以再继续发明 `soft pull / delayed pull / 更花的 pairwise loss`，收益大概率都不如先把参数化改正。

---

# 几何分支怎么处理

这里我赞成你现在的判断：

## 1. `readout-only`

降级成审计 helper，不进默认训练。
你已经实证证明它不是主因。

## 2. `geometry-only`

保留，而且继续当主训练底座。
你这轮结果已经说明：

* `approx_kl_bw`、`clip_frac_bw` 下来了
* deterministic 端略有改善
* det-vs-stoch gap 缩小

这就足够说明：

**PPO 几何确实有问题，而且 `per-UAV surrogate + raw ratio` 是正方向。**

## 3. “normalized logprob 直接算 PPO ratio”

我同意你，不建议采用。
这一点现在不该再往前推。

---

# 具体实施顺序

## Phase A：先改 BW 参数化，不改训练主线

只动：

* `structured_actor.py`
* 如果必要，再轻改 `distributions_logistic_normal.py`

不动：

* env
* reward
* target
* geometry-only 的 PPO 分支

### 你要做的最小改动

1. 删掉 per-user `log_scale_head`
2. 新增：

   * `score_head`（per-user）
   * `alpha_head`（per-UAV scalar）
   * `tau_head`（per-UAV scalar）
   * `log_scale_scalar_head`（per-UAV scalar）
3. deterministic action 改成：

   * masked uniform + masked softmax(score/tau) 的 convex mixture
4. stochastic 分布的 mean 改成由这个 deterministic mean 派生

---

## Phase B：先只做离线 broad2local 审计

继续用你现在已经成熟的 `audit_bw_broad2local_offline.py`。

但这次只比较三组：

1. baseline
2. geometry-only actor + 新参数化
3. geometry-only actor + 新参数化 + dense ranking

**不要先加 pull。**

### 这一步的验收标准

如果新参数化是对的，你应该看到：

* dense ranking 提升 `pairwise_acc / spearman`
* 同时也提升 `det_score / det_x_acc`
* `gap_to_best` 下降
* `pre_backlog` 不再明显恶化

也就是：

**排序改善终于和 deterministic 质量同向。**

如果还是只有排序涨、不带动 deterministic，那才说明还得继续动分布本体。

---

## Phase C：再决定要不要回 live PPO

如果离线 broad2local 已经证明：

* 新参数化 + dense ranking 能同时提升排序和 det quality

这时再进 live：

* 保留 `geometry-only`
* 不加 readout 改动
* 先不开 pull
* 先看 `u100/u140` matched eval

### 这一步重点看

* deterministic 是否继续缩小与 stochastic 的差
* deterministic 是否更稳定地接近 heuristic
* `pre_backlog` 是否不再像现在这样一碰 dense ranking 就变差

---

# 如果这版还不行，下一步该怀疑什么

如果你做完这版以后，仍然出现：

* 排序提升很多
* deterministic 质量还是不上去

那时我会把怀疑进一步收窄到：

## `MaskedLogisticNormal` 的 ALR 参考坐标本身不适合这个任务

也就是要进入下一层：

### Phase D：把分布坐标从“最后 valid 分量做 reference”改成“无 reference 的 centered basis”

比如：

* CLR + 投影基
* Helmert basis
* 或别的 zero-sum latent basis

这样可以把现在这个“reference component 特权”去掉。
但我把这一步放在 **新参数化之后**，不放在现在第一刀。

因为你现在已经有一个更直接、更低成本的可疑点：**per-user `log_scale` + mode 不吃 `log_scale`**。

---

# 最后一句话

我现在基本同意你这轮整理后的核心结论，而且我会再往前推半步：

**当前最可疑的病根，不只是“BW 参数化本身”，而是更具体的——当前 BW 头把 `log_prob` 学习和 deterministic mode 学习拆开了：dense ranking 可以主要通过 `log_scale` 学排序，hard pull 可以单独拉 mode，但现有 `loc + per-user log_scale + ALR reference` 这套几何不擅长把两者统一。**   

所以我现在最推荐的主线不是再改环境、不是再改 reward、也不是再改 readout，而是：

**保留 geometry-only；把 BW 头重做成“score + sharpness + scalar scale”的参数化；先用你现有离线 broad2local 表验证‘排序改善能否终于带动 deterministic 动作改善’。**

如果你要，我下一条就直接把这版改动写成**代码级改造清单**：`structured_actor.py` 里哪些函数改什么签名、哪些旧字段删掉、`BwPolicyOutput` 里该加什么新字段、以及怎么最小改动接回当前 `MaskedLogisticNormal`。
