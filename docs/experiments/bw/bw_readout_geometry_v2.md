根据你这批 **2026-04-05/06 的诊断数据** 和 **2026-04-06 上传的当前代码**，我会把结论明确改成：

## 结论

**现在主问题已经不再是“bw 动作对目标影响太小”。**
更准确地说：

1. **broad leverage 是存在的，而且不小**。
   旧环境、policy-independent 审计里，`K=2 reward gap` 的 `p50≈0.221`、`mean≈0.242`，而且 `82.1%` 的状态满足 `gap>=0.1`。这已经足够反驳“高 leverage 状态极稀少”这个解释。

2. **当前真正卡住的是：broad leverage 没能被翻译成更好的 deterministic bw 动作。**
   你的 local probe 里，`positive_best_reward_gain_rate≈0.979`，说明附近通常有更好方向；但 `best_reward_gain p50≈0.0194`、`top1_hit≈0.229`、`pairwise_acc≈0.609`，说明当前 policy 局部排序浅、也不够稳。更关键的是，`loc` 和 `drop_gain` 的相关性明显高于它和 `reward_gain` 的相关性，说明现在学到的方向有点“偏子目标”。
   within-state full-episode 诊断也支持这一点：在同一状态里，把动作做得更“sharp”或更“flat”，`selected_return` 差只有大约 `1e-3` 量级，说明当前 deterministic readout 周围确实很平。

3. **所以 diagnosis 方向没错，但 root cause 比“local signal 浅/脏”更深一层。**
   现在最可疑的已经不是环境 leverage，也不是主 reward，而是：

   * **BW 参数化**
   * **deterministic readout / mode**
   * **PPO 对 BW 头的更新几何**

你自己那轮离线 broad2local 结果，其实已经把这个层次关系说得很清楚了：
**dense ranking 能明显修“脏”，但修完排序以后，`det_score / x_acc` 还是没同步起来。**
这非常像“排序学到了，但当前参数化和 deterministic readout 没法把它转成更有用的动作”。

---

## 我现在最同意你的哪一部分

我同意你这句：

> diagnosis 大体对，但真正的 root cause 可能更深，已经开始指向 BW 参数化 / deterministic mode / PPO 更新几何。

我现在的补充是：

### 不是三者并列同权，优先级应该是

**第一优先级：deterministic readout / mode**
**第二优先级：PPO 更新几何**
**第三优先级：BW 参数化本身**

原因是：你现在这轮 dense-ranking 离线试验，不经过 live env leverage 也不主要依赖 critic，就已经出现了“排序上去了，但最终 deterministic 动作质量没上去”的现象。
这首先指向 **actor 输出怎么被读成 deterministic bw**，其次才是 PPO 怎么把它更新过去。

---

## 代码层面，当前最可疑的地方在哪里

### 1) 当前 deterministic bw 不是“这个分布在 simplex 上的真正 mode”

现在 `BwPolicy.forward()` 在 deterministic 分支里直接做：

* 构造 `MaskedLogisticNormal`
* `action = dist.mode()`。

而 `MaskedLogisticNormal.mode()` 的实现，是：

* 先取 `latent = loc`
* 再通过 `_flat_probs_from_latent(...)` 映射回 simplex。

但同一个分布的 `log_prob()` 真正优化的是：

* latent Gaussian log-density
* **减去 Jacobian 的 `log_det`**
* 而且还显式依赖 `log_scale`。

所以当前这个 `mode()` 本质上更接近：

**“把 latent mean 推过 softmax-like 映射后的动作”**

而不是：

**“在 simplex 上最大化当前分布密度得到的动作”**。

这点非常关键。因为你现在的离线结果恰好就是：

* 排序学得更好了
* 但最终 deterministic action 质量没有跟着涨

这和“deterministic readout 不是对的代表动作”是高度一致的。

更重要的是，评估代码里 deterministic bw 也是直接走 `actor.act_bw(..., deterministic=True)`，所以 eval 看到的就是这个 readout，不是别的东西。

---

### 2) 当前 PPO 的 BW log-prob 几何，仍然会随 valid_count / latent_count 变形

`MaskedLogisticNormal.log_prob()` 在 event 内部已经把 latent 维度求和了。也就是说，对单个 UAV 而言，valid user 越多，latent_count 越大，`log_prob` 的尺度天然越大。

而 `StructuredMAPPO` 里，stage actor eval 又把 `out.logprob` 在 UAV 维度再求和，形成 joint `new_logprob` / `old_logprob`，再去做 ratio 和 clip。

所以现在 BW 的 PPO 比例几何其实是：

* **先沿 latent dims 累加**
* **再沿 UAV dims 累加**

这会带来两个后果：

第一，**同一 batch 里 valid_count 不同的样本，更新尺度不一致**。
而你本地 probe 里 valid user count 的波动并不小，均值约 `8.56`，`p90≈14`。

第二，**一个 UAV 上的 log-ratio 变化，可能被另一个 UAV 的变化抵消或放大**。
这对 `bw` 尤其不友好，因为它本来就是更局部的动作。

你之前文档里提过“`|log_ratio_bw|` 不该随 `valid_count` 病态放大”，现在看，这条担心是有代码支撑的。

---

### 3) 当前参数化缺少一个“显式控制非均匀程度”的旋钮

现在 `BwPolicy` 主要输出的是：

* 每个 user 的 `loc`
* 每个 user 的 `log_scale`。

但当前 deterministic action 只用 `loc`，**完全不看 `log_scale`**。

这意味着：

* 你可以把排序学得更对
* 但不一定能把动作变得“更有用地不均匀”

也就是说，当前参数化里：

**“哪个 user 该更大”** 和 **“整体该有多尖/多平”** 没有被一个稳定的 deterministic readout 同时表达出来。

这正好解释了你现在看到的现象：

* dense ranking 把 pairwise / spearman 明显拉起来
* 但 `det_score / x_acc` 不同步改善

---

## 所以现在应该怎么做

我给你的这版完整方案，不再从环境 leverage 开刀，也不再继续发明 reward。
我把主线改成：

# `bw_readout_geometry_v2`

目标只有一个：

**先把“学会排序”真正转成“学出更好的 deterministic bw 动作”。**

---

# Phase 0：先定死这轮不做什么

这轮先不做：

* 不改环境
* 不改 arrival / hotspot / active mask
* 不改 association
* 不改主 reward
* 不改 `D_sys`
* 不继续做新的 flow proxy 变体
* 不继续做新的 broad2local loss 花样

因为你现在已经有足够证据表明：

* broad leverage 够
* diagnosis 方向也大体对
* 继续在外层改环境/target，只会继续混因果

---

# Phase 1：先修 deterministic readout，不碰 live PPO

这是第一刀，而且我认为必须先做。

## 1.1 把当前 `mode()` 从主 readout 里退下来

不要再把当前 `MaskedLogisticNormal.mode()` 当 deterministic bw 的标准输出。

建议改成两套 readout：

### A. `latent_mean_pushforward`

就是你现在的旧 `mode()`，保留做对照。

### B. `simplex_argmax_logprob`

在 simplex 上直接求

[
a^* = \arg\max_a \log p_\theta(a \mid s)
]

做法不需要太重：

* 每个样本 5~10 步 projected gradient ascent
* 只在 **eval 和离线 broad2local 审计** 里用
* 不进 live rollout 主循环

这样你能先回答一个非常关键的问题：

> dense ranking 学会以后，问题到底是“actor 没学到”，还是“当前 deterministic readout 读错了”？

## 1.2 先只重跑你已经有的离线 broad2local 表

不要新发明实验，直接复用你现成流程：

* baseline
* 单对 `panel_pref`
* dense ranking `top-k all-pairs`
* delayed soft pull

唯一变化：

* deterministic readout 改成 `simplex_argmax_logprob`

### 这一步的通过标准

如果只改 readout，就能让：

* dense ranking 的 `det_score`
* `x_acc`
* `pre_backlog`

明显比现在好，

那说明主问题首先在 **mode/readout**，而不是环境、critic 或者 broad2local 思路本身。

---

# Phase 2：改 PPO 更新几何，只动 BW stage

如果 Phase 1 仍然出现“排序明显提升，但 deterministic/online 仍不跟”，下一刀就不是再加 loss，而是改 PPO 几何。

## 2.1 不要再对 BW 用 joint-summed logprob 做 ratio

现在 `_stage_actor_eval(_from_batch)` 对 bw 的 `logprob` 是：

* 先每 UAV 内部按 latent dims 求和
* 再 across UAV 求和。

我建议 BW stage 改成：

### `per-UAV PPO surrogate`

对每个 sample、每个 UAV 单独算：

[
\log r_{u} = \log \pi_\theta(a_u|s_u) - \log \pi_{\theta_{old}}(a_u|s_u)
]

然后做 per-UAV clip surrogate，再在 UAV 维度平均：

[
L_{bw} = - \frac{1}{U}\sum_u \min(r_u A, \text{clip}(r_u) A)
]

先仍然用 team advantage 都可以，重点是：

**不要再把三个 UAV 的 bw log-ratio 先乘成一个 joint ratio。**

这一步的意义非常大：
它会把“一个 UAV 学对了，另一个没学对”从相互抵消的 joint 乘法里解开。

## 2.2 BW 的 logprob / entropy 先按 latent_count 归一

因为 `log_prob` 和 `entropy` 都天然随 latent_count 增长。

所以对 BW stage，我建议直接改成：

[
\log \pi^{norm}_u = \frac{\log \pi^{raw}_u}{\max(\text{latent_count}_u,1)}
]

[
H^{norm}_u = \frac{H^{raw}_u}{\max(\text{latent_count}_u,1)}
]

然后：

* PPO ratio 用 `logpi_norm`
* entropy regularization 用 `H_norm`

这不是在改分布本体，而是在改 **PPO 对 BW 头的目标尺度**。

## 2.3 Buffer 也要跟着改

当前 buffer 里存的是 stage transition 的单个 `old_logprob` 标量。

如果你要做 per-UAV surrogate，就要让 BW stage 额外存：

* `old_logprob_bw_per_agent`
* `old_logprob_bw_norm_per_agent`
* 或至少 `old_logprob_bw_raw_per_agent + latent_count_per_agent`

否则 update 阶段拿不到旧策略的 per-UAV ratio。

---

# Phase 3：broad2local 只保留 dense ranking，先别再玩 pull 花样

这一步是对你当前离线 broad2local 结果的直接响应。

你已经测出来了：

* 单对 `panel_pref` 太弱
* dense ranking 能明显修“脏”
* pull 系列会和 dense ranking 打架
* delayed soft pull 只是稍微没那么糟，但也没把最终质量真正带起来

所以我建议：

## 3.1 主 aux 就保留一个

只保留：

**`top-k all-pairs panel_pref`**

不要再默认叠：

* hard pull
* soft pull
* low soft pull
* delayed pull

## 3.2 pull 只在 readout 修完以后再回来

如果 Phase 1 修完 deterministic readout 之后，dense ranking 仍然“排序升了但动作质量差一点”，这时才考虑加一个很轻的 pull。

而且 pull 的 target 也别再用“旧 mode”或某个 heuristic action，
而要用：

* panel-best action
* 或新 deterministic readout 的目标形式

---

# Phase 4：如果 1~3 都做完，还不行，再改参数化

这一层我不建议现在立刻动，但要提前把方向定好。

如果：

* readout 修了
* PPO geometry 也修了
* dense ranking 还是只能提升排序，不能提升动作质量

那就说明问题真正落到 **BW 参数化本身**。

## 4.1 我建议的参数化改法

不是直接推翻成别的动作空间，而是给 BW 一个显式“尖锐度”旋钮。

例如 deterministic mean 改成：

[
a = (1-\alpha),u + \alpha ,\text{softmax}(s/\tau)
]

其中：

* `u` 是 valid users 上的 uniform
* `s` 是 ranking scores
* `\alpha` 控制“离 uniform 有多远”
* `\tau` 控制“有多尖”

这样：

* 排序由 `s` 决定
* 非均匀程度由 `\alpha, \tau` 决定
* deterministic readout 和训练语义终于分开了

这会比当前“全靠 `loc` 间接推 simplex 形状”更适合你这个任务。

但这一步我只放在 Phase 4，不作为现在第一刀。

---

# 这版方案怎么验证

## 验证 1：readout 诊断

只改 deterministic readout 后，重跑你现有离线 broad2local 表。

看三件事：

* dense ranking 下 `pairwise_acc / spearman` 是否还能保持
* `det_score` 是否明显上升
* `x_acc / pre_backlog` 是否同步变好

如果只改 readout 就有效，主因就坐实了。

## 验证 2：geometry 诊断

改成 per-UAV + per-latent normalized PPO 后，先看：

* `|log_ratio_bw|` 与 `valid_count / latent_count` 的相关性是否下降
* `approx_kl_bw`、`clip_frac_bw` 是否不再随 valid_count 病态扩大
* `stochastic - deterministic` gap 是否缩小

## 验证 3：live matched eval

最后才看你最熟的 matched eval：

* deterministic `policy/policy/policy`
* stochastic `policy/policy/policy`
* heuristic bw

你那条 `u100` 的 fullchain 线里，deterministic `x_acc=0.9874`，heuristic 是 `1.0037`，stochastic 是 `0.9986`。我会把目标定成：**先缩 deterministic 与 stochastic 的差，再缩 deterministic 与 heuristic 的差。**

---

# 最后压缩成一句话

**现在最可信的结论是：问题已经不是“bw 对目标影响太小”，而是“broad leverage 明明存在，但当前 BW 头把排序学到以后，没法稳定地把它读成一个更好的 deterministic simplex 动作；同时 PPO 还在用随 latent_count / UAV 数共同放大的 joint log-ratio 去更新它”。**

所以我现在给你的主线不是再改环境、也不是再改 reward，而是：

1. **先修 deterministic readout**
2. **再修 BW 的 PPO 几何（per-UAV、per-latent normalized）**
3. **broad2local 只保留 dense ranking，不再同时堆 pull**
4. **如果还不行，再改 BW 参数化**

这次我认为这条线比继续折腾 leverage / target 更贴近你现在的真实病根。
