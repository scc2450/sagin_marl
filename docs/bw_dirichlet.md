根据 **2026-04-06/07 的当前代码** 和你这轮实验结果，我会把方案再往前收一层：

## 我现在最同意你的结论

**主线已经不该再放在环境 leverage、readout，甚至也不该再放在“继续堆 dense-ranking/pull loss”。**
现在最可信的主因是：

**`alpha-only` 只修了 deterministic mean 这半边，但 live PPO 训练时真正被优化的 stochastic policy 几何，仍然基本沿着旧的 ALR logistic-normal 在走。**

这不是一句抽象判断，代码上是连起来的：

* 新分支 `score_alpha_scalar_scale` 先算
  `det_mean = uniform + alpha * (softmax(score) - uniform)`，
  然后又把这个 `det_mean` 通过 `_loc_from_probs(...)` 反推回 **ALR `loc`**；`log_scale` 仍然是一个 scalar 再 broadcast 到各 latent 维度。
* `BwPolicy.forward()` 在训练时仍然是
  `dist = MaskedLogisticNormal(loc, log_scale, mask)`，
  然后 `action = dist.rsample()`，`logprob = dist.log_prob(action)`。也就是说，**live PPO 真正在优化的，还是这个 distribution 的 sample/log_prob**，不是 `det_mean` 本身。
* 而 `MaskedLogisticNormal` 现在仍然是 **ALR 坐标 + “最后一个 valid 分量作 reference”**；`mode()` 只是把 latent `loc` 推回 simplex，`log_prob()` 也显式依赖 reference 分量和 Jacobian `log_det`。

所以你这轮现象其实非常一致：

* `alpha-only` 离线 broad2local 有正信号：说明 **mean path** 的确被你改对了一部分。
* live PPO 到 `u0120` 只是混合信号：说明 **stochastic training path** 还没有被一起改对。

一句话说，就是：

**你已经把“mean 怎么长”改好了半步，但“PPO 实际在学哪种分布”这半步还没改。**

---

## 这也解释了为什么三类实验会是现在这个形状

### 1. `readout-only` 没打中

这很合理。
因为问题不在“最终 deterministic 读出时怎么求 mode”，而在**训练过程中 sample/log_prob 本身的几何**。所以 `simplex_argmax_logprob` 只能当审计 helper，不该进主线。这个判断我同意你保留。

### 2. `geometry-only` 有帮助，但不是根治

这也和代码一致。
你现在的 `StructuredMAPPO` 已经支持 BW stage 的 `per-UAV surrogate`，而且仍然是 **raw log-prob** 做 ratio，不走你之前更激进的 normalized-ratio 分支。你这轮看到 `approx_kl_bw`、`clip_frac_bw` 下去，det gap 缩小，但 stochastic 没一起变好，这很像：

* PPO 几何确实更顺了
* 但 policy family 本身还没对齐 mean path。

### 3. `geometry + dense ranking` 还是分裂

这反而更说明问题已经收束到参数化/分布层了。
因为 dense ranking 主要是在教 **ordering**；可当前 live 训练里，ordering 最终还得通过 `det_mean -> ALR loc -> logistic-normal sample/log_prob` 这条链来变成策略更新。只要这条链没对齐，你就会继续看到：

* ordering 指标上去
* deterministic 质量不一定跟着上去

这和你现在观察到的现象是同一个故事。

---

## 所以我现在给的完整方案

我会把主线改成这个：

# **保留 `geometry-only`，停止扩展 readout 和 loss 花样；下一步专门改 BW 的 stochastic 分布层。**

不是再改环境，不是再改 reward，也不是再继续堆 dense-ranking/pull。
就是把 **“actor 输出的 deterministic mean”** 和 **“PPO 实际优化的 stochastic policy”** 统一起来。

---

# 方案分四步

## Phase 0：先把当前结论锁死

这几条先当作固定结论，不再回头：

* `readout-only` 只保留成审计工具，不进默认训练。
* `geometry-only` 保留，而且继续作为当前主线底座。
* `normalized logprob 直接算 PPO ratio` 不采用。
* `tau` 继续只做 ablation，不作为主线。
* `dense ranking` 先只保留离线审计用途，不直接再塞回 live 主训练。

你现在最不需要做的，就是再回去折腾 leverage / reward / hotspot / curriculum。

---

## Phase 1：先做一个很便宜但很关键的检查

### 查 `alpha` 有没有“过早饱和”

当前 `alpha` 的实现是：

[
\alpha = \alpha_{\max}\cdot \sigma(\text{alpha_head}(\cdot))
]

而且初始化时，`alpha_head` 最后一层 bias 会直接填成 `alpha_init_bias`。现在默认 bias 是正值，`alpha_max` 默认又不超过 1。换句话说，**`alpha` 很可能一开始就不低**。

这和你 live 里“stochastic 略好、deterministic backlog 更差”的现象是相容的：
mean path 可能被拉得**过尖、过早远离 uniform** 了。

### 所以这一步只做两件事

在当前 alpha-only 分支上，把这些日志加出来：

* `bw_alpha_mean / p90 / saturation_rate`
* `bw_log_scale_scalar_mean`
* `bw_det_l1_to_uniform_mean`
* `bw_det_entropy_mean`

你现在 `BwPolicyOutput` 已经把 `alpha / tau / det_mean / log_scale_scalar` 都暴露出来了，所以这一步很便宜。

如果你一看就发现：

* `alpha` 很快贴近 `alpha_max`
* `det_mean` 很快比 legacy 更尖

那先做一个**极小的、不是大扫超参**的 sanity ablation：

* `alpha_init_bias`: 试 `0.0` 和 `1.0`
* 或者 `alpha_max`: 试 `0.6~0.8`

这不是主线，只是为了先排掉“mean 太尖太早”这个低成本可能性。

---

## Phase 2：主线改动——换掉 BW 的 stochastic 分布层

这一步才是核心。

### 现在的问题

你已经有了一个更合理的 deterministic mean：

[
p_{\text{det}} = (1-\alpha)u + \alpha \cdot \text{softmax}(score)
]

但你现在又把它反推成 ALR `loc`，再交给 `MaskedLogisticNormal` 去 sample / 算 log_prob。
所以 mean path 和 stochastic path 还是两套几何。

### 下一步应该怎么改

**保留 `score + alpha (+ scalar scale)` 这套 actor 头，不动环境；只把 `MaskedLogisticNormal` 从 BW 训练路径里撤下来。**

也就是说：

* deterministic action 直接就是 `det_mean`
* stochastic policy 也要**直接围绕 `det_mean` 定义**
* 不再经过 `_loc_from_probs(det_mean)` 这条 ALR/reference 路

---

## Phase 2A：先做最短路径的诊断实现

我建议先实现一个：

# `MaskedMeanConcentrationDirichlet`

参数只有两个：

* `p = det_mean`
* `kappa`：每个 UAV 一个 scalar concentration

构造类似：

[
\alpha_i^{dir} = \kappa , p_i + \epsilon
]

然后：

* deterministic action 仍然是 `p`
* stochastic sample / log_prob / entropy 都来自这个分布

### 为什么我现在反而建议你先试这个

不是因为我要你回到“旧 Dirichlet 方案”，而是因为：

**旧问题不是 Dirichlet 这三个字本身，而是“per-user alpha 同时承担排序、浓度和几何尺度”。**

现在这版不一样：

* 排序由 `score`
* 离 uniform 多远由 `alpha`
* stochastic 宽度只由一个 scalar `kappa`

这三个已经拆开了。
所以它不是旧病复发，而是一个**最短、最便宜、最能直接验证“ALR/reference 是否仍是主阻塞”**的诊断分布。

如果这个分布一上去，live 里就出现：

* deterministic 不再比 legacy 差
* stochastic 还能保持甚至更好

那你就基本坐实了：
**真正卡住你的不是 mean 头，而是旧 ALR logistic-normal 的 stochastic geometry。**

---

## Phase 2B：如果 2A 有效，再决定最终版本

如果 `MaskedMeanConcentrationDirichlet` 这个诊断版有效，那再决定最终保留什么。

### 两个方向

第一个方向，直接保留它。
优点是实现简单，mean path 和 stochastic path 完全统一。

第二个方向，如果你不想长期留 Dirichlet，
那就把最终版做成：

# `MaskedCenteredLogisticNormal`

也就是：

* 不再用“最后一个 valid 分量作 reference”的 ALR
* 换成 zero-sum / centered basis
* mean 直接围绕 `det_mean` 定义
* exploration 仍只保留一个 scalar `sigma`

这个版本更“干净”，但实现成本更高。
所以我建议顺序一定是：

**先用 mean-concentration Dirichlet 做诊断，再决定要不要上 centered logistic-normal。**

---

## Phase 3：PPO 这层怎么配

这里我反而建议**少动**。

你这轮已经说明：

* `per-UAV surrogate + raw ratio` 是正方向
* 但 normalized-ratio 不该上

所以 Phase 3 只保留：

* `geometry-only`
* `per-UAV surrogate`
* `raw logprob ratio`

不要再同时改 PPO 口径。
这一步的目的是让你把因果关系锁死：

如果 Phase 2 改完以后 live 好了，
那功劳就属于 **distribution alignment**，不是 PPO 口径。

---

## Phase 4：dense ranking 什么时候再回来

不是现在。

### 正确顺序应该是：

1. 先把 stochastic 分布层换掉
2. 先看 **只靠 PPO + 新分布**，live 有没有改善
3. 如果这时 deterministic 和 stochastic 仍有差，但方向已经一致
   再把 `dense ranking` 作为小权重 aux 放回来

因为你现在已经知道：
在旧分布层下，dense ranking 会把 ordering 教好，但不保证 det quality 一起好。
所以在 Phase 2 之前，继续往 live 主线里塞它，意义不大。

---

# 这版方案的验收标准

## 第一关：离线 broad2local

对比这三组就够：

* legacy + geometry
* alpha-only + geometry
* alpha-only + 新 distribution + geometry

看四个量是否**第一次一起同向**：

* `pairwise_acc`
* `spearman`
* `det_score`
* `det_gap_to_best`
* 再加 `det_pre_backlog`

如果新 distribution 是对的，你应该看到：

**ordering 变好时，deterministic 质量也终于一起变好。**

## 第二关：live geometry-only 到 `u60/u120`

我建议只盯这个形状：

* stochastic 不能比 legacy 差
* deterministic 至少不能再在 backlog 上明显差于 legacy
* det-vs-stoch gap 继续缩小，而不是只有 stochastic 变好

## 第三关：只在 1、2 都过以后，再决定要不要把 dense ranking 放回 live

---

# 最后一句最重要的话

我现在会把这件事总结成：

**你这轮 `alpha-only` 的正信号，已经足够说明“mean 参数化方向是对的”；但它之所以没在 live 里单独解决问题，不是因为你又找错了根因，而是因为 live PPO 训练的 stochastic policy 仍然沿着旧的 `ALR loc + scalar log_scale + reference component` 几何在走。**

所以我现在最推荐的不是再改环境、再改 reward、再改 readout，而是：

## **保留 geometry-only；把 BW 的 stochastic 分布从当前 `MaskedLogisticNormal(ALR)` 换成“直接围绕 `det_mean` 定义的 mean-concentration 分布”；先用一个 scalar-κ 的诊断版做最短验证。**

如果你愿意，我下一条就直接把这个方案落成**代码级改造清单**，按你现在的仓库结构写成：

* `structured_actor.py` 里哪些字段保留、哪些删掉
* 新分布类该放在哪个文件
* `BwPolicyOutput` / `StructuredMAPPO` / `structured_parallel_eval.py` 需要改哪些接口
* 以及第一版 `MaskedMeanConcentrationDirichlet` 怎么最小接回你当前训练路径。
