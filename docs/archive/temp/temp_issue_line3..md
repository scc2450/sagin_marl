可以开始了，**现在不需要你再补材料**。

我先给你一个明确结论：

## 结论

你这次要做线路 3，方向是对的；而且**第一刀不该先砍 reward，也不该先做大而全的网络升级**。
我建议的线路 3 起手式是：

**先改 actor 的观测表达方式：从“整块 flatten 后直接进 MLP”，改成“分块编码 + mask 聚合 + 再融合输出 accel”。**

也就是：

* **优先级 1：obs 表达重构**
* **优先级 2：动作/安全接口收紧**
* **优先级 3：再看是否需要更复杂策略结构**

不是三条一起上，先只动第一条。

---

## 为什么我现在把“obs 表达”排第一

因为你现在的主要矛盾，已经不是“动作头太多”了，而是：

### 1. 当前 run 实际只学 2 维 accel，但仍然学不稳

配置里这次就是：

* `enable_bw_action: false`
* `fixed_satellite_strategy: true`
* `train_accel: true`
* `train_bw: false`
* `train_sat: false` 

这说明问题不是“任务头太复杂”；是**只剩 2 维连续控制，这套表示仍然不稳定**。

### 2. 你的 actor 输入方式确实太“硬拼”

`flatten_obs()` 直接把

* `own`
* `users`
* `users_mask`
* `sats`
* `sats_mask`
* `nbrs`
* `nbrs_mask`

全部拼成一个长向量，再送进两层 MLP。

这会带来 4 个问题：

* **异构块混在一起**：用户、卫星、邻居、本机状态语义完全不同，却共享同一套前几层线性权重。
* **mask 被当普通数值**：不是结构化屏蔽，只是多拼几个 0/1。
* **集合结构被丢掉**：`users/sats/nbrs` 本来是“元素集合”，现在变成“固定槽位表格”。
* **排序/数量变化全让 MLP 硬学**：尤其 `nearest` 候选集下，槽位语义并不稳定。

这类问题非常符合你现在的现象：
**能把前期 backlog 清掉，但长期几何控制不稳。**

### 3. 你这份 obs 的尺度不齐，确实严重

材料里和样本都说明了：

* `users[..., eta]` 可以到 `8+`
* `sats[..., doppler/nu_max]` 实际可到 `23` 左右
* 其他很多块又在 `[-0.1, 1]` 或更小范围附近 

虽然 actor 前面有 `LayerNorm(obs_dim)`，但那只是**整条向量统一归一**，不是按语义块做稳定编码。
这对 flatten+MLP 的伤害很真实。

### 4. fixed 零加速度都能打过 learned

你文档里这条非常关键：当前 learned accel 比 `fixed` 的 `zero_accel_policy()` 还差，而 fixed 根本不读 obs，只输出全 0 accel。

这说明：

* 不是 baseline 有额外先验
* 而是 learned policy 的“可动性”目前更像负资产

这通常不是 reward 小调能解释的，更像**表示 + 控制接口**的问题。

---

## 为什么我不把“动作参数化”排第一

它确实有问题，但我把它排第二。

### 你现在的动作接口确实有错位风险

policy 输出 accel，环境里再做：

1. `clip(action, -1, 1) * a_max`
2. 叠加 avoidance repulsion
3. 再 clip
4. 再积分到速度/位置 

也就是说，真正执行到环境里的 accel 不完全等于 policy 想要的 accel。
这会造成 credit assignment 变差。

但是，从你给的失败 episode 看，**safety takeover 不是时时刻刻都在强接管**，而是更像在临近危险时突然介入；这说明它是问题的一部分，但不像是当前最上游的问题。你文档里也明确记了：这个坏例子主要是“晚期碰撞，不是 backlog 崩”。

所以我的判断是：

* **安全层接口错位：是真的**
* **但先修 obs 表达，收益更大**

---

## 我对线路 3 的正式判断

我现在会把你这个系统的问题分成三层：

### 第一层：主问题

**actor 的输入表示不适合这个任务。**

具体说：

* 集合观测被 flatten
* mask 没有结构化使用
* 不同物理量尺度混杂
* 多智能体几何关系被弱表达

### 第二层：次问题

**policy action 和实际执行 action 存在后处理错位。**

### 第三层：暂时不是第一刀

**critic 太简单 / 需要 attention / 需要更大模型**
这些以后可能成立，但现在还没到非做不可。

---

## 我建议你做的“线路 3 v1”

不要大改全系统。先做一个**最小但结构上正确**的版本。

### 改法

把 actor 从现在的：

`flatten(obs) -> LayerNorm -> MLP -> accel`

改成：

`own_encoder`
`users_encoder + masked pooling`
`sats_encoder + masked pooling`
`nbrs_encoder + masked pooling`
`concat(own_feat, users_feat, sats_feat, nbrs_feat) -> fusion MLP -> accel`

### 具体怎么收

#### own

`own(7)` 直接过一个小 MLP。

#### users

对每个 user row 的 5 维特征做共享小 MLP，得到 per-user embedding；
再用 `users_mask` 做 masked mean / masked max pooling，压成一个 `users_summary`。

#### sats

同理，对每个 sat row 的 9 维特征做共享小 MLP，再 masked pooling。

#### nbrs

同理，对每个 neighbor row 的 4 维特征做共享小 MLP，再 masked pooling。

#### 最后

把这 4 个 summary 拼起来，再过两层 fusion MLP 出 accel 均值。

---

## 为什么这个版本比直接上 attention 更合适

因为它有 3 个优点：

### 1. 改动小

只动 `policy.py` 的 actor 编码逻辑，
环境、reward、训练主循环基本都不用动。

### 2. 直接打中主矛盾

它会同时解决：

* 异构块混拼
* mask 使用粗糙
* 槽位语义不稳定
* 尺度混杂早期污染

### 3. 验证成本低

你只需要对比：

* 是否还会频繁输给 fixed
* collision rate 是否明显下降
* 在 queue 已经清空后，是否更少出现晚期碰撞

---

## 这版先不要做什么

为了避免线路 3 失控，我建议你**先不要同时做**下面这些：

* 不要同时改 reward
* 不要同时改 critic 结构
* 不要同时改 PPO 超参
* 不要同时把动作从 accel 改成目标速度/航向
* 不要同时上 attention / transformer

不然你很难知道到底是哪一刀起作用。

---

## 我对“有人给你的结论”的最终评价

那份总结的主判断，我现在可以比较明确地说：

**大方向正确，而且已经足够支撑线路 3 开工。**

尤其下面几条，我认为是成立的：

* 现在的问题不只是 reward 侧
* 当前 best run 的失败模式里，几何稳定性是一级嫌疑
* flatten + 尺度不齐 + 异构块混拼，是非常强的结构性问题
* learned accel 打不过 zero-accel baseline，说明当前控制自由度没有被学成有效能力

---

## 你现在要做什么

你现在不用再找材料了。
下一步最合理的是：

**直接让我按这个“线路 3 v1：分块编码 actor”给你出具体 patch 方案。**

到那一步，我会把要改的东西收成很小一包，主要集中在 `policy.py`，并告诉你：

* 要新增哪些模块
* `flatten_obs()` 怎么保留兼容
* `ActorNet` 怎么改
* 训练侧接口需不需要同步动

这样你就不是在“讨论该不该做线路 3”，而是直接进入“怎么做第一版”。
