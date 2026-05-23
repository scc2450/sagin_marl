# BW Environment Design: V1 / V2 (2026-04-11)

这份文档的目标不是直接给出最终系统，而是把目前关于 “BW-only 环境应该怎么设计，才能既有真实物理意义，又能让 PPO / planner 真正学到东西” 的讨论收成两版：

- V1：先验证 **BW action leverage** 是否足够强
- V2：在 V1 跑通后，再逐步逼近更真实的在线系统

## 1. 设计目标

BW 动作的本质不是创造资源，而是在同一个时刻、同一个 UAV 关联的多个 GU 之间重新分配固定总带宽。

所以环境设计真正要满足的，不是 “GU 看起来差异很大”，而是：

**把一点带宽从 GU-i 挪到 GU-j，未来回报要出现真实、稳定、可观测、可控的变化。**

可以把这件事拆成 6 条要求：

1. 差异要大  
   这里的大，不是观测值差异大，而是 `marginal value` 差大。  
   即：多给 `j` 一点 BW，比多给 `i` 一点 BW，长期收益明显更高。

2. 差异要持续几步  
   不能只是一帧尖峰。高价值 / 低价值状态最好持续几步，不然 credit 很容易被冲掉。

3. 差异要能被观测到  
   真正决定边际价值的状态量，要尽量进入 observation，或者能由短期历史推断。

4. 差异要真能被 BW 改变  
   如果 BW 改动不会明显改变服务量、队列、掉包或送达，actor 就没有真正可利用的 signal。

5. 资源竞争要真实存在  
   总 BW 要偏紧，让 “该给谁” 成为真问题，而不是大家都分一点也差不多。

6. 差异要符合真实物理 / 业务逻辑  
   差异最好来自真实因素，而不是人为标签：
   - arrival / burstiness
   - 链路质量 / 速率
   - queue backlog / headroom
   - deadline / urgency
   - downstream congestion / backhaul availability

## 2. 推荐的总体策略

我建议把环境设计分成两个阶段。

### V1：先验证 leverage

目标：
- 先证明这个 BW 子问题本身有足够大的 action-level signal
- 先看清 `init / heuristic / learned / planner` 能否真正拉开
- 先避免把问题变成 “高耦合 partial observability + admission control + long-horizon closed-loop 全都一起学”

原则：
- 允许合理简化
- 但不伪造物理机制
- 优先保证 actor 能看见足够多的、现实里说得通的决定性状态量

### V2：再逼近真实系统

目标：
- 逐步收紧观测
- 引入 age / delay / hidden regime / history / recurrent memory
- 恢复更多真实系统耦合

原则：
- 只在 V1 已经确认 leverage 足够、teacher / planner 确实有 closed-loop 价值之后再做

## 3. V1：第一版环境定义

### 3.1 问题范围

V1 建议先做：

- `BW-only`
- `association` 慢时标固定
- `UAV / SAT` 其他控制先固定或用简单规则
- 先只检验 “已关联 GU 之间如何分带宽”

这样做的原因不是最终系统就该这么简化，而是先隔离出：

**“BW reallocating signal 本身够不够强。”**

### 3.2 动作定义

对每个 UAV，在当前已关联且 valid 的 GU 集合上输出一条 simplex 分配：

```text
a_t = [alpha_1, ..., alpha_m], alpha_k >= 0, sum_k alpha_k = 1
```

V1 不建议第一版就加入 `idle share`。

原因：
- 那会把问题从 “带宽怎么分” 变成 “带宽要不要用”
- 等于把 admission control 一起引进来
- 会额外放大动作空间，并稀释我们当前最关心的 GU 间分配 signal

`idle share` 只有在 V1 已经证明 “分配子问题有 leverage” 之后，再考虑是否纳入 V2。

### 3.3 真实动力学

环境真实更新应当使用 **真实服务率 / 真实链路容量**，而不是 actor 观测里的估计值。

也就是说：

- `r_true` 用于环境物理演化
- `r_est` 可以进入 observation
- 但不能拿 `r_est` 直接做真实队列更新

否则环境本身会退化成 “按照估计值演化”，会把物理系统误差和观测误差混在一起。

### 3.4 队列更新

所有队列更新都应使用 **admitted service**，而不是直接把理论容量硬灌进下游队列。

形式上应当像：

```text
s_access = min(q_GU, capacity_access(alpha, r_true))
q_GU_next = min(B_GU, q_GU - s_access + arrival)
drop_GU = max(q_GU - s_access + arrival - B_GU, 0)
```

UAV / SAT 侧同理：

```text
s_backhaul = min(q_UAV, capacity_backhaul)
q_UAV_next = min(B_UAV, q_UAV - s_backhaul + sum_g s_access_g)
drop_UAV = max(q_UAV - s_backhaul + sum_g s_access_g - B_UAV, 0)
```

这样才能避免 “队列里没有这么多数据，但系统却把这么多数据推进下游” 的 ghost traffic。

### 3.5 观测设计

V1 的原则是：

**actor 先看见足够多、现实里又说得通的决定性状态量。**

对每个已关联 GU，建议至少给：

- 当前 backlog 或 backlog ratio
- 当前可达速率估计 / CQI
- urgency：例如 deadline slack / HOL TTL / 即将过期比例
- 最近几槽实际服务量 / ACK 摘要

对 UAV 自身，建议给：

- 本机队列占用
- 本机 headroom
- downstream 回传能力估计
- 上一槽动作
- 上一槽本地 delivered / drop 摘要

如果 downstream 瓶颈确实重要，还应给每个 UAV 一个现实可实现的拥塞反馈，例如：

- SAT queue level 的粗量化广播
- congestion flag
- backpressure-like price
- delayed aggregate ACK

V1 不建议一开始就把 actor 观测削得太狠。

更稳的顺序是：

1. 先用更完整但仍物理合理的 actor 观测，把 leverage 验出来
2. 再逐步删减成部署期最严格观测

### 3.6 奖励设计

V1 奖励建议保持 KPI 对齐，不要引入标签式 shaping。

一个稳妥的形式可以是：

```text
r_t =
  + w1 * delivered_goodput
  - w2 * drop_GU
  - w3 * drop_UAV
  - w4 * drop_SAT
  - w5 * expire
  - w6 * normalized_queue_occupancy
```

注意点：

- 惩罚项不要极端大到压倒一切
- occupancy 项建议先做轻惩罚，而不是变成主导项
- 服务结算尽量连续，不要让动作轻微变化却几乎总是对 packet-level 结果无影响

### 3.7 差异应来自哪里

V1 最适合拿来调出 leverage 的，是这 5 类因素：

1. arrival burstiness 和 dwell time  
   高负载状态要持续几槽，不要一帧尖峰。

2. 链路质量异质性和相关时间  
   “当前是好链路窗口” 也要持续几槽。

3. GU / UAV / SAT buffer 大小相对平均服务率的比例  
   太大就永远不紧，太小就全在瞎掉包。

4. downstream 回传容量的波动和共享程度  
   这样 “多收接入流量到底值不值” 才会依赖下游状态。

5. deadline / slack 的分布  
   让 urgency 差异存在，但不要极端到一眼写死规则。

### 3.8 V1 暂不建议加入的东西

以下内容不是永远不该做，而是 **不建议第一版就一起上**：

- `idle share`
- 强 partial observability
- 隐藏 burst regime 标签驱动的 reward，但 actor 完全看不见对应 proxy
- GRU / LSTM 作为第一针
- 恢复完整的 UAV 位置 / SAT 选择 / association 耦合

原因是：这些改动都会把问题迅速从 “设计一个有 leverage 的 BW 子问题” 推成 “高复杂度闭环系统识别 + 控制”。

## 4. V2：更真实的环境定义

在 V1 已经满足以下条件之后，再进入 V2：

- `marginal-value gap` 在很多状态里明显非零
- `planner / teacher` 有实际 closed-loop 价值
- `learned - init` 能显著拉开
- heuristic 不是已经几乎最优

### 4.1 观测收紧

V2 可以逐步把 actor 观测收成更真实的在线可得形式，例如：

- 不再给即时真实 backlog，只给上次 BSR + BSR age
- 不再给即时真实速率，只给 CQI estimate + CQI age
- 给 delayed ACK / delivered history
- 给 coarse congestion feedback，而不是精确共享状态

### 4.2 记忆机制

当关键状态通过 age / delay / history 才能恢复时，再加：

- frame stack
- short history
- GRU / LSTM

V2 中记忆的角色是：

**补足真实可观测性不足，而不是替代环境 leverage 本身。**

### 4.3 更真实的耦合

V2 可以逐步恢复：

- UAV 位置变化
- SAT 选择变化
- association 改变
- 更长时域
- 多 UAV 共享瓶颈

但每加一层，都要重新检查：

- leverage 是否还在
- actor 观测是否仍能推断关键因果
- 新难点是不是把原来的 BW signal 淹没了

### 4.4 可选扩展

当确认系统里 “不用满全部 BW” 也是合理动作时，再考虑：

- `idle share`
- admission control / throttling
- richer backpressure signals

## 5. 推荐的验收测试

正式跑 RL 前，建议先做 4 个测试。

### 5.1 Marginal-value gap test

从同一个 state 出发，用相同随机数，把一小份 BW 从 `i` 挪到 `j`：

```text
DeltaG = G(s, a + eps * e_j - eps * e_i) - G(s, a)
```

希望看到：

- 很多状态下 `DeltaG` 明显不为 0
- `DeltaG` 的符号会随状态变化

这说明环境里真的存在状态相关的 reallocating leverage。

### 5.2 Persistence test

检查 `DeltaG` 的优势是否能持续几槽，而不是只闪一下。

### 5.3 Observability test

只用 actor 可见的 observation / history，训练一个小监督模型预测：

- 下一槽 drop
- queue jump
- downstream congestion

如果这些都几乎预测不住，那 actor 大概率也学不住。

### 5.4 Baseline test

测试几种简单规则：

- max backlog
- min slack
- backlog × rate
- backlog × downstream-headroom

希望看到：

- 它们应当是强 baseline
- 但不应已经接近最优

## 6. 一句话收束

V1 和 V2 的分工可以总结成：

- **V1**：先保证 “把一点 BW 从 i 挪到 j 值不值” 这件事在环境里是大、真、稳、可见、可控的
- **V2**：再逐步逼近真实系统的部分可观测性、长时域和多模块耦合

当前阶段最重要的不是把环境一次做得最真实，而是先把下面这件事钉死：

**UAV 只凭现实里说得通的 backlog / rate / urgency / downstream feedback，就能判断带宽该往谁那里偏，而且这种判断会在几槽内真实改变送达、掉包和队列。**
