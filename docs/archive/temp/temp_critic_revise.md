是的，**critic 就应该按这个系统本身来改**，不是套一个通用“大点的 MLP”就完了。

你现在最关键的结构矛盾很明确：
actor 是按 **每个 UAV 的局部决策问题**建的，输入里有 `own(10) + danger_nbr + users(20x5) + sats(6x12) + nbrs(4x4)`，而且 `users/sats/nbrs` 都经过 set-pool；但 critic 还是只吃一个 **142 维全局 state MLP**，并且卫星侧只截了 `sat_state_max=9`。这意味着 actor 在做“当前 UAV 面对哪些 GU/卫星候选”的细粒度判断，而 critic 给 advantage 时看不到同粒度的信息。

这类错配在部分可观测 MARL 里确实是高风险点：近期对 centralized critic 的分析表明，**critic 中心化不一定天然更好**，而且在部分可观测环境里，纯 state-based value 可能引入额外 bias 和 variance。MAPPO 的经验论文则说明，PPO/MAPPO 常用的是 centralized value function，但 value 输入的设计对效果很关键。([arXiv][1])

所以我给你的建议不是“给 critic 塞更多原始量”，而是：

## 你应该把 critic 改成“agent-centered centralized critic”

也就是：

[
V_{\text{team}} = \operatorname{Agg}_i ; V_i(g,\ell_i,d_i)
]

其中

* (g)：全局摘要
* (\ell_i)：第 (i) 个 UAV 的局部集合观测摘要
* (d_i)：少量按系统机理构造的派生特征

不是再用单个 `V(g)` 去硬撑三个阶段。

---

## 我建议的具体结构

### 1）保留一个 global branch

保留你现在 `get_global_state()` 的思路，但它不再是 critic 的全部，而只是一个分支。

输入还是你现有那套全局量：

* UAV pos/vel/queue
* GU pos/queue
* 截断后的 SAT pos/vel/queue
* time

这个分支的作用不是替代局部信息，而是提供：

* 全局交通压力
* 三层队列基线
* 全局几何大势
* episode phase（`t/T_steps`）

这一支你可以继续用 MLP。

### 2）新增一个 per-agent local branch

这才是最重要的。

对每个 UAV，critic 也吃一份和 actor 同语义的局部输入：

* `own`
* `danger_nbr`
* `users + users_mask + bw_valid_mask`
* `sats + sats_mask + sat_valid_mask`
* `nbrs + nbrs_mask`

但**不要直接复用 actor 的 frozen backbone 参数**。
critic 最好有自己的一套 encoder；结构可以和 actor 类似，但参数独立。原因很简单：actor 的表示是为选动作服务的，critic 的表示是为估值服务的，这两个目标不一样。

最简单的做法是：

* `own_encoder_c`
* `danger_encoder_c`
* `users_encoder_c + masked mean/max`
* `sats_encoder_c + masked mean/max`
* `nbrs_encoder_c + masked mean/max`

得到一个 `local_ctx_i`。

### 3）再加一个 derived-feature branch

这里就回答你“能不能用计算过的量而不是原始量”：

**能，而且应该用，但只能作为补充，不能替代原始集合。**

原则是：

* **候选排序和比较**相关的东西，保留原始 set 输入
  例如各候选用户的 `eta/queue/prev_assoc`，各候选卫星的 `spectral_efficiency/sat_queue/load_count/projected_count/stay_flag`。这些不能只压成几个统计量，不然 critic 还是看不见“top1 和 top2 到底差多少”。
* **瓶颈和耦合强度**相关的东西，适合加派生量
  因为这些是价值判断最关心、但纯 MLP 不一定容易自己学出来的。

我建议你给每个 UAV 加一小组 `d_i`，大概 16–32 维就够了。

---

## 我建议保留的派生特征

### A. 对 Stage 1 很重要的几何/服务特征

这些你 actor 里已经部分有了，critic 也该看到：

* `assoc_count / num_gu`
* `assoc_centroid_rel_x, assoc_centroid_rel_y`
* 当前候选 GU 的 `eta` 均值、最大值、top1-top2 gap
* 当前候选 GU 队列均值、最大值
* 与最近邻 UAV 的距离、closing speed
  这些都直接对应 accel 的价值判断。

### B. 对 Stage 2 很重要的带宽分配压力特征

* `n_valid_bw_users`
* 有效用户 queue 总和 / 均值 / max
* 有效用户 `eta` 加权均值
* “高队列用户”和“高链路用户”是否重合的一个简单 gap
  例如 top-queue user 的 eta 与 top-eta user 的 eta 差异

Stage 2 的本质是“谁该吃带宽”，critic 需要更容易看出这一步分配是否会缓解前端 backlog。

### C. 对 Stage 3 最重要的选星压力特征

这部分最关键：

* `n_valid_sats`
* 候选卫星 `spectral_efficiency` 的 mean/max/top1-top2 gap
* 候选卫星 `sat_queue` 的 mean/min/max
* 候选卫星 `load_count` 的 mean/max
* 候选卫星 `projected_count` 的 mean/min
* `stay_flag` 的个数
* 一个“链路-拥塞冲突度”
  例如：best-eta sat 的 queue 与 best-lowqueue sat 的 eta 差异

你前面 candidate scan 已经说明，sat 候选并不是没差别，而是“链路 vs 拥塞”的权衡在那儿。所以 critic 最需要的，不是更多卫星全局状态，而是**当前这架 UAV 面前 6 颗候选的冲突结构**。

### D. 全局 bottleneck 特征

这里用计算量比 raw 更好：

* `Q_gu_sum / queue_max_gu_total`
* `Q_uav_sum / queue_max_uav_total`
* `Q_sat_sum / queue_max_sat_total`
* `Q_sat_sum / (Q_gu_sum + Q_uav_sum + eps)` 这类层间占比
* 当前 step 的 arrival reference / processed reference
* 当前全局活跃卫星数
* 每层 drop ratio 的 EMA 或最近一步值

这些特征本质上是在告诉 critic：
**现在是前端堵、后端堵，还是层间失衡。**

---

## 我建议的最终 critic 公式

对每个 UAV (i)：

[
z_i = [,g_{\text{ctx}},; local_ctx_i,; d_i,; id_i,]
]

其中 `id_i` 可以是一个很小的 agent id embedding，哪怕只有 3 维 one-hot。
虽然你是参数共享，同质 UAV 仍然可能需要一点点“我是第几架机”的对称性破缺。

然后：

[
v_i = f_{\text{value}}(z_i)
]

最后聚合成团队 value：

[
V_{\text{team}} = \frac{1}{N}\sum_i v_i
]

我更建议先用 **mean**，不要用 sum。
因为 shared reward 下，mean 的尺度更稳，换 agent 数时也更干净。

---

## 为什么我不建议你直接上 Q critic

因为你现在还是 PPO / GAE 路线。
如果你突然把 critic 改成 (Q(s,a)) 或 joint-action critic，那就不只是“改 critic 结构”，而是连 PPO 的 value target、优势计算、可能连训练接口都要重做，工程量会陡增。

你现在最合适的是：

* 仍然保留 **state-value critic**
* 但把它从 `V(g)` 改成 **agent-centered pooled value**：`Agg_i V_i(g, l_i, d_i)`

这已经足够把你当前最核心的 credit-assignment 问题改善一大截。

---

## 哪些东西不要做

### 不要只靠 handcrafted feature 替代原始集合

这是最容易踩的坑。

如果你把 sat 候选全压成
“max eta / min queue / mean projected_count”
critic 还是看不到真正的候选结构。
派生特征应该是 **shortcut**，不是 **replacement**。

### 不要把 actor 的 frozen backbone 直接拿来当 critic backbone

因为你现在 Stage 2/3 本来就有冻结/局部解冻逻辑。critic 再挂在 actor backbone 上，会把训练语义搅乱。

### 不要继续让 critic 只看 `sat_state_max=9` 的全局截断卫星

这个保留当 global branch 可以，但绝不能继续当 sat 学习的主要信息来源。你 actor 的 sat 决策核心明明是“每架 UAV 当前 6 颗候选卫星”，critic 就必须显式看到这一层。

---

## 按工程量最小的落地版本

如果你想先做一个 **v1 critic**，我建议这样切：

### Critic v1

* `global_mlp(g) -> g_ctx (128)`
* `local_setpool(obs_i) -> l_i (128)`
* `derived_mlp(d_i) -> d_i' (32)`
* 拼接后 `value_head([g_ctx, l_i, d_i']) -> v_i`
* `V_team = mean_i(v_i)`

这版已经能同时覆盖：

* Stage 1 accel
* Stage 2 bw
* Stage 3 sat

而且你几乎不用改 actor。

### Critic v2

如果 v1 有效果，再考虑：

* 给 sat 分支加一个更强的 attention pooling
  让 critic 更显式地建模“这 6 颗星里谁和谁竞争”
* 或者再加一个轻量 recurrent / temporal summary
  但这不是第一步

---

## 我现在对“是否需要更多信息”的回答

**做设计层建议，我现在不缺关键信息。**
你上传的新 `model_architecture.md` 已经足够说明：

* actor 的局部输入结构是什么
* critic 现在为什么太粗
* Stage 1/2/3 的训练语义分别是什么。

真正还缺的，只有在你准备**让我把它写成代码级改造方案**时，我才需要看两处：

* `sagin_marl/rl/critic.py`
* `mappo.py` 里 critic 前向、GAE/value target 的调用位置

因为到那一步我得按你现有 tensor shape、buffer 格式和 batch 组织来写，不能只讲结构图。

一句话总结：

**critic 要按你的系统本身来设计；原始集合特征必须保留，计算过的量要作为“价值判断的捷径”补进去；最适合你的不是更大的单个 global MLP，而是“global + per-agent local + derived summary”的 agent-centered centralized critic。**

[1]: https://arxiv.org/abs/2408.14597 "[2408.14597] On Centralized Critics in Multi-Agent Reinforcement Learning"
