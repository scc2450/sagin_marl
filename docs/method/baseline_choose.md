可以，重新整理后我的建议是：

**主 learning baseline：IPPO**
**主 non-learning baseline：Topology-aware One-Step DPP（拓扑感知的一步式 Lyapunov / drift-plus-penalty 控制器）**

## 当前实现状态（2026-06）

当前分支已经有可运行的规则/MaxWeight 类 baseline，并迁入了一个 structured Python 版 `topology_dpp`。它复用历史 `lyapunov-dpp` 分支的思想，但按当前 structured observation/action 接口重新接入；目前还不是 native CUDA 快路径。

当前可直接跑的非学习式 baseline 分成三层：

| 方法 ID | 含义 | 当前定位 |
|---|---|---|
| `static_uniform` / `random_feasible` | 静止/随机可行动作 | sanity lower bound |
| `link_priority` / `demand_priority` | 单因子规则 | 辅助 lower bound |
| `queue_aware` / `cluster_center_queue_aware` | 队列与拓扑启发式 | 当前强 heuristic baseline |
| `maxweight_lyapunov` | 当前 native 可运行的 stage-wise MaxWeight/Lyapunov 控制器 | 当前强 non-learning baseline |
| `dpp_no_mobility` | `maxweight_lyapunov` 去掉移动控制 | 轻量消融 |
| `dpp_equal_bw` | `maxweight_lyapunov` 改用 uniform BW | 轻量消融 |
| `dpp_greedy_sat` | `maxweight_lyapunov` 改用 queue-aware SAT | 轻量消融 |
| `topology_dpp` | 枚举候选 UAV 动作、预测拓扑并联合打分 access/backhaul/BW/SAT 的 one-step DPP | 主 non-learning benchmark 候选，structured Python fallback |

兼容说明：旧名 `lyapunov` 仍然可用，但新实验和论文表格建议写成 `maxweight_lyapunov`，避免和完整拓扑枚举 DPP 混淆。

真正要作为论文主 non-learning benchmark 的，是 `topology_dpp`：枚举候选 UAV 动作，预测移动后拓扑，再做 access/backhaul/BW/SAT coupling 的一步式 DPP 优化。当前 `maxweight_lyapunov` 仍然是强规则基线和快速 native 参照。

这两个一起用，论文说服力会比较强，因为它们分别回答两件不同的事：

* **MAPPO vs IPPO**：你这个任务里，集中式 critic / CTDE 到底有没有带来真实收益。
* **MAPPO vs DPP controller**：learning 到底有没有超过一个强的、队列/时延导向的在线控制器。

这比“MAPPO vs random”或者“MAPPO vs nearest/equal-split”强得多。JCC-SAGIN 的近期综述也把这两条路线分得很清楚：一条是学习类方法，另一条是传统优化，里面专门把**任务队列管理**和 **Lyapunov drift-plus-penalty** 单列为重要方法。([arXiv][1])

---

## 一、为什么 learning baseline 选 IPPO，而不是 DDPG / MADDPG

先说结论：
**不是因为 DDPG 不能做 baseline，而是因为对你这篇论文来说，IPPO 是更“干净”的主 baseline。**

### 1）你真正要回答的问题，不是“哪种 RL 名字更常见”

你现在主方法是 **MAPPO**。
所以最关键的科学问题其实是：

> 你的收益来自“PPO 这条算法线本身”，还是来自“多智能体协同 + centralized critic + CTDE”？

要回答这个问题，最自然的对照就是 **IPPO**：
它和 MAPPO 同属 PPO 家族，actor 形式、on-policy 数据流、clip 目标、GAE、熵正则、参数共享方式都可以尽量保持一致，只把 **critic 从 centralized 改成 local / independent**。这样对比结果最容易解释。CTDE 的近期综述也明确把 MAPPO 和 IPPO 并列成 cooperative MARL 里的代表性方法，并指出两者在很多标准 benchmark 上都表现很好，差异常常集中在 critic 信息设计上。

### 2）PPO 家族本身就是强 baseline，不是“低配选项”

PPO 在 cooperative MARL 里并不弱。
《The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games》这篇很经典的工作就专门说明：PPO 系多智能体方法在多个基准上能达到很强的效果，而且相对一些 off-policy 方法，最终回报和样本效率都可能是竞争性的。([arXiv][2])

所以在你的论文里，用 **MAPPO vs IPPO** 并不是“拿一个弱 baseline 凑数”，而是在做一个很标准、很有解释力的 PPO-family 内部对照。

### 3）为什么不把 DDPG / MADDPG 作为主 learning baseline

因为它会同时改掉太多东西。

DDPG 的理论出发点就是**连续动作 deterministic policy gradient**；原始 DDPG 和 MADDPG 都依赖对动作 (a) 的梯度，经典表述直接要求动作空间连续。MADDPG 论文里也明确写了这一点。([arXiv][3])

而你的动作不是一个“纯连续、无结构”的盒子动作，至少包含这些特点：

* 2D 加速度是连续的；
* 带宽分配虽然可以编码成连续 `bw_logits`，但还带有**候选 GU 集合变化、可服务 GU 数上限**这些结构；
* 卫星选择带有**可见性约束**和**同时连接数量限制**，本质上是离散/组合决策结构。

这意味着如果你硬上 DDPG / MADDPG，通常会把论文里真正比较的对象，从
“MAPPO vs another baseline”
变成
“PPO-family + 当前动作建模 + 当前训练细节”
对上
“deterministic off-policy + replay buffer + target network + hybrid-action 额外工程”。
这样最后就很难说清楚，性能差异到底是来自 centralized critic，还是来自算法范式和动作工程本身。

### 4）“别的论文不是也会拿 PPO 和 DDPG 比吗？”

会，而且最近也有。
比如一些 2025 年的 UAV/MEC 论文会把 PPO、DDPG、TD3 放在一起比；也有工作专门因为“**离散和连续变量并存**”而对 PPO 做 hybrid-action 扩展。还有 LEO satellite edge computing 的工作用 MADDPG 做主方法，并把 PPO、DDPG 作为对照。([科学直达][4])

但那类比较通常是在回答另一个问题：

> “在这个任务里，哪一整套 DRL 路线更强？”

而你现在更该优先回答的是：

> “在同一条 PPO 路线下，MAPPO 比 independent learner 强多少？”

对于你这种已经有一套成熟 MAPPO 实现、而且动作与执行逻辑都比较复杂的系统，**IPPO 是更公平、更好解释的主 learning baseline；DDPG/MADDPG 更适合当可选附加 baseline，而不是主 baseline。**

### 5）所以 learning baseline 的最终建议

我建议你论文里这样定：

* **主 learning baseline：IPPO**
* **可选附加 learning baseline：MADDPG 或 TD3-family**

  * 只有在你算力和篇幅都够、而且愿意处理 hybrid action 工程时再加
  * 它可以证明“不是只有 PPO 系能做这题”，但不应该替代 IPPO 的主对照地位

---

## 二、为什么 non-learning baseline 选 DPP / Lyapunov，而不是简单 heuristic

你现在的主评估目标是

[
D_{\text{sys_report}}
=====================

\frac{Q_{\text{gu,sum}} + Q_{\text{uav,sum}} + Q_{\text{sat,sum}}}
{\max(\text{sat_processed_bits}, \epsilon)}
]

这本质上是一个**总积压 / 实际处理能力**的时延 proxy。
而 Lyapunov optimization / drift-plus-penalty 的核心，就是在随机到达、有限处理能力、存在队列的系统里，每个时隙做在线决策，尽量压低未来拥塞并保持队列稳定。JCC-SAGIN 的近期综述明确把它作为队列管理的重要工具，并强调它的标准做法就是把长期问题拆成每时隙的实时问题。([arXiv][1])

所以它和你的目标是天然对口的。
这不是“我给你随便编了个 queue-aware heuristic”，而是一个有明确理论来源的 baseline 家族。

---

## 三、这个 non-learning baseline 应该叫什么

我建议你不要写成泛泛的 “queue-aware heuristic”。
正式一点，写成：

**Topology-aware One-Step Drift-Plus-Penalty Controller**
中文可以写：**拓扑感知的一步式 DPP 控制器**

之所以加 “Topology-aware”，是因为你的网络拓扑每步都在变：

* UAV 一移动，GU 的最优接入 UAV 会变；
* UAV 一移动，可见卫星集合会变；
* 不同 UAV 选到同一颗星，还会改变回传带宽的均分关系；
* 每个 UAV 可同时服务的 GU 数量、可同时连接的 SAT 数量都有限制。

所以这个 baseline 不能写成一句“谁队列大就服务谁”就完了，而要写成：

> **每步先根据候选动作诱导出当前拓扑，再在这个拓扑上做一步式的队列导向在线优化。**

这就严谨了。

---

## 四、它的原理，尽量用你能直接理解的话讲

### 1）先看它想压的是什么

定义一个拥塞能量：

[
L(t)=\frac12\left(
\sum_g Q_g^2(t)+
\sum_u Q_u^2(t)+
\sum_s Q_s^2(t)
\right)
]

它不是你最终汇报的指标，但它有个好处：
如果每一步都尽量让它别涨、最好往下掉，系统通常就会往“低积压、低时延 proxy”的方向走。

### 2）一步式思想是什么

DPP 不需要未来星历。
它只看**当前时刻**的：

* 当前队列
* 当前链路质量
* 当前可见卫星
* 当前安全约束
* 当前可服务 GU / SAT 数量限制

然后在这一步选动作，让“服务高压力队列”的收益尽量大。

直觉上就是两条差压原则：

* **GU→UAV**：优先让“GU 队列大、UAV 队列没那么堵、而且接入链路又好”的流量先上来；
* **UAV→SAT**：优先把“UAV 队列大、卫星队列相对小、而且回传链路好”的流量送上去。

这就是为什么它会天然对齐你的时延目标。

---

## 五、按你的系统，整个流程应该怎么写

下面这版是我建议你论文里使用的正式流程。它不是玩具 heuristic，而是一套**每步在线控制器**。

### Step 0：输入当前状态

当前 step 拿到：

* 所有 (Q_g, Q_u, Q_s)
* 所有 UAV 当前位姿
* GU 到各 UAV 的当前信道条件 / 路损
* 各 UAV 当前可见 SAT 集合
* 各 UAV 到可见 SAT 的当前回传率、是否多普勒超限
* 安全模块参数
* 每个 UAV 的最大服务 GU 数 (K_u^{\max})
* 每个 UAV 的最大连接 SAT 数 (M_u^{\max})

### Step 1：为每个 UAV 生成一小组候选加速度

不是在连续空间里暴力优化，而是给每个 UAV 一个有限动作集，例如：

* 零动作
* 上下左右
* 四个对角
* 再加一圈小幅动作

每个候选先经过和主方法一样的**安全过滤 / clip / 干预模块**。
这样 baseline 和 learning 方法在安全层面是公平的。

### Step 2：枚举一个 joint accel 候选后，先更新 UAV 位置

先把 UAV 按这个 joint accel 走一步。
因为你的拓扑是在“先移动”之后决定的，这一步顺序必须放前面。

### Step 3：在移动后的拓扑上，重算 GU 候选接入关系

对每个 GU：

1. 计算它到每个 UAV 的当前链路质量；
2. 选最优 UAV 作为候选接入对象；
3. 如果最优链路仍不满足门限，则这一时刻不接入任何 UAV。

这样，每个 UAV 会得到一个候选 GU 集合 (\mathcal C_u)。

### Step 4：每个 UAV 在候选 GU 里，先选“值得服务”的那一批

因为一个 UAV 可同时服务的 GU 数有限，不能把所有候选都接进来。
所以对每个候选 GU (g \in \mathcal C_u)，计算一个接入优先级：

[
W_{g,u}^{\text{acc}}
====================

[Q_g - Q_u]*+ \cdot \hat r*{g,u}
]

其中 (\hat r_{g,u}) 是当前链路下的单位带宽速率或满带宽参考速率。
然后从大到小选前 (K_u^{\max}) 个，作为本步真正参与带宽分配的 GU 集合 (\mathcal S_u)。

这个量很好理解：

* (Q_g) 越大，说明前端排队越急；
* (Q_u) 越大，说明 UAV 自己也堵，不能无脑继续吸入；
* (\hat r_{g,u}) 越大，说明这条链路“同样一份带宽更值钱”。

### Step 5：在选中的 GU 上分配带宽

接下来解一个小的 per-UAV 子问题：

[
\max_{{b_{g,u}}}
\sum_{g\in \mathcal S_u}
[Q_g - Q_u]*+ , x*{g,u}(b_{g,u})
]

满足：

[
b_{g,u}\ge 0,\qquad
\sum_{g\in \mathcal S_u} b_{g,u}\le 1
]

其中 (x_{g,u}(b_{g,u})) 是这一步实际从 GU 发到 UAV 的量。

如果你想做得最强，就按你环境里的真实速率公式做一个小规模数值求解。
如果你想先做一个工程上容易跑的版本，可以用近似：

[
b_{g,u}
\propto
W_{g,u}^{\text{acc}}
]

再归一化。
但论文里我更建议你写成“小规模实时优化”，这样 baseline 更强、更好辩护。

### Step 6：在移动后的拓扑上，重算每个 UAV 的可见卫星集合

对每个 UAV (u)，得到当前可见 SAT 集合 (\mathcal V_u)。
不可见、或多普勒超限的卫星直接视为不可选。

### Step 7：联合决定每个 UAV 连接哪些卫星

这里要特别注意：
因为一颗卫星的总带宽要在连到它的 UAV 之间均分，所以一个 UAV 连某颗卫星的收益，取决于**别的 UAV 也连了谁**。

所以这一步不应该各自贪心，而应做**联合枚举**。
如果每个 UAV 最多连 (M_u^{\max}) 颗 SAT，就枚举所有满足

[
|\mathcal T_u| \le M_u^{\max}, \qquad \mathcal T_u \subseteq \mathcal V_u
]

的组合 ({\mathcal T_u})，然后对每个组合计算真实回传能力。

对某个候选 SAT 子集，定义该 UAV 的回传收益为：

[
W_{u}^{\text{sat}}
==================

\sum_{s\in \mathcal T_u}
[Q_u - Q_s]*+ , y*{u,s}
]

其中 (y_{u,s}) 用你环境里的真实均分规则和速率公式计算。
最后选使全系统总回传收益最大的 SAT 组合。

如果你现在的实现实际上每个 UAV 只能选 1 颗卫星，那这一步就退化成“枚举单星组合”；如果能同时连多颗，公式和流程也不需要改，只是组合从单星变成子集。

### Step 8：给这个 joint action 打分

对当前这个 joint accel + BW + SAT 组合，计算一步式 DPP 打分：

[
J_t
===

\sum_{u}\sum_{g\in \mathcal S_u}
[Q_g-Q_u]*+, x*{g,u}
+
\sum_{u}\sum_{s\in \mathcal T_u}
[Q_u-Q_s]*+, y*{u,s}
--------------------

\lambda_a \sum_u |a_u|^2
]

你也可以再加一个很轻的平滑项，避免加速度太抖。
然后在所有候选 joint action 里选 (J_t) 最大的。

### Step 9：输出 baseline 动作

最终输出：

* `accel`
* `bw_logits` 或对应的带宽比例
* `sat_choice` / `sat_subset`

再交给和主方法一样的安全模块、环境执行链路去跑。

---

## 六、为什么这个 baseline 是“强 baseline”，而不是简单 heuristic

因为它同时满足这几条：

### 1）它不是瞎规则，而是有理论来源

它直接来自 Lyapunov / drift-plus-penalty 这条队列优化路线；而这条路线在 SAGIN 的近期综述里就是正经方法，不是边角料。([arXiv][1])

### 2）它直接对齐你的目标

你的目标是小积压、小时延 proxy。
它每步就是围绕“谁更该被服务、服务后能不能真正减堵”来决策。

### 3）它用的是当前真实拓扑，不是静态启发式

它会显式考虑：

* GU 候选接入随 UAV 位置变化；
* 每 UAV 可服务 GU 数有限；
* 可见卫星集合随位置变化；
* 每 UAV 可同时连的卫星数有限；
* 多 UAV 共连一颗星会发生耦合。

所以它比 `nearest-sat`、`equal-split`、`centroid` 这一类基线强很多。

### 4）它不需要未来信息，和你的方法信息集公平

你已经说了：baseline 不能看未来星历，你的方法自己也不看。
DPP 正好是逐时隙在线决策，不依赖未来。([arXiv][1])

---

## 七、最终建议你在论文里怎么摆

我建议你把 baseline 部分收成下面这套：

### 主 baseline

* **Learning:** IPPO
* **Non-learning:** Topology-aware One-Step DPP Controller

### 可附加弱基线

* Equal-split BW + nearest-valid SAT + centroid / fixed accel
* Random policy
* Queue-aware BW + nearest SAT 这种“拆开组件”的消融型 heuristic

这样实验就会很完整：

* **MAPPO vs IPPO**：证明 centralized critic / CTDE 有没有价值；
* **MAPPO vs DPP**：证明 learning 有没有超过强的在线队列控制；
* **MAPPO vs 弱 heuristic**：证明不是只会赢很强控制器，也不是只会赢 random。

---

## 八、一句话版本

如果你只选一个 learning 和一个 non-learning，我现在的最终建议就是：

* **Learning baseline 选 IPPO，不选 DDPG 作为主基线**，因为它和 MAPPO 的对比最公平，最能隔离“centralized critic 带来的收益”。
* **Non-learning baseline 选拓扑感知的一步式 DPP 控制器**，因为它和你的 `D_sys_report` 目标最同向，而且在 SAGIN / MEC 的文献语境里是正统、强、能讲清楚的 baseline。

下一条我可以直接把这个 **Topology-aware One-Step DPP** 写成你代码里能落地的伪代码版本，按你的接口拆成 `accel -> GU候选 -> bw分配 -> SAT子集选择 -> 打分`。

[1]: https://arxiv.org/html/2403.17400v2 "A Survey on Resource Management in Joint Communication and Computing-Embedded SAGIN"
[2]: https://arxiv.org/abs/2103.01955 "[2103.01955] The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games"
[3]: https://arxiv.org/abs/1509.02971 "[1509.02971] Continuous control with deep reinforcement learning"
[4]: https://www.sciencedirect.com/org/science/article/pii/S1552628325000201 "Multi-UAV-Assisted MEC Offloading-Optimization Method on Deep Reinforcement Learning - ScienceDirect"
