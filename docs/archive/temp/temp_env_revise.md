可以。既然你现在明确要的是**标准化系统参数**，那我建议顺序改成：

**先把 env 改成一个语义自洽的标准化版本，冻结后，再重做 critic。**

原因很简单：你现在的 Stage 配置里，`carrier_freq=2e9`、`sat_height=500 km` 这些本身是可以放进 3GPP NTN/FR1 语境里的；3GPP 官方概述里，NTN 的 FR1 覆盖 410 MHz–7125 MHz，并明确包含 S-band 的 n256，LEO 典型高度也是 500–2000 km。可你真正不“标准化”的，不是 2 GHz 或 500 km，而是 **带宽/资源预算的语义**：你当前训练用的 `b_acc=2.2e6`、`b_sat_total=8.75e5`，如果把它们直接叫做 NR 载波带宽，就低于 3GPP 官方列出的 FR1 标准化带宽集合下限 5 MHz；所以它们现在更像“有效资源池”，不是标准意义上的 carrier bandwidth。 ([3GPP][1])

你说 `scripts/analysis/estimate_throughput.py` 和真实跑出来的不一致，这个判断我同意，而且原因也比较明确：
它做的不是“真实 episode 吞吐预测”，而是**reset 时刻的一步静态容量探针**。代码里它在 `env.reset(seed)` 后，只做一次 `_associate_users()`，再用全零的 `accel / bw_logits / sat_logits` 去算 `_compute_access_rates()`、`_select_satellites()` 和 `_compute_backhaul_rates()`，最后把 `access_cap`、`backhaul_cap`、`compute_cap_eff` 取最小当 bottleneck。它没有模拟真实 policy 的移动、带宽分配、sat 选择、队列累积、初始队列、溢出/drop，也没有看整条 episode 的动态演化，所以它和实际 eval 不一致是正常的。它更适合叫 **static capacity probe**，不适合当 ground truth。

所以，重新讲一遍 **env 应该怎么改**，我建议你不要再从“把几个数随便调大”开始，而是按下面这套顺序来。

### 1. 先把资源参数改成“标准化语义”

你现在最该做的第一刀，不是改 `task_arrival_rate`，而是改**资源字段本身的含义**。

现在这两个字段最容易误导：

* `b_acc`
* `b_sat_total`

把 backhaul 侧拆成：

* `b_sat_link`：单条 UAV→SAT 链路的标准 carrier bandwidth
* `N_RF`：同时并发的链路数
* 总 backhaul 资源 = `N_RF * b_sat_link` 或按你实现里的 active links 再做分配

这样你才能把 `5/10/15/20 MHz` 这种 3GPP FR1 标准化带宽，真正对应到 env 里。3GPP 官方页面给出的 FR1 标准化带宽集合就是从 5 MHz 起，到 100 MHz。([3GPP][2])

### 2. 给你一套我建议的“标准化 v1”起点

先别追求完美，先做一个**语义清楚的 v1**。

我建议你把 env 先改成这套：

* `carrier_freq = 2.0e9`
  这个可以保留。它落在 FR1 里，也和 3GPP NTN 概述里的 S-band / 2 GHz 一类部署语境相容。([3GPP][1])
* `subcarrier_spacing = 15e3`
  也可以保留。
* `sat_height = 500000.0`
  也可以保留；500 km 本来就在 3GPP NTN 给的典型 LEO 范围里。([3GPP][1])
* `b_acc = 10e6`
  先用 10 MHz，不要再用 2.2 MHz。
* backhaul 侧
  * **改代码语义**：新增 `b_sat_link = 5e6`，`N_RF = 2`，总 backhaul 资源通过 `2 × 5 MHz` 来形成。

这套的核心不是“10 MHz 一定比 5 MHz 更正确”，而是：
**你终于把资源参数拉回了 3GPP FR1 的标准化带宽集合里。**([3GPP][2])

### 3. 功率和天线不要一起乱改，只先做一件事：选定“终端类型”

你当前 Stage 配置里 `gu_tx_power=0.18 W`、`uav_tx_gain=300`、`sat_rx_gain=300`。其中 `0.18 W` 大约 22.6 dBm，这个量级本身不刺眼；真正需要你先拍板的是：**UAV→SAT 这条链路到底按什么终端来建模。**

你必须二选一：

**口径 1：把 UAV 当作“带定向 backhaul 天线的机载终端”。**
那高增益可以留一部分，但我建议先从 `15–20 dBi` 这一档起步，而不是直接继续用 `300` 这种线性增益值。这样仍然是“定向链路”，但不会过于激进。

**口径 2：把 UAV 当作“更接近普通 3GPP handheld / mounted UE”的终端。**
那就该把天线增益明显往下收，别再用现在这种高增益 backhaul 口径。

这一刀很重要，因为它直接决定：
你最后看到的“UAV 位置对 sat 比对 GU 更敏感”，到底是任务本身如此，还是你把 backhaul 链路建得过强造成的。

我选择口径1

### 4. 到达率必须在资源参数定完后再调

这点你前面直觉就是对的。
如果你把 `b_acc` 和 `b_sat_total` 拉回标准化带宽档位，`task_arrival_rate` 就**绝对不能原封不动**。你当前 Stage 用的是 `task_arrival_rate=1.3e5` bits/slot、`b_acc=2.2e6`、`b_sat_total=8.75e5`；一旦你把带宽提到 10 MHz 档，原来的到达率大概率会把系统直接推到另一个 regime。

但这里我不建议你再用 `scripts/analysis/estimate_throughput.py` 来定到达率。
更稳的办法是：

1. 先把资源参数改成标准化 v1。
2. 用一个**固定 baseline rollout** 跑 20 episode，比如 `Stage 2 + queue_aware_sat`。
3. 记下这 20 episode 的平均 `sat_processed_bits_per_step` 或你最认可的 end-to-end processed bits。
4. 再按目标负载系数设总到达量：

[
\lambda_{\text{total}} = \rho \cdot T_{\text{ref}}
]

[
\text{task_arrival_rate} = \frac{\lambda_{\text{total}}}{\text{num_gu} \cdot \tau_0}
]

其中：

* `T_ref` = 你 baseline rollout 的平均 end-to-end processed bits/step
* `ρ = 0.8~1.0`：适合“先把策略学出来”
* `ρ = 1.1~1.3`：适合压力测试

也就是说：
**用真实 rollout 的 end-to-end 处理量，来反推到达率。**
不要再用一步静态探针去反推。

### 5. 队列上限不要按“bit 数直觉”填，要按“缓冲时长”设

你现在 Stage 里 `queue_max_gu=3e7`、`queue_max_uav=1e9`、`queue_max_sat=2e9`，再叠加 `queue_init_gu_steps=16`、`queue_init_uav_steps=40`、`queue_init_sat_steps=32`，实际上把系统推成了一个**很深的长队列环境**。这不是不能用，但它会让 backlog 非常强势，reward 也更容易被“慢变量”主导。

标准化 v1 里我建议你改成“按缓冲时长设队列”：

* GU 队列：先按 **30–60 step** 的平均 GU 入流来定
* UAV 队列：先按 **60–100 step** 的平均 UAV 入流来定
* SAT 队列：先按 **100–150 step** 的平均 SAT 入流来定

也就是：

[
Q^{\max}*{\text{layer}} = H*{\text{layer}} \cdot \bar{\lambda}_{\text{in,layer}}
]

这里的 (\bar{\lambda}_{\text{in,layer}}) 不要拍脑袋，用 20-episode baseline rollout 的实测每层平均入流来估。
初始队列也建议先收浅一些，例如：

* `queue_init_gu_steps = 4~8`
* `queue_init_uav_steps = 8~16`
* `queue_init_sat_steps = 12~24`

先别一上来就用 16 / 40 / 32 这么深的预加载。

### 6. 所以你现在具体该怎么动 env

我会建议你直接做一个新版本，比如 `env_std_v1`，只改这些：

第一组，直接改：

* `b_acc = 10e6`
* `b_sat_total = 10e6`，或者更好地改成 `b_sat_link = 5e6, N_RF = 2`
* `task_arrival_rate`：删掉旧值，用 baseline rollout 反推
* `queue_max_*`：删掉旧 bit 数，改成按层平均入流 × 缓冲时长
* `queue_init_*_steps`：整体调浅

第二组，先别动：

* `carrier_freq`
* `sat_height`
* `uav_height`
* `doppler/fading/interference`
* reward
* observation
* actor / critic 结构

第三组，只做“口径决策”，先不细抠数：

* `uav_tx_gain`
* `sat_rx_gain`

也就是：
**先把“资源语义”和“负载语义”改正确，再决定 UAV→SAT 链路是高增益机载 backhaul 口径，还是更接近普通 UE 口径。**


