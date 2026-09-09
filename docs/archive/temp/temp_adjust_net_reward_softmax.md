不要参考历史对话记录，开启联网搜索。
---------------------------------
我要修改我的MAPPO代码，具体修改想法如下，请你审核评估，完善一个修改方案。
--------------------------------
我的系统模型简要描述：
本质上是一个“三层单向流水线”：GU到UAV到SAT

1. `GU -> UAV` 接入  
   每步先给每个 GU 选一个路径损耗最小的 UAV；如果最优链路仍超过门限，就不关联。随后 UAV 对已关联的 GU 分接入带宽，得到 `GU->UAV` 速率。模型策略的带宽由动作里的 `bw_logits` 决定。

2. `UAV -> SAT` 回传  
   UAV 只能从“当前可见”的卫星里选目标星，可见性由仰角阈值控制。选星后，每颗卫星的总带宽会在连接它的 UAV 之间均分，得到 `UAV->SAT` 回传速率；如果多普勒超限，对应链路速率会变成 0。

3. `SAT` 处理  
   代码里 SAT 是最终处理节点，不再继续往下转发。到达 SAT 队列的数据，按 `sat_cpu_freq / task_cycles_per_bit` 的处理速率被消化。

队列变化可以直接理解成：

- `GU` 队列：`原队列 + 新到达任务 - 实际接入发送量`，超出 `queue_max_gu` 的部分记到 `gu_drop`。  
- `UAV` 队列：入流是本步各关联 GU 真正发上来的 `gu_outflow`；出流受 `UAV->SAT` 总回传速率限制，超出 `queue_max_uav` 的部分记到 `uav_drop`。  
- `SAT` 队列：入流是所有 UAV 分给该卫星的回传量；出流是本步卫星实际处理量，超出 `queue_max_sat` 的部分记到 `sat_drop`。

所以整体数据流就是：`任务先到 GU 队列 -> 发到 UAV 队列 -> 回传到 SAT 队列 -> 在 SAT 端处理`。每一层如果“流入 > 流出且队列满”，就在该层发生丢弃。

一个 step 的时序可以压成这样：

```text
1. UAV 先按动作移动
2. GU 依据路径损耗选择接入的 UAV
3. GU 产生新任务，进入 GU 队列
4. GU 队列按接入速率把一部分数据发到对应 UAV，形成 UAV 入流
5. UAV 从可见 SAT 里选星，按回传速率把一部分 UAV 队列数据发到 SAT，形成 SAT 入流
6. SAT 按自身计算速率处理队列里的数据
```

对应的队列更新就是：

```text
GU队列  = 旧GU队列  + 到达量 - GU到UAV发送量
UAV队列 = 旧UAV队列 + 来自GU的入流 - UAV到SAT发送量
SAT队列 = 旧SAT队列 + 来自UAV的入流 - SAT处理量
```

如果某层更新后超过本层 `queue_max_*`，超出的部分就记为 `drop`，队列本身被截断到上限。

-----------------------------
我的系统调整的是每个UAV的策略，包括2D加速度、分配给不同GU的带宽、卫星选择。
我希望用一个合理的指标衡量我的系统的能力。

我想参考以下内容，它的系统模型和我不一样，我主要是想参考它衡量时延的方式，请你结合联网搜索的知识审核是否合理。
----
这篇文章基于**队列状态（Queue-Aware）**以及**处理/传输速率**来量化系统延迟。由于任务通常被拆分为“本地计算”和“边缘卸载”两部分并同时执行，因此设备的总延迟取决于这两条路径中耗时较长的一条。

为了契合离散时间模型，文章对所有单项延迟的计算都使用了 $\min(\text{实际所需时间}, \tau)$ 进行限制（其中 $\tau$ 为单个时间时隙的长度），以确保计算出的延迟不会超过当前时隙。

整个延迟的推导过程可以分解为以下四个步骤：

**1. 本地计算延迟 (Local Computing Delay)**
*   **推导**：物联网设备（IoTD）处理留在本地队列中的任务所需的时间。计算方法是将**本地队列中的任务数据量**乘以**每比特所需处理的CPU周期数**，然后除以**设备本地的CPU计算频率**。

**2. 卸载路径延迟 (Offloading & Edge Computing Delay)**
如果任务被卸载到空中基站（无人机 AAV 或 高空平台 HAPS），会产生两项基础延迟：
*   **卸载传输延迟**：将任务从地面设备发送到空中基站的时间。等于**卸载队列任务量**除以设备到基站的**上行数据传输速率**。
*   **边缘计算延迟**：空中基站处理这些任务的时间。等于**边缘队列任务量**乘以对应的处理复杂度（CPU周期数），再除以**空中基站分配给该设备的计算资源（CPU频率）**。

**3. 中继延迟 (Relaying Delay)（仅在需要时）**
当关联的无人机（AAV）由于负载过高或资源不足，将任务进一步中继给更高层的高空平台（HAPS）时，还会额外叠加两项延迟：
*   **中继传输延迟**：等于无人机上的中继队列任务量，除以无人机到HAPS的传输速率。
*   **HAPS计算延迟**：等于HAPS队列中的任务量除以HAPS所分配的处理资源。

**4. 汇总求出总延迟 (Total Delay)**
*   **推导**：因为本地计算和卸载处理是同步进行的，所以系统将某一设备的**总延迟**定义为：**本地计算延迟** 与 **卸载总延迟**（即：卸载传输延迟 + 边缘计算延迟 + 中继传输延迟 + HAPS计算延迟）两者之间的**最大值**。
*   此外，文章结合了利特尔法则（Little's Law），确保了长期的平均队列延迟不超过系统设定的容忍上限，并将其作为模型必须满足的硬性约束条件。

利用**利特尔法则 (Little's Law)**，将抽象的“延迟”转化为了具体的、归一化的“队列长度”，并以此来计算强化学习中的惩罚项。

以下是具体的推导和转化过程：

**1. 理论基础：利特尔法则 (Little's Law)**
文章指出，根据利特尔法则，排队延迟（Queuing Delay）实际上与**“队列长度”与“任务到达率”的比值**成正比。这就为将延迟转化为队列长度提供了理论依据。

**2. 归一化队列长度的计算**
为了衡量长期的平均排队延迟，系统首先统计了不同处理路径上**随时间平均的任务到达率**（例如本地队列的平均任务到达率 $\overline{j}_n^l(t)$，卸载队列和边缘队列的到达率等）。

接着，文章将实际的**当前队列长度**除以对应的**平均任务到达率**，从而得到了一个“归一化的队列长度”（这在数学上等效于长期的平均排队延迟）。文章针对系统中的三种队列分别设定了这种归一化的延迟上限：
*   **本地计算延迟上限**：$\overline{\mathcal{Q}}_n^l(t) \le \mathcal{Q}_{max}^l$
*   **卸载传输延迟上限**：$\overline{\mathcal{L}}_{n,m}^o(t) \le \mathcal{Q}_{max}^o$
*   **边缘计算延迟上限**：$\overline{\mathcal{Q}}_{n,m}^e(t) \le \mathcal{Q}_{max}^e$

**3. 如何转化为强化学习的惩罚函数 (Penalty)**
在强化学习中，智能体的目标是最大化奖励，如果动作导致上述归一化的队列长度超出了最大容忍上限（即 $\mathcal{Q}_{max}^l$、$\mathcal{Q}_{max}^o$ 或 $\mathcal{Q}_{max}^e$），就需要受到惩罚。

为了实现这一点，文章使用了深度学习中常见的 **ReLU (Rectified Linear Unit)** 激活函数来计算惩罚值。以物联网设备 (IoTD) 为例，其延迟惩罚项的计算方式为：
$$p_n(t) = \psi_1 ReLU(\overline{\mathcal{Q}}_n^l(t) - \mathcal{Q}_{max}^l) + \psi_2 ReLU(\overline{\mathcal{Q}}_{n,m}^o(t) - \mathcal{Q}_{max}^o)$$

*   **如果未超时**：当当前的归一化队列长度 $\overline{\mathcal{Q}}_n^l(t)$ **小于** 上限 $\mathcal{Q}_{max}^l$ 时，括号内为负数，ReLU 函数的输出为 0，**不产生任何惩罚**。
*   **如果已超时**：当队列严重积压，导致归一化队列长度 **大于** 上限时，括号内为正数，ReLU 函数会输出超出的具体数值。超出得越多（积压越严重），系统扣除的惩罚分就越大，且会乘以一个权重因子（如 $\psi_1, \psi_2$）来调整延迟约束的相对重要性。无人机 (AAV) 和高空平台 (HAPS) 的边缘队列延迟惩罚也是采用类似的方式计算的。

通过这种方式，模型只需要观测各种缓冲区的队列长度变化，就能通过归一化直接评估系统是否满足了严苛的延迟要求。
----
我的系统和它不一样，是GU到UAV到SAT，没有跨层，只有SAT具备CPU计算的能力。我希望最小化时延，而不是设定一个容忍上限，是否合理？

-----------------------------------------------
我的模型结构如下：
# 模型结构说明

MAPPO 框架：

- actor 是 `ActorNet`
- critic 是 `CriticNet`
- actor 编码器类型为 `actor_encoder_type: set_pool`
- actor 隐层宽度为 `actor_hidden: 256`
- set encoder 嵌入维度为 `actor_set_embed_dim: 64`
- critic 隐层宽度为 `critic_hidden: 256`
- 输入归一化开启了 `input_norm_enabled: true`

## 2. 整体训练结构

当前训练是标准的集中式 critic、分散式 actor：

- actor 输入是每个 UAV 的局部观测 `obs`
- critic 输入是环境全局状态 `env.get_global_state()`
- 每个 UAV 共用同一个 actor 网络参数
- critic 输出单个标量状态价值 `V(s)`

## 3. Actor 输入定义

### 3.1 输入来源

actor 的输入通过 `flatten_obs()` 展平成一维向量。

这三个配置下，展平顺序固定为：

1. `own`
2. `danger_nbr`
3. `users`
4. `users_mask`
5. `sats`
6. `sats_mask`
7. `nbrs`
8. `nbrs_mask`

因为当前三个配置都启用了：

- `danger_nbr_enabled: true`
- `users_obs_max: 20`
- `sats_obs_max: 6`
- `nbrs_obs_max: 4`

所以 actor 的实际输入维度固定为：

```text
obs_dim
= own(7)
+ danger_nbr(5)
+ users(20 * 5)
+ users_mask(20)
+ sats(6 * 12)
+ sats_mask(6)
+ nbrs(4 * 4)
+ nbrs_mask(4)
= 230
```

### 3.2 `own`，形状 `(7,)`

当前 UAV 自身状态：

1. `uav_pos_x / map_size`
2. `uav_pos_y / map_size`
3. `uav_vel_x / v_max`
4. `uav_vel_y / v_max`
5. `uav_energy / uav_energy_init`
6. `uav_queue / queue_max_uav`
7. `t / T_steps`


### 3.3 `danger_nbr`，形状 `(5,)`

这是一个“最危险邻机摘要”，不是邻机集合本身。

特征定义：

1. 邻机距离 `dist / map_size`
2. 接近速度 `closing_speed / v_max`
3. 相对方向 `dir_x`
4. 相对方向 `dir_y`
5. 有效标记 `1.0/0.0`

### 3.4 `users`，形状 `(20, 5)`，配合 `users_mask`

每个 UAV 最多观察 `20` 个候选地面用户。当前配置中：

- `candidate_mode: nearest`
- `candidate_k: 20`
- `users_obs_max: 20`

单个用户特征定义：

1. `rel_x / map_size`
2. `rel_y / map_size`
3. `gu_queue / queue_max_gu`
4. 接入链路频谱效率 `eta`
5. 上一步是否关联到当前 UAV 的标记 `prev_association == u`

`users_mask[i] = 1` 表示第 `i` 个槽位有效，否则该槽位是 padding。

注意：

- actor 看到的是候选用户列表，不是“已经关联成功的用户列表”
- 真正做带宽分配时，只会对其中已经关联到当前 UAV 的用户做 masked softmax

### 3.5 `sats`，形状 `(6, 12)`，配合 `sats_mask`

每个 UAV 最多观察 `6` 个候选卫星。

这三个配置都满足：

- `visible_sats_max: 6`
- `sats_obs_max: 6`

因此 actor 的卫星输入槽位和卫星动作输出槽位是一一对应的。

候选卫星由环境先筛选可见卫星，再按候选规则排序并截断。当前配置使用的是：

- stage1/stage2：默认 `sat_candidate_mode: elevation`
- stage3：显式设置 `sat_candidate_mode: elevation`

单个卫星特征定义：

1. 相对位置 `rel_pos_x / (r_earth + sat_height)`
2. 相对位置 `rel_pos_y / (r_earth + sat_height)`
3. 相对位置 `rel_pos_z / (r_earth + sat_height)`
4. 相对速度 `rel_vel_x / (r_earth + sat_height)`
5. 相对速度 `rel_vel_y / (r_earth + sat_height)`
6. 相对速度 `rel_vel_z / (r_earth + sat_height)`
7. 多普勒归一化 `nu / nu_max`
8. 当前链路频谱效率 `spectral_efficiency`
9. 卫星队列归一化 `sat_queue / queue_max_sat`
10. 当前负载计数归一化 `load_count / num_uav`
11. 预计接入后负载倒数 `1 / projected_count`
12. 是否是当前已连接卫星 `stay_flag`

`sats_mask[i] = 1` 表示该候选槽位有效，否则为 padding。

即使 stage1 和 stage2 不训练卫星动作头，actor 仍然会看到这组卫星观测，因为 UAV 加速度和带宽决策也会用到这些卫星侧上下文。

### 3.6 `nbrs`，形状 `(4, 4)`，配合 `nbrs_mask`

每个 UAV 最多观察 `4` 个邻近 UAV，按距离排序截断。

单个邻机特征定义：

1. `rel_pos_x / map_size`
2. `rel_pos_y / map_size`
3. `rel_vel_x / v_max`
4. `rel_vel_y / v_max`

`nbrs_mask[i] = 1` 表示该槽位有效，否则为 padding。

## 4. Actor 骨干结构

### 4.1 `set_pool` 编码器

- 对标量/小向量输入单独编码
- 对可变长集合输入先逐元素编码
- 再通过带 mask 的 mean pooling + max pooling 聚合
- 最后把所有分支特征拼接后送入融合 MLP

### 4.2 各子编码器

当前参数：

- `actor_set_embed_dim = 64`
- `actor_hidden = 256`
- `input_norm_enabled = true`

因此 actor 各分支为：

#### `own_encoder`

```text
LayerNorm(7)
Linear(7 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

输出维度：`64`

#### `danger_nbr_encoder`

```text
LayerNorm(5)
Linear(5 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

输出维度：`64`

#### `users_encoder`

对每个用户槽位独立编码：

```text
LayerNorm(5)
Linear(5 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

然后结合 `users_mask` 做两种池化：

- masked mean pooling，得到 `64`
- masked max pooling，得到 `64`

拼接后得到用户集合特征：`128`

#### `sats_encoder`

对每个候选卫星槽位独立编码：

```text
LayerNorm(12)
Linear(12 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

再做 masked mean + max pooling，得到卫星集合特征：`128`

#### `nbrs_encoder`

对每个邻机槽位独立编码：

```text
LayerNorm(4)
Linear(4 -> 64)
ReLU
Linear(64 -> 64)
ReLU
```

再做 masked mean + max pooling，得到邻机集合特征：`128`

### 4.3 特征融合层

当前配置启用了 `danger_nbr`，因此最终拼接维度为：

```text
fusion_in_dim
= own_feat(64)
+ danger_nbr_feat(64)
+ users_pool(128)
+ sats_pool(128)
+ nbrs_pool(128)
= 512
```

融合 MLP 为：

```text
Linear(512 -> 256)
ReLU
Linear(256 -> 256)
ReLU
```

其输出就是所有动作头共享的 actor backbone 表示。

## 5. Actor 输出头定义

### 5.1 加速度头 `accel`

所有阶段都存在 `accel` 头：

```text
mu_head: Linear(256 -> 2)
log_std: trainable parameter, shape=(2,)
```

动作分布：

- 对每一维使用高斯分布 `Normal(mu, std)`
- `log_std` 先 clamp 到 `[-5, 2]`
- 再通过 `exp(log_std)` 得到 `std`
- 训练时使用 `rsample()`
- 最终经过 `tanh` squash 到 `[-1, 1]`

输出含义：

- actor 输出的是归一化加速度 `a_norm in [-1, 1]^2`
- 环境执行前会乘上 `a_max = 5.0`
- 即实际物理加速度范围是 `[-5, 5] m/s^2`

注意执行层还有额外安全处理：

- 避碰修正
- 边界硬约束
- pairwise hard filter

因此真正执行的加速度不一定等于策略原始输出。

### 5.2 带宽头 `bw`

只有 stage2 和 stage3 启用了 `enable_bw_action: true`，因此这两个阶段 actor 中存在 `bw` 头：

```text
bw_head: Linear(256 -> 20)
bw_log_std: trainable parameter, shape=(20,)
```

当前配置的特殊设置：

- `bw_head_zero_init: true`
- `bw_log_std_init: -1.5`
- `bw_log_std_trainable: false`

也就是说：

- stage2/stage3 的 `bw_head` 初始权重和偏置都为 0
- `bw_log_std` 初始化为 `-1.5`
- 训练时 `bw_log_std` 不更新

更重要的是，当前 stage2/stage3 不是“直接输出最终带宽分配”，而是 residual 带宽策略：

- `exec_bw_source: heuristic_residual`
- `bw_residual_alpha: 0.5`
- `bw_residual_clip: 1.0`

因此 actor 的 `bw` 头建模的是一个 residual logit `delta_bw`，其采样范围为：

```text
delta_bw = tanh(z_bw) * 1.0
```

即每个槽位都在 `[-1, 1]` 内。

真正送入环境执行的带宽 logit 为：

```text
bw_exec = clip( bw_heuristic + 0.5 * clip(delta_bw, [-1, 1]), [-5, 5] )
```

其中：

- `bw_heuristic` 来自 `queue_aware_policy()`
- `[-5, 5]` 来自 `bw_logit_scale`

而 PPO 计算 `logprob` 时，对应的是 residual 动作 `delta_bw`，不是上面融合后的 `bw_exec`。

### 5.3 卫星头 `sat`

只有 stage3 启用了：

- `fixed_satellite_strategy: false`
- `train_sat: true`

所以只有 stage3 actor 中存在卫星头：

```text
sat_head: Linear(256 -> 6)
sat_log_std: trainable parameter, shape=(6,)
```

其分布形式与 `bw` 类似：

```text
sat_logits = tanh(z_sat) * sat_logit_scale
```

当前 `sat_logit_scale = 5.0`，所以每个候选卫星槽位的输出范围是 `[-5, 5]`。

这些 `sat_logits` 与 `sats` 输入中的 6 个候选卫星槽位一一对应。

## 6. 动作向量拼接与环境语义

### 6.1 actor 内部动作向量顺序

在 `ActorNet` 内部，动作向量的拼接顺序固定为：

1. `accel(2)`
2. `bw(20)`，如果启用
3. `sat(6)`，如果启用

因此三阶段动作维度分别是：

- stage1：`2`
- stage2：`2 + 20 = 22`
- stage3：`2 + 20 + 6 = 28`

### 6.2 `bw_logits` 如何转成实际带宽分配

环境并不会把 `bw_logits` 直接当作带宽比例，而是做如下处理：

1. 先根据路径损耗阈值完成 GU 到 UAV 的关联
2. 对当前 UAV 的候选用户列表取出已关联子集
3. 仅在该子集上做 masked softmax
4. 得到带宽份额 `beta`
5. 用 `beta * b_acc * spectral_efficiency` 计算接入速率

因此：

- `bw` 输出本质上是“关联后用户子集上的相对优先级 logit”
- 没有关联成功的用户即使在观测里出现，也不会得到带宽

如果 `enable_bw_action: false`，环境会对已关联用户做平均分配。

### 6.3 `sat_logits` 如何转成卫星选择

stage3 中：

1. 环境先得到当前 UAV 的可见卫星候选
2. 只保留前 `sats_obs_max = 6` 个槽位
3. 读取与这 6 个槽位一一对应的 `sat_logits`
4. 按 `sat_select_mode: sample` 做抽样
5. 最多选择 `N_RF = 2` 颗卫星，且无放回

如果启用了多普勒过滤，则多普勒超限的候选卫星会被强行置为无效。

stage1 和 stage2 中，由于 `fixed_satellite_strategy: true`，不使用 `sat` 头，环境会在当前保留的候选卫星集合里选择距离最近的一颗卫星。

## 7. Critic 输入定义

critic 使用 `SaginParallelEnv.get_global_state()` 输出的一维全局状态。

当前三个配置下，global state 的拼接顺序为：

1. `uav_pos.flatten() / map_size`
2. `uav_vel.flatten() / v_max`
3. `uav_queue / queue_max_uav`
4. `uav_energy / uav_energy_init`
5. `gu_pos.flatten() / map_size`
6. `gu_queue / queue_max_gu`
7. `sat_pos.flatten() / (r_earth + sat_height)`
8. `sat_vel.flatten() / (r_earth + sat_height)`
9. `sat_queue / queue_max_sat`
10. `t / T_steps`

### 7.1 当前实际 state 维度

当前三个配置虽然 `num_sat` 不同：

- stage1/stage2：`num_sat = 72`
- stage3：`num_sat = 144`

但它们都设置了：

- `sat_state_max = 9`

因此 critic 不会看到全部卫星，只会保留“对所有 UAV 来说最大仰角最高”的前 9 颗卫星进入 global state。

所以这三个配置的全局状态维度相同：

```text
state_dim
= uav_pos(3 * 2 = 6)
+ uav_vel(3 * 2 = 6)
+ uav_queue(3)
+ uav_energy(3)
+ gu_pos(20 * 2 = 40)
+ gu_queue(20)
+ sat_pos(9 * 3 = 27)
+ sat_vel(9 * 3 = 27)
+ sat_queue(9)
+ time(1)
= 142
```

## 8. Critic 结构

当前 critic 没有使用 set encoder，而是一个普通 MLP：

```text
LayerNorm(142)
Linear(142 -> 256)
ReLU
Linear(256 -> 256)
ReLU
Linear(256 -> 1)
```

最后经过 `squeeze(-1)` 输出标量状态价值。

即：

- 输入：全局状态 `s`
- 输出：`V(s)`

----------------------------------------------

我觉得我的模型结构有问题，三种动作差别很大，这个结构能把三种动作都训练出来吗？网络规模合不合适？请你结合联网搜索的知识判断。

-----------------------------------------------

我目前的奖励函数如下：

```text
R = c_access * throughput_access_norm
  + c_backhaul * throughput_backhaul_norm
  - c_queue * gu_queue_arrival_steps
```

其中这些量都是全局求和后再归一化得到的，不是单个 UAV 各自单独算的。

按三个阶段具体是：

- stage1: `reward_mode: throughput_only`，但没有单独写 `throughput_only_*` 系数，所以走代码默认值，等价于 `R = throughput_access_norm + throughput_backhaul_norm`。
- stage2: `R = 1.0 * throughput_access_norm + 0.0 * throughput_backhaul_norm - 0.05 * gu_queue_arrival_steps`。[stage2 配置]
- stage3: `R = 0.0 * throughput_access_norm + 1.0 * throughput_backhaul_norm - 0.0 * gu_queue_arrival_steps`，也就是基本只看回传吞吐。[stage3 配置]

至于“一个 UAV 一个奖励函数还是全局一个”，答案是：全局一个。`_compute_reward()` 只算一次标量 `reward`，然后在 `step()` 里直接复制给所有 agent：

```python
rewards = {agent: reward for agent in self.agents}
```

我觉得我的奖励函数有问题，一个是上文说的需要改成时延的指标，一个是所有UAV agent用同一个reward合理吗？请你结合联网搜索的知识判断。

---------------------------------------------------

还有一件事，我的带宽分配和卫星选择两类动作用到softmax函数，我看有下文这种处理方式（它是别的文章的，和我的系统不一样），请你结合联网搜索的知识判断这样是否正确，是否有必要。
-----
在这篇文章中，Softmax 函数主要被用来处理智能体的动作输出（如计算**资源分配比例**、带宽分配比例和**任务卸载比例**），以确保这些输出的值都落在 区间内，并且总和恰好等于 1。

但是，直接使用标准的 Softmax 会在强化学习训练中带来一些不稳定性，文章针对这一点进行了专门的数学处理。

**1. 标准 Softmax 面临的问题（为什么会网络震荡？）**
文章指出，如果神经网络输出的多个动作偏好值（即传给 Softmax 的输入值）完全相同或非常接近，标准的 Softmax 函数就会输出一个**均匀的概率分布**（比如分配给三个服务器的比例变成了均等的 33%、33%、33%）。在多智能体不断试错的训练过程中，这种“犹豫不决”的均等输出会导致**网络震荡（oscillation）和发散（divergence）**，使得模型很难收敛到一个确定的最优策略。

**2. 文章采取的解决办法（如何改造 Softmax？）**
为了稳定训练过程，作者对输入给 Softmax 的数值进行了三步特殊的“包装”处理：

*   **第一步：乘上缩放因子（$\iota$）来“放大差异”**。作者引入了一个常数缩放因子 $\iota$，将其乘在神经网络的原始输出值（$\chi$）上。这一步的核心目的是**人为放大各个输出选项之间的差异（amplify the output differentiation）**。差异被放大后，Softmax 就能给出更倾向于某一方的明确决策，而不是给出平均主义的结果。
*   **第二步：减去平移项（$-\iota/2$）**。在乘上缩放因子后，统一减去 $\iota/2$ 对数值区间进行中心化调整。
*   **第三步：拼接一个固定基准值（0）**。这是最关键的一步防震荡措施。作者将经过上述“放大和平移”处理的各个网络输出值组合起来后，强制在末尾**拼接一个固定值 0**，最后再把这组包含了 0 的序列打包丢给 Softmax 去计算最终的比例。

**直观的公式形态：**
经过改造后，计算资源分配或卸载比例的 Softmax 结构变成了类似这样：
$$Softmax(\{ \iota * \chi_1 - \iota/2, \ \iota * \chi_2 - \iota/2, \ \dots, \ 0 \})$$

**总结来说**，文章通过**“引入缩放因子放大动作差异”**以及**“强制拼接固定零值”**的操作，避免了因 Softmax 遇到相似输入而输出均等概率所引发的训练崩溃问题。

在神经网络中，**给 Softmax 函数的输入拼接一个固定值 0，其核心道理是为了提供一个“固定锚点（基准值）”，从而消除 Softmax 函数自带的“冗余性”，防止网络训练时发生数值漂移和震荡**。

**1. Softmax 的“相对性”与“漂移”问题**
Softmax 函数的特点是它只看重输入数值之间的**差值**，而不看重绝对值。
*   例如：输入 ``，Softmax 计算出的概率分布大约是 `[27%, 73%]`。
*   如果你把输入同时加上 100 变成 ``，Softmax 计算出的概率分布**依然是 `[27%, 73%]`**。

这就导致了一个问题：在多智能体不断试错的训练中，神经网络的输出值可能会因为梯度的累积，整体不受控制地变大（比如全变成 1000）或变小（全变成 -1000）。由于它们之间的差值不变，策略虽然没变，但巨大的数值会导致网络发生剧烈的**震荡（oscillation）甚至发散崩溃（divergence）**。

**2. 为什么相同的值会导致“犹豫不决”？**
文章中提到，如果神经网络对所有动作的偏好输出相同（例如都是 `c`），那么输入给 Softmax 的就是 `[c, c, ..., c]`。根据 Softmax 的原理，这会强制输出一个**完全平均的概率分布**（比如 `[33%, 33%, 33%]`）。这种平均主义在强化学习中意味着智能体“完全不知道该怎么选”，会进一步加剧训练的不稳定性。

**3. 拼接固定值 0 的“锚点”作用**
为了解决上述问题，作者改变了做法：如果有 $N$ 个选择，神经网络只负责输出 $N-1$ 个值，然后在最后一个位置强行拼上一个固定的 `0`。
此时，输入给 Softmax 的序列变成了 `[输出1, 输出2, ..., 0]`。
*   这个 `0` 就像海平面一样，提供了一个**绝对的参照系**。
*   如果网络输出相同的值（比如都是 `5`），带入 0 后变成 ``，Softmax 会明确知道前两项概率大，最后一项概率小，不再输出平均的分布。
*   如果网络输出都是 `-5`，变成 `[-5, -5, 0]`，Softmax 就会把绝大多数概率分给最后一项。

**总结来说**，拼接一个固定的 0，剥夺了神经网络输出值集体“随波逐流”自由漂移的权利。网络必须以 0 为基准，实打实地输出相对 0 来说更高或更低的明确数值，这样就完美消除了导致网络震荡的隐患，稳定了整个训练过程。

------
--------------------------------------------------

我担心我的系统训练不出来收敛不了。上文很长，请你一个问题一个问题回答。给我一个修改方案。
