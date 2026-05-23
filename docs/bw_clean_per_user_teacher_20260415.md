# 干净版 Per-User BW Teacher 方案（2026-04-15）

## 1. 目标

为当前 `1 UAV` 的 structured BW 线路设计一条**干净的 per-user teacher** 主线，不再混用：

- `branch_delta` 的 actor-advantage PPO
- 当前的 `proxy_score` shortcut
- 当前基于 stick-breaking latent 的 `per-slot PPO`
- heuristic imitation
- critic/value 路径

这份文档定义的是一条**独立的 BW 训练路线**：

- policy：deterministic、score-only
- teacher：多步 per-user marginal utility
- loss：直接做动作匹配，不走 aux loss


## 2. 这份方案明确排除什么

这份方案**不是**前面那条 shortcut 版 marginal teacher。

明确排除：

- `target-only proxy_score`
  也就是“往某个 target 多挪一点 BW，同时其余用户按比例一起缩”
- `proxy_score -> dot(delta_action, proxy_score)` 这种 sample credit
- 当前 `per-slot surrogate`
  因为当前 `slot` 是 stick-breaking latent factor，不是用户维度
- 把 `pairwise n*n` 当作第一版 teacher
  因为 simplex 上 `m` 个 valid 用户只有 `m-1` 个自由度
- BW 头里的 `kappa/tau` 浓度学习
- 第一版就用 `KL/CE`


## 3. 策略形式

对干净版 per-user 路线，BW policy 应该是：

- 每个 valid 用户一个 `score_i`
- 在 valid 用户上做 `masked softmax(score)`
- 只用 deterministic 动作

形式上，对 valid 用户集合 `V`，

\[
a_\theta(s) = \text{masked-softmax}(score_\theta(s))
\]

这条路线**不使用**：

- stick-breaking BW sampling
- `kappa`
- `tau`
- per-slot PPO 的 log-prob 分解


## 4. Snapshot 和参考动作

在 BW 决策状态 `s_t` 上，先构造当前 deterministic 参考动作：

\[
a_{ref} = a_\theta(s_t)
\]

后续所有 teacher 计算都基于**同一个当前 BW snapshot**：

- 同一个当前 `assoc`
- 同一个当前 `sat_selection`
- 同一个当前 queue
- 同一个当前 EMA-derived reward cost
- 同一个 valid-slot mask

所有 branch rollout 在第一步之后都使用**同一个 deterministic follow policy**。


## 5. 多步 Per-User Marginal Utility

设 `V` 为 valid 用户集合，`m = |V|`。

选一个参考 donor：

\[
r = \arg\max_{j \in V} a_{ref,j}
\]

它只是用来在 simplex 上定义一个一致的局部基准方向。

对每个 `i \in V, i \neq r`，构造一个合法的一步小扰动：

\[
a^{(i)} = a_{ref} + \delta_i (e_i - e_r)
\]

其中：

- `e_i` 是用户 `i` 的 one-hot 基向量
- `delta_probe > 0` 是一个小的 probe 半径
- 实际可用的扰动量为

\[
\delta_i = \min(\delta_{probe}, a_{ref,r})
\]

如果某个样本里 `delta_i` 数值太小，可以直接跳过。

### 5.1 Return 的定义

对每个分支，只有**第一步 BW 动作**不同。

参考分支：

- 第 `t` 步执行 `a_ref`
- 第 `t+1 ... t+h-1` 步都执行当前 deterministic BW policy 的闭环动作

用户 `i` 分支：

- 第 `t` 步执行 `a^(i)`
- 第 `t+1 ... t+h-1` 步同样执行当前 deterministic BW policy 的闭环动作

因此：

\[
G_h^{ref} = \sum_{k=0}^{h-1}\gamma^k r_{t+k}^{ref}
\]

\[
G_h^{(i)} = \sum_{k=0}^{h-1}\gamma^k r_{t+k}^{(i)}
\]

定义 per-user marginal utility：

\[
u_i = \frac{G_h^{(i)} - G_h^{ref}}{\delta_i}, \quad i \neq r
\]

并令：

\[
u_r = 0
\]

说明：

- 除以 `delta_i` 是为了把 reward 差变成**单位带宽的边际效用**
- 如果所有方向上的 `delta_i` 都完全一样，那除不除只差一个公共缩放；如果有的方向被裁短了，则必须除
- 这一定义天然可以扩展到多步 `h`


## 6. 从 Utility 到合法的 Teacher 方向

在 valid 用户内先把 `u` 做中心化：

\[
\tilde u_i = u_i - \frac{1}{m}\sum_{j \in V} u_j
\]

再把它归一化成一个零和方向：

\[
d_i = \frac{\tilde u_i}{\|\tilde u\|_1}
\]

对 `i \in V` 使用上式，对 invalid 用户令 `d_i = 0`。

这个方向满足：

- `sum_{i \in V} d_i = 0`
- `d_i > 0` 表示用户 `i` 应该多拿一点
- `d_i < 0` 表示用户 `i` 应该少拿一点

这就是干净的 per-user 方向信号。

这里不需要先做 `pairwise` 展开。


## 7. Teacher 目标动作

从 `a_ref` 出发，沿着方向 `d` 走一个合法的小步，得到 teacher target action。

最大的可行步长为：

\[
\rho_{max} = \min_{i:d_i<0} \frac{a_{ref,i}}{-d_i}
\]

再取一个保守比例：

\[
\rho = \beta \rho_{max}, \quad \beta \in (0,1)
\]

第一版建议：

\[
\beta = 0.5
\]

于是 teacher target action 定义为：

\[
a_{target} = a_{ref} + \rho d
\]

因为 `d` 是零和方向、且 `rho <= rho_max`，所以这个目标动作仍然是合法 simplex 动作：

- valid 用户上非负
- invalid 用户仍为 0
- 总和保持不变


## 8. 训练目标

这条路线里，BW **不再使用 PPO advantage**。

而是直接让 BW actor 的动作去拟合 `a_target`：

\[
a_\theta(s_t) \approx a_{target}
\]

第一版推荐的损失是：

\[
L_{BW} = \text{Huber}(a_\theta(s_t), a_{target})
\]

为什么第一版先用 Huber：

- teacher 是通过有限扰动和有限 horizon 构造出来的
- `a_target` 是一个局部改进目标，不是精确的全局最优动作
- 相比 KL，Huber 对 teacher 的近似误差更宽容，更稳

这里的 `L_BW` 是这条路线的**主损失**，不是 auxiliary loss。


## 9. 第一版推荐超参数

对第一条干净版 per-user 训练，建议：

- follow policy：deterministic
- BW head：score-only + masked softmax
- probe 半径：小的固定 `delta_probe`
  第一版可以复用当前 BW 小扰动量级，比如 `0.02`
- teacher horizon：先用一个中等长度的多步值
  推荐第一版：`h = 10`
- target 步长比例：`beta = 0.5`
- loss：动作空间上的 `Huber`

为什么不第一版就用 `h = 30`：

- 现在每个 BW snapshot 要做 `m-1` 个 counterfactual rollout
- 中等 horizon 更适合作为“per-user teacher 本身有没有价值”的第一轮验证


## 10. 与当前最好 baseline 的关系

当前最好的 baseline 仍然是：

- `Reward-Obs + branch_delta h=30`

这条线应继续保留，作为单独对照。

本文定义的是一条**新的 BW 学习路线**，不是对当前 shortcut marginal-teacher 实现的修补。


## 11. 最小对照计划

当后续实现时，建议先做这两条的干净对照：

1. 当前最好 baseline
   - `Reward-Obs + branch_delta h=30`

2. 干净版 per-user 路线
   - score-only BW policy
   - deterministic follow policy
   - 多步 per-user marginal utility
   - 用 Huber 直接匹配 target action

这样可以直接判断：

- 干净版 per-user supervision 能不能接近或超过当前 direct branch teacher 路线


## 12. 一句话总结

干净版 per-user BW 设计就是：

> 以当前 deterministic BW 动作为局部基点，逐个 valid 用户方向计算多步 marginal utility，把这个 utility 变成 simplex 上一个合法的小改进目标动作，然后直接训练一个 deterministic、score-only 的 BW policy 去拟合这个目标动作。
