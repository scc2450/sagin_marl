# Structured SAT Pairwise Ranker 方案（2026-04-16）

## 1. 为什么要从显式 SAT rollout teacher 转向新方法

最近这轮 `sat` 局部换星时效检查已经说明，`sat` 动作的价值主要体现在后续闭环，而不是当步：

- `H=1` 的 best gap 均值很小
- `H=5`、`H=10` 的 gap 明显更大
- 第 0 步 reward 在总差值里的占比很小

这说明当前 `sat_clean_joint v1` 的核心假设不成立：

- 单步 teacher 标签不是正确的监督目标
- 如果把显式 rollout teacher 的 horizon 拉长，训练成本会迅速变高

但这里还有一个很重要的点：

**问题并不是“原 PPO 只看到了即时回报”。**

当前 structured 训练本来就已经在用多步 return：

- rollout 时 `accel/sat` 先记 `reward=0`，环境 reward 记在 `bw`
  - [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:1516)
- 后面算 `step_level return` 时，又把整步 return 回填给 `accel` 和 `sat`
  - [structured_buffer.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_buffer.py:489)
- 训练入口固定用的是 `target_mode="step_level"`
  - [train_structured.py](/d:/研三上/毕设/sagin_marl/scripts/train_structured.py:1137)

所以真正的问题是：

1. `sat` 虽然拿到了长时延 return，但那只是单条轨迹上的高噪声样本。
2. 它没有“同一个 snapshot、同一个随机性下，只改 sat 动作”的配对对照。
3. 好的 `sat` 动作可能是“当步更差、后面更好”，PPO 很难靠采样稳定学到。
4. `sat` 动作空间是组合动作，真正有效的动作本来就不容易采样到。

所以“环境里存在明显长时延收益”并不意味着原 PPO 就一定学得出来。

## 2. 为什么不直接改成 SAT 专用 Q / delta critic

`bw` 这边已经给了一个很强的负面经验：  
如果直接去回归绝对 `Q(s,a)` 或者绝对 delta，误差可能很大，训练会很不稳。

所以 `sat` 这边不应该直接把方向押在：

- 绝对 `Q(s,a)` 回归
- 绝对 long-horizon delta 回归

我不能保证 `sat` 不会踩和 `bw` 一样的坑。

但 `sat` 和 `bw` 也有一个关键差别：

- `bw` 是连续分配动作
- `sat` 在 structured 里是离散 legal-subset 动作
- 当前 `sat` actor 本来就是 categorical over subset
- 我们真正需要的往往是“哪个动作更好”，而不是“它到底值多少”

所以相比绝对值回归，`sat` 更适合先走：

**排序 / 偏好学习（ranking / preference）**

而不是一上来做绝对值 critic。

## 3. 核心思路

把“每个 SAT context 都用显式长 horizon rollout 找 teacher”的主训练路径，改成：

1. 用少量 matched rollout 构造偏好标签
2. 训练一个 `sat` 联合动作 ranker
3. actor 用 pairwise preference 信号更新，而不是靠绝对 `Q`

这里的 ranker **不是绝对 critic**。

它只需要回答一个问题：

- 在同一个 `sat-stage snapshot` 下
- 动作 `a` 和动作 `b`
- 哪个更好

也就是说，它学的是**排序关系**，不是绝对回报值。

## 4. 哪些东西可以保留

下面这些基础设施都不用推倒：

- 当前正式的 `sat clean-only` 路线
- 固定 partner：
  - `exec_accel_source = cluster_center_queue_aware`
  - `exec_bw_source = queue_aware`
- `weighted_workload_level` 作为环境侧目标
- `sat-stage snapshot + RNG` 一致 replay
  - [structured_driver.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/structured_driver.py:528)
- 当前 structured `sat` actor
  - [structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:242)
- 当前 legal-subset top-k helper
  - [structured_actor.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py:371)
- `uav_id_norm` 这种轻量破对称特征

真正要换的是：  
`sat` 的训练目标和训练信号。

## 5. pairwise 样本怎么构造

### 5.1 snapshot 来源

继续只用**当前 update 的 fresh rollout context**。

也就是说：

- 先在当前 rollout 里抽 `sat-stage snapshot`
- 这些 snapshot 继续来自当前 SAT clean 路线里已经缓存的 `sat_stage_state`
  - [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:1329)

抽样规则也可以先不大改，继续沿用：

- 当前 SAT actor 的归一化 entropy
- 加少量均匀随机样本

### 5.2 候选 panel

对每个被抽中的 `sat` context：

1. 用当前 SAT actor 取每个 UAV 的 top-M legal subset
2. 把 rollout 里实际执行的动作单独保留成 anchor
3. 用这些动作构造一个 pruned joint panel

建议起步：

- 每个 UAV 取 `M=3` 或 `4`
- 如果 executed action 不在 top-M 里，再额外并进去

executed action 要保留的原因很直接：

- 它是当前策略真正做出来的动作
- 最适合拿来当 pairwise 比较基线

### 5.3 pair 怎么挑

不要在线上把 panel 里所有动作两两全比一遍，这样太贵。

每个 snapshot 只构造少量最有信息量的 pair：

1. `executed action` vs `top-1 candidate`
2. `executed action` vs `top-2 candidate`
3. `top-1` vs `top-2`
4. 如果 panel 足够多样，再补一个分歧更大的 pair

这样在线校准成本接近 `O(panel)`，而不是 `O(panel^2)`。

### 5.4 pair 的真值标签怎么来

只对少量 calibration context 做真实 matched rollout：

1. 从同一个 `sat-stage snapshot` 出发
2. 恢复同一个 RNG
3. 分别执行动作 `a` 和动作 `b`
4. 后续都走固定 `queue_aware_sat + queue_aware_bw`
5. 计算长时延 score 差值

标签规则：

- 如果 `score(a) - score(b) > margin`，记成 `a > b`
- 如果 `score(b) - score(a) > margin`，记成 `b > a`
- 如果差值太小，视为 tie，直接跳过

这里的关键是：

- 真实 rollout 仍然要做
- 但它不再是每个 context 主训练内环都要做的 teacher 搜索
- 而是只做**稀疏校准**

## 6. ranker 长什么样

### 6.1 输出

用一个 SAT 联合动作打分器：

- `u_phi(snapshot, joint_sat_action)`

它看起来有点像 `Q`，但训练目标完全不同。

它不需要预测绝对回报值，只需要满足：

- 对同一个 snapshot，
- 好动作分数高于坏动作

### 6.2 为什么这仍然不是绝对 Q

这个 ranker 不要求去拟合：

- 动作真实回报是多少
- long-horizon delta 的精确数值有多大

它只要求满足排序关系：

- `u(s, a_plus) > u(s, a_minus)`

这样就避开了绝对值回归最难的那部分校准问题。

### 6.3 输入

ranker 最好尽量复用当前 SAT actor 已经有的表示：

- `LocalSatState` 里的 snapshot 侧特征
- 每个 UAV 选中的 subset index
- 被选 subset 对应的 subset embedding
- 跨 UAV 的交互特征

这里最关键的是**跨 UAV 交互特征**。  
如果没有它，ranker 很容易退化成独立 per-UAV 打分，处理不了你最关心的多 UAV 协调问题。

建议至少加这些 joint features：

1. 每个 UAV 选中 subset 的 embedding
2. 被选 SAT 的 occupancy count
3. 被选 SAT 的 projected fan-in count
4. 被选 SAT 的 queue / load 汇总
5. UAV 两两之间是否共享 SAT，共享几颗

### 6.4 最小可行结构

`v1` 建议这样做：

1. 复用 SAT actor 的 subset encoder
2. 对每个 UAV，取它被选 subset 的 embedding
3. 拼接：
   - 各 UAV 的 chosen-subset embedding
   - 两两 UAV 之间的交互特征
   - 被选 SAT 的汇总统计
4. 过一个小 MLP
5. 输出一个标量分数

这个 ranker 建议单独放新文件，不要硬塞进 actor head 里。

推荐新文件：

- `sagin_marl/rl/sat_pairwise_ranker.py`

## 7. ranker 的损失函数

建议直接用 pairwise logistic / Bradley-Terry 风格的损失：

- 给定同一个 snapshot 下的 `(a_plus, a_minus)`
- 定义 `d = u(s, a_plus) - u(s, a_minus)`
- 损失写成：`-log sigmoid(d / tau)`

可选再加权：

- 按真实 rollout gap 大小截断加权
- 按 pair 置信度加权

但 `v1` 我建议最简单：

- 不加回归头
- 不加绝对值监督
- 只做 pairwise ranking loss

## 8. ranker 怎么接到现在的 SAT actor

这里有两种接法，更推荐第一种。

### 8.1 推荐：pairwise policy loss

对一个标注好的偏好 pair `(a_plus, a_minus)`：

1. 评估 actor 在 `a_plus` 下每个 UAV 的 `log pi`
2. 把各 UAV 的 `log pi` 加起来，得到 joint log-prob：
   - `L_plus = sum_u log pi_u(a_plus_u | s_u)`
3. 同样算 `a_minus` 的 `L_minus`
4. 用下面的损失更新 actor：
   - `L_actor_pref = -log sigmoid(beta * (L_plus - L_minus))`

这句的含义其实很直白：

- 提高好动作的联合概率
- 压低坏动作的联合概率

为什么它比 hard CE 更适合这里：

- 它直接利用成对偏好信息
- 不需要把每个 snapshot 硬压成一个唯一 teacher 类别
- 和 pairwise 标签来源天然一致

### 8.2 备用方案：ranker 先挑 best，再做 CE

如果想先做一版更简单的，也可以这样：

1. ranker 给 panel 里的动作打分
2. 挑出分数最高的动作
3. 还是用 CE 去学这个 best action

但这会丢掉 pairwise 信息，所以我更建议把它当 fallback，不当主方案。

## 9. 真实 rollout 还需不需要

需要，但作用变了。

它不再是每个 context 主循环都要做的 teacher 搜索，而是变成：

**稀疏校准器**

推荐分工：

1. 大多数 SAT context：
   - 不做显式长 horizon rollout
   - 直接由 ranker 给 panel 打分
   - actor 用 ranker 产生的 pairwise preference 更新
2. 少量 calibration context：
   - 每隔 `K` 个 update，或者每次抽 `N` 个 fresh context
   - 做 matched rollout
   - 生成真实偏好标签
   - 用来训练 / 校准 ranker
   - 监控 ranker 是否漂掉

这才是相比显式长 horizon teacher 的主要算力节省点。

## 10. replay 和 stale label 问题

旧 actor 派生的 teacher 标签会过时，这个你前面已经指出过。

但 pairwise 标签不一样。

如果样本里存的是：

- `sat-stage snapshot`
- 动作 `A`
- 动作 `B`
- 在固定 partner 下、同 snapshot 同 RNG 的真实偏好标签

那么这个标签是环境真值。  
即使 actor 漂了，它也不会像“旧 teacher best-action 标签”那样直接失效。

所以：

- old best-action bank 不安全
- pairwise replay bank 相对安全很多

## 11. 它怎么替代当前 SAT clean 路线

### 当前 SAT clean 路线

当前实现在 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3562) 附近，大致是：

1. entropy 抽样 context
2. 用 actor top-M subset 构造候选 panel
3. 显式 rollout 给 panel 打分
4. 选一个 best action
5. actor 用 hard CE 学这个动作

### 新路线

新路线改成：

1. entropy 抽样 context
2. 用 actor top-M subset 构造候选 panel
3. ranker 快速给 panel 打分
4. 从 panel 里构造 pairwise preference
5. actor 用 pairwise policy loss 更新
6. 少量真实 matched rollout 只负责校准 ranker

也就是说：

- SAT clean 的 replay、snapshot、采样基础设施都保留
- 只是把“teacher 搜索”从主训练内环挪成“稀疏校准”

## 12. 推荐推进顺序

### 阶段 0：保留现有 SAT clean 基础设施

保留：

- snapshot export/load
- subproc worker replay
- SAT context 抽样
- SAT actor top-k helper

这些都已经做了，不需要推倒。

### 阶段 1：先做 pairwise 数据集

先只做：

- 从 calibration context 里导出 pairwise 标签
- 暂时不改 actor 更新

目标是先验证：

- pairwise 标签稳不稳
- tie 比例高不高
- 值得学习的排序样本多不多

### 阶段 2：先训 ranker，不动 actor

新增：

- `sat_pairwise_ranker.py`
- ranker 训练入口
- 指标：
  - pair accuracy
  - panel top-1 hit rate
  - calibration gap separation

这一阶段先别把 ranker 接到 actor 更新上。

### 阶段 3：再接 actor 的 pairwise preference update

先从最稳的一种 pair 开始：

- `best-ranked candidate` vs `executed action`

等这条稳定了，再扩展成：

- `best-ranked` vs `second-best`
- 一个 context 多个 pair

## 13. 主要风险

1. ranker 可能过拟合当前 candidate-panel 分布。
2. 如果 joint interaction 特征太弱，它会退化成独立 per-UAV 打分。
3. 如果 calibration rollout 太稀，ranker 会和环境真值慢慢漂开。
4. decentralized shared SAT actor 在高度对称状态下仍然可能学不稳。

第 4 点也是为什么 `uav_id_norm` 这类轻量破对称特征要继续保留。

## 14. 验收标准

不要只看 ranker loss。

更应该看：

1. calibration pair accuracy
2. panel top-1 hit rate 对真实 rollout 的命中率
3. fixed-partner 下 `weighted_workload_level` 是否提升
4. `processed_ratio_eval` 不能恶化
5. `pre_backlog_steps_eval` 不能恶化

## 15. 最终建议

`sat` 下一步更合理的方向是：

- 不把显式长 horizon rollout teacher 放进主训练内环
- 不把希望押在绝对 `Q / delta critic` 回归上
- 改成：
  - 少量 matched rollout 做偏好标签
  - 一个 SAT 联合动作 pairwise ranker
  - actor 用 pairwise preference loss 更新

这是在下面两条都不理想的路之间，一个更稳妥的中间方案：

- 一条是：显式长 horizon teacher，算力太贵
- 一条是：绝对 long-horizon value regression，风险太高
