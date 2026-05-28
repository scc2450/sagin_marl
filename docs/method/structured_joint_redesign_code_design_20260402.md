# 结构化联合控制重设计：代码设计版

## 1. 文档目的

本文档给出一套可落地到代码层的重设计方案，用于替换当前 `accel`、`bw`、`sat` 并列多头的 factorized MAPPO 结构。

该方案必须同时满足三类要求：

1. **问题对齐**
   设计必须直接针对当前已经确认的系统问题，而不是迁就已有代码开关或历史实现路径。

2. **算法一致性**
   设计仍保持在 centralized critic + decentralized actor + PPO/GAE 的 actor-critic 范式中，不依赖外部启发式监督作为主方案。

3. **速度可接受**
   设计不能通过环境克隆、全量反事实搜索、或指数级结构展开来换取“理论上更干净”的 credit。
   方案必须在现有多环境并行训练框架下具备实际可运行性。


## 2. 设计目标

### 2.1 要解决的问题

当前系统的核心问题不是单一动作头失效，而是以下三层叠加：

1. **动作结构失配**
   当前任务是强耦合、条件化、部分组合化的联合控制问题，但 actor 仍按并列多头建模。

2. **跨头 credit 过粗**
   共享标量 advantage 会把“其他头的成功”错误地奖励给“当前头的局部坏动作”。

3. **`bw` 头内优化病灶**
   当前 `bw` 的参数化与 PPO `log_prob` 几何耦合过强，导致 `valid_count / concentration` 异常。

### 2.2 本文档不试图保留的东西

下列内容不应因为“当前代码里已经有”而默认保留：

- 现有 `env.step()` 内部动作执行顺序
- 现有 `get_global_state()` 的 critic 输入形式
- 现有 `ActorNet` 的并列多头结构
- 现有 `HybridActionDist`
- 现有 `MaskedDirichlet`
- 现有 `set_pool` 式早期 mean/max pooling critic


## 3. 总体思路

### 3.1 从单步联合动作改为阶段化控制过程

每个物理时间步不再被视为一次“同时采样三个头”的动作，而是拆成三个语义明确的决策阶段：

1. `accel`
2. `sat_pair`
3. `bw`

只有三者都完成后，才执行一次真实环境推进。

因此，一个物理时间步对应一个三阶段扩展 MDP：

```text
z_accel --a_accel--> z_sat --a_satpair--> z_bw --a_bw--> z_accel(next)
```

### 3.2 训练框架仍然是 PPO + GAE

保留：

- centralized critic
- decentralized actor
- PPO clipped objective
- value baseline + GAE

改变的是：

- 状态表示
- actor 因子分解
- critic 的阶段输入
- rollout 缓冲区单位


## 4. 环境与控制语义重设计

## 4.1 新的环境控制周期

环境层面需要支持新的控制语义，而不是把旧的 `step()` 顺序当成不可改的事实。

新的单步控制流程定义如下：

1. **阶段 A：运动控制**
   输入 `accel`
   输出 post-motion world state

2. **阶段 B：卫星对选择**
   在 post-motion 状态下，为每个 UAV 选择一个无序卫星对

3. **阶段 C：带宽分配**
   在 post-motion + selected-sat-pair 状态下，为每个 UAV 在其 valid users 上分配 `bw`

4. **阶段 D：物理执行**
   根据完整联合动作执行：
   - access transmission
   - backhaul transmission
   - queue update
   - reward computation
   - terminal/truncation update

### 4.2 为什么顺序是 `accel -> sat_pair -> bw`

这是从任务因果结构出发做的选择：

- `accel` 先改变几何和可行性
- `sat_pair` 决定 relay/backhaul 支撑条件
- `bw` 应在已知 relay context 下做资源分配

这和当前环境中“先 access，再 sat”的执行顺序不同，但更符合 end-to-end 控制语义。

### 4.3 环境 API 重设计

需要新增一个结构化控制接口，而不是复用旧 `step(actions)` 直接塞三头动作。

建议环境侧提供以下 API：

```python
class StructuredSAGINEnv:
    def build_stage_accel_state(self) -> StructuredWorldState
    def apply_stage_accel(self, accel_action) -> StructuredWorldState
    def build_sat_pair_candidates(self, state_after_accel) -> SatPairCandidateBatch
    def apply_stage_sat_pair(self, sat_pair_action) -> StructuredWorldState
    def build_bw_valid_context(self, state_after_sat_pair) -> BwDecisionContext
    def execute_stage_bw_and_step(self, bw_action) -> StepResult
```

这套 API 表示：

- 阶段状态由环境显式构建
- 阶段间的状态转移由环境显式负责
- actor/critic 不再自己猜测阶段状态


## 5. 状态表示设计

## 5.1 基本原则

critic 输入必须是 actor 关键观测信息的**超集**，而不是子集。

同时，状态表示必须保留：

- 节点信息
- 边信息
- 动作前缀诱导的结构变化

不能在早期池化后丢失逐元素信息。

## 5.2 统一的结构化世界状态

引入统一数据结构：

```python
@dataclass
class StructuredWorldState:
    uav_nodes: FloatTensor      # [B, U, Du]
    gu_nodes: FloatTensor       # [B, G, Dg]
    sat_nodes: FloatTensor      # [B, S, Ds]

    uav_gu_edges: FloatTensor   # [B, U, G, D_ug]
    uav_sat_edges: FloatTensor  # [B, U, S, D_us]
    uav_uav_edges: FloatTensor  # [B, U, U, D_uu]

    gu_mask: BoolTensor         # [B, G]
    sat_mask: BoolTensor        # [B, S]
    uav_gu_mask: BoolTensor     # [B, U, G]
    uav_sat_mask: BoolTensor    # [B, U, S]
    uav_uav_mask: BoolTensor    # [B, U, U]

    stage_id: IntTensor         # [B]
```

这里：

- `U` 是 UAV 数
- `G` 是 GU 数
- `S` 是阶段相关 satellite 子集大小

### 5.3 节点与边特征定义

#### UAV nodes

- position
- velocity
- energy
- UAV queue
- 当前时间归一化
- 关联统计
- 阶段前缀动作写回字段

#### GU nodes

- position
- GU queue
- arrival / demand fields
- previous association status

#### SAT nodes

- orbital / motion state
- SAT queue
- load / usage
- remaining or projected service context

#### UAV-GU edges

- relative position
- access spectral efficiency / channel quality
- candidate flag
- valid flag
- continuity-related indicators

#### UAV-SAT edges

- relative position / velocity
- Doppler-related fields
- relay spectral efficiency / projected bandwidth
- satellite queue / load reflected on edge
- visibility / validity
- selected-pair indicators when relevant

#### UAV-UAV edges

- relative position
- relative velocity
- safety distance / danger relation fields


## 6. 速度约束下的状态规模控制

速度约束必须作为一等设计目标处理。

### 6.1 不使用全量反事实环境克隆

训练中不允许采用以下高成本路径作为主方案：

- 每步克隆环境多次
- 对所有可能 `sat` 组合做在线反事实 rollout
- 对所有可能 `bw` 分配做反事实估值

这些只可用于离线诊断，不可作为主训练机制。

### 6.2 actor 与 critic 使用不同粒度的状态

这是速度优化的关键。

#### actor：局部阶段状态

actor 仍然是 decentralized 的，因此每个 UAV 只应看到与自己决策相关的局部子图：

- 本 UAV 节点
- 相关 UAV 邻接子图
- 所有 GU，或 candidate GU 集
- 所有可见 SAT，或当前阶段的 SAT 候选子集

#### critic：中央结构化状态

critic 是 centralized 的，但仍然不必无条件用全量静态大图。
应采用**语义一致的稀疏化**：

- GU：使用全量 GU，因数量本就较小
- UAV：使用全量 UAV
- SAT：使用“当前至少被一架 UAV 可见”或“与任一决策相关”的活动 SAT 子集

这种稀疏化是物理语义驱动的，不是为了迁就旧代码。

### 6.3 动态 SAT 子集而不是全 144 星密集图

对于 critic，原则上应使用与当前时刻决策相关的 SAT 子集，而不是盲目对全部 `num_sat` 建完整图。

定义：

- `active_sat_set = union_u visible_sats(u)`

critic 与 actor 都基于 `active_sat_set` 建图。

理由：

- 不可见卫星对当前步的决策没有直接作用
- 这是物理语义上的裁剪，不是工程偷懒
- 可大幅降低 `U x S` 边计算成本

### 6.4 保留逐元素 token，推迟 pooling

速度优化不能靠提前把集合池化成单向量。

正确策略是：

- 先在 token 级别做 1-2 层轻量关系更新
- 再在 stage-specific readout 时做 late pooling

这样既保留逐元素信息，又控制深层 attention/graph 计算量。


## 7. Actor 代码设计

## 7.1 总体接口

```python
class StructuredActor(nn.Module):
    def act_accel(self, local_state: LocalAccelState) -> AccelPolicyOutput
    def act_sat_pair(self, local_state: LocalSatState) -> SatPairPolicyOutput
    def act_bw(self, local_state: LocalBwState) -> BwPolicyOutput

    def evaluate_accel(...)
    def evaluate_sat_pair(...)
    def evaluate_bw(...)
```

不再提供“一个 `act()` 同时返回三头拼接动作”的主接口。

## 7.2 `AccelPolicy`

输入：

```python
@dataclass
class LocalAccelState:
    ego_uav: FloatTensor
    nbr_uavs: FloatTensor
    all_gu_nodes: FloatTensor
    all_gu_edges: FloatTensor
    visible_sat_nodes: FloatTensor
    visible_sat_edges: FloatTensor
    masks: ...
```

输出：

```python
@dataclass
class AccelPolicyOutput:
    action: FloatTensor        # [B, 2]
    logprob: FloatTensor       # [B]
    entropy: FloatTensor       # [B]
    aux: dict
```

## 7.3 `SatPairPolicy`

输入：

```python
@dataclass
class LocalSatState:
    ego_uav_after_accel: FloatTensor
    visible_sat_nodes: FloatTensor
    visible_sat_edges: FloatTensor
    sat_pair_tokens: FloatTensor
    pair_mask: BoolTensor
```

这里 `sat_pair_tokens` 不是提前池化后的 satellite summary，而是每个合法卫星对的逐对 token：

```text
[pair_count, D_pair]
```

输出：

```python
@dataclass
class SatPairPolicyOutput:
    pair_index: LongTensor
    pair_members: LongTensor   # [B, 2]
    logprob: FloatTensor
    entropy: FloatTensor
    aux: dict
```

## 7.4 `BwPolicy`

输入：

```python
@dataclass
class LocalBwState:
    ego_uav_after_accel: FloatTensor
    selected_sat_pair_context: FloatTensor
    candidate_gu_nodes: FloatTensor
    candidate_gu_edges: FloatTensor
    valid_user_mask: BoolTensor
```

输出：

```python
@dataclass
class BwPolicyOutput:
    alloc: FloatTensor         # [B, K_valid padded]
    logprob: FloatTensor
    entropy: FloatTensor
    aux: dict
```

## 7.5 `BwPolicy` 的分布实现

引入新分布：

```python
class MaskedLogisticNormal:
    def sample(...)
    def mode(...)
    def log_prob(...)
    def entropy(...)
```

实现建议：

1. 仅对 valid users 构造 latent Gaussian
2. 使用 log-ratio / centered-log-ratio 坐标
3. 通过 softmax 类变换映射回 simplex
4. invalid users 直接置零

需要单独实现：

- 动态 valid-set packing/unpacking
- 数值稳定的 Jacobian log-det
- 单用户/空集合边界行为


## 8. Critic 代码设计

## 8.1 总体接口

```python
class StructuredCritic(nn.Module):
    def value_accel(self, world_state: StructuredWorldState) -> FloatTensor
    def value_sat(self, world_state: StructuredWorldState) -> FloatTensor
    def value_bw(self, world_state: StructuredWorldState) -> FloatTensor
```

critic 不再以：

- `global_state: [B, D]`
- `obs_step: [B, N, Dobs]`

这种旧接口作为主设计。

## 8.2 编码器结构

推荐结构：

1. 类型特定输入投影：
   - UAV node encoder
   - GU node encoder
   - SAT node encoder
   - 三类 edge encoder

2. 轻量关系更新层：
   - cross-attention 或 graph message passing
   - 至少保留节点与边 token

3. 阶段特定读出：
   - `readout_accel`
   - `readout_sat`
   - `readout_bw`

### 8.3 读出方式

阶段读出不应该只做单次全局 mean pooling。

建议：

- 先按 UAV 聚合其相关边上下文
- 再在 UAV 级别做团队 value 聚合

这样 `bw` 与 `sat` 的逐元素信息仍可进入最终标量 value。


## 9. 环境 API 与状态构造设计

## 9.1 结构化 step API

新增一个新的环境驱动器，而不是在旧 `step()` 内塞更多条件逻辑。

```python
class StructuredControlDriver:
    def begin_step(env) -> StructuredWorldState
    def run_accel_stage(env, accel_action) -> StructuredWorldState
    def run_sat_stage(env, sat_pair_action) -> StructuredWorldState
    def run_bw_stage_and_commit(env, bw_action) -> StepResult
```

### 9.2 本地状态构造器

为 actor 提供单独的本地状态生成器：

```python
build_local_accel_states(world_state) -> list[LocalAccelState]
build_local_sat_states(world_state) -> list[LocalSatState]
build_local_bw_states(world_state) -> list[LocalBwState]
```

这一步应在环境/driver 侧完成，而不是在 actor 内部反向猜测。

### 9.3 SAT pair token 构造

新增：

```python
build_sat_pair_tokens(visible_sat_nodes, visible_sat_edges) -> (pair_tokens, pair_mask, pair_members)
```

输出每个 UAV 当前所有合法 pair 的 token。

### 9.4 BW valid context 构造

新增：

```python
build_bw_context(world_state_after_sat) -> LocalBwState
```

它必须重新计算：

- valid user set
- candidate user edges
- relay-aware backhaul context summary


## 10. Rollout Buffer 设计

## 10.1 新缓冲区基本原则

旧 `RolloutBuffer` 以“单个物理 step 一条记录”为单位，不再适合。

新缓冲区按“阶段 transition”存储：

```python
class StructuredRolloutBuffer:
    add_stage_transition(...)
    finalize_env_step(...)
    as_stage_batches(...)
```

## 10.2 建议字段

```python
@dataclass
class StageTransition:
    stage_id: int

    world_state: StructuredWorldState
    local_actor_state: object

    action: Tensor
    old_logprob: Tensor
    value: Tensor

    reward: float
    terminated: bool
    truncated: bool

    next_world_state: StructuredWorldState
```

### 10.3 速度考虑

为了控制内存：

- actor 本地状态和 critic 世界状态应拆开存储
- 对大张量使用定长 padded tensor + mask
- 不存环境对象，不存可执行克隆


## 11. 扩展 MDP 下的 GAE 与 PPO

## 11.1 三阶段 transition

每个物理时间步对应三条 transition：

1. `z_accel --a_accel--> z_sat`
2. `z_sat --a_satpair--> z_bw`
3. `z_bw --a_bw--> z_accel(next)`

## 11.2 reward 定义

默认：

- `r_accel = 0`
- `r_sat = 0`
- `r_bw = r_env`

原因：

- 中间两个阶段只是一个物理 control cycle 的内部决策阶段
- 系统目标仍在完整物理推进后定义
- 这样不会偷偷改任务目标

## 11.3 折扣定义

```text
gamma_stage = 1
gamma_env = gamma
```

因此：

- `accel -> sat` 与 `sat -> bw` 用 `gamma_stage`
- `bw -> next accel` 用 `gamma_env`

## 11.4 目标值与优势

对三阶段分别计算：

```text
delta_accel = 0 + 1 * V_sat(z_sat) - V_accel(z_accel)
delta_sat   = 0 + 1 * V_bw(z_bw) - V_sat(z_sat)
delta_bw    = r_env + gamma * V_accel(z_next) - V_bw(z_bw)
```

再按扩展轨迹做 GAE。

## 11.5 PPO 损失

```text
L_actor =
  L_ppo_accel
+ L_ppo_sat
+ L_ppo_bw

L_value =
  L_v_accel
+ L_v_sat
+ L_v_bw
```

这里每个阶段只使用自己的：

- old/new logprob
- stage advantage
- stage entropy


## 12. 速度优化策略

## 12.1 不允许的高成本设计

以下方案不作为主训练设计：

- 全量在线反事实枚举
- 每阶段环境克隆
- 对所有 SAT 组合或所有 simplex 候选做 rollout 搜索
- 在 PPO epoch 内重复构造大图状态

## 12.2 必须采用的优化

### A. 阶段状态缓存

每个物理 step 内：

- `z_accel`
- `z_sat`
- `z_bw`

只构造一次并缓存，训练阶段直接读取缓冲区。

### B. 活动 SAT 子集裁剪

critic 与 actor 均只对当前活动 SAT 子集建图，而不是固定全量 `num_sat`。

### C. 逐阶段轻量编码器

每个阶段的 actor 不共享“错误的并列 trunk”，
但可以共享部分低层类型编码器权重，例如：

- UAV node stem
- GU node stem
- SAT node stem

高层决策模块与 readout 则分阶段独立。

这样既避免旧 shared-head 污染，又控制参数量和计算量。

### D. late pooling

只在最终 value readout 做聚合，不在输入端过早池化。

### E. 向量化 stage builder

环境阶段转换应尽量用批量 NumPy / Torch 运算完成，不在 Python 循环中频繁重建对象。


## 13. 文件与模块重构建议

### 13.1 新增模块

- `sagin_marl/rl/structured_types.py`
  - dataclass 定义

- `sagin_marl/rl/structured_actor.py`
  - `StructuredActor`
  - `AccelPolicy`
  - `SatPairPolicy`
  - `BwPolicy`

- `sagin_marl/rl/structured_critic.py`
  - `StructuredCritic`

- `sagin_marl/rl/structured_buffer.py`
  - `StructuredRolloutBuffer`

- `sagin_marl/env/structured_driver.py`
  - 阶段化控制驱动器

- `sagin_marl/rl/distributions_logistic_normal.py`
  - `MaskedLogisticNormal`

### 13.2 旧模块处理

- `policy.py`
  - 保留为 legacy 或逐步退役

- `critic.py`
  - 保留为 legacy baseline，不作为新核心

- `buffer.py`
  - 不再作为新训练主缓冲区

- `mappo.py`
  - 新增 `structured_mappo.py`
  - 不建议在旧文件里继续堆条件分支

### 13.3 环境层处理

- `sagin_env.py`
  - 保留底层物理/队列更新函数
  - 但新增结构化控制驱动接口
  - 不建议继续把所有阶段逻辑都塞在旧 `step()` 中


## 14. 第一阶段实现顺序

推荐按以下顺序实施：

1. 新建结构化类型与阶段化环境状态构造器
2. 实现 `SatPairPolicy` 与 pair token builder
3. 实现 `MaskedLogisticNormal`
4. 实现 `StructuredCritic`
5. 实现 `StructuredRolloutBuffer`
6. 实现新的 `structured_mappo.py`
7. 用 smoke config 先跑三阶段训练闭环


## 15. 验证标准

验证不应只看 reward。

### 15.1 `sat` 机制验证

- pair 排序与一步/多步 relay quality 正相关
- 局部坏 relay pair 不再系统性拿到正 advantage
- 之前 `u0100` 的 backhaul 崩坏明显缓解

### 15.2 `bw` 机制验证

- `|log_ratio_bw|` 不再随 `valid_count` 病态放大
- concentration 主导异常不再出现
- 执行表现不再严重落后于 heuristic `bw`

### 15.3 系统验证

- learned `accel` 不再轻易把系统推入同样坏 regime
- 结构化方案优于旧 factorized baseline
- 训练时间相对旧系统不应出现不可接受级别的增长


## 16. 结论

当前系统不适合继续围绕旧并列多头 MAPPO 做局部补丁。

更合理的代码层方案是：

1. 重写环境控制周期，
2. 用结构化 actor 取代并列多头 actor，
3. 用关系型阶段 critic 取代伪 global critic，
4. 用三阶段扩展 MDP 取代单步 joint action rollout，
5. 用 masked logistic-normal 取代当前 `bw` 的主训练分布。

这套方案既直接对应当前问题分析，也把执行速度作为设计约束纳入了主方案，而不是事后再补优化。