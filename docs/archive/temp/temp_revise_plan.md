请你审核以下修改方案，按需调整后执行，注意，它给出的代码仅为示意，命名之类的都和现有代码不一致，有些变量现有代码已经定义过了，按现有代码来。
---------------------------------------------------
修改方案

### 1. 奖励与评估口径

#### 1.1 固定归一化分母

统一定义：

```text
A_ref = 所有 GU 的理论平均总到达比特数/步
```

如果每个 GU 的泊松到达均值是已知的，就直接求和。
这样所有比率的单位都统一成“相对于平均外生到达量”。

#### 1.2 训练奖励

在每个环境 step 结束后计算：

```text
x_acc  = gu_outflow_sum / (A_ref + eps)
x_rel  = uav_to_sat_inflow_sum / (A_ref + eps)

B_pre_t   = (Q_gu_sum + Q_uav_sum)_t
B_pre_tp1 = (Q_gu_sum + Q_uav_sum)_{t+1}
g_pre = (B_pre_tp1 - B_pre_t) / (A_ref + eps)

d_pre = (gu_drop + uav_drop) / (A_ref + eps)
```

训练奖励为：

```text
r_train
= 0.5 * x_acc
+ 0.5 * x_rel
- 1.0 * d_pre
- 0.2 * relu(g_pre)
```

先不要再分 stage1 / stage2 / stage3 改主目标。
三阶段都用同一个 `r_train`，区别只在于**逐步解锁动作头**。

#### 1.3 评估主指标

每次 checkpoint eval 固定输出：

```text
processed_ratio_eval   = sat_processed_bits / (A_ref + eps)
drop_ratio_eval        = (gu_drop + uav_drop + sat_drop) / (A_ref + eps)
pre_backlog_steps_eval = (Q_gu_sum + Q_uav_sum) / (A_ref + eps)
```

另外单独报告：

```text
D_sys_report = (Q_gu_sum + Q_uav_sum + Q_sat_sum) / max(sat_processed_bits, eps)
```

选最优 checkpoint 的顺序：

1. `processed_ratio_eval` 高
2. `drop_ratio_eval` 低
3. `pre_backlog_steps_eval` 低

---

### 3. `bw` 动作：改成 Dirichlet policy

#### 3.1 为什么改

你的带宽分配本质上是：

* 每个权重非负
* 权重和为 1
* 只在有效关联用户子集上分配

这是标准 simplex 动作。Dirichlet policy 天然定义在 simplex 上，而相关工作指出 Gaussian 不天然适配这类约束分配，Gaussian-softmax 还会带来训练偏差；Dirichlet 更适合 allocation task。

#### 3.2 怎么改

actor 不再输出高斯 `bw_logits`，而是对每个候选用户输出一个标量分数，再变成 Dirichlet concentration：

```text
score_u[k] = BWHead(...)
alpha_u[k] = softplus(score_u[k]) + alpha_floor
```

建议：

```text
alpha_floor = 0.2 或 0.5
```

环境先根据关联结果得到有效用户子集 `valid_users`，然后：

* 若 `n_valid = 0`：带宽全 0，`log_prob = 0`
* 若 `n_valid = 1`：唯一用户权重为 1，`log_prob = 0`
* 若 `n_valid >= 2`：在 `alpha_valid` 上构造 `Dirichlet(alpha_valid)`

训练时 sample，评估时取均值：

```text
beta_valid = alpha_valid / alpha_valid.sum()
```

#### 3.3 一个必须同步改的点

把 heuristic residual 那套执行逻辑关掉。
也就是不要再“policy 输出 residual，再和启发式融合”。
policy 直接输出最终带宽份额，否则优化的分布和实际执行动作还是不一致。这个问题在你文件里已经存在。

---

### 4. `sat` 动作：改成策略内 masked categorical

#### 4.1 为什么改

你的卫星选择本质上是：

* 只在当前可见且有效的候选卫星里选
* 最多选 `N_RF = 2`
* 无放回

这就是离散选择问题，不是连续控制问题。你文件里现在是连续 `sat_logits` 再由环境 sample，这会让 policy 的概率模型和真实执行语义脱节。

#### 4.2 怎么改

policy 直接输出 6 个候选槽位的 logits。
对无效候选做 mask 后，顺序采样两次：

第一次：

```text
a1 ~ Categorical(masked_logits)
```

第二次：

* 把 `a1` mask 掉
* 再从剩余有效候选里采 `a2`

总 logprob：

```text
logprob_sat = log p(a1) + log p(a2)
```

---

### 5. actor 网络结构：改成“共享上下文 + 逐候选打分”

你文件里当前 actor 的关键问题是：

* `users` / `sats` 先被池化成集合摘要
* 然后直接从全局 summary 线性输出 20 个 `bw` 槽位和 6 个 `sat` 槽位

这对 `accel` 还行，但对 `bw/sat` 这种“要给每个候选对象分别打分”的动作不理想。Deep Sets 适合做 permutation-invariant set summary；而你这里 `bw/sat` 更需要元素级打分。 

#### 5.1 新结构

```text
ctx = ContextEncoder(
    own_obs,
    danger_nbr,
    pooled_nbrs,
    pooled_users,
    pooled_sats
)

e_user[k] = UserEncoder(user_k)
e_sat[m]  = SatEncoder(sat_m)

accel = AccelHead(ctx)

score_u[k] = BWHead([ctx, e_user[k], raw_user_k])
logit_s[m] = SatHead([ctx, e_sat[m], raw_sat_m])
```

#### 5.2 各头建议

* `accel`：继续高斯连续头
* `bw`：逐用户 scorer，输出 Dirichlet `alpha`
* `sat`：逐卫星 scorer，输出 categorical logits
* `critic`：第一轮先不改，继续全局 state MLP

---

### 6. 训练与日志修改

新增并固定记录下面这些量：

```text
A_ref
x_acc
x_rel
g_pre
d_pre

processed_ratio_eval
drop_ratio_eval
pre_backlog_steps_eval
D_sys_report
```

训练曲线重点看：

* `x_acc`
* `x_rel`
* `g_pre`
* `d_pre`

评估表重点看：

* `processed_ratio_eval`
* `drop_ratio_eval`
* `pre_backlog_steps_eval`

这样你能直接判断：

* 是接入没学会；
* 还是回传没学会；
* 还是前端越来越堵；
* 还是系统最终处理能力没改善。

---

---------------------------------------------------
下面是代码级修改清单，同样注意，它给出的代码仅为示意，命名之类的都和现有代码不一致，有些变量现有代码已经定义过了。
-------------------------------------------------
你文件里当前的关键现状是：

* actor 动作顺序是 `accel(2) + bw(20) + sat(6)`；stage1/2/3 动作维度分别是 `2 / 22 / 28`。
* `bw` 现在是 **Gaussian residual logit**，再和 heuristic 融合后送进环境；PPO 优化的是 residual 的 `logprob`，不是最终执行的 `bw_exec`。
* `sat` 现在也是连续 `sat_logits`，然后由环境按 sample 模式从 6 个槽位里抽样选 2 颗星。
* reward 现在三阶段主目标不一致，而且所有 UAV 共享同一个全局 reward。

我要你改成的最终状态是：

* **训练奖励统一**成“前两段可控流量 + 前两段拥塞约束”
* **评估指标单独记录**系统最终能力
* `bw` 改成 **Dirichlet**
* `sat` 改成 **策略内 masked categorical**
* actor 改成 **共享上下文 + 逐候选打分**
* env 不再对 `bw` 做 softmax，不再自己对 `sat_logits` 抽样
* PPO 里改成 **hybrid action**：连续 accel + simplex bw + 离散 sat

---

# 一、建议的改动顺序

按这个顺序做，最不容易一次改崩：

1. 先改 **reward / metrics**，不改动作分布
2. 再给 observation 增加 **`bw_valid_mask` / `sat_valid_mask`**（我注：这个不是本来就有吗？）
3. 再改 **actor 结构**，先做逐候选 scorer，但先不动 sat
4. 再把 **stage2 的 bw 改成 Dirichlet**
5. 最后把 **stage3 的 sat 改成 masked categorical**
6. 最后再整理 **trainer / evaluate / config**

不要一上来同时改 6 处。

---

# 二、环境与配置：先把“数据契约”改清楚

## 1. `config.py` 里新增/替换配置项（我注：下面代码中的变量好多都定义过了，只是和它命名不同，还是按现有的来）

在环境配置和训练配置里新增这些字段：

```python
# reward
reward_mode = "controllable_flow"
reward_w_access = 0.5
reward_w_relay = 0.5
reward_w_pre_drop = 1.0
reward_w_pre_growth = 0.2

# normalization
arrival_ref_mode = "expected_arrival"   # 用理论平均总到达量
use_queue_max_norm = False

# policy distribution
bw_policy = "dirichlet"
bw_alpha_floor = 0.2
sat_policy = "masked_categorical"
sat_num_select = 2

# observation
append_action_masks_to_obs = True

# stage switches
train_accel = True
train_bw = False / True
train_sat = False / True
```

同时把这些旧配置标记为废弃或忽略：

```python
exec_bw_source
bw_residual_alpha
bw_residual_clip
bw_logit_scale
bw_log_std_init
bw_log_std_trainable

sat_logit_scale
sat_log_std
sat_select_mode
```

因为它们对应的是你现在的 residual bw 和连续 sat 方案。

---

## 2. 在 `sagin_env.py` 初始化时固定 `A_ref`

在 env 初始化时，根据你已知的泊松到达率，直接算出：

```python
self.arrival_ref_bits_per_step = float(sum(self.gu_arrival_rate_bits_per_step))
```

如果每个 GU 有不同到达率，就求和。
如果到达率单位还要乘时隙长度 `tau0`，就在这里统一乘好。

**不要**再用 `queue_max_*` 做主 reward 归一化。
之后所有主指标都除以这个 `arrival_ref_bits_per_step`。

---

## 3. observation 里新增两个 mask

你后面要做 Dirichlet 和 masked categorical，**policy 必须在采样时知道哪些动作是有效的**。

所以在 actor obs 末尾追加：

```python
bw_valid_mask : shape = (users_obs_max,)   # 20
sat_valid_mask: shape = (sats_obs_max,)    # 6
```

建议放到 observation flat vector 的最后面，这样最少改 wrapper / vec env。

### `bw_valid_mask` 怎么算

对当前 UAV 的 20 个 candidate user 槽位，标 1 表示：

* 槽位里有真实 GU
* 当前满足可关联条件
* 允许参与带宽分配

### `sat_valid_mask` 怎么算

对 6 个 satellite 槽位，标 1 表示：

* 槽位里有真实卫星
* 当前可见
* 若启用多普勒过滤，则多普勒也合法

你文件里现在环境本来就在内部做这些有效性判断，只是没有显式暴露给策略。

### 一定要同步更新

* observation dim
* obs slice index
* global state dim（如果 global state 用到了 obs 统计，也同步检查）

---

# 三、reward 和 metrics：先改这个

## 4. 在 `env.step()` 前后记录前两层 backlog

在执行动作之前记录：

```python
b_pre_t = q_gu_sum + q_uav_sum
```

执行一步环境转移后记录：

```python
b_pre_tp1 = q_gu_sum_next + q_uav_sum_next
```

这里都用**全局求和**，保持和你当前 shared reward 口径一致。

---

## 5. 在 env 里新增这几个一步指标

每步都算：

```python
A_ref = self.arrival_ref_bits_per_step + 1e-9（我注：这个还用每步都算吗？）

x_acc = gu_outflow_sum / A_ref
x_rel = uav_to_sat_inflow_sum / A_ref

g_pre = (b_pre_tp1 - b_pre_t) / A_ref
d_pre = (gu_drop_bits + uav_drop_bits) / A_ref
```

其中：

* `gu_outflow_sum`：这一步所有 GU 实际发给 UAV 的比特量
* `uav_to_sat_inflow_sum`：这一步所有 UAV 实际发给 SAT 的比特量
* `gu_drop_bits` / `uav_drop_bits`：这一步对应层的丢弃比特量

这些量都应该能从你现有 env 统计链路里拿到；拿不到就把中间变量补存一下。

---

## 6. 用统一训练奖励替换三阶段 reward

把 `_compute_reward()` 改成只算一个共享标量：

```python
reward = (
    cfg.reward_w_access * x_acc
    + cfg.reward_w_relay * x_rel
    - cfg.reward_w_pre_drop * d_pre
    - cfg.reward_w_pre_growth * max(g_pre, 0.0)
)
```

建议默认值：

```python
reward_w_access = 0.5
reward_w_relay = 0.5
reward_w_pre_drop = 1.0
reward_w_pre_growth = 0.2
```

然后继续：

```python
rewards = {agent: reward for agent in self.agents}
```

不要一开始就上 local shaping。你当前更大的问题不是这个。

---

## 7. 新增评估指标，不放进训练主 reward

在 env metrics / evaluate 汇总里新增并固定输出：

```python
processed_ratio_eval = sat_processed_bits / A_ref
drop_ratio_eval = (gu_drop_bits + uav_drop_bits + sat_drop_bits) / A_ref
pre_backlog_steps_eval = (q_gu_sum + q_uav_sum) / A_ref

D_sys_report = (q_gu_sum + q_uav_sum + q_sat_sum) / max(sat_processed_bits, 1e-9)
```

其中：

* `processed_ratio_eval`：系统最终能力
* `drop_ratio_eval`：整体可靠性
* `pre_backlog_steps_eval`：UAV 相关前两层拥塞
* `D_sys_report`：只做报告，不做主训练奖励

checkpoint 选择顺序建议改成：

1. `processed_ratio_eval` 高
2. `drop_ratio_eval` 低
3. `pre_backlog_steps_eval` 低

---

# 四、policy：actor 改成“共享上下文 + 逐候选打分”

## 8. 不要再让 `bw/sat` 直接从 pooled summary 线性出 20/6 槽位

你文件里的当前结构是 pooled set summary 后直接 `Linear(256 -> 20)` 和 `Linear(256 -> 6)` 出槽位，这正是要改的点。

建议把 actor 改成下面结构。

---

## 9. 保留共享上下文 trunk

先保留你现有的：

* own encoder
* nbr encoder + pooling
* user encoder + pooling
* sat encoder + pooling
* shared backbone / trunk

得到：

```python
ctx: [B, n_agent, C]
```

这个 `ctx` 继续给 accel head 用，也给 bw/sat scorer 用。

---

## 10. 新增逐用户 scorer

新增模块：

```python
self.bw_user_encoder = MLP(user_feat_dim, embed_dim)
self.bw_scorer = MLP(ctx_dim + embed_dim + user_feat_dim, 128, 1)
```

前向时：

```python
user_raw = ...                  # [B, A, 20, F_u]
user_emb = self.bw_user_encoder(user_raw)      # [B, A, 20, E]

ctx_u = ctx.unsqueeze(-2).expand(-1, -1, 20, -1)
bw_in = torch.cat([ctx_u, user_emb, user_raw], dim=-1)

bw_score = self.bw_scorer(bw_in).squeeze(-1)   # [B, A, 20]
bw_alpha = F.softplus(bw_score) + cfg.bw_alpha_floor
```

这里的 `bw_alpha` 不是最终动作，而是 Dirichlet 参数。

### 速度建议

不要写三层 Python for-loop。
用 `[B, A, 20, *]` 一次性向量化。

---

## 11. 新增逐卫星 scorer

新增模块：

```python
self.sat_encoder = MLP(sat_feat_dim, embed_dim)
self.sat_scorer = MLP(ctx_dim + embed_dim + sat_feat_dim, 128, 1)
```

前向时：

```python
sat_raw = ...                   # [B, A, 6, F_s]
sat_emb = self.sat_encoder(sat_raw)            # [B, A, 6, E]

ctx_s = ctx.unsqueeze(-2).expand(-1, -1, 6, -1)
sat_in = torch.cat([ctx_s, sat_emb, sat_raw], dim=-1)

sat_logits = self.sat_scorer(sat_in).squeeze(-1)   # [B, A, 6]
```

---

## 12. accel 头先保持原样

`accel` 先不改语义：

* 继续 `mu/log_std`
* 继续 tanh squash
* 继续输出 2 维归一化加速度

你文件里 accel 这一段本身没有最优先要修的问题。

---

# 五、动作分布：新增 hybrid distribution

## 13. 新建 `distributions.py`（或同类模块）

新增 3 个分布包装器：

### 13.1 `MaskedDirichlet`

输入：

```python
alpha: [B, A, 20]
mask : [B, A, 20]  # 0/1
```

需要实现：

* `sample()`
* `mode()`：返回均值 `alpha / alpha.sum()`
* `log_prob(action_bw)`
* `entropy()`

### 特殊情况要单独处理

* `n_valid == 0`：返回全 0，`log_prob = 0`
* `n_valid == 1`：唯一位置为 1，`log_prob = 0`

### `sample()` 返回

返回最终 `bw_alloc`：

```python
bw_alloc: [B, A, 20]
```

要求：

* invalid 槽位恒为 0
* valid 槽位和为 1

---

### 13.2 `MaskedSequentialCategorical`

输入：

```python
logits: [B, A, 6]
mask  : [B, A, 6]
k = 2
```

需要实现：

* `sample()`：顺序无放回采样 2 次
* `mode()`：取 valid 槽位 top-k
* `log_prob(idx_pair)`
* `entropy()`

`sample()` 返回：

```python
sat_indices: [B, A, 2]     # 无效位置填 -1
sat_select_mask: [B, A, 6] # 多热向量，最多 2 个 1
```

### 特殊情况

* `n_valid == 0`：`[-1, -1]`
* `n_valid == 1`：`[idx, -1]`

---

### 13.3 `HybridActionDist`

把三部分组合起来：

* `accel_dist`
* `bw_dist`
* `sat_dist`

提供统一接口：

```python
sample()
mode()
log_prob(actions, sat_indices=None)
entropy()
```

总 logprob：

```python
logprob_total = logprob_accel + logprob_bw + logprob_sat
```

总 entropy 也这样合起来。

---

# 六、action 语义：不要再让 env 自己 softmax / sample

## 14. `action_assembler.py` 改成只做“打包”，不再做 heuristic residual

把现有这类逻辑删掉或旁路掉：

* `bw_exec = heuristic + residual`
* `clip(..., [-5, 5])`
* env 内对 `bw_logits` 再 masked softmax
* env 内对 `sat_logits` 再 sample

因为这会继续造成“policy 优化的动作”和“env 真执行的动作”不一致。你文件里这个问题现在就存在。

---

## 15. 新的 env action 语义

建议保持 env 入口还是**扁平向量**，这样少改 vec env。

### stage1

```python
env_action = concat([accel])         # 2
```

### stage2

```python
env_action = concat([accel, bw_alloc])   # 2 + 20 = 22
```

### stage3

```python
env_action = concat([accel, bw_alloc, sat_select_mask])  # 2 + 20 + 6 = 28
```

这样 stage 维度和你当前一样，但语义变了。

---

## 16. env 里改动作解析逻辑

在 `sagin_env.py` 解析 action 时：

### `bw`

读取的是**最终分配结果**，不是 logit：

```python
bw_alloc = action[..., 2:22]
beta = bw_alloc * bw_valid_mask
s = beta.sum()
if s > eps:
    beta = beta / s
elif valid_count > 0:
    beta = bw_valid_mask / valid_count
else:
    beta = 0
```

也就是说 env 只做安全兜底，不再做 softmax。

### `sat`

读取的是 `sat_select_mask`：

```python
sat_sel = action[..., 22:28] > 0.5
sat_sel = sat_sel & sat_valid_mask
```

如果超过 `N_RF=2`，只保留前 2 个；
如果一个都没选到，则 fallback 到“最近合法卫星”或“top-1 valid”。

---

# 七、PPO / rollout：改成存 hybrid action

## 17. rollout buffer 新增 `sat_indices`

因为 env_action 里只有 `sat_select_mask`，但 PPO 重新算 `log_prob` 时需要知道**顺序采样得到的两个索引**。

所以在 rollout buffer 里新增：

```python
sat_indices: int64, shape [T, N_env, N_agent, 2]
```

stage1/2 可以填 `-1`。

---

## 18. collect rollout 时 policy 返回结构化结果

在 `mappo.py` 采样动作时，让 policy 返回：

```python
{
    "env_action": flat_action,
    "accel_action": accel_action,
    "bw_action": bw_alloc,
    "sat_indices": sat_indices,
    "sat_select_mask": sat_select_mask,
    "logprob": logprob_total,
    "entropy": entropy_total,
}
```

然后：

* `env_action` 送给环境
* `sat_indices` 单独存进 buffer
* `logprob` 继续按原 PPO 流程存

---

## 19. PPO update 时用 `evaluate_actions()`

在 `policy.evaluate_actions(obs, env_action, sat_indices)` 中：

* 重新从 obs 解析 `bw_valid_mask` / `sat_valid_mask`
* 重新前向得到：

  * accel dist 参数
  * bw `alpha`
  * sat logits
* 用 buffer 里的：

  * accel action
  * bw alloc
  * sat indices
    重新计算 `log_prob`

### 具体对应关系

* `accel_action`：从 `env_action[:2]`
* `bw_action`：从 `env_action[2:22]`
* `sat_indices`：从 buffer 单独字段，不从 `env_action[22:28]` 反推

---

## 20. 删掉 `bw_log_std` / `sat_log_std` 相关训练逻辑

你文件里现在 `bw` 和 `sat` 都还有各自的 Gaussian `log_std` 参数。

改完后：

* `bw` 不再需要 `bw_log_std`
* `sat` 不再需要 `sat_log_std`

所以要删掉：

* 参数注册
* clamp / exp
* optimizer 参数组里的对应项
* entropy / logprob 相关 Gaussian 分支

---

## 21. entropy 建议拆开记

在 PPO 日志里新增：

```python
entropy_accel
entropy_bw
entropy_sat
```

总 entropy bonus 先仍然合成一个：

```python
entropy_total = entropy_accel + entropy_bw + entropy_sat
```

但日志一定分开记，不然以后你没法判断到底是哪一头在乱探索。

---

# 八、warm start：三阶段衔接要改

## 22. stage 间加载权重要用 `strict=False`

因为三个阶段 head 不同：

* stage1 只有 accel
* stage2 多了 bw
* stage3 多了 sat

所以在加载 `actor.pt` 时，不要要求全量严格匹配：

```python
state = torch.load(...)
actor.load_state_dict(state, strict=False)
```

### 建议初始化规则

* stage2：继承 stage1 的 trunk + accel head，`bw_head` 新初始化
* stage3：继承 stage2 的 trunk + accel + bw，`sat_head` 新初始化

如果你愿意更稳一点，stage3 开始时可以先把新 sat head 的学习率设小一点。

---

# 九、`evaluate.py`：推理时不要 sample

## 23. 确定性评估模式改掉

现在评估时：

* `accel`：用均值动作
* `bw`：用 Dirichlet 的均值
* `sat`：用 valid 槽位 top-k logits

也就是：

```python
bw_eval = alpha / alpha.sum()
sat_eval = topk(masked_logits, k=2)
```

不要在 eval 中继续 sample，不然 checkpoint 排名噪声太大。

---

## 24. 输出新评估表头

在 `evaluate.py` / csv 汇总里固定加上：

```python
processed_ratio_eval
drop_ratio_eval
pre_backlog_steps_eval
D_sys_report

x_acc_mean
x_rel_mean
g_pre_mean
d_pre_mean
```

这样你能直接看到“训练改好了没有”和“系统最终变好了没有”是不是一致。

---

# 十、YAML：三阶段应该怎么写

## 25. stage1 YAML

```yaml
train_accel: true
train_bw: false
train_sat: false

reward_mode: controllable_flow
reward_w_access: 0.5
reward_w_relay: 0.5
reward_w_pre_drop: 1.0
reward_w_pre_growth: 0.2

bw_policy: dirichlet
sat_policy: masked_categorical
append_action_masks_to_obs: true
```

虽然 stage1 不训练 bw/sat，但 reward 也先统一。

---

## 26. stage2 YAML

```yaml
train_accel: true
train_bw: true
train_sat: false

bw_policy: dirichlet
bw_alpha_floor: 0.2

init_actor: <stage1 actor>
init_critic: <stage1 critic>
```

并确保旧这些项关掉：

```yaml
exec_bw_source: none
bw_residual_alpha: 0.0
bw_residual_clip: 0.0
```

或者直接删掉。

---

## 27. stage3 YAML

```yaml
train_accel: true
train_bw: true
train_sat: true

bw_policy: dirichlet
sat_policy: masked_categorical
sat_num_select: 2

init_actor: <stage2 actor>
init_critic: <stage2 critic>
```

并把：

```yaml
fixed_satellite_strategy: false
```

保留为 false。

---

# 十一、每改完一步要做的最小自测

## 28. reward 改完后先测这个

跑 1～3 个 update，只看日志是否合理：

* `x_acc` 有波动，不是常数
* `x_rel` 有波动，不是常数
* `g_pre` 正负都会出现
* `d_pre` 大多数时候接近 0，但拥塞场景会升高

如果这里全是常数，后面不用继续改。

---

## 29. mask 加完后检查这个

随机拿一个 batch 打印：

* `bw_valid_mask.sum(dim=-1)` 不全是 0，也不总是 20
* `sat_valid_mask.sum(dim=-1)` 和可见星数量一致
* 多普勒过滤开启时，非法星槽位确实是 0

---

## 30. Dirichlet 加完后检查这个

对 stage2：

* `bw_alloc[invalid] == 0`
* `bw_alloc[valid].sum() == 1`
* `log_prob` 有限，不是 `nan/inf`
* eval 模式下 `bw_alloc` 平滑，不全是 one-hot

---

## 31. sat categorical 加完后检查这个

对 stage3：

* `sat_indices` 都在合法范围内或是 `-1`
* `sat_select_mask.sum() <= 2`
* 选中的一定是 valid sat
* `log_prob_sat` 有限

---

## 32. PPO update 后检查这个

重点看：

* `ratio` 不爆
* `approx_kl` 不异常飙升
* `entropy_bw` / `entropy_sat` 不是 0，也不是极大常数
* `policy_loss` 可回传，没有 shape mismatch

---

# 十二、最关键的两句提醒

第一，**这次重构的关键不是“把连续动作换成两个新分布”本身，而是让“policy 优化的动作”与“env 真执行的动作”重新一致。**
你文件里当前 `bw residual -> heuristic fusion` 和 `sat logits -> env 内抽样` 的问题，正是这一点。

第二，**`bw_valid_mask` 和 `sat_valid_mask` 一定要在采样时就让 policy 看见。**
不然你又会回到“policy 采一个动作，env 再替你改成另一个动作”的老问题。

--------------------------------------------------

总之，以上的修改方案只能看大意，涉及到代码变量等很多细节与现有代码不完全一致，请你仔细审核检查，按需修改后执行。