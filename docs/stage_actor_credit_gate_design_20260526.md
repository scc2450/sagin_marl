# Stage Actor Credit Gate 设计

## 1. 目标

当前 joint MC-GAE 训练里，每一轮都会经历：

```text
rollout -> critic 训练 -> 计算 advantage -> actor PPO 更新
```

只看 critic 的 explained variance 不够。EV 只能说明：

```text
V(s) 是否拟合了 MC return
```

但 actor 训练真正需要的是：

```text
这次 actor update 是否把高 advantage 的动作概率提高了
```

所以 gate 的目标不是判断最终策略好坏，而是判断每一轮、每个 stage 的 actor update 是否值得执行。

## 2. Gate 总览

本设计只保留三类 gate：

```text
critic gate: EV
advantage gate: advantage 是否退化
policy response gate: actor 更新方向是否和 advantage 一致
```

不单独记录 `resid_std_ratio`。它和 EV 基本重复：

```text
EV = 1 - Var(G_MC - V) / Var(G_MC)
```

## 3. Gate 1：Critic Fit Gate

### 3.1 指标

每个 stage critic 训练后计算：

```text
EV_stage = explained_variance(V_stage(s), G_MC)
```

这里的 `G_MC` 是当前训练流程里用于 critic 的 finite-horizon MC return。

### 3.2 规则

```text
if EV_stage >= 0.96:
    critic 通过

if 0.90 <= EV_stage < 0.96:
    额外训练 critic 3 epochs
    最多 retry 2 次
    重新计算 EV

if EV_stage < 0.90 after retries:
    跳过该 stage 的 actor update
```

### 3.3 成本

这个 gate 当前基本已经存在。额外成本只来自 critic extra epochs。

## 4. Gate 2：Advantage Health Gate

### 4.1 目的

这个 gate 不判断 advantage 是否“正确”，只判断它是否适合拿来做一次 PPO 更新。

它拦截两类明显坏情况：

```text
1. advantage 几乎没有信号
2. advantage 被极少数 outlier 样本支配
```

### 4.2 使用哪个 advantage

设：

```text
A_raw  = actor update 前的原始 advantage
A_norm = PPO 实际使用的 advantage
```

如果开启 advantage normalization，则 `A_norm` 是归一化后的值；否则 `A_norm = A_raw`。

判断“有没有信号”用 `A_raw`。

判断“是否被 outlier 支配”用 `A_norm`，因为 actor 实际吃的是它。

### 4.3 指标

只保留四个指标：

```text
raw_std = std(A_raw)
norm_std = std(A_norm)
ess_frac = (sum |A_norm|)^2 / (N * sum A_norm^2)
top10_share = sum(top10 |A_norm|) / sum |A_norm|
```

含义：

```text
raw_std:
    原始 advantage 是否几乎全一样

norm_std:
    PPO 实际输入是否退化

ess_frac:
    有效样本比例。它的范围大约是 1/N 到 1。

    如果所有样本权重差不多：
        ess_frac 接近 1

    如果只有 m 个样本主导更新：
        ess_frac 大约接近 m / N

    所以 ess_frac=0.10 可以理解为：
        这次更新的有效样本数只相当于 batch 的 10%

top10_share:
    最大的 10 个样本占总 advantage 权重的比例
```

`ess_frac` 使用 `|A_norm|`，是因为 PPO policy gradient 里每条样本对 actor 的一阶影响大致和 advantage 绝对值成正比。

### 4.4 非有限值不是 gate

`NaN/Inf` 不属于 gate 条件。

gate 的语义是：

```text
这批 credit 是否足够可靠，是否值得更新 actor
```

`NaN/Inf` 的语义是：

```text
训练张量已经坏了，继续训练会污染参数或 optimizer state
```

所以处理位置应该在 actor update 热路径里，而不是 gate 里：

```text
A_raw / A_norm 生成后:
    检查 finite

old_logprob / replay_logprob 生成后:
    检查 finite

actor backward 后、optimizer.step() 前:
    检查 grad norm / gradient finite

如果失败:
    RuntimeError
    不执行 optimizer.step()
```

也就是说，`NaN/Inf` 不是：

```text
skip this actor update
```

而是：

```text
停止训练，先修数值错误
```

### 4.5 第一版干预规则

第一版不是“只记录不干预”。上线时就按下面规则干预，同时把所有指标写入 metrics。

```text
if raw_std < 1e-8:
    skip actor update

if norm_std < 1e-6:
    skip actor update

if ess_frac < 0.10:
    skip actor update

if top10_share > 0.50:
    skip actor update

if 0.10 <= ess_frac < 0.20:
    this_update_actor_lr *= 0.5

if 0.30 < top10_share <= 0.50:
    this_update_actor_lr *= 0.5
```

这些阈值不是理论常数，是第一版保守初值。含义如下：

```text
raw_std < 1e-8:
    原始 advantage 基本是常数，几乎没有可区分的 credit。

norm_std < 1e-6:
    PPO 实际输入基本没有尺度，更新只会靠数值噪声。

ess_frac < 0.10:
    有效样本少于 10%，先硬拦，避免极少数样本支配策略。

0.10 <= ess_frac < 0.20:
    不是完全不可用，但风险偏高，所以只降低本轮 lr。

top10_share > 0.50:
    最大 10 个样本占了一半以上权重，明显 outlier 主导。

0.30 < top10_share <= 0.50:
    outlier 影响偏大，降低本轮 lr。
```

### 4.6 阈值如何校准

校准不是在同一次训练过程中自动改阈值。

第一版训练时使用固定阈值：

```text
按固定阈值干预
同时完整记录触发原因和指标
```

阈值校准发生在训练结束后，或多组 seed / K / reward 对照之后。也就是：

```text
run A:
    使用阈值版本 v1
    训练过程中不改阈值
    记录 gate 指标、rollback/skip、rollout/eval 曲线

离线分析:
    判断 v1 是否过严或过松

run B:
    使用阈值版本 v2
```

每个 stage 需要记录：

```text
ess_frac
top10_share
gate_action
policy_response_credit_t
rollout_episode_reward_mean
deterministic_eval_reward, 如果该轮有 eval
```

离线校准规则：

```text
如果某个 gate 高频触发，但触发后的后续 eval / rollout 趋势仍稳定变好:
    说明 gate 过严
    放宽该 stage 的对应阈值

如果某类坏 update 被 Policy Response Gate 回滚，
但 Advantage Health Gate 之前没有提前降 lr 或 skip:
    说明 advantage gate 对这种坏样本不敏感
    收紧对应阈值或新增记录项

如果某个 stage 长期不触发 advantage gate，
但 credit_t 长期接近 0:
    说明不是 advantage 退化，而是 actor 响应弱
    不调整 advantage gate，转向 policy response / lr / 参数化问题
```

阈值版本更新要写进配置或 run metadata：

```text
threshold_old
threshold_new
calibration_reason
```

### 4.7 成本

成本很低。都是已经算出的 advantage 上的 reduction。

`top10_share` 需要一次 `topk(10)`，对当前几千到一万多行 actor samples 可以忽略。

## 5. Gate 3：Policy Response Gate

### 5.1 目的

这是最重要的 gate。它不看 reward，也不做 branch replay，只检查这次 PPO 更新本身是否把 advantage 写进策略。

对于同一批 actor samples：

```text
logp_before = log π_old(a | s)
actor update
logp_after  = log π_new(a | s)
```

定义：

```text
delta_logp_i = logp_after_i - logp_before_i
credit_i = A_norm_i * delta_logp_i
```

如果 `A_norm_i > 0`，希望 `delta_logp_i > 0`。

如果 `A_norm_i < 0`，希望 `delta_logp_i < 0`。

所以：

```text
credit_i > 0
```

表示这条样本的策略变化方向和 advantage 一致。

### 5.2 指标

不要只用全 batch 的样本级显著性。原因是 rollout 不是独立样本集合。

同一个 env 的 250 个时间步来自同一条轨迹。它们共享同一个初始状态、同一段 traffic/random tape、同一批队列历史，所以这些 step 之间高度相关。

如果直接把 `64 env * 250 step` 当成 16000 个独立样本，问题是：

```text
credit_mean 本身不会变
但估计“这个 mean 有多可靠”的标准误会过小
```

可以用一个简化公式理解：

```text
如果每个 env 内有 T 个 step
同一 env 内 step-step 相关系数约为 rho

单个 env 的有效独立样本数约为：

T_eff = T / (1 + (T - 1) * rho)
```

当 `T=250` 时：

```text
rho = 0:
    T_eff = 250

rho = 0.05:
    T_eff ≈ 18.6

rho = 0.20:
    T_eff ≈ 4.9
```

也就是说，只要同一轨迹内部有一点相关性，`16000` 个 step 的有效独立证据就会远小于 `16000`。

按 env 先平均是一种保守估计：

```text
把 64 条 rollout trajectory 当成 64 条近似独立证据
而不是把 16000 个 correlated step 当成 16000 条独立证据
```

这样做不改变：

```text
credit_mean
```

只改变：

```text
credit_t = 这个 mean 到底有多可靠
```

所以先把每个 env 内部平均，再把 env 当成近似独立样本：

```text
credit_env[e] = mean_i_in_env credit_i
credit_mean = mean_e credit_env[e]
credit_std = std_e credit_env[e]
credit_t = credit_mean / (credit_std / sqrt(num_envs) + eps)
```

同时记录：

```text
weighted_sign_agree =
    sum |A_norm_i| * 1[sign(delta_logp_i) == sign(A_norm_i)] / sum |A_norm_i|

mean_abs_delta_logp = mean |delta_logp_i|
```

核心判据是 `credit_t`。

`weighted_sign_agree` 的含义是：

```text
按 advantage 绝对值加权后，有多少权重的样本满足：

    A_norm > 0 时 logprob 上升
    A_norm < 0 时 logprob 下降
```

它只作为辅助诊断，不直接决定 rollback。原因是它只看符号，不看幅度；少量大幅正确更新可能比大量微小符号正确更重要。

### 5.3 第一版干预规则

Policy response gate 发生在 actor update 之后。

第一版就干预，同时记录全部指标。

```text
if credit_t < -2 and credit_mean < 0:
    rollback this stage actor parameters
    rollback this stage optimizer state
    stage_actor_lr *= 0.5

elif -2 <= credit_t < 1:
    keep update
    mark weak_response

elif credit_t >= 1:
    keep update
```

连续弱响应时：

```text
if weak_response_count >= 10 and KL < 0.3 * target_kl:
    mark stage as stalled
```

`stalled` 不等于立刻增大学习率。它表示：

```text
advantage 没被 actor 明确吸收
且 KL 又很小
```

这时要记录为 plateau 证据。是否增大学习率，要看同一 stage 的 KL/clip/credit_t 历史，而不是在这里自动动作。

如果后续决定启用自动 grow，也必须写成独立规则，例如：

```text
if weak_response_count >= 10
and mean_KL < 0.3 * target_KL
and recent rollback_count == 0:
    stage_actor_lr *= grow_factor
```

但这个 grow 规则不是 policy response gate 的一部分。

这里的 `credit_t` 使用类似 t-statistic 的经验阈值：

```text
credit_t < -2:
    env-level 平均方向明显为负，说明这次更新大概率在降低高 advantage 动作概率。

-2 <= credit_t < 1:
    没有明确负向，但正向证据也不强，记为 weak_response。

credit_t >= 1:
    有正向响应，可以认为这轮 actor update 被 advantage 吸收。
```

`weak_response_count` 是每个 stage 独立维护的连续计数：

```text
如果本轮 stage 是 weak_response:
    weak_response_count += 1
否则:
    weak_response_count = 0
```

它不是硬失败，只表示该 stage 连续多轮 actor 更新没有明显响应，用来判断是否进入 plateau。

### 5.4 成本

每个 stage 多两次 actor logprob forward：

```text
1. update 前 logp_before
2. update 后 logp_after
```

没有 backward。

当前 actor forward 相比 critic 训练便宜很多，成本可接受。比 branch replay 低得多。

## 6. 整体流程

每轮每个 stage 按下面顺序：

```text
1. 训练 critic
2. Critic Fit Gate
3. 计算 A_raw 和 A_norm
4. Advantage Health Gate
5. 保存 actor 参数和 optimizer state 的临时副本
6. 计算 logp_before
7. 执行 PPO actor update
8. 计算 logp_after
9. Policy Response Gate
10. 必要时 rollback actor 和 optimizer
```

## 7. 启用方式

gate 上线时就同时做两件事：

```text
1. 按规则干预训练
2. 完整记录触发原因和全部指标
```

每次 actor update 都记录：

```text
EV_stage
raw_std
norm_std
ess_frac
top10_share
credit_mean
credit_t
weighted_sign_agree
mean_abs_delta_logp
gate_action
gate_reason
weak_response_count
stage_actor_lr_before
stage_actor_lr_after
```

其中：

```text
gate_action ∈ {keep, skip, halve_lr, rollback}
```

如果后续发现某个阈值过严或过松，根据这些记录在下一次 run 调整阈值，而不是在同一次训练中边跑边改。

## 8. 实现清单

### 8.1 当前已有

已经补上的硬检查：

```text
old_logprob 复算后:
    必须 finite

actor_advantage 进入 PPO 前:
    必须 finite

grad_norm 在 optimizer.step() 前:
    必须 finite
```

这些检查失败时直接 `RuntimeError`，不会执行 `optimizer.step()`。

### 8.2 还需要实现

还没有完整实现的部分：

```text
1. 记录 A_raw
   现在 actor update 只拿到 A_norm。
   需要把 stage_adv 原始值和 stage_adv_norm 一起传给 actor update。

2. Advantage Health Gate
   在 actor update 前计算 raw_std / norm_std / ess_frac / top10_share。
   根据固定阈值决定 keep / skip / this_update_lr_scale。

3. Actor / optimizer snapshot
   每个 stage update 前保存：
       stage actor 参数
       stage optimizer state_dict
       stage 当前 lr

4. logp_before / logp_after
   update 前用同一批 actor samples 算 logp_before。
   update 后再算 logp_after。

5. Policy Response Gate
   计算 delta_logp、credit_i、env-level credit_t。
   如果触发 rollback：
       恢复 actor 参数
       恢复 optimizer state
       恢复/调整 lr
       同步 native actor binding

6. metrics
   把 gate 指标、动作、原因、rollback/skip 次数写进每轮 metrics。
```

### 8.3 rollback 能否实现

可以实现，但必须同时回滚 actor 和 optimizer。

只回滚 actor 参数是不够的，因为 Adam optimizer 里有：

```text
step
exp_avg
exp_avg_sq
```

如果 actor 参数回去了，但 Adam 动量还保留坏 update 的痕迹，下一轮仍可能沿坏方向继续推。

正确流程：

```text
actor update 前:
    actor_snapshot = 当前 stage actor 参数 clone
    optim_snapshot = 当前 stage optimizer state_dict deepcopy
    lr_before = 当前 stage lr

actor update 后:
    计算 credit_t

if rollback:
    load actor_snapshot
    optimizer.load_state_dict(optim_snapshot)
    设置新的 stage lr
    sync native actor CUDA bindings
```

这里保存的是 stage-specific 参数，不需要保存整个 actor：

```text
accel: accel_policy.*
sat:   sat_subset_policy.*
bw:    bw_policy.*
```

## 9. 训练后如何调整阈值

阈值调整是离线版本更新，不在同一次训练过程中自动改。

也就是说：

```text
run v1:
    使用固定 gate 阈值
    训练中按阈值干预
    完整记录 gate 指标和训练/eval 结果

分析 v1:
    判断阈值是否过严或过松

run v2:
    修改配置里的阈值
    重新训练或继续做对照
```

### 9.1 训练后先做三张表

每个 stage 分开统计：

```text
1. gate 触发表
   skip_count
   halve_lr_count
   rollback_count
   weak_response_count 最大连续长度

2. gate 指标分布
   ess_frac: p10 / p25 / median
   top10_share: median / p75 / p90
   credit_t: p10 / median / p90
   weighted_sign_agree: mean / p10

3. 训练结果分段
   每 50 updates 的 rollout_episode_reward_mean
   每 50 updates 的 mc_return_mean
   每 50 updates 的 deterministic eval, 如果有
   每 50 updates 的 stage-combo eval, 如果有
```

不要只看单个 update。一个 gate 是否合理，要看它触发前后的一段趋势。

### 9.2 判断阈值过严

如果出现：

```text
某 stage 经常 skip / halve_lr / rollback
但该阶段之后 rollout reward 或 deterministic eval 仍稳定上升
```

说明 gate 可能过严。

调整方向：

```text
ess_frac 过严:
    skip 阈值:  0.10 -> 0.05
    halve 阈值: 0.20 -> 0.10 或 0.15

top10_share 过严:
    skip 阈值:  0.50 -> 0.70
    halve 阈值: 0.30 -> 0.50
```

`raw_std < 1e-8` 和 `norm_std < 1e-6` 一般不要先放宽。它们触发时更像是 advantage 计算或 return/value 口径出了硬问题。

### 9.3 判断阈值过松

如果出现：

```text
某 stage 很少触发 gate
但 rollout reward / deterministic eval 明显变差
并且 credit_t 经常接近 0 或为负
```

说明 gate 可能过松，或者 gate 没覆盖真正的问题。

先收紧 Advantage Health Gate：

```text
ess_frac 过松:
    skip 阈值:  0.10 -> 0.15
    halve 阈值: 0.20 -> 0.30

top10_share 过松:
    skip 阈值:  0.50 -> 0.40
    halve 阈值: 0.30 -> 0.25
```

如果 advantage gate 指标都正常，但 `credit_t` 长期弱：

```text
不是 advantage 分布坏
而是 actor 没有吸收 advantage
```

这时优先查：

```text
stage actor lr 是否太低
KL 是否长期过低
policy 参数化是否进入 plateau
old/new logprob 是否正确
```

不要盲目继续收紧 `ess_frac/top10_share`。

### 9.4 Policy Response Gate 怎么调

第一版不建议先动：

```text
credit_t < -2 and credit_mean < 0 -> rollback
```

因为这是比较明确的负向证据。

如果 rollback 很频繁，先问：

```text
为什么 Advantage Health Gate 没有提前 skip 或 halve lr？
```

优先调整顺序：

```text
1. 先调 ess_frac / top10_share 的 skip/halve 阈值
2. 再看 weak_response_count 和 KL
3. 最后才考虑改 credit_t 的 rollback 阈值
```

只有当离线统计显示：

```text
credit_t < -2 的 rollback 后，原本被回滚的 update 实际经常对应后续 eval 上升
```

才考虑把 rollback 阈值从：

```text
-2 -> -3
```

否则不要放宽。

### 9.5 最终判断表

训练后按下面四类归因：

```text
gate 少触发，reward/eval 上升:
    阈值基本可保留

gate 常触发，reward/eval 仍上升:
    阈值过严，下一版放宽

gate 少触发，reward/eval 下降:
    阈值过松，或 gate 没覆盖真正问题

gate 常 rollback，reward/eval 比无 gate 更稳:
    rollback 有用，保留

gate 常 weak_response，KL 很低:
    不是坏 update，而是 actor 响应弱；
    下一步看 lr、参数化、advantage 口径
```

## 10. 这套 gate 不解决什么

它不直接回答：

```text
这个 reward 是否最优
这个 stage 是否应该用 GAE 或 MC residual
这个 policy 是否超过所有规则策略
```

它只回答当前训练过程中最关键的局部问题：

```text
这轮 actor update 是否有可信 advantage
这轮 actor update 是否真的沿 advantage 方向改变了策略
```

所以它适合做训练控制和诊断，不替代最终 deterministic eval。
