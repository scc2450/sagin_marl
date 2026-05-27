# Stage Actor 重要性采样设计

## 1. 目的

当前 joint MC-GAE 训练里，每个 stage 的 actor update 默认对该 stage 的所有 row 做均匀平均：

```text
L = mean_i PPO_loss_i
```

这个做法稳定，但可能存在一个问题：

```text
大量 row 的 advantage 很小，或者 logprob 对 policy 输出不敏感，
它们会稀释真正有 actor 梯度信号的 row。
```

因此这里考虑一种不改变 PPO 目标口径的采样方式：

```text
每个 actor epoch 只抽一部分 row；
抽样概率偏向更可能产生有效 policy gradient 的 row；
loss 乘重要性权重，尽量校正回原来的 uniform-over-rows PPO objective。
```

这个方案用于提高 actor update 的样本效率，不改变 reward、critic target、GAE 计算方式。

## 2. 原目标和采样目标

原始 actor loss 对 stage 内所有 row 均匀平均：

```text
L_uniform = (1 / N) * sum_i loss_i
```

如果改成按分布 `q_i` 抽样，抽到 `M` 个 row，则无偏重要性采样形式是：

```text
L_IS = (1 / M) * sum_{i sampled from q} w_i * loss_i
w_i = 1 / (N * q_i)
```

实际训练中为了降低极端权重带来的方差，会使用归一化和截断：

```text
w_i = w_i / mean(w_i in sampled rows)
w_i = clamp(w_i, w_min, w_max)
```

这会引入轻微偏差，但通常比完全不校正更安全。

## 3. Row priority

每个 stage 独立计算 priority：

```text
priority_i = |A_i| * score_norm_i
```

其中：

```text
A_i:
  当前 actor update 使用的 advantage。

score_norm_i:
  policy 输出层对该 row 的 logprob score norm。
```

这里的 score norm 不是完整 actor 参数梯度：

```text
不是 || ∇_θ log π(a_i | s_i) ||
```

而是 policy 输出层近似：

```text
score_norm_i = || ∇_η log π(a_i | s_i) ||
```

其中 `η` 是当前 stage actor 直接用于构造 action distribution 的输出参数。

选择这个近似的原因是：

```text
完整 per-row actor 参数梯度需要对每个 row 做一次 backward，训练 hot path 成本太高。
policy 输出层 score norm 成本低，并且能反映这个 row 的 logprob 对当前分布参数是否敏感。
```

## 4. Accel score norm

当前 Accel actor 的采样形式是：

```text
z ~ Normal(mu, std)
a_policy = radial_squash(z)
```

PPO logprob 对应的是 squash 前的 latent sample：

```text
log π = log Normal(z_sample | mu, std)
```

当前配置中：

```text
accel_log_std_trainable = true
```

因此 Accel 的 policy 输出层参数是：

```text
η_accel = (mu, log_std)
```

score 为：

```text
score_mu = (z - mu) / std^2
score_log_std = (z - mu)^2 / std^2 - 1
```

row 级 score norm：

```text
score_norm_accel =
sqrt(sum_over_uav_and_dims(score_mu^2 + score_log_std^2))
```

注意：

```text
score norm 使用 policy sample z_sample。
不要使用 safety shield 后的执行动作 a_exec 来算 logprob。
```

## 5. SAT score norm

当前 SAT actor 不是逐个 SAT sequential categorical，而是 subset categorical。

代码路径是：

```text
SatSubsetPolicy._compute_logits()
SatSubsetPolicy._select()
```

其形式是：

```text
每个可见 SAT 先得到 item_logit
每个合法 subset 的 logit 为：

subset_logit =
  sum(item_logits of subset members)
  + count_logit[subset_size]

subset ~ Categorical(legal subset logits)
```

因此 SAT 的 policy 输出层参数是：

```text
η_sat = legal subset logits
```

对每个 row：

```text
p = softmax(legal subset logits)
score = one_hot(chosen_subset) - p
score_norm_sat = ||score||
```

如果 legal subset 数量小于等于 1，当前 logprob 本身置 0，因此：

```text
score_norm_sat = 0
```

## 6. BW score norm

当前 BW actor 的分布形式是：

```text
score = score_head(gu_h)
tau_raw = tau_head(ctx)
tau = tau_min + (tau_max - tau_min) * sigmoid(tau_raw)

det_mean = masked_softmax(score / tau)

kappa_raw = kappa_head(ctx)
kappa = kappa_min + (kappa_max - kappa_min) * sigmoid(kappa_raw)

a_bw ~ Dirichlet(alpha)
alpha = det_mean * kappa
```

当前 YAML 中：

```text
bw_fixed_tau = null
bw_fixed_kappa = null
bw_tau_min = 0.5
bw_tau_max = 2.0
bw_kappa_min = 16.0
bw_kappa_max = 64.0
```

所以当前 BW 的 `tau` 和 `kappa` 都是可学习的有界输出。

BW 的 policy 输出层参数应定义为：

```text
η_bw = (valid GU score logits, tau_raw, kappa_raw)
```

对应 score norm：

```text
score_norm_bw =
|| ∇_(score_valid, tau_raw, kappa_raw)
   log Dirichlet(a_bw | det_mean(score_valid, tau_raw), kappa(kappa_raw)) ||
```

注意：

```text
不要把它简写成 “logits + kappa + tau”。
更准确的是：

valid GU score logits
+ tau_head 的 raw output
+ kappa_head 的 raw output

并且通过当前 BwPolicy._params() 的真实变换进入 logprob。
```

如果以后设置：

```text
bw_fixed_tau != null
```

则 `tau_raw` 不参与 score norm。

如果以后设置：

```text
bw_fixed_kappa != null
```

则 `kappa_raw` 不参与 score norm。

## 7. Priority 到采样概率

每个 stage 内单独把 priority 转成采样概率：

```text
q_i = (priority_i + eps)^alpha / sum_j (priority_j + eps)^alpha
```

第一版建议：

```text
alpha = 0.5
eps = 1e-6 * mean(priority)
```

含义：

```text
alpha = 0.5:
  温和偏向高 priority row。
  alpha = 1 太激进，alpha = 0 退化成均匀采样。

eps:
  防止 priority 为 0 的 row 永远没有采样概率。
```

如果 `mean(priority)` 本身为 0，则应 fallback 到 uniform sampling。

## 8. Row 抽样比例

第一版按用户当前想看的设置：

```text
不使用全部 batch。
不混 uniform。
全部 sampled rows 都按 priority 重采样。
```

配置语义：

```text
sample_frac = 0.5
uniform_frac = 0.0
priority_frac = 1.0
```

若 stage 内共有 `N` 个 row，则每个 actor epoch 抽：

```text
M = ceil(sample_frac * N)
```

这 `M` 个 row 全部按 `q_i` 抽样。

## 9. 重要性权重

因为目标仍然是 uniform-over-rows PPO objective，所以 sampled row 的权重是：

```text
w_i = 1 / (N * q_i)
```

实际使用：

```text
w_i = w_i / mean(w_i in sampled rows)
w_i = clamp(w_i, 0.25, 4.0)
```

actor loss：

```text
L = mean_sampled [ w_i * PPO_loss_i ]
```

如果使用 clipped PPO，则 `PPO_loss_i` 是原来的 per-row clipped surrogate loss。

## 10. 三个 stage 独立采样

Accel、SAT、BW 不混在一个池子里采样。

应该是：

```text
Accel:
  用 accel rows 算 accel priority
  按 accel q_i 抽 accel rows
  更新 accel actor

SAT:
  用 sat rows 算 sat priority
  按 sat q_i 抽 sat rows
  更新 sat actor

BW:
  用 bw rows 算 bw priority
  按 bw q_i 抽 bw rows
  更新 bw actor
```

原因：

```text
三个 stage 的 action distribution 不同，
score_norm 尺度不同，
advantage 尺度也可能不同。
混合采样会让某个 stage 的尺度主导整体抽样。
```

## 11. 第一版配置

建议新增或使用如下配置：

```yaml
stage_actor_importance_sampling_enabled: true

stage_actor_is_sample_frac: 0.5
stage_actor_is_alpha: 0.5
stage_actor_is_eps_scale: 1.0e-6

stage_actor_is_weight_clip_min: 0.25
stage_actor_is_weight_clip_max: 4.0

stage_actor_is_uniform_frac: 0.0
```

其中：

```text
stage_actor_is_uniform_frac 第一版可以不实现，
直接固定为 0。
```

## 12. 诊断指标

每个 stage 每轮记录：

```text
priority_mean
priority_p95
priority_max

sampled_priority_mean
all_priority_mean

is_weight_mean
is_weight_p95
is_weight_max

ess_frac

actor_kl
clip_frac
actor_loss
entropy
```

其中：

```text
ess_frac = (sum w)^2 / (sum(w^2) * M)
```

含义：

```text
ess_frac 越接近 1，说明重要性权重越均匀。
ess_frac 很低，说明实际有效样本数远小于 M。
```

重点观察：

```text
sampled_priority_mean 是否明显高于 all_priority_mean
ess_frac 是否太低
is_weight 是否大量撞到 clip 上界
KL / clip_frac 是否异常
训练 reward / deterministic eval reward 是否改善
```

## 13. 判断和调整

如果出现：

```text
ess_frac < 0.2
is_weight 经常撞到 4.0
actor KL 变大
clip_frac 变大
reward 或 eval reward 变差
```

说明采样太偏，应调保守：

```text
alpha: 0.5 -> 0.25
或 sample_frac: 0.5 -> 0.75
或 weight clip: [0.25, 4.0] -> [0.5, 2.0]
```

如果出现：

```text
sampled_priority_mean 和 all_priority_mean 差不多
```

说明 priority 没有明显区分度，重要性采样可能不会带来收益。

## 14. 当前方案要验证的问题

这套重要性采样要验证的是：

```text
当前 actor update 是否被大量低 credit / 低 score row 稀释。
```

如果有效，应看到：

```text
1. sampled_priority_mean > all_priority_mean
2. ESS 没有过低
3. KL / clip_frac 不异常
4. 同等 update budget 下 reward 或 eval reward 更好
```

如果无效或变差，则说明当前训练瓶颈不主要是 row 稀释，而可能是：

```text
advantage 本身不够准
policy parameterization 梯度方向不稳定
stage 间联动改变了 rollout 分布
或 priority 定义没有抓住真正有效的 credit
```

