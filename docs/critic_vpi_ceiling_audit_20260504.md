# Critic 是否能学到 Vπ(s)：验证方案

## 1. 问题

critic 训练时看到的是单条轨迹产生的样本：

```text
(s_t, G_t)
```

## 18. 2026-05-05 SAT critic path 诊断与结构改法

这一节只讨论 critic 是否能学固定 policy 下的 `Vπ(s)`，不混入后续 actor credit / `G - V` 的问题。

### 18.1 为什么继续查 SAT path

前面发现 SAT fixed-policy critic 的 `Vhat` 拟合不稳定：有些 heldout seed 上 EV 很高，有些 seed 上明显差。

这不能直接解释成：

```text
critic 输入缺少 last-step 信息
```

因为代码和 probe 已确认：

```text
SAT actor local 输入有 last selected / last load / last outflow。
critic world 输入也有 last selected / last load / last outflow。
SAT stage 的 current/prefix selected 为空，是因为当前 SAT action 尚未选择，这是正常的。
```

因此这轮重点改为检查：

```text
1. SAT actor local 输入是否比 critic world 输入多出关键可解释信息。
2. critic world raw 输入是否因为高维、尺度、stage 不相关信息混入而条件很差。
3. 某些 heldout seed 预测差，是不是因为状态外的 future random tape / seed regime。
```

### 18.2 local/world 同状态对照

脚本：

```text
scripts/audit_sat_local_world_vhat_probe.py
```

artifact：

```text
runs/diagnostics/critic_relearn_20260505/sat_vhat_artifact_s5_r6_a6_c6.pt
```

replay 校验：

```text
5 个 heldout seed 的 world replay max abs diff 全部为 0.0
```

所以这次 local/world probe 是同一批 SAT state 上的对照。

强正则 ridge 下的结果：

| feature set | EV(Vhat) | corr | MAE |
| --- | ---: | ---: | ---: |
| critic global scalars | 0.471 | 0.689 | 15.85 |
| critic world raw | 0.501 | 0.749 | 13.20 |
| SAT local core, no action-space | 0.536 | 0.766 | 11.70 |
| SAT local full, with action-space | 0.534 | 0.765 | 11.71 |

解释：

```text
1. SAT local 比 critic world 略好，但不是碾压。
2. full action-space 信息没有比 local core 更好，说明 candidate/subset 表本身不是主要缺口。
3. 不能说 critic 缺了 SAT actor 才有的关键 last-step 信息。
```

### 18.3 raw world probe 的病态不是“world 没信息”

`critic_world_raw` 是把整个 `StructuredWorldState` 直接摊平成约 2499 维特征后做 ridge probe。

弱正则时：

| ridge | critic_world_raw EV |
| ---: | ---: |
| 1e-4 | -4.894 |
| 1 | -3.906 |
| 100 | 0.090 |
| 1000 | 0.398 |
| 10000 | 0.501 |

解释：

```text
2499 维 raw world 特征高维、冗余、尺度混杂。
弱正则线性 probe 会在训练集上学到病态解，heldout 上预测爆掉。
这不是 world 输入完全没信息，而是 raw 输入条件数很差。
```

这件事会影响 critic 网络，但不是“同一个线性 probe 错误直接等于 critic 错误”。

更准确的判断是：

```text
world 输入里有信息；
但如果 critic 把所有 stage 的高维细节无差别混入同一个 system context，
优化会变难，且容易被 stage 不相关信息拖累。
```

### 18.4 分块 probe：SAT value 主要靠哪类信息

同一份 artifact、同一份 `Vhat` 上，按 world 输入块拆分：

| input block | EV(Vhat) | corr | MAE |
| --- | ---: | ---: | ---: |
| `uav_sat_edges` only | 0.756 | 0.873 | 10.08 |
| `sat_nodes` only | 0.494 | 0.715 | 12.75 |
| `global_scalars` only | 0.471 | 0.689 | 15.85 |
| `sat_ids` only | 0.225 | 0.497 | 20.30 |
| `gu_nodes` only | -0.055 | 0.236 | 22.99 |
| `uav_nodes` only | -0.209 | 0.240 | 22.53 |
| `uav_gu_edges` only | -0.722 | -0.059 | 29.13 |

组合/剔除结果：

| feature set | EV(Vhat) | corr | MAE |
| --- | ---: | ---: | ---: |
| `global + uav_sat_edges` | 0.716 | 0.849 | 10.77 |
| `global + sat_nodes` | 0.567 | 0.760 | 11.12 |
| `world minus uav_gu_edges` | 0.647 | 0.808 | 9.79 |
| full `world_raw` | 0.501 | 0.749 | 13.20 |

关键结论：

```text
SAT value 的主要解释信息在 UAV-SAT/backhaul/SAT 侧。
full uav_gu_edges 逐 pair access 细节对 SAT Vhat 是拖累项。
这不是说 GU/access 不重要，而是它不该以高维 pair edge 的形式强混进 SAT value path。
GU/access 对 SAT value 更适合以系统级 backlog/flow summary 进入。
```

### 18.5 坏 seed 不是简单 OOD

对 heldout rows 做训练分布 z-score：

```text
global_scalars: |z| > 3 的比例为 0
world_raw:      |z| > 3 约 0.3%~0.6%
SAT local:      |z| > 3 约 0.5%~1.1%
```

坏 seed 没有明显跑出训练分布。

所以某些 seed 预测差，不应简单解释为：

```text
heldout 状态分布外，critic 没见过。
```

### 18.6 future tape 对 Vhat 的影响

当前 branch replay / `Vhat` 生成会从原 rollout 复制未来 random tape。

因此 `Vhat` 更接近：

```text
E_policy_actions,continuations [G | s, fixed future tape]
```

不完全等同于 critic 理论上要学的：

```text
Vπ(s) = E_future_noise,policy [G | s]
```

把当前 state 看不到的 future arrival/hotspot/fading summary 加入线性 probe：

| feature set | EV(Vhat) | corr | MAE |
| --- | ---: | ---: | ---: |
| global only | 0.546 | 0.741 | 15.66 |
| hidden future tape summary only | 0.579 | 0.771 | 12.49 |
| global + hidden future tape summary | 0.657 | 0.816 | 11.01 |

解释：

```text
Vhat 中确实有一部分来自当前 state 不可见的 future tape / seed regime。
这会降低任何只看 state 的 critic 对该 Vhat artifact 的可达 EV。
但 hidden future summary 也没有完全解释掉坏 seed，
所以仍然存在 critic 结构/训练覆盖的问题。
```

### 18.7 结构结论：不要再用单一 shared system context

当前 `StructuredCritic` 的结构本质是：

```text
StructuredWorldState
  -> GU/UAV/SAT nodes + all edge types encoders
  -> 同一套 relational blocks
  -> 一个 system_token
  -> accel/sat/bw 三个 value head
```

这对 SAT value 不合适。

更合理的结构是：

```text
StructuredWorldState 不变

内部拆成：
  AccelCriticPath
  SatCriticPath
  BwCriticPath

三条 path 不共享 encoder、不共享 interaction block。
```

原因：

```text
同一个字段在不同 stage value 中语义不同。

uav_nodes:
  accel: mobility / safety / control state
  sat: relay demand source / backhaul requester
  bw: access serving context

uav_gu_edges:
  bw: 核心竞争/接入关系
  sat: 只应提供系统需求摘要，不应作为高维 pair edge 强混入 SAT path

uav_sat_edges:
  sat: 核心候选链路/回传关系
  bw: 只应通过 backhaul summary 影响 BW value
```

### 18.8 SAT path 的确定蓝图

SAT path 的目标：

```text
预测 SAT stage 的 Vπ(s)，也就是在当前 SAT action 尚未选择时，
当前系统状态和可见候选回传结构对应的 expected return。
```

输入仍然读同一个 `StructuredWorldState`，但 SAT path 只让以下信息做高维交互：

```text
global_scalars
uav_nodes
sat_nodes
uav_sat_edges
uav_sat_mask
sat_mask
SAT 相关 local summaries
```

不进入 SAT 高维 interaction 的信息：

```text
full gu_nodes token set
full uav_gu_edges pair edge set
full uav_uav_edges pair edge set
```

这些信息只通过 summary 进入：

```text
GU total queue / drop / expected arrival / last outflow
UAV queue / last inflow / last outflow / last drop
last weighted workload
remaining horizon
last selected sat load
non-token SAT residual workload/drop/processed
```

#### SAT path 模块

建议结构：

```text
SatCriticPath
  sat_global_encoder(global_scalars)
  sat_uav_encoder(uav_nodes)
  sat_node_encoder(sat_nodes)
  sat_us_edge_encoder(uav_sat_edges)
  sat_summary_encoder(sat_summary_features)
  sat_relational_blocks
  sat_system_readout
  sat_value_head
```

其中：

```text
sat_system_token = learned_sat_system_token
                 + sat_global_encoder(global_scalars)
                 + sat_summary_encoder(sat_summary_features)
```

`sat_relational_blocks` 只在：

```text
UAV tokens
SAT tokens
UAV-SAT edge tokens
system token
```

之间传递信息。

#### SAT summary features

SAT path 需要的 summary 不应重新改外部 schema，直接从 `StructuredWorldState` 派生：

```text
global summary:
  world_state.global_scalars

UAV-side summary:
  sum/mean/max UAV_QUEUE_STEPS
  sum/mean/max UAV_LAST_INFLOW_STEPS
  sum/mean/max UAV_LAST_OUTFLOW_STEPS
  sum/mean/max UAV_LAST_DROP_STEPS

SAT-side summary:
  sum/mean/max SAT_QUEUE_STEPS over token SATs
  sum/mean/max SAT_LAST_PROCESSED_STEPS over token SATs
  sum/mean/max SAT_LAST_DROP_STEPS over token SATs
  last selected SAT load mean/max

GU/system demand summary:
  total GU queue/drop/arrival/last outflow from global scalars
  no full GU token interaction in SAT path
```

#### SAT relational block

SAT block 不用 full graph message passing，而是 typed bipartite interaction：

```text
UAV -> SAT:
  message from uav_token[u], us_edge[u,s] to sat_token[s]

SAT -> UAV:
  message from sat_token[s], us_edge[u,s] to uav_token[u]

UAV/SAT -> system:
  masked attention or gated pooling into system token

system -> UAV/SAT:
  global conditioning/gating
```

不做：

```text
GU -> UAV -> SAT 多跳细粒度传播
UAV-UAV pair interaction
full all-token self-attention
```

理由：

```text
SAT value 的分块 probe 已显示核心是 UAV-SAT / SAT 侧。
GU/access 细节以 full pair edge 混入会拖累 SAT value。
```

#### SAT readout

SAT value 只输出一个系统级标量：

```text
V_sat(s) = sat_value_head(sat_system_context)
```

不做 per-UAV value 再平均。

`sat_system_context` 来自：

```text
sat_system_token after sat_relational_blocks
```

而不是：

```text
mean/max concat of all tokens
```

原因：

```text
系统级 value 应该由一个专门的 system token 汇总，
而不是靠固定 mean/max pooling 去猜哪些 token 重要。
```

### 18.9 实现边界

第一版实现只改 critic 内部：

```text
不改 StructuredWorldState
不改 native env writer
不改 actor
不改 reward
不改 rollout buffer
```

新增配置：

```yaml
critic_stage_specific_paths_enabled: true
critic_sat_message_layers: 1
critic_sat_embed_dim: 256
critic_sat_hidden_dim: 256
critic_sat_value_head_hidden: 256
critic_sat_value_head_layers: 2
```

保留旧路径可回退：

```yaml
critic_stage_specific_paths_enabled: false
```

验证标准：

```text
1. 在同一 Vhat artifact 上，SAT actual critic EV 应接近或超过 uav_sat_edges/global/sat_nodes probe 的组合水平。
2. 不能只看 overall EV，还要看 255210 / 9055210 这类坏 seed 是否改善。
3. 如果 SAT path 仍无法改善，而 hidden future tape 能改善，则优先修 Vhat/target 口径，不继续盲目加网络。
```

### 18.10 SAT path pilot 结果

实现了可配置的 SAT-specific path：

```yaml
critic_stage_specific_paths_enabled: true
```

第一版只让 `uav_sat_edges` 通过 UAV/SAT token 间接进入 system token，结果不够好：

| path | critic_lr | epochs | Vhat EV | heldout MC EV |
| --- | ---: | ---: | ---: | ---: |
| shared blocks=1 baseline | 1e-3 | 30 | 0.898 | 0.913 |
| sat path v1 | 1e-3 | 30 | 0.858 | 0.869 |

随后补了直接的 `UAV-SAT edge -> system` message，不是 linear baseline，而是 SAT relational block 内的 edge-to-system 汇聚：

```text
uav_token[u], sat_token[s], us_edge[u,s], system_token
  -> us_to_system(...)
  -> masked mean over valid UAV-SAT edges
  -> system_update(...)
```

结果：

| path | critic_lr | epochs | Vhat EV | heldout MC EV |
| --- | ---: | ---: | ---: | ---: |
| sat path + edge-to-system | 1e-3 | 30 | 0.876 | 0.900 |
| sat path + edge-to-system | 3e-3 | 30 | 0.883 | 0.912 |
| sat path + edge-to-system | 3e-3 | 60 | 0.893 | 0.924 |
| shared blocks=1 baseline | 3e-3 | 60 | 0.907 | 0.918 |

当前解释：

```text
1. SAT-specific path 并没有立刻超过 shared baseline。
2. edge-to-system 是必要的；否则 SAT path 丢掉了 uav_sat_edges probe 中最强的信息。
3. SAT path 在 heldout MC EV 上可以追到甚至略高于 shared baseline，
   但在 Vhat artifact 上仍略低于 shared baseline。
4. 因此现在不能说“SAT path 已经解决 critic 问题”。
```

更谨慎的结论：

```text
stage-specific SAT path 的方向仍合理，
但第一版不能只靠删掉 uav_gu_edges；
必须保证 UAV-SAT edge 分布能强进入 system readout。
下一步若继续做，应比较 per-seed 结果，
尤其看 255210 / 9055210 是否改善，而不是只看 overall EV。
```

但它应该输出的是条件均值：

```text
Vπ(s_t) = E[G_t | s_t]
```

所以单条 `G_t` 不是精确标签，而是 noisy label。MSE 理论上仍能学到 `Vπ(s)`，前提是：

```text
E[G_t | s_t] = Vπ(s_t)
```

并且状态覆盖、网络容量、输入信息、训练样本量都足够。

## 2. 不能怎么验证

不能只做：

```text
固定一小批 rollout batch
critic 训练很多很多 epoch
看 train loss 是否下降
```

这只能说明 critic 能不能拟合这批 noisy label，不能证明它学到了 `Vπ(s)`。

也不能把：

```text
Vθ(s) vs 单条 heldout return G_t 的 EV
```

当作最终判据。单条 `G_t` 里有后续动作和外生随机噪声，EV 低不一定说明 critic 没学到条件均值。

## 3. 正确验证结构

固定当前 actor policy，不更新 actor。

### A. 多 seed 收集训练状态

```text
train bank:
  多个 reset seed
  多个 rollout
  收集 stage world state s
  target = 当前 PPO 口径的 train-time GAE return

heldout bank:
  不同 seed
  同样收集 state/target
```

这里的 `train-time GAE return` 是当前 PPO 实际给 critic 的训练目标。

### B. 只训练 critic

冻结 actor，只优化 critic：

```text
loss = MSE(Vtheta(s), train_gae_return)
```

记录：

```text
train noisy-label EV/MSE
heldout noisy-label EV/MSE
```

这些指标只看训练健康，不作为最终是否能学 `Vπ(s)` 的唯一标准。

### C. 小子集估计去噪 V_hatπ(s)

在 heldout bank 中抽一小批 state，对每个 state：

```text
采 K 个 action a ~ π(.|s)
每个 action 跑 C 条 continuation
从当前 t 跑到 finite-horizon episode 末尾
```

得到：

```text
V_hatπ(s) = mean_{a samples, continuations} G(s,a,future)
```

这才是 `Vπ(s)` 的有限样本估计。

然后比较：

```text
Vθ(s) vs V_hatπ(s)
```

核心指标：

```text
EV(Vθ, V_hatπ)
corr(Vθ, V_hatπ)
MSE(Vθ, V_hatπ)
```

## 4. 判断标准

| 结果 | 解释 |
| --- | --- |
| `Vθ` 对 `V_hatπ` EV 高 | critic 输入/结构有能力学固定 policy 的条件均值 |
| `Vθ` 对 noisy return EV 低，但对 `V_hatπ` EV 高 | 单条 return 噪声大，但 critic 可能已经学到平均值 |
| `Vθ` 对 `V_hatπ` EV 也低 | critic 输入/结构/训练不足，PPO 的 V 路线基础不稳 |
| `Vθ` 能学 `V_hatπ`，但 `G_single - Vθ` 仍和 `A_qpi` 不对齐 | 问题主要是单条 rollout advantage 方差大，不是 V 表达能力 |

## 5. 成本含义

critic 路线不是一次性成本。

固定 policy 下学到的是：

```text
Vπ_old(s)
```

actor 更新后目标理论上变成：

```text
Vπ_new(s)
```

所以这个审计只回答第一步：

```text
当前 critic 是否有能力学固定 policy 的 Vπ(s)
```

如果固定 policy 都学不到，继续调 PPO 很难。

如果固定 policy 能学到，下一步才看在线训练中 critic 能不能跟上 policy 变化。

## 6. 当前实现

脚本：

```text
scripts/audit_stage_critic_only_fit.py
```

关键参数：

```text
--target train_gae
--train_rollouts N
--critic_epochs E
--vpi_rows R
--vpi_policy_action_samples K
--vpi_continuations C
--vpi_follow stochastic
```

输出同时包含：

```text
heldout noisy-label fit
V_hatπ ceiling probe
```

## 7. Critic 后面还要查什么

`Vθ` 能不能接近 `V_hatπ(s)` 只是第一步。后面还有四个断点，不能省。

### D. 用学好的 Vθ 构造 MC advantage

定义：

```text
A_mc(s_t,a_t) = R_t^MC - Vθ(s_t)
```

其中：

```text
R_t^MC = 从当前 t 累积到 finite-horizon episode 末尾的真实 return
```

要检查：

```text
corr(A_mc, A_qpi_rollout)
sign_agree(A_mc, A_qpi_rollout)
用 A_mc 做一次 probe actor update 后，高 Qπ candidate logprob 是否上升
```

如果 `Vθ` 已经接近 `Vπ`，但 `A_mc` 仍和 `A_qpi` 对不上，说明问题主要是：

```text
单条 MC return 里的 future action / 外生随机噪声太大。
```

这不是 critic 表达能力问题，而是单样本 policy-gradient estimator 方差问题。

### E. 用学好的 Vθ 构造 GAE(V) advantage

定义：

```text
δ_t = r_t + γ Vθ(s_{t+1}) - Vθ(s_t)
A_gae(V)_t = δ_t + γλδ_{t+1} + γ²λ²δ_{t+2} + ...
```

这里的关键是：

```text
Vθ 必须是已经用 MC 或其他方式学好的 full-horizon Vπ 近似，
不能再用初始坏 V 生成短视 train-GAE target。
```

要检查：

```text
corr(A_gae(V), A_qpi_rollout)
sign_agree(A_gae(V), A_qpi_rollout)
用 A_gae(V) 做一次 probe actor update 后，高 Qπ candidate logprob 是否上升
```

如果 `A_mc` 对齐而 `A_gae(V)` 不对齐，问题在 GAE 计算/λ/terminal/bootstrap 口径。

如果两者都不对齐，但 candidate-wise `A_qpi` 对齐，问题在：

```text
单条 rollout advantage 的噪声太大；
需要多样本 action-contrast 或更低方差 estimator。
```

### F. Actor update probe

对同一批 heldout state/action/candidate，不真正跑新环境训练，只做 probe update。

比较三种 advantage：

```text
1. A_mc = R_mc - Vtheta
2. A_gae(V)
3. A_qpi = Q_hatπ(s,a) - mean_candidate Q_hatπ(s,a)
```

输出：

```text
q-shift = mean[A_qpi(candidate) * Δlogπ(candidate|s)]
per_state_pos = frac_s[sum_a A_qpi(s,a) * delta_logpi(s,a) > 0]
best_up = best-Qπ candidate logprob 上升比例
margin_up = best-Qπ 相对 worst-Qπ logprob margin 上升比例
```

这一步回答：

```text
如果给 actor 正确/更低噪声 advantage，它是否能朝 Qπ objective 改？
```

### G. 在线跟踪

即使 fixed-policy 下 critic 可以学 `Vπ_old(s)`，actor 更新后目标会变：

```text
Vπ_old(s) -> Vπ_new(s)
```

所以还要做小步在线验证：

```text
固定小 actor_lr / 小 KL
每次 actor 更新后重新收 rollout
critic 用 MC 或 high-lambda target 跟踪
看 V_hatπ EV 是否保持
看 A_mc / A_gae(V) 是否继续对齐 A_qpi
```

这一步才回答：

```text
critic 路线在真实训练中能不能持续跟踪 policy 变化。
```

## 8. 重要修正：一次性 train-GAE target 不等于 Vπ

pilot 结果显示：

```text
heldout train-GAE target mean ≈ -0.146
heldout V_hatπ full-horizon mean ≈ -2.012
```

这不是小误差，而是口径差异。

原因是当前 PPO critic target 是 bootstrapped λ-return：

```text
target = GAE return = reward + gamma/lambda bootstrapping through old V
```

如果 old V 初始很差，单次生成的 GAE target 不会等于 full finite-horizon `Vπ(s)`。它更像短有效 horizon 的 TD(λ) 训练标签。

所以 critic 验证要分两步：

```text
one-shot GAE fit:
  固定一次 train-GAE target 后多训 critic。
  只能验证 critic 能否拟合当前 PPO 这一轮的 noisy/bootstrapped label。

fixed-policy fitted evaluation:
  actor 固定。
  多轮收 rollout。
  每轮用当前 critic 重新生成 train-GAE target。
  训练 critic。
  最后再看 Vθ 是否接近 full-horizon V_hatπ。
```

如果 fixed-policy fitted evaluation 仍然不能接近 `V_hatπ`，说明当前 critic/GAE 训练链路确实难以学到固定 policy 的 long-horizon value。

如果 fitted evaluation 能接近，而 one-shot 不行，说明主要问题是在线训练里 critic 需要多轮 bootstrap 传播，早期 actor 更新会拿到很差的 advantage。

## 9. 当前 SAT pilot 结果

配置：

```text
stage = sat
reward_mode = sat_relay_processed
num_envs = 8
rollout_env_steps = 250
heldout V_hatπ:
  rows = 8
  policy action samples = rollout sampled action + 4 extra samples
  continuations = 4
  follow = stochastic
```

结果：

| run | target | train setup | heldout target mean | value pred mean | heldout noisy EV | V_hatπ mean | EV(Vθ,V_hatπ) | corr(Vθ,V_hatπ) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one-shot | train_gae | 8 rollouts, 30 epochs | -0.146 | -0.151 | 0.200 | -2.012 | -0.006 | -0.235 |
| fitted | train_gae | 3 rounds, 4 rollouts/round, 10 epochs/round | -0.149 | -0.198 | 0.096 | -2.012 | 0.008 | 0.185 |
| fitted | train_gae | 8 rounds, 4 rollouts/round, 5 epochs/round | -0.149 | -0.193 | 0.110 | -2.014 | 0.003 | 0.086 |
| control | MC | 8 rollouts, 30 epochs | -1.698 | -1.548 | 0.989 | -2.012 | 0.997 | 0.999 |

解释：

```text
1. 当前 critic 结构/输入不是完全学不了 Vπ。
   MC target 对照下，Vθ 对 V_hatπ 的 EV 到 0.997。

2. 当前 PPO 口径的 train-GAE target 和 full-horizon Vπ 尺度严重不一致。
   train-GAE heldout target mean 约 -0.15；
   V_hatπ mean 约 -2.01。

3. 多轮 fitted train-GAE 只把 value mean 推到约 -0.19~-0.20，
   仍远离 -2.01。

4. 因此当前 SAT critic 的主要问题不像是网络表达能力不够，
   而是 PPO 当前使用的 bootstrapped GAE/λ-return target 没有有效学到 long-horizon Vπ。
```

这会影响 actor：

```text
如果 Vθ 只在短 horizon / λ-return 尺度，
那么 A_ppo = G_single - Vθ
不太可能稳定等于当前 action 的 long-horizon Qπ advantage。
```

这也解释了前面链路审计结果：

```text
Qπ candidate credit 存在；
candidate-wise Qπ update 能推对；
但 A_ppo 和 A_qpi 对齐弱。
```

还有一个重要点：

```text
MC critic 对 V_hatπ 学得很好以后，
单条 residual = G_single - Vθ
和 A_qpi 的相关仍然不高。
```

这不说明 MC critic 没学好，而是说明：

```text
G_single - Vπ(s)
```

仍然包含后续 action / 外生随机的单样本噪声。它理论上靠很多样本平均才是可用 policy-gradient estimator。

所以当前证据把问题分成两层：

```text
critic 学 Vπ:
  网络本身可以，MC target 可以；
  当前 train-GAE target 不行或传播太慢。

actor credit:
  即使 Vπ 学好，单条 rollout residual 仍可能很吵；
  candidate-wise Qπ averaging 明显更直接。
```

还没完成的检查是：

```text
1. 用 MC-trained critic 明确计算 A_mc = R_mc - Vθ，并和 A_qpi 对齐。
2. 用同一个 MC-trained critic 计算 A_gae(V)，并和 A_qpi 对齐。
3. 分别用 A_mc / A_gae(V) / A_qpi 做 actor probe update，看 q-shift。
4. 如果 fixed-policy 下可行，再做小步在线跟踪。
```

## 10. MC-trained critic 后的 advantage probe

在 MC target 训练 critic 后，继续用同一 heldout bank 做 D/E/F 检查。

配置：

```text
critic target = MC
train rollouts = 8
critic epochs = 30
advantage probe rows = 8
policy action samples = rollout sampled action + 4 extra samples
continuations = 4
follow = stochastic
```

结果：

| advantage | corr(adv, A_qpi_rollout) | sign agree | q-shift | per_state_pos | best_up | margin_up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_mc = R_mc - Vθ | -0.568 | 1.000 | 0.0024 | 0.500 | 0.750 | 0.750 |
| A_gae(Vθ) | 0.308 | 1.000 | 0.0024 | 0.375 | 0.750 | 0.750 |
| A_qpi selected | 1.000 by definition | n/a | 0.0006 | 0.250 | 0.875 | 0.250 |
| A_qpi all-candidate | candidate-wise | n/a | 0.0024 | 0.750 | 0.750 | 0.750 |

注意：

```text
这次 sampled rows 里 rollout action 的 A_qpi_rollout 全部为正，
所以相关系数不能单独解释成“方向完全反”。
sign agree 和 actor probe 更有参考价值。
```

当前 pilot 支持的判断：

```text
1. MC-trained critic 能把 Vπ 学到正确尺度。
2. A_mc 和 A_gae(Vθ) 在这批样本上没有符号反掉。
3. 但它们的 q-shift 很小，只是弱正向。
4. candidate-wise A_qpi 仍然是更直接的 actor credit。
```

所以现在不能说：

```text
MC-trained critic + MC/GAE advantage 已经证明能训练。
```

只能说：

```text
MC-trained critic 解决了 V 尺度问题；
但单条 rollout advantage 到 actor update 的信号仍偏弱，需要更多 seed/rows 或直接短训验证。
```

## 11. 2026-05-04 追加：reward/return 口径对齐与 `A_qpi_V` 检查

本次追加不是替换前面的 pilot，而是修正诊断口径并补一项更直接的 credit 检查。

### 11.1 诊断脚本的 rollout begin 口径

发现旧版 `scripts/audit_stage_critic_only_fit.py` 在同一个 native rollout program 中连续收多条 rollout；真实训练主路径则是每个 update 都重新：

```text
begin_native_rollout(...)
collect rollout
update
```

这个差异会让 heldout/continuation 进入不同的 runtime/random/history 口径。影响很大：

```text
同一批 stage row:
  fresh credit_chain rollout:           Qπ return 约 56~116
  old critic_only_fit heldout path:     Qπ return 约 -1~-2
  credit_chain --pre_collect_rollouts 1 也能复现 -1~-2
```

对应修正：

```text
scripts/audit_stage_critic_only_fit.py:
  _collect_one_rollout(...) 内部先 begin_native_rollout(...)

scripts/audit_stage_credit_chain.py:
  新增 --selected_stage_samples，用来指定 exact rows
  新增 --pre_collect_rollouts，仅用于复现旧错误
```

修正后，同一批 rows 的 Qπ return 与 `credit_chain` fresh 口径对齐。

### 11.2 clean run 结果

配置：

```text
stage = sat
reward_mode = sat_relay_processed
target = mc
train_rollouts = 8
critic_epochs = 30
vpi_rows = 8
policy_action_samples = 4
continuations = 4
follow = stochastic
seed = 45210
```

输出文件：

```text
runs/diagnostics/critic_vpi_ceiling/sat_relay_processed_mc_train8_epoch30_beginfix_advprobe_seed45210.json
```

结果：

```text
heldout single-return EV:      0.875
Vθ vs sampled V_hatπ EV:       0.672
Vθ vs sampled V_hatπ corr:     0.821
```

这说明在 return 口径对齐后，MC target 下的 critic 不是完全学不动；它能学到一部分 state 难度。

### 11.3 `A_qpi_mean` vs `A_qpi_V`

用户提出的检查是关键的：

```text
A_qpi_mean = Qπ(s,a) - mean_candidate Qπ(s,a)
A_qpi_V    = Qπ(s,a) - Vθ(s)
```

`A_qpi_mean` 更接近“同一 state 下哪个 action 更好”的局部理想 credit。`A_qpi_V` 则模拟“用 critic baseline 扣掉 state 难度”后还剩下什么。如果 `Vθ(s)` 的 row-level 残差比 action gap 大，即使 EV 看起来不错，也会破坏 action credit。

clean run 中：

```text
std(A_qpi_mean selected)      ≈ 0.81
std(q_row_mean - Vθ)          ≈ 11.22
corr(A_qpi_mean, A_qpi_V)     ≈ 0.058
sign agreement all-candidate  ≈ 0.575
sign agreement selected       ≈ 0.500
```

解释：

```text
Vθ 的 state-level 残差比同 state 内 action gap 大一个数量级以上。
```

所以这里不能只看：

```text
EV(Vθ, V_hatπ)
```

还必须看：

```text
baseline_error_scale = std(q_row_mean - Vθ)
action_gap_scale     = std(Qπ(s,a) - mean_candidate Qπ(s,a))
```

只有当 baseline residual 小于或至少接近 action gap 尺度时，`Qπ - Vθ` 才可能可靠保留动作 credit。当前这组结果说明：critic 已经比初始好很多，但 residual 仍足以把局部 action credit 洗掉。

## 12. 2026-05-04 追加：critic 训练充分性拆分实验

目的：

```text
拆开判断 EV(Vθ,V_hatπ)≈0.67 是因为：
1. critic epochs 不够；
2. train rollouts / 状态覆盖不够；
3. V_hatπ continuation 估计太 noisy；
4. 还是当前设置下继续加这几项也很难明显超过这个区间。
```

固定条件：

```text
stage = sat
reward_mode = sat_relay_processed
target = MC
config = configs/tmp/structured_single_sat_3uav_20gu_t250_ppo.yaml
num_envs = 8
rollout_env_steps = 250
vpi_rows = 8
vpi_policy_action_samples = 4
seed = 45210
```

输出目录：

```text
runs/diagnostics/critic_vpi_split_20260504/
```

### 12.1 epoch sweep

固定：

```text
train_rollouts = 8
vpi_continuations = 4
```

| run | critic epochs | heldout single-return EV | EV(Vθ,V_hatπ) | corr(Vθ,V_hatπ) |
| --- | ---: | ---: | ---: | ---: |
| epoch00_r8_c4 | 0 | 0.000 | -0.000 | -0.211 |
| epoch10_r8_c4 | 10 | 0.365 | -0.077 | 0.406 |
| epoch30_r8_c4 | 30 | 0.875 | 0.672 | 0.821 |
| epoch60_r8_c4 | 60 | 0.889 | 0.676 | 0.822 |

判断：

```text
30 -> 60 epochs 基本没有提升。
所以当前 0.67 左右的 V_hatπ EV 不像是“只差多训几轮 optimizer”。
```

### 12.2 train rollout / 状态覆盖 sweep

固定：

```text
critic_epochs = 30
vpi_continuations = 4
```

| run | train rollouts | train samples | heldout single-return EV | EV(Vθ,V_hatπ) | corr(Vθ,V_hatπ) |
| --- | ---: | ---: | ---: | ---: | ---: |
| rollout02_e30_c4 | 2 | 4000 | 0.823 | 0.356 | 0.659 |
| rollout04_e30_c4 | 4 | 8000 | 0.930 | 0.605 | 0.807 |
| epoch30_r8_c4 | 8 | 16000 | 0.875 | 0.672 | 0.821 |
| rollout16_e30_c4 | 16 | 32000 | 0.864 | 0.684 | 0.827 |

判断：

```text
2 -> 4 -> 8 rollouts 对 V_hatπ EV 有明显帮助。
8 -> 16 只小幅提升。

所以训练样本/状态覆盖确实是因素，
但在这组预算里加到 16 rollouts 也没有把 EV 推到接近 0.9。
```

### 12.3 V_hatπ continuation precision sweep

固定：

```text
train_rollouts = 8
critic_epochs = 30
```

| run | V_hatπ continuations | heldout single-return EV | EV(Vθ,V_hatπ) | corr(Vθ,V_hatπ) |
| --- | ---: | ---: | ---: | ---: |
| vhat_c2_r8_e30 | 2 | 0.875 | 0.673 | 0.822 |
| epoch30_r8_c4 | 4 | 0.875 | 0.672 | 0.821 |
| vhat_c8_r8_e30 | 8 | 0.875 | 0.671 | 0.821 |

判断：

```text
continuations 从 2 到 8 几乎不改变 EV/corr。
所以这次 0.67 不是因为 V_hatπ continuation 太少导致的评估噪声。
```

### 12.4 当前结论

这轮拆分后，比较稳的结论是：

```text
1. 不是单纯 critic epochs 不够。
2. train rollouts / 状态覆盖会影响 V_hatπ EV，但 8~16 rollouts 后开始平台化。
3. V_hatπ continuation 数不是当前主要瓶颈。
4. 当前 critic 在 MC target 下能学到明显 state 难度，但还没学到足够精确的 Vπ。
```

这意味着前面 `A_qpi_mean` vs `A_qpi_V` 的低相关不能解释成“critic 路线必然失败”，但可以解释成：

```text
当前预算下的 Vθ residual 仍然太大，
不足以作为 action-credit baseline。
```

下一步如果还要继续 critic 路线，应优先查：

```text
1. 更宽 heldout state / 多 seed 下这个 0.65~0.70 平台是否稳定；
2. 增加 train_rollouts 到 32 是否继续提升，还是完全平台；
3. critic 输入/结构是否缺少能解释 Vπ residual 的状态量。
```

不应优先继续单纯把 epoch 从 60 加到更大。

## 13. 2026-05-04 追加：critic 网络本身的 residual attribution

前面的表明：

```text
critic 能学到 Vπ 的粗结构；
但是否是“输入缺失 / 网络结构丢信息 / value head 读出不够”，还没拆开。
```

因此做一个固定 policy 的 residual attribution probe：

```text
stage = sat
reward_mode = sat_relay_processed
target = MC
train_rollouts = 8
critic_epochs = 30
V_hatπ rows = 24
V_hatπ continuations = 2
```

输出：

```text
runs/diagnostics/critic_vpi_split_20260504/residual_attribution_sat_mc_r8e30_vpi24c2_resetfix.json
```

比较四种读法：

```text
actual critic:
  当前 StructuredCritic 的真实输出。

global linear:
  只用 world_state.global_scalars 做 ridge linear probe。

raw-all linear:
  把 nodes / edges / masks / global 全部 flatten 后做 ridge linear probe。
  这个维度很高，样本相对少，容易数值/过拟合，不作为主要结论。

system_context linear:
  用训练后 critic encoder 输出的 system_context，
  重新拟合一个线性 value readout。
```

结果：

| readout | EV vs V_hatπ | corr vs V_hatπ |
| --- | ---: | ---: |
| actual critic | 0.854 | 0.926 |
| global linear | 0.917 | 0.958 |
| raw-all linear | -273.990 | -0.286 |
| system_context linear | 0.883 | 0.940 |

解释：

```text
1. 这次用 24 rows 后，actual critic 对 V_hatπ 的 EV 到 0.854。
   之前 8 rows 的 0.67 很可能受 heldout row 抽样影响较大，
   不应再把 0.67 当成稳定平台结论。

2. global_scalars 线性 probe 反而比 actual critic 更好。
   这说明 Vπ 的主要可解释部分至少有相当一部分已经在 global_scalars 里；
   不是“输入完全缺失”。

3. system_context 重新拟合线性 readout 比 actual critic 稍好。
   说明 value head/readout 有一点损失，
   但差距不大，不像唯一主因。

4. 更关键的问题是：
   当前 relational critic 没有稳定地把简单全局系统量作为强 baseline 直通到 value。
   它经过 token/message/system_context 后，反而弱于 global linear baseline。
```

因此当前更像一个结构问题：

```text
V(s) 的大头是系统级 workload / remaining horizon / queue/drop/load 这些全局量；
当前 critic 把它们先 embed 到 system token，再和 relational message passing 混合，
最后只从 system_context 出 value。

这个路径能学，但不如直接 global baseline 稳。
```

下一步更合理的 critic 结构不是继续单纯加 epoch，而是：

```text
V(s) = global_baseline(global_scalars) + relational_residual(system_context)
```

其中：

```text
global_baseline:
  专门预测系统级大趋势；

relational_residual:
  只负责局部交互、prefix、route/load、竞争关系带来的残差。
```

这样不会否定 relational critic，而是避免它把最容易、最稳定的全局 baseline 也绕进复杂 message passing 里。

## 14. 2026-05-05 固定 critic benchmark

前面的 `8 rows / 24 rows` 都只是临时 probe，容易被抽到哪些 heldout state 影响。这里固定一个更明确的 benchmark：

```text
stage = sat
reward_mode = sat_relay_processed
train_rollouts = 8
critic_epochs = 30
benchmark V_hatπ rows = 64
V_hatπ policy action samples = 4
V_hatπ continuations = 2
```

输出：

```text
runs/diagnostics/fixed_critic_benchmark_20260505/sat_mc_r8e30_rows64_c2.json
runs/diagnostics/fixed_critic_benchmark_20260505/sat_mc_r8e30_rows64_c2_layerwise.json
```

注意：

```text
global_linear 和 system_context_linear 都不是在这 64 个 benchmark rows 上训练的。
它们都只用 critic train bank 的 16000 个 MC target 样本拟合，
然后在固定 benchmark rows 上评估。
```

所以这组结果不能简单解释成“linear probe 在小批 benchmark rows 上过拟合”。

结果：

| readout | EV vs V_hatπ | corr vs V_hatπ | MAE |
| --- | ---: | ---: | ---: |
| actual critic | 0.779 | 0.883 | 11.484 |
| global linear | 0.902 | 0.950 | 8.489 |
| system_context linear | 0.827 | 0.909 | 10.209 |

进一步做 layerwise probe，定位 `global_scalars` 信息在哪里变差：

| layer/readout | EV vs V_hatπ | corr vs V_hatπ |
| --- | ---: | ---: |
| raw global_scalars linear | 0.902 | 0.950 |
| global_embed linear | 0.802 | 0.899 |
| system_initial linear | 0.802 | 0.899 |
| after block1 linear | 0.846 | 0.920 |
| after block2 / system_context linear | 0.827 | 0.909 |
| actual critic output | 0.779 | 0.883 |

这个结果比之前更具体：

```text
1. 最大的信息损失不是最后才发生，而是一开始：
   raw global_scalars -> global_embed 从 0.902 掉到 0.802。

2. block1 并不是单纯破坏信息，反而把 EV 从 0.802 拉到 0.846。

3. block2 又从 0.846 掉到 0.827。

4. actual value head 再从 system_context linear 的 0.827 掉到 0.779。
```

所以“global 信息丢在哪里”的当前答案是：

```text
第一处：global_scalar_encoder 本身已经让 raw global_scalars 的 Vπ 可读性明显下降。
第二处：第二个 relational block 有进一步下降。
第三处：value head 没有达到同一 system_context 上的 linear readout。
```

这说明问题不是简单“旁边加 global_scalars”可以解释完，而是当前 critic 的基础结构确实有三段损伤。

同一组模型在 heldout single-return 上：

| readout | EV vs single MC return | corr |
| --- | ---: | ---: |
| actual critic | 0.875 | 0.944 |
| global linear | 0.910 | 0.955 |
| system_context linear | 0.905 | 0.955 |

解释：

```text
1. fixed benchmark 下，actual critic 确实弱于 global linear。
   这不是只看 8 rows 的偶然现象。

2. system_context linear 也强于 actual critic。
   这说明训练后的 system_context 里有 actual value head 没读干净的信息。

3. 但 system_context linear 仍弱于 global linear。
   这说明 system_context 本身已经相对 raw global_scalars 损失了一部分
   对 Vπ 很有用的简单全局信息。
```

因此这不是一句“加 residual head”能解决的问题。更准确的结构诊断是：

```text
当前 critic 的 value 路径：
  global_scalars -> global_embed
  global_embed + token/message passing -> system_context
  system_context -> value_head

这个路径没有把 raw/global 系统级 baseline 作为稳定直通项保留到 value 输出。
```

代码上看，`global_embed` 会参与每层 `system_update`：

```text
system_token = system_token + system_update([system_token, agg_g, agg_u, agg_s, global_embed])
```

这有残差，但没有一条：

```text
raw global_scalars 或 global_embed -> value
```

的直接读出路径。也就是说，全局系统级主导量必须经过多层 MLP/message mixing 后才能影响 value。

这解释了为什么：

```text
global_scalars linear > system_context linear > actual critic
```

当前最保守的判断：

```text
critic 输入中有强 Vπ 信息；
当前 system_context 不是完全坏，但它没有比 raw global_scalars 更好；
actual value head 又没有完全读出 system_context 里已有的信息。
```

所以如果要改结构，优先不是假设 `system_context` 能神奇预测 residual，而是改成显式保留全局 baseline：

```text
V(s) = ValueHead([global_embed, system_context])
```

或更强一点：

```text
V(s) = GlobalHead(global_scalars or global_embed)
     + GraphHead([global_embed, system_context])
```

这里的关键不是“让 system_context 预测 residual”，而是：

```text
不要让系统级全局 baseline 只能通过 message-passing 后的 system_context 间接进入 value。
```
## 15. 2026-05-05 Head-only 与 LR sweep

这一步专门回答两个问题：

```text
1. actual critic 弱于 linear probe，是不是典型过拟合？
2. actual value head 是否本身读不出 system_context？
3. full critic 是否因为学习率太低或卡平台？
```

脚本：

```text
scripts/audit_critic_head_lr_sweep.py
```

输出：

```text
runs/diagnostics/critic_head_lr_sweep_20260505/sat_mc_head_lr_rows32_c2.json
```

### 15.1 过拟合判断

先补了 train-side `V_hatπ` 对照。结果：

| readout | train V_hatπ EV | heldout V_hatπ EV |
| --- | ---: | ---: |
| actual critic | 0.868 | 0.863 |
| global linear | 0.935 | 0.888 |
| system_context linear | 0.902 | 0.893 |

这不是典型过拟合。典型过拟合应该是：

```text
train 很高，heldout 明显低。
```

但 actual critic 在 train/heldout 上几乎一样，所以当前更像：

```text
actual critic 在 train 上也没有读干净已有信息。
```

### 15.2 Head-only 对照

固定训练后的 `system_context`，只训练新的 head：

```text
linear
relu MLP
silu MLP
linear + relu residual
```

训练目标仍是 train bank 的 MC return，评估到 heldout `V_hatπ`。

结果：

| head | lr | V_hatπ EV | heldout MC EV | train MC EV |
| --- | ---: | ---: | ---: | ---: |
| linear | 3e-4 | 0.888 | 0.907 | 0.807 |
| linear | 1e-3 | 0.889 | 0.910 | 0.807 |
| relu MLP | 3e-4 | 0.888 | 0.907 | 0.807 |
| relu MLP | 1e-3 | 0.889 | 0.909 | 0.807 |
| silu MLP | 3e-4 | 0.890 | 0.911 | 0.807 |
| silu MLP | 1e-3 | 0.890 | 0.911 | 0.807 |
| linear+relu residual | 3e-4 | 0.887 | 0.905 | 0.807 |
| linear+relu residual | 1e-3 | 0.889 | 0.909 | 0.807 |

结论：

```text
在冻结 system_context 后，head-only 不是主要瓶颈。
ReLU head 没明显输给 linear，也没明显赢过 linear。
```

所以 `system_context linear > actual critic` 更像是完整 critic 联合训练路径没有稳定到最佳读出，而不是“当前 head 结构单独无法表达 linear”。

### 15.3 Full critic LR sweep

同一套固定 train/heldout bank，同一套 heldout `V_hatπ`，只改 full critic 的学习率。

| critic lr | V_hatπ EV | heldout MC EV | train MC EV |
| ---: | ---: | ---: | ---: |
| 1e-4 | 0.827 | 0.860 | 0.782 |
| 3e-4 | 0.899 | 0.913 | 0.808 |
| 1e-3 | 0.886 | 0.898 | 0.802 |
| 3e-3 | 0.903 | 0.918 | 0.802 |

epoch 曲线显示：

```text
1e-4 明显太慢。
3e-4 已经可用。
1e-3 前期更快，但最后不如 3e-4/3e-3。
3e-3 没有明显发散，在这套固定 bank 上最好。
```

当前判断：

```text
critic 不是典型过拟合；
head-only 不是单独表达瓶颈；
full critic 对 LR 明显敏感，当前 3e-4 不是灾难，但 1e-4 太低，3e-3 在固定 MC benchmark 上更好。
```

这还不能直接说明在线 PPO 应该用 `3e-3`，因为在线训练里 value target 会随 policy 和 buffer 变。但它说明：

```text
当前 critic 卡平台至少有一部分是优化/学习率问题，
不是单纯网络没有 Vπ 信息。
```

### 15.4 Target 标准化、AdamW、input norm 对照

在同一套固定 train/heldout/V_hatπ bank 上继续做 optimizer 变体。输出：

```text
runs/diagnostics/critic_head_lr_sweep_20260505/sat_mc_optimizer_variants_rows32_c2.json
```

变体格式：

```text
optimizer:target_mode:lr:weight_decay:input_norm
```

结果：

| variant | V_hatπ EV | heldout MC EV | train MC EV |
| --- | ---: | ---: | ---: |
| adam:raw:3e-4:0:0 | 0.903 | 0.928 | 0.809 |
| adam:norm:3e-4:0:0 | 0.886 | 0.891 | 0.802 |
| adam:norm:1e-3:0:0 | 0.822 | 0.846 | 0.781 |
| adam:norm:3e-3:0:0 | 0.874 | 0.895 | 0.797 |
| adamw:raw:3e-4:1e-4:0 | 0.901 | 0.915 | 0.805 |
| adamw:raw:3e-3:1e-4:0 | 0.908 | 0.939 | 0.800 |
| adamw:norm:3e-4:1e-4:0 | 0.888 | 0.895 | 0.806 |
| adamw:norm:1e-3:1e-4:0 | 0.897 | 0.897 | 0.814 |
| adamw:norm:3e-3:1e-4:0 | 0.886 | 0.915 | 0.812 |
| adam:raw:3e-4:0:1 | 0.828 | 0.822 | 0.802 |
| adam:norm:3e-4:0:1 | 0.873 | 0.890 | 0.814 |

解释：

```text
1. 固定 target normalization 没有改善。
   在这套 benchmark 里，normalized target 普遍不如 raw target。

2. raw input LayerNorm 仍然不好。
   adam:raw:3e-4:input_norm=1 的 V_hatπ EV 从 0.903 掉到 0.828。
   这和之前“raw physical feature 上不要硬 LayerNorm”的判断一致。

3. AdamW 本身不是决定性变化，但 AdamW + raw target + 较高 LR 最好。
   adamw:raw:3e-3:wd=1e-4 达到 V_hatπ EV 0.908、heldout MC EV 0.939。

4. 当前最靠谱的 critic 优化方向不是 dropout，也不是 target 标准化。
   更像是：
   raw target,
   raw input norm 关闭,
   critic_lr 提高到 1e-3~3e-3 区间再做在线小步验证,
   可选 AdamW weight_decay=1e-4。
```

注意：

```text
这仍然只是 fixed-policy / fixed-bank critic 回归结果。
它说明 critic 回归本身能被更好的 optimizer/LR 推高，
但不能直接等价成 PPO 在线训练一定改善。
下一步如果要进在线训练，应优先做短训对照：
  baseline Adam lr=3e-4
  AdamW lr=1e-3 wd=1e-4
  AdamW lr=3e-3 wd=1e-4
并观察 EV、value loss、actor reward 是否同时改善。
```

## 16. 2026-05-05 固定 artifact 下的 block/init 对照

前面不同脚本重复生成 `V_hatπ`，导致 `0.906` 和 `0.872` 这种数字不能直接横比。这里改成先生成固定 artifact：

```text
runs/diagnostics/critic_artifacts_20260505/sat_mc_r8_rows32_c2.pt
```

artifact 固定了：

```text
train_world / train_target
heldout_world / heldout_target
heldout V_hatπ rows
```

后续所有对照只读这个 artifact，不再重复 branch replay。

### 16.1 Blocks=2 vs Blocks=1，三组初始化

共同设置：

```text
stage = sat
reward_mode = sat_relay_processed
critic_lr = 1e-3
critic_epochs = 30
critic_minibatches = 8
```

结果：

| setting | init | actual V_hatπ EV | system_context linear EV | heldout MC EV |
| --- | ---: | ---: | ---: | ---: |
| blocks=2 | 45210 | 0.8721 | 0.8982 | 0.8978 |
| blocks=2 | 45211 | 0.8820 | 0.9024 | 0.8831 |
| blocks=2 | 45212 | 0.8636 | 0.9097 | 0.8762 |
| blocks=1 | 45210 | 0.8984 | 0.9016 | 0.9132 |
| blocks=1 | 45211 | 0.9070 | 0.9158 | 0.9246 |
| blocks=1 | 45212 | 0.8932 | 0.9022 | 0.9183 |

均值：

| setting | actual V_hatπ EV mean | range | system_context linear mean | heldout MC EV mean |
| --- | ---: | ---: | ---: | ---: |
| blocks=2 | 0.8726 | 0.8636-0.8820 | 0.9034 | 0.8857 |
| blocks=1 | 0.8995 | 0.8932-0.9070 | 0.9065 | 0.9187 |

这次是干净对照，因为数据、`V_hatπ`、训练 target 都固定。

### 16.2 Layerwise 解释

固定 artifact 下的 layerwise：

| setting | init | global linear | global_embed linear | block1 linear | block2 linear | actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blocks=1 | 45210 | 0.8885 | 0.9321 | 0.9016 | n/a | 0.8984 |
| blocks=1 | 45211 | 0.8885 | 0.8792 | 0.9158 | n/a | 0.9070 |
| blocks=1 | 45212 | 0.8885 | 0.9047 | 0.9022 | n/a | 0.8932 |
| blocks=2 | 45210 | 0.8885 | 0.9205 | 0.9017 | 0.8982 | 0.8721 |
| blocks=2 | 45211 | 0.8885 | 0.8622 | 0.8913 | 0.9024 | 0.8820 |
| blocks=2 | 45212 | 0.8885 | 0.9238 | 0.9022 | 0.9097 | 0.8636 |

解释：

```text
1. 固定 artifact 后，不再支持“global_scalar_encoder 一定伤信息”的说法。
   global_embed linear 随 init 波动，有时高于 global linear，有时低于 global linear。

2. blocks=1 稳定优于 blocks=2。
   actual critic 的 V_hatπ EV 平均从 0.8726 提到 0.8995。

3. system_context linear 在 blocks=1/2 之间差别不大：
   0.9065 vs 0.9034。

4. 所以 block2 的问题不只是“让 system_context linear 可读性变差”。
   更像是两层 message passing 后，actual value head/联合优化更难把 V 读出来。

5. blocks=1 下 actual critic 已经接近 system_context linear：
   mean actual 0.8995 vs mean system linear 0.9065。
   这比 blocks=2 的 gap 小很多。
```

当前 critic 结构层面的结论：

```text
对当前 SAT fixed-policy critic benchmark，
critic_message_layers=1 比 2 更稳、更好。

这不是由 V_hatπ 重新采样造成的，
因为这次所有 init/block 对照都使用同一个 artifact。
```

### 16.3 value_head_hidden=512 对照

在更好的 `critic_message_layers=1` 上继续测试：

```text
critic_lr = 1e-3
critic_epochs = 30
critic_value_head_hidden = 512
init = 45210 / 45211 / 45212
```

和 256 对比：

| hidden | init | actual V_hatπ EV | system_context linear EV | heldout MC EV |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 45210 | 0.8984 | 0.9016 | 0.9132 |
| 256 | 45211 | 0.9070 | 0.9158 | 0.9246 |
| 256 | 45212 | 0.8932 | 0.9022 | 0.9183 |
| 512 | 45210 | 0.8727 | 0.9002 | 0.8703 |
| 512 | 45211 | 0.8923 | 0.9117 | 0.9108 |
| 512 | 45212 | 0.8811 | 0.9240 | 0.9080 |

均值：

| hidden | actual V_hatπ EV mean | range | heldout MC EV mean |
| ---: | ---: | ---: | ---: |
| 256 | 0.8995 | 0.8932-0.9070 | 0.9187 |
| 512 | 0.8820 | 0.8727-0.8923 | 0.8964 |

结论：

```text
value_head_hidden=512 没有稳定更好，反而三组 init 都低于 256。
所以当前不把 512 设为默认。

critic_message_layers=1 已经稳定优于 2，
因此默认值改成 1。
```

## 17. 2026-05-05 blocks=1 后续 advantage/update probe

使用目前较好的 critic 设置：

```text
stage = sat
reward_mode = sat_relay_processed
target = MC
critic_lr = 1e-3
critic_message_layers = 1
critic_value_head_hidden = 256
critic_epochs = 30
train_rollouts = 8
advantage_probe_rows = 16
policy_action_samples = rollout action + 4 policy samples
continuations = 4
follow = stochastic
```

输出：

```text
runs/diagnostics/critic_artifacts_20260505/sat_blocks1_advantage_probe_r16.json
```

critic fit：

| metric | value |
| --- | ---: |
| heldout MC EV | 0.913 |
| heldout MC corr | 0.960 |
| heldout MC MAE | 8.155 |

advantage 和同状态 Qπ rollout action advantage 的对齐：

| item | value |
| --- | ---: |
| corr(A_mc_norm, A_qpi_rollout) | -0.144 |
| sign agree(A_mc_norm, A_qpi_rollout) | 0.3125 |
| corr(A_gae_norm, A_qpi_rollout) | -0.002 |
| sign agree(A_gae_norm, A_qpi_rollout) | 0.3125 |
| qpi_mean_adv_scale | 0.709 |
| q_value_baseline_error_scale | 11.402 |
| baseline_error / action_gap_scale | 16.09 |

这里的关键不是 critic 完全没学。相反，critic 对 heldout MC return 已经拟合得不错。但对 actor credit 来说：

```text
V baseline 的剩余误差尺度仍然约为同状态 action gap 的 16 倍。
```

所以 `G - V` 在这批 row 上没有稳定还原当前 action 相对同状态其他 action 的优势。

actor probe：

| update advantage | q-shift | corr(delta_logprob, Qπ adv) | per-state positive | best up | margin up |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_mc | 0.0254 | 0.097 | 0.625 | 0.625 | 0.625 |
| A_gae(V) | 0.0296 | 0.098 | 0.6875 | 0.625 | 0.750 |
| A_qpi selected | 0.0355 | 0.104 | 0.6875 | 0.5625 | 0.5625 |
| A_qpi all-candidate | 0.0358 | 0.109 | 0.6875 | 0.625 | 0.625 |

解释：

```text
1. A_mc / A_gae 和 A_qpi 的逐样本相关仍然不好。
2. 但它们做一次 actor probe 后 q-shift 为正，说明平均更新不是完全反向。
3. A_qpi all-candidate 仍然是最直接、最稳定的更新信号。
4. 当前主要瓶颈不是 critic EV 完全学不上去，而是 residual baseline error 相对 action gap 仍太大。
```

当前不能推出：

```text
blocks=1 + MC critic 就能在线训练好。
```

只能推出：

```text
blocks=1 明显改善 critic fit；
但用单条 rollout 的 G - V 做 action credit，仍然很吵。
```

### 17.1 actor probe lr sweep

同一批 rows、同一批 Qπ branch returns、同一个训练好的 blocks=1 critic 下，扫 actor probe lr：

```text
1e-4 / 3e-4 / 1e-3 / 3e-3
```

输出：

```text
runs/diagnostics/critic_artifacts_20260505/sat_blocks1_advantage_probe_lrs_r16.json
```

结果：

| lr | update | q-shift | per-state positive | best up | margin up |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1e-4 | A_mc | 0.0100 | 0.6875 | 0.6250 | 0.6250 |
| 1e-4 | A_gae(V) | 0.0118 | 0.6875 | 0.6250 | 0.6875 |
| 1e-4 | A_qpi selected | 0.0149 | 0.7500 | 0.6250 | 0.5625 |
| 1e-4 | A_qpi all | 0.0155 | 0.7500 | 0.6250 | 0.5625 |
| 3e-4 | A_mc | 0.0259 | 0.6875 | 0.6250 | 0.6250 |
| 3e-4 | A_gae(V) | 0.0302 | 0.7500 | 0.6250 | 0.7500 |
| 3e-4 | A_qpi selected | 0.0362 | 0.7500 | 0.5625 | 0.5625 |
| 3e-4 | A_qpi all | 0.0365 | 0.7500 | 0.6250 | 0.6250 |
| 1e-3 | A_mc | 0.0499 | 0.7500 | 0.3125 | 0.8125 |
| 1e-3 | A_gae(V) | 0.0593 | 0.7500 | 0.3125 | 0.7500 |
| 1e-3 | A_qpi selected | 0.0693 | 0.7500 | 0.3125 | 0.6875 |
| 1e-3 | A_qpi all | 0.0676 | 0.7500 | 0.3125 | 0.7500 |
| 3e-3 | A_mc | 0.0847 | 0.6875 | 0.3125 | 0.6875 |
| 3e-3 | A_gae(V) | 0.1008 | 0.7500 | 0.3125 | 0.6875 |
| 3e-3 | A_qpi selected | 0.1266 | 0.6875 | 0.3125 | 0.6250 |
| 3e-3 | A_qpi all | 0.1177 | 0.7500 | 0.3125 | 0.6250 |

解释：

```text
1. actor_lr=3e-4 时，A_qpi all 的 q-shift 最大，但只比 A_gae/A_mc 大一截，
   不是数量级差异。

2. 增大 actor_lr 会放大所有 q-shift，
   但 best-up 从 0.625 降到 0.3125。
   所以不能用“加 actor_lr”来解决 credit 噪声。

3. A_mc/A_gae 的逐样本 sign agree 低，
   但 batch 平均 q-shift 为正。
   这说明它们不是固定反向，而是 noisy/弱正。

4. A_qpi all-candidate 仍是最干净的参考，
   但由于 actor 参数共享、分布约束和单步小更新，
   q-shift 不会比其他方法大一个数量级。
```
## 18. Vhat future random tape 口径修正

之前的 branch/Vhat replay 默认复用了 official rollout history 里的未来 random tape。
这对 same-noise action contrast 是合理的，但对 `Vπ(s)` 审计不对，因为它估计的是：

```text
E_policy_action,continuation [ G | s, fixed future arrival/hotspot/fading/doppler tape ]
```

而不是更接近 critic 需要的：

```text
Vπ(s) = E_future_noise,policy [ G | s ]
```

这会让 `Vhat` 里混入当前 actor/critic 输入看不到的未来 tape 信息，导致一些 probe 看起来像
“critic 对某些 heldout seed 学不好”，但其中一部分其实是 label 口径被固定未来随机量污染。

代码修正：

```text
sagin_marl/env/structured_batch_env_core.py:
  prepare_native_branch_replay_from_history(..., future_random_mode="copy|resample")

scripts/audit_stage_qpi_action_credit.py:
  _branch_returns_qpi(..., future_random_mode="copy|resample")

scripts/audit_stage_critic_only_fit.py:
  Vπ ceiling probe 默认 future_random_mode="resample"

scripts/audit_critic_vhat_artifact.py:
  --future_random copy|resample，默认 resample
```

语义：

```text
copy:
  旧行为。窗口复制 rollout history 中同一段未来 tape。
  用于 exact same-noise action contrast / vs-ref gate。

resample:
  先从 history 恢复 stage snapshot，再基于恢复后的 branch state
  重新生成未来 arrival/hotspot/fading/doppler rollout tape 和 reset followup tape。
  用于估计 Vπ / critic ceiling。
```

注意：`resample` 只接入 branch replay/audit，不改变 PPO 训练 rollout 热路径。

smoke：

```text
copy smoke:
  runs/diagnostics/smoke_vhat_copy.pt
  vhat = 1.3242
  vhat_se = 0.0474

resample smoke:
  runs/diagnostics/smoke_vhat_resample.pt
  vhat = 0.8959
  vhat_se = 0.1455
```

两者不同是预期结果：`copy` 固定未来 tape，方差更小；`resample` 对未来随机性重新积分，才更接近
critic 的 `Vπ(s)` 目标。

## 19. resampled Vhat 只做评估，不作为 critic 训练标签

这里修正一个容易混掉的口径：

```text
critic 训练仍然只能用普通 rollout 数据：
  MC return 或 train-time GAE return

resampled Vhatπ(s) 只作为 heldout evaluation target：
  用来判断 critic 从普通 rollout 数据里学到的东西，
  是否接近固定 policy 下的条件均值 Vπ(s)。
```

本轮重新生成 artifact：

```text
runs/diagnostics/critic_relearn_20260506/sat_vhat_resample_s5_r3_a3_c4.pt
stage = SAT
reward = sat_relay_processed
heldout seeds = 245210,255210,305210,9045210,9055210
rows_per_seed = 3
policy action count = rollout action + 3 samples = 4
continuations = 4
future_random = resample
```

Vhat 统计：

| item | value |
| --- | ---: |
| rows | 15 |
| Vhat mean | 76.61 |
| Vhat std | 25.76 |
| Vhat SE mean | 2.79 |
| single MC mean | 74.50 |
| single MC - Vhat std | 16.85 |

然后用同一份 heldout Vhat，比较两种“普通 rollout 训练标签”。

### 19.1 critic 用 MC return 训练

命令输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_vhat_resample_eval_mc_r8_e30_lr1e3_relational.json
train_target = MC
train_rollouts = 8
critic_epochs = 30
critic_lr = 1e-3
critic_value_mode = relational
critic_message_layers = 1
critic_value_head_hidden = 512
```

结果：

| metric | value |
| --- | ---: |
| train MC EV | 0.867 |
| critic vs resampled Vhat EV | 0.913 |
| critic vs resampled Vhat corr | 0.957 |
| critic vs single MC EV | 0.370 |
| global-linear baseline vs Vhat EV | 0.864 |

解释：

```text
1. critic 不是在“直接拟合 Vhat”。
   它只用普通 MC rollout target 训练。

2. 训练后对 resampled Vhat 的 EV=0.913，
   高于对单条 MC return 的 EV=0.370。

3. 这说明 critic 确实更像学到了条件均值结构，
   而不是记住单条 future noise。
```

### 19.2 critic 用当前 train-time GAE target 训练

命令输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_vhat_resample_eval_gae_r8_e30_lr1e3_relational.json
train_target = train_gae
其他参数同上
```

结果：

| metric | value |
| --- | ---: |
| train GAE EV | 0.823 |
| critic pred mean on Vhat rows | 4.25 |
| resampled Vhat mean | 76.61 |
| critic vs resampled Vhat EV | -0.042 |
| critic vs resampled Vhat corr | -0.244 |

解释：

```text
1. critic 能拟合 train-time GAE target 本身，train EV=0.823。
2. 但这个 GAE target 的尺度和 full finite-horizon Vπ 明显不一致。
3. 所以当前 PPO 口径下 critic 学到的是短尺度/bootstrapped label，
   不是 full-horizon Vπ。
```

当前更稳的结论：

```text
critic 结构/输入不是完全学不了固定 policy 的 Vπ；
用 MC return 训练时，它对 resampled Vhatπ(s) 的评估已经能到 EV≈0.91。

当前 PPO 使用的 train-time GAE target 才是更大的断点：
它能被 critic 拟合，但不接近 full finite-horizon Vπ。
```

## 20. MC-trained critic 的 A_mc actor-credit 检查

这一步不再问 critic 能不能学 `Vπ(s)`，而是问：

```text
A_mc(s,a) = G_mc(s,a,future_sample) - Vθ(s)
```

作为 actor 更新信号时，是否会提高同状态下更高 `Qπ` action 的 logprob。

设置：

```text
stage = SAT
reward = sat_relay_processed
critic train target = MC return
train_rollouts = 8
critic_epochs = 30
critic_lr = 1e-3
critic_value_mode = relational
critic_message_layers = 1
critic_value_head_hidden = 512
advantage probe rows = 12
policy action count = rollout action + 3 samples = 4
continuations = 4
```

critic 训练结果：

```text
heldout MC EV = 0.8796
```

### 20.1 与 resampled expected-Q 对照

输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_mc_advantage_resample_r12.json
advantage_future_random = resample
```

alignment：

| metric | value |
| --- | ---: |
| corr(A_mc_norm, A_qpi_rollout) | -0.099 |
| sign agree(A_mc_norm, A_qpi_rollout) | 0.333 |
| q_value_baseline_error / action_gap_scale | 2.61 |

actor probe：

| lr | update | q-shift | per-state positive | best-up |
| ---: | --- | ---: | ---: | ---: |
| 1e-4 | A_mc | -0.035 | 0.333 | 0.167 |
| 1e-4 | A_gae(V) | 0.045 | 0.667 | 0.667 |
| 1e-4 | A_qpi all-candidate | 0.058 | 0.667 | 0.833 |
| 3e-4 | A_mc | -0.095 | 0.333 | 0.167 |
| 3e-4 | A_gae(V) | 0.107 | 0.667 | 0.667 |
| 3e-4 | A_qpi all-candidate | 0.139 | 0.667 | 0.750 |
| 1e-3 | A_mc | -0.167 | 0.333 | 0.167 |
| 1e-3 | A_gae(V) | 0.228 | 0.667 | 0.333 |
| 1e-3 | A_qpi all-candidate | 0.258 | 0.667 | 0.500 |

解释：

```text
在这个 heldout probe 里，A_mc 不仅弱，而且平均方向为负。
这说明“critic 能学 Vπ”并不自动等于“单条 MC residual 能给 actor 好 credit”。
```

### 20.2 same-noise copy 对照

输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_mc_advantage_copy_r12.json
advantage_future_random = copy
```

结果：

| metric | value |
| --- | ---: |
| corr(A_mc_norm, A_qpi_rollout) | -0.299 |
| sign agree(A_mc_norm, A_qpi_rollout) | 0.583 |
| q_value_baseline_error / action_gap_scale | 13.85 |
| A_mc q-shift @1e-4 | -0.0056 |
| A_qpi all-candidate q-shift @1e-4 | 0.0077 |

解释：

```text
same-noise 口径下 A_mc 也没有变成稳定正向。
所以这不是单纯 “resampled expected-Q 太难” 的问题。
```

当前结论：

```text
MC-trained critic 可以学到 Vπ 的大结构；
但 A_mc = single MC return - Vθ(s) 作为 actor credit 仍然不可用或至少很不稳定。

如果继续走 critic 路线，更可能要用 MC-trained Vθ 计算更局部/更短噪声的 GAE(V)，
而不是直接 full-horizon MC residual。

如果目标是最干净的 actor credit，A_qpi all-candidate 仍然最直接，但成本高。
```

## 21. policy 更新一次后 critic 重新跟踪需要多大规模

这一步检查的是：

```text
先在 π0 下用 MC return 训练 critic；
用这个 critic 计算 A_gae(V)，做一次 SAT actor full-stage PPO 风格更新，得到 π1；
然后在 π1 下重新收 MC rollout，训练 critic；
评估 critic 重新接近 π1 下 resampled Vhatπ(s) 需要多少数据/epoch。
```

设置：

```text
stage = SAT
reward = sat_relay_processed
pre critic target = MC return
pre_train_rollouts = 8
pre_critic_epochs = 30
actor update = A_gae(V), full-stage, 1 epoch, 1 minibatch
actor lr = 3e-4
critic lr = 1e-3
critic_value_mode = relational
critic_message_layers = 1
critic_value_head_hidden = 512
Vhat probe = resampled future random
```

π0 critic 训练结果：

```text
pre train EV = 0.891
pre heldout EV = 0.868
```

actor 更新强度：

```text
update_samples = 2000
q-shift vs Qπ probe = 0.218
best-up = 0.583
margin-up = 0.500
```

这不是完全微小的 no-op update。

### 21.1 π1 下 post_train_rollouts = 8

输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_relearn_after_gae_fullppo_post8.json
```

| epoch | train MC EV | heldout MC EV | resampled Vhat EV | Vhat MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.862 | 0.756 | 0.750 | 8.39 |
| 1 | 0.781 | 0.634 | 0.687 | 53.47 |
| 3 | 0.843 | 0.752 | 0.819 | 13.17 |
| 5 | 0.882 | 0.792 | 0.960 | 5.70 |
| 10 | 0.918 | 0.801 | 0.827 | 5.64 |
| 20 | 0.942 | 0.766 | 0.918 | 12.39 |
| 30 | 0.955 | 0.771 | 0.848 | 7.69 |

### 21.2 π1 下 post_train_rollouts = 4

输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_relearn_after_gae_fullppo_post4.json
```

| epoch | train MC EV | heldout MC EV | resampled Vhat EV | Vhat MAE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.879 | 0.756 | 0.750 | 8.39 |
| 1 | 0.844 | 0.647 | 0.671 | 16.42 |
| 3 | 0.917 | 0.721 | 0.819 | 13.78 |
| 5 | 0.891 | 0.755 | 0.878 | 11.31 |
| 10 | 0.937 | 0.652 | 0.847 | 32.60 |
| 20 | 0.957 | 0.685 | 0.804 | 8.25 |
| 30 | 0.971 | 0.739 | 0.860 | 6.11 |

解释：

```text
1. policy 更新一次后，旧 critic 在 π1 上并没有完全失效：
   epoch 0 的 Vhat EV 已经约 0.75。

2. 重新训练不需要几十个 epoch 才开始有用：
   4-8 rollouts 下，epoch 3-5 已经能回到比较可用的区间。

3. 继续训练到 20-30 epoch 会提高 train MC EV，
   但 heldout MC / Vhat 不稳定，并不明显更好。

4. 当前 Vhat rows 只有 8，单点 EV 会抖；
   所以不要把 epoch 5 的 0.960 解读成精确最优，
   更稳的结论是“几轮 epoch 足够恢复，长训收益不稳定”。
```

当前成本判断：

```text
如果采用 MC-trained critic + A_gae(V) actor update，
每次 policy update 后 critic 跟踪的量级大致是：

post rollout:
  4-8 个 rollout，每个 rollout = 8 env × 250 step

critic epochs:
  3-5 epoch 已经足够进入可用区；
  不建议默认 20-30 epoch。
```

这还不是最终训练方案，只说明：

```text
critic 跟踪 π 更新的成本没有高到“必须几百 epoch”；
更大的问题是怎样把这套 MC critic + A_gae(V) 稳定接入在线训练。
```

## 22. 更接近在线 PPO 的单批流程：64 env rollout

上面的 `pre_train_rollouts=8` 和 actor update rollout 是分开的，用于隔离审计。
真正在线训练更应该是：

```text
collect one rollout batch
用这同一批 rollout 的 MC return 训练 critic
用更新后的 critic 重新算 A_gae(V)
用这同一批 rollout 更新 actor
再收新 policy 的 rollout，训练 critic 跟踪
```

为了让一批 rollout 有足够样本，测试了：

```text
num_envs = 64
rollout_env_steps = 250
SAT stage samples = 64 × 250 = 16000
```

64 env native path 已经 smoke 和完整审计跑通。

### 22.1 未开启 advantage normalization 的错误口径

输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_online64_refit20_actor5_relearn.json
actor_advantage_normalize_enabled = false
stagewise_advantage_norm_enabled = false
```

结果：

```text
critic refit on actor batch:
  20 epochs 后 train EV = 0.841

actor update:
  actor_update_epochs = 5
  selected A_gae mean = -45.17
  selected A_gae std = 20.32
  q-shift = -0.173
  best-up = 0.25
```

解释：

```text
这个口径不是我们想要的 PPO actor update。
整批 advantage 没有中心化，样本 advantage 大多为负，
导致 full-stage update 在降低许多 sampled action 的概率。
```

### 22.2 开启 stagewise advantage normalization 后

输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_online64_refit20_actor5_advnorm_relearn.json
actor_advantage_normalize_enabled = true
stagewise_advantage_norm_enabled = true
```

## 23. 64 env 与 8 rollout 的同口径对照

这次要澄清的问题是：

```text
8 rollouts × 8 envs 和 1 rollout × 64 envs 样本数一样，
为什么之前 critic refit 20 epoch 只有 train EV≈0.818，
而早前小实验看起来有 0.89+？
```

对照脚本输出：

```text
runs/diagnostics/critic_relearn_20260506/sat_mc_bank_shape_compare_20260506.json
```

共同条件：

```text
stage = sat
reward = sat_relay_processed
target = MC return
critic_value_mode = relational
critic_message_layers = 1
critic_value_head_hidden = 512
critic_lr = 1e-3
samples = 16000
```

结果：

| bank | seed_base | epoch20 train EV | epoch30 train EV |
| --- | ---: | ---: | ---: |
| 8 env × 8 rollout | 45210 | 0.841 | 0.867 |
| 64 env × 1 rollout | 45210 | 0.862 | 0.878 |
| 64 env × 1 rollout | 945210 | 0.825 | 0.843 |

结论：

```text
1. 64 env × 1 rollout 本身不是问题。
   同 seed_base=45210 时，它不比 8 env × 8 rollout 差。

2. 之前的 0.818 主要对齐的是 actor-batch 那类 seed bank，
   并且只训练到 epoch20。
   这次 64 env × 1 rollout, seed_base=945210, epoch20 为 0.825，
   与 0.818 基本一致。

3. 早前说的 0.89+ 不是同一个指标：
   0.867 是 8×8 bank 的 epoch30 train MC EV；
   0.913 是训练好的 critic 对 resampled Vhatπ artifact 的 EV。
   不能直接拿它和 actor-batch refit epoch20 train EV 比。
```

所以当前更准确的说法是：

```text
critic refit 20 epoch 到 0.818/0.825 并不是 64-env 采样方式坏了；
它是 seed bank/工作点差异 + epoch 数差异。
如果按同 seed_base 比，64 env 与 8 rollout 没有发现结构性劣化。
```

## 24. actor 更新 5 epoch 后，critic 是否能跟上

这次用同一套更明确的统计方式重新检查：

```text
runs/diagnostics/critic_relearn_20260506/sat_online64_actor5_vhat24_relearn_20260506.json
```

设置：

```text
stage = sat
reward = sat_relay_processed
num_envs = 64
rollout_env_steps = 250

critic refit before actor:
  target = MC return
  epochs = 20
  minibatches = 8

actor update:
  advantage = normalized stagewise A_gae(V)
  full-stage samples = 16000
  actor_update_epochs = 5
  actor_update_minibatches = 1
  actor_lr = 3e-4

post-update Vhat probe:
  rows = 24
  actions = rollout action + 4 policy samples
  continuations = 4
  future_random_mode = resample
```

actor update 方向：

```text
corr(selected A_gae, Qπ rollout adv) = 0.493
sign agree = 0.750
q-shift = +0.126
best-up = 0.583
margin-up = 0.750
```

actor 更新后，在 π1 下用同一批 resampled Vhat rows 评估 critic：

| critic relearn epoch | heldout MC EV | resampled Vhat EV | Vhat MAE |
| ---: | ---: | ---: | ---: |
| 0 | 0.811 | 0.907 | 10.53 |
| 1 | 0.767 | 0.833 | 23.58 |
| 3 | 0.813 | 0.879 | 18.87 |
| 5 | 0.818 | 0.893 | 18.01 |
| 10 | 0.830 | 0.919 | 14.08 |
| 20 | 0.839 | 0.911 | 10.33 |

结论：

```text
1. 如果用 resampled Vhatπ 作为评估标准，
   actor_update_epochs=5 后 critic 并没有明显跟不上。

2. epoch0 已经有 Vhat EV≈0.907，
   说明这次 actor 更新幅度不大，π0 critic 在 π1 下仍然可用。

3. 继续在 π1 rollout 上用 MC target 训练，
   epoch5 Vhat EV≈0.893，epoch10≈0.919，epoch20≈0.911。
   所以这组 evidence 不支持“critic 需要非常多 epoch 才能重新跟上”。

4. heldout MC EV 和 Vhat EV 不完全同步。
   判断 Vπ 是否学到，应优先看同一批 resampled Vhat rows；
   heldout MC EV 更多反映单条 continuation 噪声下的拟合情况。
```

设置：

```text
critic refit on actor batch:
  MC target
  20 epochs
  8 minibatches

actor update:
  A_gae(V)
  full-stage samples = 16000
  actor_update_epochs = 5
  actor_update_minibatches = 1
  actor_lr = 3e-4
```

结果：

```text
critic refit on actor batch:
  20 epochs 后 train EV = 0.818

actor update:
  selected A_gae mean = -0.086
  selected A_gae std = 0.728
  corr(selected A_gae, Qπ rollout adv) = 0.469
  sign agree = 0.750
  q-shift = +0.166
  per-state positive = 0.750
  best-up = 0.667
  margin-up = 0.667
```

这说明：

```text
MC-trained critic + normalized A_gae(V)
在一整批 16000 SAT samples 上做 5 epoch actor update，
这次是朝 Qπ probe 的正方向推的。
```

### 22.3 actor 更新后 critic 跟踪

同一组 `advnorm` 实验里，π1 下重新收一批 64 env rollout，然后训练 critic：

| epoch | heldout MC EV | resampled Vhat EV | Vhat MAE |
| ---: | ---: | ---: | ---: |
| 0 | 0.806 | 0.859 | 14.49 |
| 1 | 0.734 | 0.755 | 11.00 |
| 3 | 0.812 | 0.753 | 19.02 |
| 5 | 0.816 | 0.794 | 12.15 |
| 10 | 0.830 | 0.828 | 19.19 |

解释：

```text
1. actor 更新后，旧 critic 在 π1 下并没有崩：
   epoch 0 already has heldout MC EV ≈ 0.806, Vhat EV ≈ 0.859。

2. 用一批 64 env rollout 继续训练 3-5 epoch，
   heldout MC EV 维持在约 0.81 左右。

3. Vhat EV/MAE 会抖，因为 Vhat rows 仍只有 8；
   但没有出现“必须大量 rollout/几十 epoch 才能跟上”的迹象。
```

当前在线方案的更精确版本：

```text
first actor update:
  collect 64 env × 250 step
  critic MC train about 20 epochs
  recompute normalized stagewise A_gae(V)
  actor full-stage PPO update, 5 epochs, 1 minibatch

later updates:
  collect 64 env × 250 step
  critic MC train about 3-5 epochs may be enough
  recompute normalized stagewise A_gae(V)
  actor update
```

关键要求：

```text
actor_advantage_normalize_enabled = true
stagewise_advantage_norm_enabled = true
```
