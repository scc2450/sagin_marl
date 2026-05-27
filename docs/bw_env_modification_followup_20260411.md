# BW Environment Modification Follow-up (2026-04-11)

这份文档整理的是：在收到“环境应该围绕 BW leverage 重新设计”的意见之后，围绕 `BW-only + T=10 + 1 UAV + 5 GU` 这条线做过的所有主要环境改动、对应的实现、数值结果、尚未做的项，以及对后续方向的判断。

关联文档：
- `docs/bw_env_design_v1_v2_20260411.md`
- `docs/bw_critic_path_feedback_experiments_20260411.md`

当前主用基线配置：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo.yaml`

目标不是把环境一次改成最终联合系统，而是先回答两个更具体的问题：

1. 未来 joint 也合理的前提下，能不能把 `BW leverage` 做强、做稳、做得 actor 看得见。
2. 即使 leverage 变强了，它对当前 plain `state-value PPO` 来说，到底强得够不够。


## 1. 判断标准

这条线里其实一直有两个不同的判断标准，后面必须分开看。

### 1.1 宽标准：环境里是否存在可利用信号

这看的是：
- `planner` 能不能比 `heuristic` 更好
- `learned` 能不能至少好于 `fresh init`
- `slot_gap_mean`、`mean_abs_delta_vs_heuristic` 这些 leverage 指标是否明显增大

这个标准回答的是：

`环境里有没有真实 leverage`

### 1.2 严标准：这个信号对 plain state-value PPO 是否够大

这看的是：
- 动作层收益量级
  - `policy_local_reward_gap`
  - `branch_delta_abs_mean`
  - `h10 mean_abs_delta_vs_heuristic`
- critic 误差量级
  - online critic holdout `RMSE`
  - fresh supervised critic holdout `RMSE`

这个标准回答的是：

`这个 leverage 会不会继续被 V(s) 误差盖住`

后面的结论里，两种标准会分开写。


## 2. 改动总览

主要代码改动集中在：
- `sagin_marl/env/config.py`
- `sagin_marl/env/sagin_env.py`
- `sagin_marl/env/structured_driver.py`
- `scripts/diagnose_structured_bw_env_leverage.py`

主要新增/使用的配置：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_only.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_leverage_only.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v11_norelay.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v11_relay01.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v11.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v12_urgency50.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v13_servicegap.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild.yaml`


## 3. 基线环境

基线是：
- `1 UAV`
- `5 GU`
- `T_steps = 10`
- `train_accel = false`
- `train_sat = false`
- `train_bw = true`
- `exec_accel_source = zero`
- `exec_sat_source = zero`
- `exec_bw_source = policy`
- `traffic_model = sticky_subset_hotspot`
- `hotspot_rho = 8`
- `hotspot_on/off_mean_steps = 20 / 5`
- `queue_max_gu_steps = 30`
- `reward_w_relay = 0`

这个基线的问题是：
- leverage 有，但不强
- actor 只能看到 `queue + eta` 这类量，缺少更直接的 traffic/service proxy
- 没有更强的 per-GU urgency 机制

对应 leverage 基线结果在：
- `runs/analysis/bw_env_leverage_ablation_20260411/summary.csv`
- `runs/analysis/bw_env_leverage_compare_20260411/structured_bw_sanity_1uav_static_gap_debug_t10_ppo/summary.json`

关键数值：

| config | h2 slot_gap | h5 slot_gap | h10 slot_gap | h10 mean_abs_delta_vs_heuristic | h10 heuristic_minus_uniform | h5->h10 corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.1065 | 0.0847 | 0.0789 | 0.0444 | 0.0342 | 0.9746 |


## 4. 改动 A：只补 observation proxy

### 4.1 想解决的问题

之前的意见里，一个关键点是：

`差异不只是要存在，还要 actor 看得见`

所以第一步先只补 actor 可见的 proxy，不碰环境动力学，检查这些特征是否“偷改了环境”。

### 4.2 具体实现

在 `config.py` / `sagin_env.py` 里新增并接入：
- `obs_user_include_arrival_rate`
- `obs_user_include_recent_arrival`
- `obs_user_include_recent_service`
- `obs_user_include_queue_headroom`

对应配置：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_only.yaml`

### 4.3 结果

`obsproxy_only` 和 `base` 的 leverage 指标完全一样：

| config | h10 slot_gap | h10 mean_abs_delta_vs_heuristic | h10 heuristic_minus_uniform | h5->h10 corr |
| --- | ---: | ---: | ---: | ---: |
| base | 0.0789 | 0.0444 | 0.0342 | 0.9746 |
| obsproxy_only | 0.0789 | 0.0444 | 0.0342 | 0.9746 |

### 4.4 结论

这是好结果。

它说明：
- 这些 proxy 没有篡改环境
- 它们只会影响未来 actor 的可观测性
- 可以保留，因为未来 joint 也合理


## 5. 改动 B：温和增强 leverage 本身

### 5.1 想解决的问题

仅靠观测增强不够，环境里的 marginal-value gap 还偏小，所以尝试温和改以下旋钮：
- 更强更久的 hotspot
- 更紧的 GU 队列
- 少量 downstream 影响

### 5.2 具体实现

相对基线：
- `hotspot_rho: 8 -> 12`
- `hotspot_on_mean_steps: 20 -> 35`
- `hotspot_off_mean_steps: 5 -> 8`
- `queue_max_gu_steps: 30 -> 24`

先做纯动力学版：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_leverage_only.yaml`

再和 observation proxy 合并：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v11_norelay.yaml`

再尝试小幅 relay：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v11_relay01.yaml`
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v11.yaml`

### 5.3 结果

对应结果：
- `runs/analysis/bw_env_leverage_ablation_20260411/summary.csv`
- `runs/analysis/bw_env_leverage_relay01_20260411/summary.csv`

关键数值：

| config | h10 slot_gap | h10 mean_abs_delta_vs_heuristic | h10 heuristic_minus_uniform | h5->h10 corr |
| --- | ---: | ---: | ---: | ---: |
| base | 0.0789 | 0.0444 | 0.0342 | 0.9746 |
| leverage_only | 0.0786 | 0.0299 | 0.0210 | 0.8096 |
| v11_norelay | 0.0786 | 0.0299 | 0.0210 | 0.8096 |
| v11_relay01 | 0.0904 | 0.0346 | 0.0232 | 0.8080 |
| v11 (relay=0.2) | 0.1023 | 0.0393 | 0.0254 | 0.8071 |

注：
- `v11_norelay` 本质上就是 `leverage_only + obsproxy`
- `slot_gap_mean` 和 `mean_abs_delta_vs_heuristic` 是不同口径，前者不是简单复写错误

### 5.4 结论

这轮的结论是：
- proxy 本身不改 leverage
- 真改 leverage 的是 `hotspot / queue / relay`
- `reward_w_relay` 小调只是插值，不是主解
- `v11` 的短时 leverage 变强了，但 persistence 明显变弱了

也就是说，这一轮是有帮助的，但还不够干净。


## 6. 改动 C：soft overflow-risk urgency

### 6.1 想解决的问题

前面的问题是：局部差异虽然变大了一点，但“现在不服务这个 GU，后面会更亏”这层 urgency 还不够直接。

### 6.2 具体实现

在 `config.py` / `sagin_env.py` 中加入：
- `obs_user_include_urgency_risk`
- `reward_w_pre_overflow_risk`
- `overflow_risk_threshold_frac`
- `overflow_risk_arrival_coef`
- `overflow_risk_service_coef`

对应配置：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v12_urgency50.yaml`

### 6.3 结果

对应结果：
- `runs/analysis/bw_env_leverage_urgency50_smooth_driverfix_20260411/summary.csv`

关键数值：

| config | h10 slot_gap | h10 mean_abs_delta_vs_heuristic | h10 heuristic_minus_uniform | h5->h10 corr |
| --- | ---: | ---: | ---: | ---: |
| v11_norelay | 0.0786 | 0.0299 | 0.0210 | 0.8096 |
| v12_urgency50 | 0.0801 | 0.0306 | 0.0228 | 0.8123 |

### 6.4 结论

方向是正的，但提升很小。

说明：
- soft risk proxy 本身不是错方向
- 但它不足以把 leverage 推到一个新量级


## 7. 改动 D：soft service-gap / latency-risk

### 7.1 想解决的问题

soft overflow risk 还是偏“快照”，所以又尝试加入“持续没被充分服务”的状态。

### 7.2 具体实现

在 `config.py` / `sagin_env.py` 中加入：
- `obs_user_include_service_gap`
- `obs_user_include_service_gap_risk`
- `reward_w_pre_service_gap`
- `service_gap_increment`
- `service_gap_relief_coef`
- `service_gap_cap_steps`
- `service_gap_risk_threshold_steps`

对应配置：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v13_servicegap.yaml`

### 7.3 结果

对应结果：
- `runs/analysis/bw_env_leverage_v13_servicegap_20260411/summary.csv`

关键数值：

| config | h10 slot_gap | h10 mean_abs_delta_vs_heuristic | h10 heuristic_minus_uniform | h5->h10 corr |
| --- | ---: | ---: | ---: | ---: |
| v11_norelay | 0.0786 | 0.0299 | 0.0210 | 0.8096 |
| v13_servicegap | 0.0786 | 0.0299 | 0.0212 | 0.8086 |

### 7.4 结论

这条线不是突破口。

`service-gap` 方向上可解释，但没有把 leverage 真正拉起来。


## 8. 改动 E：真实 deadline / slack / expiry 动力学

### 8.1 想解决的问题

前面的 urgency 都还是 proxy。要想真正让“不给某个 GU 带宽会更亏”这件事更硬，就需要真实动力学，而不是再加软 shaping。

### 8.2 具体实现

在 `config.py` / `sagin_env.py` 中加入：
- `deadline_enabled`
- `deadline_base_steps`
- `deadline_jitter_steps`
- `deadline_age_increment`
- `deadline_service_relief_coef`
- `deadline_expire_rate`
- `deadline_age_cap_steps`
- `obs_user_include_deadline_slack`
- `obs_user_include_deadline_risk`

环境里新增了：
- per-GU `deadline_age`
- per-GU `deadline_slack`
- per-GU `deadline_risk`
- 过期后真实 `expire` / `drop` 进入队列和 reward 链条

对应配置：
- 强版：`configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline.yaml`
- 温和版：`configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild.yaml`

### 8.3 结果

对应结果：
- `runs/analysis/bw_env_leverage_v14_deadline_mild_20260411/summary.csv`

关键数值：

| config | h10 slot_gap | h10 mean_abs_delta_vs_heuristic | h10 heuristic_minus_uniform | h5->h10 corr |
| --- | ---: | ---: | ---: | ---: |
| v11_norelay | 0.0786 | 0.0299 | 0.0210 | 0.8096 |
| v14_deadline | 0.2755 | 0.1039 | 0.0344 | 0.8337 |
| v14_deadline_mild | 0.1643 | 0.0596 | 0.0459 | 0.9233 |

### 8.4 结论

这是这条环境线里最有效的一次改动。

强版说明：
- 真实 deadline/expiry 的确能把 leverage 明显做大

温和版说明：
- 不必走到强版那么硬
- `deadline_mild` 已经比旧环境更像一个“真的有 leverage”的 V1 候选

但要注意：
- 它只是把动作收益从大约 `0.02~0.04` 推到 `0.03~0.06`
- 还没有发生量级翻转


## 9. 在 `deadline_mild` 上跑短训 PPO

### 9.1 想解决的问题

环境 leverage 变强以后，要看两件事：
- 这个环境是不是仍然“谁上都差不多”
- 还是它真的给 learner 留出了空间

### 9.2 具体实现

训练配置：
- `configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild.yaml`

训练 run：
- `runs/structured/structured_bw_t10_deadline_mild_ppo_u20_20260411`

比较文件：
- `runs/analysis/bw_deadline_mild_policy_compare_20260411/compare.csv`
- `runs/analysis/bw_deadline_mild_policy_compare_20260411/compare.json`

### 9.3 结果

| policy | reward | processed | drop | backlog |
| --- | ---: | ---: | ---: | ---: |
| fresh init mean | -1.0826 | 0.9588 | 0.4794 | 3.3194 |
| learned `u20` | -0.7935 | 0.9802 | 0.4618 | 3.3015 |
| heuristic | 0.5833 | 1.0746 | 0.3756 | 3.1008 |
| planner `k=1` | 0.6416 | 1.0817 | 0.3762 | 2.9771 |
| planner `k=2` | 0.8210 | 1.0927 | 0.3630 | 3.0084 |

### 9.4 结论

按宽标准看：
- `deadline_mild` 已经不是“谁上都差不多”的假强环境
- learned 比 init 好
- planner 比 heuristic 好

但这还不能推出：

`plain state-value PPO 已经有足够大的可学信号`


## 10. 在 `deadline_mild` 上检查 critic 是否仍把动作收益盖住

### 10.1 想解决的问题

环境 leverage 变强以后，最关键的问题是：

`对 plain PPO 来说，动作收益相对 V(s) 误差到底有没有改善到可用程度`

### 10.2 具体实现

value probe：
- `scripts/diagnostics/probe/probe_structured_bw_value_generalization.py`
- 输出：`runs/analysis/bw_deadline_mild_value_probe_20260411/summary.json`

credit mismatch probe：
- `scripts/diagnose_structured_bw_credit_mismatch.py`
- 输出：`runs/analysis/bw_deadline_mild_credit_mismatch_20260411/summary.json`

### 10.3 结果

#### 10.3.1 新环境里的动作收益量级

`deadline_mild`：
- `policy_local_reward_gap mean = 0.0326`
- `h10 mean_abs_delta_vs_heuristic = 0.0596`

对应文件：
- `runs/analysis/bw_deadline_mild_credit_mismatch_20260411/summary.json`
- `runs/analysis/bw_env_leverage_v14_deadline_mild_20260411/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild/summary.json`

#### 10.3.2 critic 误差量级

`deadline_mild`：
- online critic holdout `RMSE = 16.6566`
- fresh supervised critic holdout `RMSE = 3.6220`

对应文件：
- `runs/analysis/bw_deadline_mild_value_probe_20260411/summary.json`

#### 10.3.3 旧环境 vs 新环境的比值

旧环境口径：
- 动作收益取 `branch_delta_abs_mean_h10 ≈ 0.0203`
- 或 `h10 mean_abs_delta_vs_heuristic = 0.0444`

新环境口径：
- 动作收益取 `policy_local_reward_gap = 0.0326`
- 或 `h10 mean_abs_delta_vs_heuristic = 0.0596`

比值如下：

| ratio | old weak env | `deadline_mild` |
| --- | ---: | ---: |
| online RMSE / local action signal | `30.58` | `511.07` |
| fresh RMSE / local action signal | `21.10` | `111.13` |
| online RMSE / h10 mean_abs_delta | `13.99` | `279.28` |
| fresh RMSE / h10 mean_abs_delta | `9.66` | `60.73` |

旧环境文件：
- `runs/analysis/bw_value_gen_t10_branchprobe_stratified_lambda1_bank800_small_20260411/summary.json`
- `runs/analysis/bw_env_leverage_compare_20260411/structured_bw_sanity_1uav_static_gap_debug_t10_ppo/summary.json`

新环境文件：
- `runs/analysis/bw_deadline_mild_value_probe_20260411/summary.json`
- `runs/analysis/bw_deadline_mild_credit_mismatch_20260411/summary.json`
- `runs/analysis/bw_env_leverage_v14_deadline_mild_20260411/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild/summary.json`

#### 10.3.4 advantage 方向也仍不对

`deadline_mild` 的 credit mismatch：
- `corr_joint_adv_vs_policy_local_reward_gap = -0.3246`
- `corr_joint_adv_vs_policy_local_drop_gap = -0.4217`
- `positive_joint_adv_fraction = 0.3875`
- `bad_local_reward_but_positive_adv_fraction = 0.3875`

最坏样本里会出现：
- `joint_adv_raw = 7.60`
- `value_target = -3.06`
- 但同状态下最优本地重分配只比当前策略好 `0.032`

### 10.4 结论

这是这轮最重要的结论：

- 按宽标准，`deadline_mild` 比旧环境更有 leverage
- 但按严标准，它仍然远远不够 plain state-value PPO 用

也就是说：

`环境 leverage 增强了`

不等于

`plain state-value PPO 终于可学了`


## 11. 为什么 `deadline_mild` 提升这么小

这个问题最后又回到环境设计本身。

原因主要有 4 个：

1. `deadline_mild` 就是故意做成温和版  
   - `deadline_base_steps = 5`
   - `deadline_expire_rate = 0.20`
   - `deadline_service_relief_coef = 0.75`
   所以只有一部分状态会真正进入高 urgency。

2. deadline 没有单开一条特别强的 reward 通道  
   它主要还是通过 `drop` 进入已有的 `controllable_flow` reward。

3. 单步 BW 改动本来就被稀释  
   - `1 UAV + 5 GU`
   - 一次 action 是总带宽在 5 个 GU 之间重分
   - probe 里只做 `cf_delta = 0.05` 的局部搬移
   - 结果再进入所有 GU 汇总后的全局 reward

4. heuristic 已经吃掉了一部分明显信号  
   所以新环境里新增 leverage 更多是增量改进，不是把排序逻辑整体改写。

因此：
- 强版 `v14_deadline` leverage 会明显更大
- mild 版只会把信号推高一点，不会变成质变


## 12. 还有哪些没有做

下面这些还没做，或者没有做成完整版本：

### 12.1 环境侧

- 真正的 actor-only observability test  
  还没单独训练一个监督模型，只用 actor observation/history 去预测下一步 drop / queue jump / congestion。

- 更完整的 heuristic baseline test  
  还没系统跑：
  - `max backlog`
  - `min slack`
  - `backlog × rate`
  - `backlog × downstream headroom`

- aged reports / delayed feedback  
  现在补的是即时 proxy，还没做：
  - aged BSR
  - aged CQI
  - delayed ACK aggregate

- history / GRU  
  还没进入这条环境线。

- 多 UAV shared bottleneck 下的 coarse congestion feedback  
  当前 `1 UAV` 环境里这个量没法形成跨 GU leverage。

### 12.2 动作侧

- 改动作语义  
  还没有真正尝试：
  - residual over heuristic
  - top-2 focus
  - pairwise transfer
  - donor/receiver + delta

- 固定 `kappa` 的 mean-only actor 作为正式训练动作  
  之前只在 probe 里做过 `mean_only / concentration_only` 对照，还没把它变成新动作接口。


## 13. 要把环境改到什么程度，plain PPO 才可能可学

这里不能给出一个精确阈值，但可以给出量级判断。

当前 `deadline_mild`：
- 动作收益大约 `0.03 ~ 0.06`
- fresh supervised critic holdout RMSE 大约 `3.6`
- online critic holdout RMSE 大约 `16.7`

如果只看 fresh critic，想把

`RMSE / action-signal`

压到一个勉强不像当前这么离谱的水平：

- 压到 `10`：动作收益至少要到 `0.36`
- 压到 `5`：动作收益至少要到 `0.72`
- 压到 `3`：动作收益至少要到 `1.21`

这意味着相对当前 `0.03~0.06`，还要再提升大约：
- `6x ~ 12x`
- 到更激进标准甚至 `20x ~ 40x`

如果 online critic 还维持在 `16.7` 这个量级，那环境侧几乎需要把动作收益推到 `O(1)` 甚至更大，才可能盖过误差。这已经很容易把环境改得过硬、过于标签化，或者严重偏离未来 joint 主线。

所以我现在的判断是：

### 13.1 只靠继续温和改环境，不太可能把 plain PPO 救活

`deadline_mild` 已经说明：
- 未来 joint 也合理的温和改法，可以把 leverage 拉高一点
- 但离 plain PPO 所需的量级还差很远

### 13.2 如果继续只改环境，方向只能越来越硬

下一步若还想单靠环境把 PPO 顶起来，基本只剩：
- 更短 deadline
- 更高 expire/drop 代价
- 更紧 headroom
- 更少可被“平均分带宽”糊过去的状态

但这条线的副作用是：
- 物理 realism 更差
- 更容易变成“谁快过期就永远给谁”的近规则环境
- 对未来 joint 的迁移价值反而下降


## 14. 一次分配所有用户带宽比例，是不是太难了

我的判断是：**很可能是。**

当前动作接口的问题不是简单的“维度高”，而是这几件事叠在一起：

1. 每步动作是整条 simplex 分配  
   `1 UAV + 5 GU` 时，自由度是 `4`，未来 joint 里还会更高。

2. PPO 用一个 stage-level 标量 advantage 去更新整条 joint action  
   也就是一个数同时决定“5 个用户之间比例该怎么改”。

3. 动作收益本来就小  
   我们前面的 probe 一直看到本地收益增量只有 `0.03~0.06` 量级。

4. 现在 policy 还同时能改 `mean` 和 `kappa`  
   这会让“logprob 变大”并不总等于“均值分配真的变好了”。

换句话说，当前动作对 plain PPO 来说要求太高：

`它不是只学谁更重要，而是要学每个 GU 该拿多少比例、同时怎么改分布形状，而且监督只给一个总 advantage`

### 14.1 如果要改动作，我觉得最合理的顺序

#### 方案 1：residual over heuristic

让 actor 不直接输出整条绝对分配，而是输出对 `queue_aware_bw` 的有界修正。

形式可以是：

```text
a_exec = Normalize((1 - beta) * a_heuristic + beta * a_actor)
```

优点：
- 语义和未来 joint 兼容
- planner 结果已经说明 heuristic 附近存在可利用空间
- 比从零学整条 simplex 更容易

#### 方案 2：top-2 focus

让 actor 先决定：
- 哪两个 GU 应该重点倾斜
- 再决定它们之间怎么分
- 其余 GU 按 heuristic / residual 补齐

优点：
- 更贴近实际 leverage probe 里“关键步通常只改 1 到 2 个 GU”
- 大幅降低动作自由度

#### 方案 3：pairwise transfer

动作直接定义成：
- donor `i`
- receiver `j`
- transfer ratio `delta`

也就是显式地学：

`把一点 BW 从 i 挪到 j`

这和当前 leverage probe 的定义最一致，但它会把动作变成混合离散+连续，工程改动更大。

### 14.2 我不建议的顺序

不建议继续坚持：

`先把环境改得足够硬，再看全 simplex PPO 能不能自己学出来`

因为按当前证据，这条线的代价会越来越高，而收益未必高。


## 15. 当前结论

把这一轮环境改动线收成一句话：

`future-joint-safe` 的温和环境改动里，最有效的是把 soft proxy 换成真实 `deadline / slack / expiry` 动力学；但即使这样，动作收益也只从大约 `0.02~0.04` 提到 `0.03~0.06`，远远不足以盖过 plain state-value PPO 的 critic 误差。

因此，当前最稳妥的判断是：

1. 继续温和改环境有意义，但已经不是主解。
2. 如果目标是让 PPO 真正可学，下一步更该认真考虑改动作接口，而不是继续把环境往更硬、更规则化方向推。
3. 最值得优先尝试的动作改法，不是完全重做，而是：
   - `residual over heuristic`
   - 或 `top-2 focus`
   - 并先固定 `kappa`，只学 mean-like correction
