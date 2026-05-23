# BW 动作接口改造实验整理（2026-04-12）

这份文档只整理“改 BW 动作接口”这一条线，目的是把它和前面那段意见一一对上，回答下面几件事：

- 这批动作改造想解决什么问题
- 实际选了哪些改法
- 具体是怎么实现的
- 训练和测试口径是什么
- 结果数值是什么
- 新动作接口自身的动作收益有多大
- 还有哪些建议里的动作改法还没做

## 1. 背景与目标

在环境已经固定到 `deadline_mild + obs proxy` 之后，问题不再是“环境里有没有 BW leverage”，而是：

- 当前 `absolute simplex + trainable kappa` 的 BW actor，是否让 PPO 学了一个语义过重的动作；
- 即：PPO 需要用一个 stage-level 标量 advantage，同时去更新：
  - 全部 GU 的绝对带宽比例；
  - 分布的尖锐度 `kappa`；
- 这可能导致“动作语义过重 + 标量 credit 太粗”。

对应到收到的意见，这一批实验主要在验证：

1. `residual mean-only over heuristic`
2. `top-2 focus on extra mass + floor`
3. `donor / receiver + delta`

文中提到的 `PASPO / autoregressive allocation` 这类更重的分解式接口，这一轮还没有做。

## 2. 这批动作改造想解决什么问题

这批改动不是在解决“critic 准不准”，而是在解决更窄的一层：

- 旧接口里，actor 直接输出“整条 absolute simplex 分配”；
- 同时还学 `kappa`；
- 对 PPO 来说，这相当于让一个 noisy scalar advantage 去推一个很重的动作语义。

所以动作改造的共同目标是：

- 不再从零学整条 absolute simplex；
- 尽量把“要学的动作”改成“围绕一个合理基线，做局部重分配”；
- 先固定 `kappa`，只学 mean-like correction；
- 保留 continuous BW allocation 的物理语义，不做“其他 GU 直接归零”的 hard exclusion。

## 3. 固定不变的实验条件

为了隔离“动作接口本身”的影响，这一批实验里下面这些东西都故意保持不变：

- 固定环境：
  - `1 UAV`
  - `5 GU`
  - `T=10`
  - `deadline_mild + obs proxy`
- 只训练 BW：
  - `train_accel=false`
  - `train_sat=false`
  - `train_bw=true`
- critic 保持原样：
  - 仍然是 plain state-value critic
  - 没有给 critic 额外动作语义信息
  - 没有换成 action-dependent / branch teacher critic
- actor 沿用当前现实可解释的 BW 观测族：
  - `queue`
  - `eta_ref`
  - `recent arrival`
  - `recent service`
  - `queue headroom`
  - `deadline slack / deadline risk`
- 不给 actor hidden truth：
  - 不喂 `hotspot label`
  - 不喂未来 arrivals
  - 不喂 future channel

也就是说，这一批结果要解读成：

**“在同一个 critic 瓶颈下，只改 actor 的动作接口，PPO 会不会更容易学。”**

## 4. 代码改动

主要改动文件：

- [sagin_marl/rl/structured_actor.py](/D:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_actor.py)
- [sagin_marl/rl/structured_factory.py](/D:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_factory.py)
- [sagin_marl/env/config.py](/D:/研三上/毕设/sagin_marl/sagin_marl/env/config.py)
- [tests/test_structured_action_modules.py](/D:/研三上/毕设/sagin_marl/tests/test_structured_action_modules.py)

新增/扩展了 3 类 BW 参数化：

1. `score_residual_fixedkappa_dirichlet`
2. `score_focus2_fixedkappa_dirichlet`
3. `score_transfer_fixedkappa_dirichlet`

共同点：

- 基线动作都从当前 structured BW 本地特征里重建 `queue_aware` base；
- `kappa` 固定，不再训练；
- 都保留 per-GU floor；
- 最终仍输出合法 simplex action，再由固定 `kappa` 的 Dirichlet 做探索。

## 5. 训练与测试口径

### 5.1 训练

训练统一用：

- [scripts/train_structured.py](/D:/研三上/毕设/sagin_marl/scripts/train_structured.py)

这批 run 的共同 PPO 口径都沿用各自 yaml：

- `actor_lr = 3e-4`
- `critic_lr = 3e-4`
- `clip_ratio = 0.2`
- `value_coef = 0.5`
- `entropy_coef = 0`
- `ppo_epochs = 4`
- `num_mini_batch = 4`
- `buffer_size = 100`
- `bw_return_mode = gae`

训练长度：

- 每条新接口先跑 `u20`
- `focus2` 额外跑了 `u100`

### 5.2 训练内 checkpoint eval

所有配置都开了：

- `checkpoint_eval_interval_updates = 10`
- `checkpoint_eval_episodes = 32`
- `checkpoint_eval_fixed_policy = queue_aware_bw`
- `checkpoint_eval_policy_mode = stochastic`

所以训练过程中每隔 10 updates 都会拿当前 learned policy 和固定 heuristic 做一次同口径评估。

### 5.3 同口径 fresh-init / learned / heuristic 对照

训练后又补做了 same-eval compare：

- `fresh init`
- `learned final`
- `heuristic`

三者都在同一组 eval 设定下跑，用来分清：

- 提升是不是来自“训练学到了东西”
- 还是只是因为“动作接口里编码了更强 prior”

### 5.4 动作收益 probe

后面又补了一步更关键的检查：

- [scripts/diagnose_structured_bw_action_interface_leverage.py](/D:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_action_interface_leverage.py)

它不是测“环境本身”的 leverage，而是测：

**在某个动作接口允许的动作流形上，从同一批固定状态出发，这个接口自己最多能提供多大的局部长期收益。**

这里要特别说明一下顺序：

- 第一轮动作接口训练时，这个 probe 还没有单独做；
- 当时只知道环境级 leverage；
- 在用户追问“改动作之前有没有看过新接口自己的动作收益够不够”之后，才补做了这一步 interface-conditioned leverage probe。

口径是：

- 固定 `deadline_mild` 环境 snapshot panel
- `states_analyzed = 24`
- horizon 用 `h=2/5/10`
- tail rollout 用 heuristic future
- 比较：
  - `base action` vs heuristic
  - `best candidate on this interface manifold` vs heuristic
  - `best candidate` vs interface base

这一点很重要，因为它回答的是：

**“新动作接口自身的动作收益够不够大？”**

## 6. 对应意见的 3 条动作改法

### 6.1 `residual mean-only over heuristic`

对应意见里的目标：

- 不再输出整条 absolute simplex
- 改成围绕 heuristic base 做小幅零和转移
- 先只学 mean-like correction

配置：

- [structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_residual_meanonly.yaml](/D:/研三上/毕设/sagin_marl/configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_residual_meanonly.yaml)

关键实现：

- `structured_bw_parameterization: score_residual_fixedkappa_dirichlet`
- `structured_bw_fixed_kappa: 24.0`
- `structured_bw_residual_transfer_cap: 0.35`
- `structured_bw_residual_floor: 0.01`
- `structured_bw_residual_assoc_bonus: 0.3`
- `structured_bw_alpha_init_bias: -2.0`

语义上：

- `queue_aware` base 先给出一个可行分配；
- actor 只学习一个 bounded、零和的 residual；
- 最后投到带 floor 的 simplex；
- 再用固定 `kappa` 的 Dirichlet 采样。

训练 run：

- smoke: [structured_bw_t10_deadline_mild_residual_meanonly_smoke_u1_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_residual_meanonly_smoke_u1_20260412)
- u20: [structured_bw_t10_deadline_mild_residual_meanonly_u20_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_residual_meanonly_u20_20260412)

训练结果：

- `u20 checkpoint reward = 0.4727`
- heuristic checkpoint reward `= 0.5833`
- final `approx_kl_bw = 0.0867`
- final `clip_frac_bw = 0.065`

same-eval compare：

- fresh init `= 0.4763`
- learned final `= 0.4478`
- heuristic `= 0.5833`

动作收益（interface-conditioned leverage）：

- h10 `best_gap_vs_heuristic_mean = 0.08616`
- h10 `best_gap_vs_base_mean = 0.08615`
- h10 `mean_abs_delta_vs_base = 0.04045`
- h10 `best_positive_vs_heuristic_frac = 1.0`
- h10 `base_gap_vs_heuristic_mean ≈ 1.64e-05`

解释：

- 这条接口不是没 leverage；
- 但训练没有优于 fresh init；
- 它更多是在把 heuristic prior 编进动作头，而不是让 PPO 真正学出了净增益。

### 6.2 `top-2 focus on extra mass + floor`

对应意见里的目标：

- 不做 hard top-k exclusion；
- 所有 GU 仍有 base/floor；
- actor 只决定“额外稀缺质量更偏向哪两个 GU”。

配置：

- [structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_focus2_meanonly.yaml](/D:/研三上/毕设/sagin_marl/configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_focus2_meanonly.yaml)

关键实现：

- `structured_bw_parameterization: score_focus2_fixedkappa_dirichlet`
- `structured_bw_fixed_kappa: 24.0`
- `structured_bw_residual_transfer_cap: 0.8`
- `structured_bw_residual_floor: 0.01`
- `structured_bw_residual_assoc_bonus: 0.3`
- `structured_bw_alpha_init_bias: 0.0`

语义上：

- 先从 heuristic base 出发；
- actor 选一个 top-2 focus 方向；
- 把更多“额外质量”朝这两个 GU 倾斜；
- 保持 floor，不让其他 valid GU 直接归零。

训练 run：

- smoke: [structured_bw_t10_deadline_mild_focus2_meanonly_smoke_u1_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_focus2_meanonly_smoke_u1_20260412)
- u20: [structured_bw_t10_deadline_mild_focus2_meanonly_u20_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_focus2_meanonly_u20_20260412)
- u100: [structured_bw_t10_deadline_mild_focus2_meanonly_u100_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_focus2_meanonly_u100_20260412)

训练结果：

- `u20 checkpoint reward = 0.3101`
- `u20 final approx_kl_bw = 0.760`
- `u20 final clip_frac_bw = 0.280`

- `u100 checkpoint reward = 0.4354`
- `u100 final approx_kl_bw = 0.099`
- `u100 final clip_frac_bw = 0.195`

same-eval compare：

- u20:
  - fresh init `= -0.0334`
  - learned final `= 0.2535`
  - heuristic `= 0.5833`
- u100:
  - fresh init `= -0.3698`
  - learned final `= 0.4539`
  - heuristic `= 0.5833`

动作收益（interface-conditioned leverage）：

- h10 `best_gap_vs_heuristic_mean = 0.10896`
- h10 `best_gap_vs_base_mean = 0.10894`
- h10 `mean_abs_delta_vs_base = 0.11748`
- h10 `best_mass_shift_mean = 0.38380`
- h10 `best_positive_vs_heuristic_frac = 1.0`
- h10 `base_gap_vs_heuristic_mean ≈ 1.64e-05`

解释：

- 这是 3 条里最像样的一条；
- 训练增益是明确的，不只是 prior；
- `u100` 还在继续往上走；
- 但即使这样，plain PPO 还是没过 heuristic。

### 6.3 `donor / receiver + delta`

对应意见里的目标：

- 直接把动作语义改成“从谁挪给谁多少”；
- 更直接对齐 pairwise leverage 的解释。

配置：

- [structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_transfer_meanonly.yaml](/D:/研三上/毕设/sagin_marl/configs/structured_bw_sanity_1uav_static_gap_debug_t10_ppo_obsproxy_v14_deadline_mild_transfer_meanonly.yaml)

关键实现：

- `structured_bw_parameterization: score_transfer_fixedkappa_dirichlet`
- `structured_bw_fixed_kappa: 24.0`
- `structured_bw_residual_transfer_cap: 0.35`
- `structured_bw_residual_floor: 0.01`
- `structured_bw_residual_assoc_bonus: 0.3`
- `structured_bw_alpha_init_bias: 0.0`

语义上：

- 还是从 heuristic base 出发；
- actor 明确构造 donor / receiver / transfer；
- 再投影回带 floor 的 simplex。

训练 run：

- smoke: [structured_bw_t10_deadline_mild_transfer_meanonly_smoke_u1_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_transfer_meanonly_smoke_u1_20260412)
- u20: [structured_bw_t10_deadline_mild_transfer_meanonly_u20_20260412](/D:/研三上/毕设/sagin_marl/runs/structured/structured_bw_t10_deadline_mild_transfer_meanonly_u20_20260412)

训练结果：

- `u20 checkpoint reward = 0.0449`
- final `approx_kl_bw = 1.279`
- final `clip_frac_bw = 0.262`

same-eval compare：

- fresh init `= 0.3233`
- learned final `= 0.1025`
- heuristic `= 0.5833`

动作收益（interface-conditioned leverage）：

- h10 `best_gap_vs_heuristic_mean = 0.10763`
- h10 `best_gap_vs_base_mean = 0.10761`
- h10 `mean_abs_delta_vs_base = 0.06410`
- h10 `best_mass_shift_mean = 0.23955`
- h10 `best_positive_vs_heuristic_frac = 1.0`
- h10 `base_gap_vs_heuristic_mean ≈ 1.64e-05`

解释：

- 它的可达局部收益并不小，和 `focus2` 几乎同量级；
- 但训练明显更差；
- 这说明它的问题不是“接口没 leverage”，而是“在当前 PPO update 下太硬、太脆”。

## 7. 三条动作接口的直接对比

### 7.1 训练结果对比

| 接口 | fresh init | learned | heuristic | 训练结论 |
| --- | ---: | ---: | ---: | --- |
| residual | `0.4763` | `0.4478` | `0.5833` | 主要是 prior，不是学习增益 |
| focus2 u20 | `-0.0334` | `0.2535` | `0.5833` | 明显有学习增益 |
| focus2 u100 | `-0.3698` | `0.4539` | `0.5833` | 继续变好，但仍低于 heuristic |
| transfer | `0.3233` | `0.1025` | `0.5833` | 接口太硬，训练不稳 |

### 7.2 动作收益对比（h10）

| 接口 | best gap vs heuristic | best gap vs base | mean abs delta vs base | 解释 |
| --- | ---: | ---: | ---: | --- |
| residual | `0.0862` | `0.0861` | `0.0404` | leverage 有，但偏小 |
| focus2 | `0.1090` | `0.1089` | `0.1175` | leverage 最大，且动作重分配最强 |
| transfer | `0.1076` | `0.1076` | `0.0641` | leverage 接近 focus2，但训练更脆 |

这个表最关键的信息是：

- `focus2` 和 `transfer` 的**可达局部收益差不多**；
- 但训练表现差很多；
- 所以训练结果不只由 leverage 大小决定，还明显受“接口稳定性”影响。

## 8. 这一批实验告诉了我们什么

### 8.1 可以确认的事

1. **动作改造方向不是错的。**
   证据：
   - `focus2` 明显比旧的 absolute-simplex PPO 更好；
   - `focus2` 在 `u100` 还继续向上。

2. **新动作接口自身不是完全没收益。**
   证据：
   - interface-conditioned leverage probe 里，h10 `best local gain` 已经到 `0.086 ~ 0.109`；
   - 而且 `base_gap_vs_heuristic ≈ 0`，说明增益不是“换个 base 就自动赢”。

3. **只改动作接口还不够。**
   证据：
   - 3 条接口都没让 plain PPO 超过 heuristic；
   - 其中最好的 `focus2` 也还差大约 `0.13` reward。

### 8.2 不能简单下的结论

- 不能说“只要改成 residual 就能解决”；
- 不能说“只要把 absolute simplex 压成 donor/receiver 就会更稳”；
- 也不能说“动作接口本身收益还是太小，所以这条线没意义”。

更准确的说法是：

- `residual` 太小，更像换 prior；
- `transfer` leverage 够，但在当前 PPO 下太脆；
- `focus2` 是目前这批里最合理的 actor interface。

## 9. 这批实验没做的动作改法

对应原意见，还没做的有：

1. `PASPO / autoregressive allocation`
2. planner-teacher 直接接到新动作接口上
3. `focus2` 接 branch/planner teacher，而不是继续用当前 critic advantage
4. 更正式的混合动作 PPO / hierarchical PPO 接口

这里面我认为最值得优先补的不是 `PASPO`，而是：

- **`focus2 + better credit / teacher`**

因为当前结果已经说明：

- `focus2` 作为动作接口本身是对的；
- 但 plain state-value PPO 还压不住。

## 10. 当前结论

这批动作接口实验最好的总结是：

**问题不再像“absolute simplex 本身让 PPO 完全学不动”，而更像“把动作接口压缩以后，PPO 确实更容易学了，但同一个 noisy scalar advantage / critic path 仍然不够干净，无法把最好的接口彻底推过 heuristic。”**

如果只保留一句最重要的话，那就是：

**动作改造方向是对的，`focus2` 是当前最好的接口；但它只是把问题减轻了，没有把问题解决。下一步更应该是“最好接口 + 更干净的 credit / teacher”，而不是再回去拧旧的 absolute-simplex PPO。**
