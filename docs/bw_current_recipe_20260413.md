# BW Current Recipe (2026-04-13)

这份文档整理的是：**现在实际在用、已经验证能工作的 BW 训练方案**，以及**当前简化环境是什么样**。目的是让后续新对话里可以直接从这里出发，把场景改成 `1 UAV + 几个 GU`，继续验证这条 BW 方法。

## 1. 现在实际在用的 BW 方案

当前最稳、也已经验证过训练结果和并行优化前基本一致的方案是：

- `BW-only`
- `bw_actor_advantage_override_mode = branch_delta`
- `branch_delta horizon = 3`
- `branch_samples = 1`
- `num_envs = 8`
- `vec_backend = subproc`
- `device = cuda`
- `PPO actor update` 用很小的 reuse：
  - `--ppo_epochs 1`
  - `--num_mini_batch 1`
- 不开每次 update 的 `DirProbe`
- 开 `checkpoint_eval + best checkpoint`
- 最终用 `best checkpoint`，不是最后一个 checkpoint

这条线当前推荐直接用 **direct teacher**，不是 learned delta critic，也不是 teacher-student。

## 2. 为什么是这套

目前已经确认的判断是：

- 旧的 `critic/value -> return -> raw_advantage` 链会把很小的动作差值淹没掉。
- 直接用 `branch_delta_h3` 当 actor advantage，能把 PPO 拉回正常学习。
- learned delta critic 现在还不能稳定长期替代 direct teacher。
- 后期继续训练会有平台后的负漂移，所以要靠 `checkpoint_eval + best checkpoint` 截住，而不是盲目用最后一个 checkpoint。

## 3. 当前简化环境

当前验证这条方法的环境，不是复杂 joint 环境，而是一个很聚焦的 BW 简化环境。代表配置可看：

- [当前 run config](../runs/structured_short/two_gu_t10_env8_bs1_1x1_ckpteval_rerun_u30_fg/config.yaml)

关键环境设定：

- `num_uav = 1`
- `num_gu = 2`
- `T_steps = 10`
- `traffic_model = sticky_subset_hotspot`
- `hotspot_subset_size = 1`
- `hotspot_rho = 20`
- `hotspot_on_mean_steps = 20`
- `hotspot_off_mean_steps = 1`
- `arrival_mean_preserve = true`
- `task_arrival_rate = 800000`
- `task_arrival_poisson = false`

也就是：

- 2 个 GU 里每次只有 1 个 hot
- hot/cold 差异非常大
- 不是泊松到达，而是常值到达

复杂物理因素当前都关掉了：

- `fading_enabled = false`
- `interference_enabled = false`
- `atm_loss_enabled = false`
- `rain_loss_enabled = false`
- `doppler_enabled = false`
- `energy_enabled = false`

所以这不是“真实复杂环境”，而是一个专门验证 BW credit 和更新机制的简化环境。

## 4. 资源是怎么设的

这次环境不是简单把 `1 UAV + 2 GU` 塞进旧资源配置里，而是专门做了资源缩放，目的是让：

- 均分 BW 时，hot user 还是不够
- cold user 会出现浪费

当前打开了：

- `resource_scale_enabled = true`

参考的是：

- `resource_scale_ref_num_uav = 3`
- `resource_scale_ref_num_gu = 20`
- `resource_scale_ref_task_arrival_rate = 490000`

缩放逻辑在：

- [config.py::_apply_resource_scaling](../sagin_marl/env/config.py#L803)

核心是按负载密度缩放：

```text
cur_gu_per_uav = task_arrival_rate * num_gu / num_uav
ref_gu_per_uav = ref_arrival * ref_num_gu / ref_num_uav
acc_scale = cur_gu_per_uav / ref_gu_per_uav
```

然后据此缩 `b_acc` 和 `b_sat_total`。

当前这套 1UAV/2GU 简化环境最终得到：

- `b_acc ≈ 2.204e6`
- `b_sat_total ≈ 3.673e6`

这点很重要，因为之前很多 `1 UAV + 多 GU` 的旧实验，问题恰恰是：

- 业务差异不够强
- 资源也没缩到“均分会明显错”的 regime

## 5. GU 位置 reset 时会不会变

会变。

`reset()` 每次都会重新 `_init_state()`，而 `_init_state()` 里每次都会重新跑一次 `thomas_cluster_process(...)` 来生成 `gu_pos`：

- [sagin_env.py::_init_state](../sagin_marl/env/sagin_env.py#L1335)
- [sagin_env.py::reset](../sagin_marl/env/sagin_env.py#L1605)

所以不是“整次训练里 GU 位置固定不变”。

但当前生成机制仍然是同一种分布：

- `gu_init_num_clusters = 1`
- `gu_init_cluster_std = 320`

也就是每个 episode 都重采样，但都是在同一类单簇分布下采样。

## 6. branch_delta 现在是怎么训练的

当前实际使用的是 **direct teacher**：

```text
branch_delta_h3
= Q_3(s_t, a_sampled ; follow current policy)
- Q_3(s_t, a_det ; follow current policy)
```

其中：

- `a_sampled` 是 rollout 里这次真实采样到的 BW 动作
- `a_det` 是当前 actor 的 deterministic BW 动作
- 从下一步开始，两条分支都跟随当前 policy
- 现在的 `follow_policy_mode` 用的是 `stochastic`

对应代码在：

- [compute_bw_branch_advantage_override](../sagin_marl/rl/structured_bw_update_direction.py#L1261)
- [_rollout_from_snapshot_states_parallel](../sagin_marl/rl/structured_bw_update_direction.py#L798)
- [_bw_actions_from_snapshots_batch](../sagin_marl/rl/structured_bw_update_direction.py#L604)

注意：当前已经做了一个重要加速优化：

- `stochastic follow` 的后续 actor 前向已经做成了**批量前向**
- 环境推进也支持 `subproc` 并行

但仍然**不要开** worker-side fused rollout。这个实验开关已经实现了，但更慢：

- `bw_actor_branch_worker_fused_rollout_enabled = false`

## 7. 训练脚本里最相关的文件

### 训练入口

- [scripts/train_structured.py](../scripts/train_structured.py)

最相关的参数入口：

- `--ppo_epochs`
- `--num_mini_batch`
- `--num_envs`
- `--vec_backend`
- `--device`
- `--bw_actor_advantage_override_mode`
- `--bw_actor_branch_horizon`
- `--bw_actor_branch_samples`
- `--bw_actor_branch_follow_policy_mode`
- `--checkpoint_eval_*`

### PPO / actor update 主体

- [sagin_marl/rl/structured_mappo.py](../sagin_marl/rl/structured_mappo.py)

这里负责：

- rollout batch 转成 actor/critic cache
- PPO loss
- advantage override 接入

### branch teacher / BW credit 计算

- [sagin_marl/rl/structured_bw_update_direction.py](../sagin_marl/rl/structured_bw_update_direction.py)

这里负责：

- `branch_delta`
- `true_adv_mc`
- branch rollout
- batched stochastic follow

### 环境分组与 subproc

- [sagin_marl/rl/structured_train.py](../sagin_marl/rl/structured_train.py)
- [sagin_marl/env/structured_vec_env.py](../sagin_marl/env/structured_vec_env.py)

### 环境本体与 reset / traffic / queue

- [sagin_marl/env/sagin_env.py](../sagin_marl/env/sagin_env.py)

### 配置项定义

- [sagin_marl/env/config.py](../sagin_marl/env/config.py)

## 8. 当前推荐训练命令骨架

下面这条不是唯一命令，但它代表当前推荐骨架。重点是参数组合，不是 run 名字。

```powershell
.venv\Scripts\python.exe scripts/train_structured.py `
  --config configs/tmp/structured_bw_sanity_1uav_2gu_t10_stronggap_probe_ppo_trueadv_rawadv.yaml `
  --run_name YOUR_RUN_NAME `
  --device cuda `
  --num_envs 8 `
  --vec_backend subproc `
  --ppo_epochs 1 `
  --num_mini_batch 1 `
  --bw_actor_advantage_override_mode branch_delta `
  --bw_actor_branch_horizon 3 `
  --bw_actor_branch_samples 1 `
  --bw_actor_branch_follow_policy_mode stochastic `
  --checkpoint_eval_interval_updates 5 `
  --checkpoint_eval_start_update 5 `
  --checkpoint_eval_episodes 32 `
  --checkpoint_eval_episode_seed_base 42000 `
  --checkpoint_eval_fixed_policy queue_aware_bw `
  --checkpoint_eval_policy_mode deterministic `
  --checkpoint_eval_reward_early_stop_enabled `
  --checkpoint_eval_reward_patience 2 `
  --checkpoint_eval_stop_on_early_stop `
  --checkpoint_eval_save_best_models `
  --checkpoint_eval_use_best_as_final
```

补充说明：

- 不用开 `--verbose_console`
- 不用开每次 update 的 `DirProbe`
- 当前默认就让 `branch teacher` 走并行 env + batched stochastic follow
- 不要开 `--bw_actor_branch_worker_fused_rollout_enabled`

## 9. 现在这条线的代表结果

推荐参考 run：

- [two_gu_t10_env8_bs1_1x1_ckpteval_rerun_u30_fg](../runs/structured_short/two_gu_t10_env8_bs1_1x1_ckpteval_rerun_u30_fg)

关键结果：

- `checkpoint_eval` 最好点在 `u25` 左右
- 最好 reward 约 `-87.359`
- `queue_aware_bw` heuristic 约 `-91.524`
- 说明这条 direct `branch_delta` BW 方法在当前简化环境里已经稳定超过 heuristic

相关文件：

- [checkpoint_eval.csv](../runs/structured_short/two_gu_t10_env8_bs1_1x1_ckpteval_rerun_u30_fg/checkpoint_eval.csv)
- [metrics.csv](../runs/structured_short/two_gu_t10_env8_bs1_1x1_ckpteval_rerun_u30_fg/metrics.csv)

## 10. 如果下个对话要改成 `1 UAV + 几个 GU`

建议直接沿用这条线，不要一下跳回复杂 joint 环境。优先改这几项：

1. `num_gu`
2. `users_obs_max`
3. `hotspot_subset_size`
4. `hotspot_rho`
5. `task_arrival_rate`
6. `resource_scale_*`
7. `gu_init_num_clusters / gu_init_cluster_std`

最重要的是继续保证这件事仍然成立：

- **均分带宽时，hot user 依然不够，cold user 会浪费**

否则就算方法本身没问题，环境也不一定会给出足够强的 BW 信号。

## 11. 改 `1 UAV + 几个 GU` 时最该先检查什么

如果换成 `1 UAV + 4/5/6 GU`，我会先检查：

1. 均分 BW 的时候，hot user 的 backlog 会不会继续堆
2. cold user 是否已经服务过量
3. heuristic `queue_aware_bw` 相对均分有没有明显优势
4. `branch_delta_h3` 的 reward / queue impulse 还主要集中在 1~3 步吗
5. 资源缩放后 `b_acc / b_sat_total` 是否仍处在“均分会错”的 regime

也就是说，先验证“环境仍然在考 BW”，再验证“这条 BW 方法还能不能学”。

## 12. 新对话里最短的交接句

如果后面开新对话，可以直接从这句开始：

> 先按 `docs/bw_current_recipe_20260413.md` 这条 direct `branch_delta` 方案，在 `1 UAV + N GU` 下改强 hotspot 差异和 resource scaling，先检查均分时 hot user 是否仍然不够、cold user 是否浪费，再跑 `num_envs=8, branch_samples=1, 1x1 PPO, checkpoint_eval + best checkpoint`。

