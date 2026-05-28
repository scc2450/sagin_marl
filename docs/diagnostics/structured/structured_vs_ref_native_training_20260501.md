# Structured Native `sample-vs-ref` 训练方案

日期：2026-05-01

本文档定义新的 accel / sat / bw 统一训练接口。目标不是继续修补 PPO critic，也不是复用旧 BW clean teacher 的 Python replay 形态，而是在**单 GPU 原生架构**里实现一个严格的 action contrast：

```text
A_stage,u(s, a_sample_u)
= G(s, a_ref_except_u + a_sample_u, deterministic_follow)
- G(s, a_ref, deterministic_follow)
```

其中只替换当前 UAV `u` 在当前 stage 的动作，其他 UAV 当前动作和所有后续动作都使用当前 actor 的 deterministic 动作。两条 branch 必须从同一个 snapshot 出发，并使用同一份后续外生随机 tape。

## 1. 必须保持的语义

训练样本单位是：

```text
(stage_id, rollout_step, env_id, uav_id)
```

不是整组 UAV，也不是整条 rollout 的 shared advantage。

当前 stage 的 ref 动作：

```text
a_ref = deterministic_actor(stage_obs)
```

当前 stage 的 sample 动作：

```text
a_sample_u ~ stochastic_actor(stage_obs_u)
```

当前 stage 的 sample branch 执行动作：

```text
a_branch[v] = a_sample_u  if v == u
a_branch[v] = a_ref[v]   otherwise
```

后续 follow policy：

```text
accel follow = deterministic current accel actor
sat follow   = deterministic current sat actor
bw follow    = deterministic current bw actor
```

return 口径：

```text
G = finite-horizon episode return from this snapshot to current episode end
```

到 `terminated` 或 `truncated` 就停止累计，bootstrap 必须为 0。不再使用 critic，不再使用 GAE，不再使用 time-limit bootstrap。

## 2. 不能做的事情

这些路径不能作为正式实现：

- 不能用 Python driver `load_*_stage_state()` 做 branch replay。
- 不能在 update 期间按 branch / row / field 用 Python for-loop 发射大量 `index_select`、`index_copy_`、`copy_` 小 op。
- 不能只给 BW 做完整 snapshot，而 accel/sat 只拿 training local obs 伪装成可恢复现场。
- 不能把整组 UAV 都 sample 后给整组一个 advantage，除非作为临时诊断，并且不得作为正式训练路径。
- 不能用 queue-aware / cluster / zero 作为 ref 或 follow policy。ref/follow 必须是当前 actor deterministic。
- 不能让 sample/ref 两条 branch 使用不同 future arrival、hotspot、fading、doppler、reset tape。
- 不能在 branch replay 里 episode done 后自动 reset 并继续累计新 episode reward。

## 3. 当前代码状态

现有代码里有几块可以复用思想，但不能原样复用实现。

### 3.1 已有可复用点

- `kernels.cu` 已有 `copy_bw_runtime_snapshot_to_history_parallel(...)`，在 `sat_to_bw_live_kernel(...)` 中把 BW stage 的 runtime state / cache / stage fields 写入 history。
- `structured_batch_env_core.py` 已有 `prepare_native_branch_replay_from_history(...)`，可以从 rollout history 恢复 BW branch replay。
- `structured_mappo.py` 已有 `_native_bw_clean_rollout_returns_from_history(...)`，逻辑上已经接近“从 snapshot 回放 branch return”。
- actor 侧已经有 native CUDA actor binding，rollout hot path 可以使用当前 actor 权重。

### 3.2 不能原样复用点

- 当前完整 runtime snapshot 只覆盖 BW。accel/sat 只有 world/local/action history，不足以恢复完整 env runtime。
- `prepare_native_branch_replay_from_history(...)` 当前通过 Python `_copy_tensor_dataclass_rows_(...)` 做大量逐字段 gather，这不符合单 GPU native 架构。
- `_BwCleanFirstActionOverrideBridge` 是 Python bridge，适合诊断，不适合作为正式训练 hot path。
- 当前 `_stage_actor_eval_from_batch(...)` 默认把一个 stage 内所有 UAV 的 logprob sum 起来；`sample-vs-ref` 需要 target UAV 的 per-agent logprob。

## 4. Native 数据结构

### 4.1 三个 stage 都要有 runtime snapshot

新增统一的 stage runtime snapshot history，而不是只给 BW 存。

推荐结构：

```text
history.runtime_state[stage_id][slot, env]
history.runtime_stage_fields[stage_id][slot, env]
history.runtime_local_obs[stage_id][slot, env, uav]
history.runtime_cache[stage_id][slot, env]   # 只有需要 cache 的 stage 填，其他为空或最小结构
```

如果 ABI 上更容易落地，也可以拆成：

```text
accel_runtime_state / sat_runtime_state / bw_runtime_state
accel_runtime_stage / sat_runtime_stage / bw_runtime_stage
accel_runtime_obs   / sat_runtime_obs   / bw_runtime_obs
bw_runtime_cache
```

关键要求是字段含义一致：每个 snapshot 都必须能把 sub-batch runtime 恢复到“当前 stage 动作执行前”的状态。

### 4.2 snapshot 写入时机

accel snapshot：

```text
prepare_initial_accel_live_kernel(...)
  sync orbit / random step tape
  prepare accel stage from state
  write accel world
  write accel local obs
  copy accel runtime snapshot to history
```

sat snapshot：

```text
accel_to_sat_live_kernel(...)
  apply accel action and avoidance
  update UAV pos/vel/energy into sat stage
  refresh sat stage derived fields
  write sat world
  write sat local obs
  copy sat runtime snapshot to history
```

BW snapshot：

```text
sat_to_bw_live_kernel(...)
  apply sat action
  prepare BW candidates/cache
  write BW world
  write BW local obs
  copy BW runtime snapshot to history
```

BW 已有类似逻辑，但要重构成统一函数，避免只有 BW 是一等公民。

### 4.3 actor ABI 不能被 history 撑爆

历史 snapshot tensors 必须继续放在 actor runtime tensor slice 之后。`bindings.py` 里 actor ABI 仍然只传 actor live inference 需要的瘦身 runtime tensors，不能因为新增 accel/sat history 让 `actor_call_args` 变大。

## 5. Native branch table

每次 update 前，在 GPU 上生成一个 branch table。

每行包含：

```text
stage_id        int32
history_row     int32   # rollout_step * num_envs + env_id
uav_id          int32
source_step     int32
source_env      int32
remaining_steps int32
valid           bool
```

采样规则：

```text
从 train_* enabled 的 stage 中抽
从对应 stage 的 valid history row 中抽
再抽 target uav
```

初始可以 uniform random，后续可以加优先级采样，但必须仍在 GPU 上一次性生成 table。不能 Python 随机挑 row 后循环 replay。

配置建议：

```yaml
structured_actor_update_mode: vs_ref
vs_ref_rows_per_update: 32
vs_ref_samples_per_row: 1
vs_ref_stage_sample_mode: per_uav
vs_ref_horizon_mode: episode_remaining
vs_ref_ref_policy: deterministic_current
vs_ref_follow_policy: deterministic_current
vs_ref_advantage_normalize: stage
vs_ref_disable_critic: true
```

如果需要分 stage 控制：

```yaml
accel_update_mode: vs_ref
sat_update_mode: vs_ref
bw_update_mode: vs_ref
```

## 6. Paired branch replay

为了保证 sample/ref 外生随机完全一致，branch replay 使用 paired lanes：

```text
lane 2*i     = ref branch
lane 2*i + 1 = sample branch
```

两个 lane 从同一个 `(history_row, uav_id)` 复制 snapshot，且使用同一份 future random tape。

### 6.1 prepare kernel

新增 native kernel：

```text
prepare_vs_ref_branch_replay_kernel(...)
```

职责：

- 根据 branch table 把对应 stage snapshot 拷贝到 sub-batch runtime。
- 每个 branch 复制两份 lane：ref lane 和 sample lane。
- 为 paired lanes 写入相同的 `source_step/source_env/episode_idx/t/global_step`。
- 为 paired lanes 绑定相同的 future arrival / hotspot / fading / doppler tape 读取口径。
- 初始化 branch alive mask 为 true。

这个 kernel 必须一次处理整个 branch batch。不能由 Python 对 dataclass 字段逐个 `_copy_tensor_dataclass_rows_`。

### 6.2 first-action kernel

每个 stage 的 first action 都按同一个模式：

1. native actor deterministic 生成全组 `a_ref`。
2. native actor stochastic 生成全组 `a_sample_all` 和 per-agent `old_logprob_sample`。
3. native first-action override kernel 写执行动作：

```text
ref lane:
  action[v] = a_ref[v] for all v

sample lane:
  action[v] = a_sample_all[u] if v == target_uav
  action[v] = a_ref[v]        otherwise
```

同时写训练用字段：

```text
train_action        = a_sample_all[target_uav]
train_old_logprob   = old_logprob_sample[target_uav]
train_stage_id      = stage_id
train_uav_id        = target_uav
train_local_obs_row = local obs of target_uav at snapshot
```

accel 必须新增 per-agent old logprob 输出。sat / bw 已经有 per-agent logprob buffer，但也要走同一套 target-uav gather。

### 6.3 follow steps

first step 之后，两个 lane 都走 deterministic current actor：

```text
for k = 1 .. remaining:
  accel deterministic
  sat deterministic
  bw deterministic
  finish step
```

branch replay 期间不能使用 stochastic follow，不然 sample/ref 差值会混入后续探索噪声。

### 6.4 episode done

branch replay 是 finite-horizon episode return，不是继续采样训练 rollout。

因此：

- 如果 lane `terminated` 或 `truncated`，后续 reward 贡献为 0。
- 不在 branch replay 中 reset 到新 episode。
- 不使用 bootstrap value。

为了避免 variable-length Python loop，可以用固定 `max_horizon = T_steps` 运行 native loop，但 kernel 内用 `alive_mask` 屏蔽已经 done 的 lane。

## 7. Return reducer

新增 native reducer：

```text
reduce_vs_ref_returns_kernel(...)
```

输入：

```text
branch rewards [horizon, 2 * branch_count]
terminated/truncated [horizon, 2 * branch_count]
branch table
gamma
```

输出：

```text
ref_return[branch]
sample_return[branch]
advantage[branch] = sample_return - ref_return
```

对 positive reward、weighted workload reward、普通 env reward 都使用同一个 reward tensor。不要在 `vs_ref` 路径里重新定义 reward。

默认：

```text
gamma = cfg.gamma
bootstrap = 0
```

如果后续要做 horizon cap，可以另加配置，但正式概念默认是到当前 finite episode 结束。

## 8. Actor update batch

branch replay 完成后，构造 contiguous GPU tensors：

```text
vs_ref_stage_id        [B]
vs_ref_uav_id          [B]
vs_ref_local_obs       stage-specific target local obs
vs_ref_action          target UAV sampled action
vs_ref_old_logprob     target UAV sampled old logprob
vs_ref_advantage       sample_return - ref_return
vs_ref_ref_return      diagnostics
vs_ref_sample_return   diagnostics
```

训练 loss 可以沿用 PPO-style clipped surrogate：

```text
ratio = exp(logprob_new - logprob_old)
loss = -mean(min(ratio * A, clip(ratio) * A))
```

但这不是 PPO critic/GAE。这里的 `A` 是 branch action contrast。

advantage normalization：

```text
默认按 stage 单独 normalize
如果某个 stage branch_count <= 1，则不 normalize
```

entropy：

可以保留 stage entropy coef，但它只作用于 target UAV 的分布，不再对整组 sum。

## 9. Per-stage logprob 口径

### 9.1 accel

需要新增：

```text
live_accel_old_logprobs_per_agent [env, uav]
hist_accel_old_logprobs_per_agent [slot, env, uav]  # optional, vs_ref training batch 更重要
```

actor eval 时只取 target UAV：

```text
logprob_new = accel_policy.evaluate(local_accel_state[target], sampled_accel[target]).logprob
```

### 9.2 sat

已有 per-agent logprob，但要训练 target UAV 的 subset action：

```text
logprob_new = sat_policy.evaluate(local_sat_state[target], sampled_subset_index[target]).logprob
```

注意 sample branch 执行时其他 UAV 的 sat subset 必须是 ref，不是 sample。

### 9.3 BW

已有 per-agent logprob。训练 target UAV 的 simplex action：

```text
logprob_new = bw_policy.evaluate(local_bw_state[target], sampled_bw[target]).logprob
```

其他 UAV 的 BW simplex 必须是 ref。

## 10. 与现有 PPO update 的关系

`vs_ref` 模式下：

- 不调用 critic update。
- 不计算 GAE。
- 不使用 `batch_view.values`。
- 不使用 `structured_step_bootstrap_stage`。
- 不使用 `bw_return_mode=bw_gae`。
- 不需要 `critic_warmup_before_actor_epochs`。

可以保留旧 PPO 代码路径，但必须通过清晰配置切换：

```text
structured_actor_update_mode == "ppo"     -> 旧 PPO
structured_actor_update_mode == "vs_ref"  -> 新 action contrast
```

不要把 `vs_ref` 塞进 `bw_actor_advantage_override_mode`，因为它不只是 BW，也不是 GAE advantage override。

## 11. 必须避免的小 op 泄漏

正式路径中禁止这些做法：

- branch replay 前用 Python list 构造每个 branch 的 snapshot。
- Python 循环 over dataclass fields 做 `index_select/index_copy_`。
- Python bridge 在每个 replay step 手写 `write_accel_action/write_sat_action/write_bw_action`。
- 对 sample/ref 分两次独立 replay，再试图用 seed 保证随机一致。
- 每个 stage 分别临时 materialize CPU numpy action。

允许的 Python 调度只有粗粒度：

```text
collect rollout
launch branch table kernel
launch paired branch replay program
launch actor update batch
log metrics
```

也就是说，Python 可以发起少数几个大 kernel / PyTorch batch op，但不能参与 branch 内部逐步拼装状态。

## 12. 实现顺序

### Step A：配置与模式开关

新增配置字段：

```python
structured_actor_update_mode: str = "ppo"  # "ppo" | "vs_ref"
vs_ref_rows_per_update: int = 32
vs_ref_samples_per_row: int = 1
vs_ref_horizon_mode: str = "episode_remaining"
vs_ref_advantage_normalize: str = "stage"  # "none" | "stage" | "global"
vs_ref_ref_policy: str = "deterministic_current"
vs_ref_follow_policy: str = "deterministic_current"
vs_ref_disable_critic: bool = True
```

如果需要单独 stage：

```python
accel_update_mode: str | None = None
sat_update_mode: str | None = None
bw_update_mode: str | None = None
```

### Step B：三 stage snapshot history

在 runtime history 中补齐：

```text
accel_runtime_state/cache/stage/local_obs
sat_runtime_state/cache/stage/local_obs
bw_runtime_state/cache/stage/local_obs
```

BW 现有 snapshot 迁移到统一接口。

新增 kernel helper：

```text
copy_stage_runtime_snapshot_to_history_parallel(stage_id, slot, env, stage_slot)
```

### Step C：native branch table

新增固定容量 branch workspace：

```text
vs_ref_branch_table
vs_ref_train_batch
vs_ref_return_buffers
```

新增 kernel：

```text
sample_vs_ref_branch_table_kernel(...)
```

### Step D：native paired replay prepare

新增 kernel：

```text
prepare_vs_ref_branch_replay_kernel(...)
```

替换当前 BW-only Python `_copy_tensor_dataclass_rows_` branch prepare。

### Step E：first-action override

新增 per-stage first-action kernel：

```text
vs_ref_first_accel_action_kernel(...)
vs_ref_first_sat_action_kernel(...)
vs_ref_first_bw_action_kernel(...)
```

或者一个 typed kernel 用 `stage_id` 分派。

### Step F：branch replay loop

新增 native replay program：

```text
run_vs_ref_branch_replay(...)
```

它应该复用主 native phase kernels，但有两个关键差异：

- first step 使用 paired first-action override。
- done 后不 reset，不继续累计新 episode。

### Step G：return reducer

新增：

```text
reduce_vs_ref_returns_kernel(...)
```

输出 actor update batch。

### Step H：actor update

在 `StructuredMAPPO.update(...)` 或新 trainer method 中加入：

```text
if structured_actor_update_mode == "vs_ref":
    generate native branch batch
    compute vs_ref advantages
    update actor only
    sync native actor binding
    return vs_ref metrics
```

建议不要继续叫 MAPPO update；可以保留类名，但函数内部要把 PPO 和 vs-ref 分支完全隔离。

## 13. 验证测试

### 13.1 common randomness test

同一个 snapshot 构造两个 ref lanes：

```text
lane A = ref
lane B = ref
```

要求：

```text
max_abs(return_A - return_B) == 0  或在 GPU 浮点容差内
```

这能抓出 future tape 不一致、done 后 reset、snapshot 污染。

### 13.2 one-UAV replacement test

构造 sample lane 后检查 first executed action：

```text
target UAV action == sampled action
non-target UAV actions == ref actions
```

accel/sat/bw 三个 stage 都要测。

### 13.3 branch no-reset test

从接近 `T_steps - 1` 的 snapshot replay：

```text
remaining episode length = 1
branch return 只包含当前 episode 最后一拍
done 后不进入新 episode
```

### 13.4 stage parity smoke

对于同一个 snapshot：

```text
ref branch first step reward == official deterministic rollout first step reward
```

分别测 accel/sat/bw snapshot。

### 13.5 no-small-op audit

开启 profiler，`vs_ref` branch generation 不应出现：

```text
O(branch_count * fields) 个 index_select/index_copy/copy_
O(branch_count * horizon) 个 Python bridge calls
```

期望形态是：

```text
O(horizon) 个 native phase kernels
+ O(1) 个 prepare/reduce kernels
+ O(stage minibatches) 个 PyTorch actor update ops
```

## 14. 预期指标

训练日志新增：

```text
vs_ref_branch_count_accel/sat/bw
vs_ref_adv_mean_accel/sat/bw
vs_ref_adv_std_accel/sat/bw
vs_ref_adv_positive_frac_accel/sat/bw
vs_ref_ref_return_mean_accel/sat/bw
vs_ref_sample_return_mean_accel/sat/bw
vs_ref_common_random_ref_ref_max_abs_diff
vs_ref_first_action_non_target_mismatch_frac
vs_ref_actor_loss_accel/sat/bw
vs_ref_approx_kl_accel/sat/bw
vs_ref_clip_frac_accel/sat/bw
vs_ref_entropy_accel/sat/bw
```

如果 `vs_ref_adv_std` 很小或 `positive_frac` 接近 0/1，说明当前 stochastic 分布或环境 leverage 有问题。

如果 common-random ref-ref diff 不为 0，先修 replay，不要看训练结果。

## 15. 最终判断标准

这个实现完成后，我们要回答的问题是：

```text
在固定同一状态、同一后续随机、同一 deterministic follow 下，
当前 actor 采样出的某个 UAV 动作是否比 deterministic ref 更好？
```

如果答案有稳定信号，actor 就有可以学习的低方差 credit。

如果答案没有稳定信号，再看：

- actor stochastic 分布是否采不到有意义动作；
- 当前动作对 full-episode return 的边际影响是否太小；
- reward / 环境尺度是否仍然让 stage action 没有 leverage；
- deterministic ref 是否已经接近局部最优。

这比继续看普通 PPO EV 更直接，因为它绕开了 critic 是否能预测整条轨迹难度的问题。
