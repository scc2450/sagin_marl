# structured native GPU rollout prefetch 修复说明

日期：2026-04-25

本文说明 `sagin_marl/env/structured_batch_env_core.py` 和 `sagin_marl/env/structured_gpu_rollout_runtime.py` 中 native structured rollout 的正确调度、history 写入、done/reset、bootstrap 和 prefetch 修改要求。

目标不是只减少一个 CUDA segment，而是保证下面三件事同时成立：

1. 中间 step 的 accel obs 不重复跑重 prepare。
2. `rollout_tail`、`episode_tail`、bootstrap 语义正确。
3. `history` 中用于训练和 return 的数据不互相污染。

## 关键概念

### `rollout_tail`

`rollout_tail` 指本次 rollout history 已提交区间的最后一个 env step。

例如 `rollout_env_steps = 250`，则本次 rollout 的 step `249` 是 `rollout_tail`。

`rollout_tail` 只表示本次 buffer 没有普通 future actor input slot 了。它不等于 episode 结束。

采用 `T+1` world storage 后，尾部有一个 bootstrap-only slot `T`，这个 slot 不是下一条 PPO transition 的 actor input slot。

### `episode_tail`

`episode_tail` 指某个 env 在某个 step 结束 episode：

```text
episode_tail = terminated or truncated
```

native core 当前定义为：

```python
energy_depleted_t = (
    bool(reward_p.energy_enabled)
    and torch.any(uav_energy_after_t <= 0.0, dim=1)
)
terminated_t = collision_t.to(dtype=torch.bool) | energy_depleted_t
truncated_t = (t_t.to(dtype=torch.float32) >= float(reward_p.t_steps - 1))
```

位置：`sagin_marl/env/structured_batch_env_core.py` 中 `_build_fast_bw_step_metrics_tensor_impl`。

含义：

```text
terminated:
  collision 或 energy depleted

truncated:
  正常 time-limit
```

`rollout_tail` 和 `episode_tail` 互相独立：

```text
rollout 中间 step 可以是 episode_tail
rollout_tail 可以不是 episode_tail
rollout_tail 也可以同时是 episode_tail
```

### `record_rollout`

当前 native runtime 只有一套 history ring 张量。它同时承担两个角色：

```text
1. env/actor step 的固定张量工作区：
   actor obs 从 history ring 的当前 slot 读取；
   action、reward、done、stage obs 也写入 history ring 的当前 slot。

2. rollout 采样结果：
   history.cursor 之前的 slot 会被 build_rollout_views 作为 PPO 更新数据读取。
```

official training rollout 中 `record_rollout=True`。当前 step 写入 history ring 的当前 slot，并在 finish 后提交为 rollout transition。

```text
record_rollout=True:
  begin 时使用 history.cursor 作为当前 slot
  finish 时 history.cursor = slot + 1
  这个 slot 进入后续 build_rollout_views / PPO update
```

环境推进和 actor I/O 都会用到 history ring，但用途不同。

权威环境状态在 `_runtime_tensor_state`：

```text
uav_pos / uav_vel / queue / energy / association / t / sat_pos / sat_vel / ...
```

history ring 是 rollout step 的张量接口和记录区：

```text
begin_accel_obs:
  从 _runtime_tensor_state 读当前环境状态
  写 accel_stage.*[k] 到 history ring

actor_accel:
  从 history ring 读 accel obs
  把 action / logprob / value 写回 history ring

accel_to_sat / sat_to_bw:
  读当前 _runtime_tensor_state 和 history ring 中的 action
  写 sat/bw obs 到 history ring

finish_bw:
  读当前 _runtime_tensor_state 和 history ring 中的 action
  计算 step 后的新环境状态
  通过 runtime.main.bw_state_out 写回 _runtime_tensor_state
  写 reward / done / next obs / terminal_next_world 到 history ring
```

因此“环境推进靠 BW finish 写 `_runtime_tensor_state`”的意思是：真正的 env state 从 `s_k` 变成 `s_{k+1}`，落在 `_runtime_tensor_state` 里。history ring 负责让 actor 读 obs、写 action，并保存 rollout transition；它不是唯一的环境状态来源。

hot replay / 验证 / profile 不能占用 official training rollout 的 runtime buffers、history ring、CUDA graph capture 或 `runtime.main`。这些用途必须创建独立的 `StructuredGpuRolloutRuntime`、独立 history ring、独立 `runtime.main` buffers，并按自己的 batch/env 数和 step 数预分配。

`record_rollout=False` 只允许用于这种独立 workspace。official training rollout runtime 不使用 `record_rollout=False` 作为 scratch slot。

当前代码中 non-record 路径把 `training_history_slot` 置为 0 的行为不能用于 official training rollout runtime；修复时应删除这类共享 scratch 语义。

### history slot delta / slot offset

当前 step 的 training slot 记为 `k`。

history row index 的基本形式是：

```text
env row   = base_env_row_ids   + slot * num_envs
local row = base_local_row_ids + slot * (num_envs * num_uav)
```

`slot_offset` 或 `slot_delta` 表示相对于当前 `training_history_slot` 的偏移：

```text
slot_offset = 0:
  写当前 step k

slot_offset = 1:
  写下一 step k+1
```

建议代码参数命名尽量明确，例如：

```python
world_history_slot_offset
local_history_slot_offset
```

不要在 CUDA hot path 里用 `.item()` 读 `history_slot_t`。slot 计算应该保持 tensor 运算。

### `runtime.main`

`runtime.main` 是 native 主内核的长生命周期 workspace 和 tensor alias 容器，不是另一套环境状态。

它保存：

```text
当前 stage fields 指针
history output buffers 指针
action/input/output 临时张量
CUDA graph replay 所需的固定 ABI 张量
prefetched next accel fields 指针
bw_state_out alias
```

读写时机：

```text
rollout 初始化:
  预分配 runtime.main 的固定 shape 张量和 history output aliases

begin_accel_obs:
  使用 runtime.main.accel_history_out 写 accel obs 到 history ring
  更新 runtime.main.accel_stage_fields

accel_to_sat:
  读取 runtime.main.accel_stage_fields 和 accel action
  写 runtime.main.sat_stage_fields / sat history rows

sat_to_bw:
  读取 runtime.main.sat_stage_fields 和 sat action
  写 runtime.main.bw_stage_fields / bw history rows

finish_bw:
  读取 runtime.main.bw_stage_fields 和 bw action
  通过 runtime.main.bw_state_out 写回 _runtime_tensor_state
  写 reward / done / terminal_next_world / next accel storage
```

`runtime.main.bw_state_out` 是指向 `_runtime_tensor_state` 字段的输出 alias。BW finish 写它时，实际更新的是权威环境状态。

## next-world 与 next actor obs

目标存储布局：

```text
accel_stage.world_batch:
  capacity = rollout_env_steps + 1
  slot 0..T-1 是 actor_accel 输入 world
  slot T 是 rollout tail bootstrap world

terminal_next_world:
  capacity = rollout_env_steps
  只在 episode_tail env 上有效

terminal_next_world_mask:
  shape = [T, num_envs]
  True 表示 transition_next_world[k, e] 来自 terminal_next_world[k, e]
  False 表示 transition_next_world[k, e] 来自 accel_stage.world_batch[k+1, e]
```

`bw_next_world_batch` 作为独立完整 world ring 删除。`transition_next_world[k, e]` 不再是一个单独 buffer，而是按 mask 解析：

```text
if terminal_next_world_mask[k, e]:
  transition_next_world[k, e] = terminal_next_world[k, e]
else:
  transition_next_world[k, e] = accel_stage.world_batch[k+1, e]
```

这条规则覆盖 rollout tail：

```text
k = T - 1 且不是 episode_tail:
  transition_next_world[k] = accel_stage.world_batch[T]
```

因此 rollout tail 不需要单独的 `bw_next_world_batch[k]`。它和普通非 done transition 一样，直接使用 `accel_stage.world_batch[k+1]`，只是 `k+1` 等于 bootstrap-only slot `T`。

episode tail 必须单独保存 terminal next-world：

```text
terminated=True:
  terminal_next_world[k, e] = BW finish 后、reset 前的 terminal post-step world
  terminal_next_world_mask[k, e] = True
  return bootstrap = 0

truncated=True and terminated=False:
  terminal_next_world[k, e] = BW finish 后、reset 前的 time-limit post-step world
  terminal_next_world_mask[k, e] = True
  return 使用 V(terminal_next_world[k, e])
```

如果 `k` 不是 rollout tail，下一步 actor 输入仍然写到：

```text
accel_stage.world_batch[k+1, e]
```

写入规则：

```text
not episode_tail:
  accel_stage.world_batch[k+1, e] = post-step world
  terminal_next_world_mask[k, e] = False

episode_tail:
  terminal_next_world[k, e] = reset 前 post-step world
  terminal_next_world_mask[k, e] = True
  如果 k 不是 rollout tail:
    accel_stage.world_batch[k+1, e] = reset 后 world
```

actor local obs 与 world 分开处理：

```text
accel_stage local obs capacity = T
accel_stage.world_batch capacity = T+1
```

rollout tail 只需要写 `accel_stage.world_batch[T]` 供 bootstrap 使用，不写 actor local obs slot `T`。

return/bootstrap 构造必须从 `terminal_next_world_mask` 解析 `transition_next_world`。`structured_buffer.py` 和 `structured_mappo.py` 中当前直接读取 `history.bw_next_world_batch` 的位置，需要改为读取：

```text
accel_stage.world_batch[k+1]
terminal_next_world[k]
terminal_next_world_mask[k]
```

## 正确目标调度

正确调度必须同时看两个条件：

```text
rollout_tail_k = step k 是否为本次 rollout buffer 的最后一步
episode_tail_k[e] = env e 在 step k 是否 terminated 或 truncated
```

### rollout 第一步

```text
begin_accel_obs
  如果没有可消费的 next accel obs：
    重 prepare，写 accel obs slot 0
  如果 slot 0 已由上一段 finish 写好：
    直接读取 slot 0，不重 prepare

actor_accel
accel_to_sat
actor_sat
sat_to_bw
actor_bw
finish_bw
  写 reward / terminated / truncated 到 slot 0
  写 transition_next_world[0]
  如果不是 rollout_tail：
    为下一 step 准备 accel slot 1
```

### rollout 中间 step k

```text
actor_accel 读 accel obs slot k
accel_to_sat
actor_sat
sat_to_bw
actor_bw
finish_bw
  写 reward / terminated / truncated 到 slot k
  写 transition_next_world[k]
  如果不是 rollout_tail：
    对每个 env 分情况准备 slot k+1
```

slot k+1 的 per-env 规则：

```text
if not episode_tail_k[e]:
  slot k+1, env e 写正常 next accel obs

if terminated_k[e]:
  return bootstrap 为 0
  下一步 env e 必须是 reset 后 obs
  slot k+1, env e 不能使用 terminal 后自然推进 obs

if truncated_k[e] and not terminated_k[e]:
  return 使用 timeout bootstrap: V(transition_next_world[k, e])
  下一步 env e 必须是 reset 后 obs
  slot k+1, env e 不能使用 time-limit 后自然推进 obs
```

### rollout 最后一步

```text
actor_accel 读 accel obs slot last
accel_to_sat
actor_sat
sat_to_bw
actor_bw
finish_bw
  写 reward / terminated / truncated 到 slot last
  写 transition_next_world[last]
  不写普通 accel_stage slot last+1
  不保留会被下一次 begin 误消费的 next accel fields
```

rollout tail 的 bootstrap 规则：

```text
if terminated:
  bootstrap = 0

elif truncated:
  bootstrap = V(transition_next_world[last])

else:
  bootstrap = V(transition_next_world[last])
```

区别在于 truncated 是 episode time-limit bootstrap，非 done rollout tail 是 rollout-cut bootstrap；二者都需要 `transition_next_world[last]`。

## 当前 done defer 条件

`sagin_marl/rl/structured_train.py` 当前有：

```python
defer_native_done_sync = bool(
    getattr(learner_cfg, "structured_native_defer_done_sync", True)
    and structured_group is not None
    and int(getattr(learner_cfg, "T_steps", 0) or 0) > int(rollout_env_steps)
    and not bool(getattr(learner_cfg, "energy_enabled", False))
)
```

含义：

```text
structured_native_defer_done_sync 为 True
structured_group 存在
T_steps > rollout_env_steps
energy_enabled 为 False
```

这个开关当前把两个概念混在了一起：

```text
1. env correctness:
   done env 是否在下一步 actor 前 reset。

2. CPU episode stats:
   done history 何时同步到 CPU 做日志统计。
```

修复后这两个概念拆开。

当前 reset 实现在 Python 侧执行：`structured_train.py` 读取 done mask 后调用 `_reset_with_optional_controller(...)`，最终进入 `reset_many(...)` 构造 reset state 并写回 runtime tensor state。因此当前代码只要中间 step 出现 `episode_tail`，下一步 actor 前要 reset 对应 env，就必须把 done mask 读回 CPU：

```text
finish step k
读取 terminated/truncated 到 CPU
CPU 决定哪些 env reset
执行 reset / refresh old caches
进入 step k+1
```

这个读取会同步 CUDA stream，也会让中间 step 退出固定 GPU replay 流程进入 Python reset 流程。

目标实现不使用中间 step 的 CPU reset。reset 在 GPU BW finish 内完成。

GPU done-aware reset 的 BW finish 顺序：

```text
1. 计算 post-step state。
2. 计算 terminated / truncated / done_mask。
3. 对 done env:
   保存 terminal_next_world[k] = reset 前 post-step world
   terminal_next_world_mask[k] = True
4. 对 not done env:
   terminal_next_world_mask[k] = False
5. 对 done env:
   从 GPU reset tape 读取 reset state
   把 reset 后状态写入 _runtime_tensor_state
6. 对 not done env:
   把 post-step state 写入 _runtime_tensor_state
7. 如果不是 rollout_tail:
   done env 的 accel_stage.world_batch[k+1] 和 local obs 写 reset 后 obs
   not done env 的 accel_stage.world_batch[k+1] 和 local obs 写 post-step obs
8. 如果是 rollout_tail:
   not done env 写 accel_stage.world_batch[T]
   done env 不写 actor local obs slot T
```

这样中间有 `episode_tail` 也不需要 GPU->CPU 同步。CPU 只在 rollout 结束后读取 done history 用于 episode 统计。

GPU reset 需要把当前 Python reset 依赖的数据 tensor 化，并在 rollout 开始前或 reset 前放到 GPU：

```text
reset_gu_pos_tape
reset_uav_pos_tape
reset_uav_vel_tape
reset_arrival_base_scale_tape
reset_deadline_steps_tape
reset_doppler_residual_tape
reset_effective_arrival_rate_tape
reset_episode_idx_tape
```

shape 按 `[rollout_env_steps, num_envs, ...]` 组织。BW finish 用当前 `rollout_step_t` 和 env row 选择对应 reset rows。

`structured_native_defer_done_sync` 不再参与 env correctness。native 主路径总是要求 GPU done-aware reset；done env 的 reset 必须在 GPU 内立即完成。

修复后处理方式：

```text
collect_env_horizon_native_tensor_policy:
  不在中间 step 做 CPU done sync
  不调用 Python reset_many
  依赖 BW finish 内 GPU done-aware reset 保证下一步 obs 正确

rollout 结束:
  CPU 读取 done history
  只用于 episode return/length 统计和日志
```

因此 `defer_native_done_sync` 这个名字对应的旧开关应删除。不能再用它决定 native GPU rollout 是否可以跨过 episode_tail。

## done/reset 与 next accel obs 的统一要求

任何实现都必须满足：

```text
step k finish 后，如果 env e done：
  在 step k+1 actor_accel 读取前，env e 已经 reset
  env e 的 accel obs slot k+1 是 reset 后 obs
```

不能让 done env 继续读取 terminal 后自然推进 obs。

目标实现是 GPU 内 done-aware reset。

### GPU 内 done-aware next-obs 路径

finish step k 在 GPU runtime 内直接处理 per-env done mask：

```text
not done env:
  写正常 next accel obs 到 slot k+1

done env:
  写 reset 后 accel obs 到 slot k+1
```

这个逻辑需要 reset tape / reset state 在 native runtime 中可用。

不能跳过 done/reset 语义。

## 需要修改的文件

主要文件：

```text
sagin_marl/env/structured_gpu_rollout_runtime.py
sagin_marl/env/structured_batch_env_core.py
sagin_marl/rl/structured_train.py
sagin_marl/rl/structured_mappo.py
sagin_marl/rl/structured_buffer.py
tests/test_structured_system_acceptance.py
```

## 修改 1：`replay_step` 接收 rollout tail 信息

文件：`sagin_marl/env/structured_gpu_rollout_runtime.py`

`StructuredGpuNativeRuntimeStepProgram.replay_step` 增加参数：

```python
def replay_step(
    self,
    *,
    actor_bridge: Any,
    record_rollout: bool,
    deterministic: bool = False,
    rollout_tail: bool,
) -> StructuredBatchStepResult:
```

finish kwargs 规则：

```python
has_next_actor_slot = not bool(rollout_tail)
finish_kwargs = {
    "bw_proxy_base_action_mode": self.bw_proxy_base_action_mode,
    "max_visible": self.fixed_visible_sat_width,
    "record_rollout": record_rollout,
    "rollout_tail": bool(rollout_tail),
    "has_next_actor_slot": bool(has_next_actor_slot),
}
if hasattr(actor_bridge, "env_phase_d_kwargs"):
    finish_kwargs.update(dict(actor_bridge.env_phase_d_kwargs()))
step_result = self._after_bw_action(**finish_kwargs)
```

`has_next_actor_slot` 不是配置项，只由 `rollout_tail` 推导：

```text
has_next_actor_slot = not rollout_tail
```

BW finish 每一步都执行 GPU done-aware reset。`has_next_actor_slot` 只决定是否写下一步 actor local obs；它不决定是否 reset。

## 修改 2：`replay_horizon` 同时处理 rollout tail 和 done

文件：`sagin_marl/env/structured_gpu_rollout_runtime.py`

正确结构：

```python
for step_index in range(steps):
    rollout_tail = bool(step_index + 1 >= steps)
    step_result = self.replay_step(
        actor_bridge=actor_bridge,
        record_rollout=record_rollout,
        deterministic=deterministic,
        rollout_tail=rollout_tail,
    )
```

BW finish 内总是处理 `terminated/truncated`：

```text
保存 terminal_next_world
写 reset 后 _runtime_tensor_state
如果不是 rollout_tail:
  写 reset 后 accel_stage.world_batch[k+1]
  写 reset 后 accel local obs[k+1]
保证 actor_accel 不会读到 terminal 后 obs
```

GPU done-aware reset 是 native GPU rollout 主路径的组成部分。未实现时，native GPU rollout 主路径不算完成。

## 修改 3：移除 native GPU rollout 的 done defer gate

文件：`sagin_marl/rl/structured_train.py`

删除 native GPU rollout 收集前的 `defer_native_done_sync` 正确性判断。native GPU rollout 主实现要求：

```text
GPU done-aware reset 已实现
```

`structured_train.py` 不再根据下面这些条件决定是否允许 native GPU rollout：

```text
T_steps > rollout_env_steps
energy_enabled 为 False
collision 不可能发生
```

原因：

```text
collision / energy depleted / time-limit 都可以在 rollout 中间发生。
这些 done 都必须由 GPU BW finish 立即 reset。
只要 GPU reset 正确，中间 done 不需要 CPU 参与。
只要 GPU reset 未实现，native GPU rollout 主路径就是未完成状态。
```

rollout 结束后的 CPU done 读取只用于 episode statistics，不用于决定 reset。

## 修改 4：删除完整 `bw_next_world_batch`

文件：`sagin_marl/env/structured_batch_env_core.py`

文件：`sagin_marl/env/structured_gpu_rollout_runtime.py`

目标 storage：

```text
accel_stage.world_batch:
  capacity = T + 1

terminal_next_world:
  capacity = T

terminal_next_world_mask:
  shape = [T, num_envs]
```

`StructuredGpuRolloutTrainingRingBuffers` 删除或废弃：

```text
bw_next_world_batch
```

新增：

```text
terminal_next_world
terminal_next_world_mask
```

`preallocate_native_main_kernel_training_ring_buffers` 中 `accel_stage.world_batch` 的 world tensor capacity 改为 `T + 1`。accel local obs / actions / values / logprob 仍是 `T`。

BW finish 写入规则：

```text
not episode_tail:
  写 accel_stage.world_batch[k+1]
  terminal_next_world_mask[k] = False

episode_tail:
  reset 前写 terminal_next_world[k]
  terminal_next_world_mask[k] = True
  如果 k 不是 rollout_tail:
    reset 后写 accel_stage.world_batch[k+1]

rollout_tail and not episode_tail:
  写 accel_stage.world_batch[T]
  terminal_next_world_mask[T-1] = False

rollout_tail and episode_tail:
  reset 前写 terminal_next_world[T-1]
  terminal_next_world_mask[T-1] = True
  不写 actor local obs slot T
```

当前 `bw_native_fused_step_next_accel_obs` 调用通过：

```python
next_world_history_out=runtime.main.bw_next_history_world_out
```

写完整 `bw_next_world_batch`。该输出改为写 `accel_stage.world_batch[k+1]` 的 world rows。

当前 rollout tail 使用的 `bw_native_fused_step` 调用传入：

```python
next_world_batch=None
```

该调用改为写 `accel_stage.world_batch[T]` 或 `terminal_next_world[T-1]`，否则 rollout tail bootstrap 缺失。

## 修改 5：GPU done-aware reset

文件：`sagin_marl/env/structured_batch_env_core.py`

文件：`sagin_marl/env/structured_gpu_rollout_runtime.py`

新增 runtime buffers：

```text
runtime.random.reset_gu_pos_rollout_tape
runtime.random.reset_uav_pos_rollout_tape
runtime.random.reset_uav_vel_rollout_tape
runtime.random.reset_arrival_base_scale_rollout_tape
runtime.random.reset_deadline_steps_rollout_tape
runtime.random.reset_doppler_residual_rollout_tape
runtime.random.reset_effective_arrival_rate_rollout_tape
runtime.random.reset_episode_idx_rollout_tape
```

这些 tape 在 rollout 开始前写入 GPU，shape 使用：

```text
[rollout_env_steps, num_envs, ...]
```

BW finish 内总是执行 done-aware reset 逻辑：

```text
done_mask = terminated | truncated
reset_step = rollout_step_t

terminal_next_world[k, done_mask] = post_step_world[done_mask]
terminal_next_world_mask[k, done_mask] = True
terminal_next_world_mask[k, ~done_mask] = False

state_for_runtime = where(done_mask, reset_state_from_tape[reset_step], post_step_state)
_runtime_tensor_state[...] = state_for_runtime
```

如果 `k` 不是 rollout tail：

```text
next_actor_world = where(done_mask, reset_world_from_tape, post_step_world)
accel_stage.world_batch[k+1] = next_actor_world
accel_stage local obs[k+1] = prepare_accel_obs(next_actor_world)
```

如果 `k` 是 rollout tail：

```text
not done env:
  accel_stage.world_batch[T] = post_step_world

done env:
  terminal_next_world[T-1] = post_step_world
  不写 actor local obs slot T
```

GPU reset 完成后，`runtime.main.prefetched_accel_stage_fields` 对 done env 也是 reset 后 obs，不是 stale terminal obs。

## 修改 6：next actor accel obs 写入 slot k+1

文件：`sagin_marl/env/structured_batch_env_core.py`

当 `has_next_actor_slot=True` 时，next actor accel obs 必须写到 slot k+1。

需要给 history row helper 增加 slot offset：

```python
def _history_slot_row_indices_tensor_impl(
    *,
    history_slot_t: torch.Tensor | None,
    base_row_ids_t: torch.Tensor | None,
    slot_offset: int = 0,
) -> torch.Tensor | None:
    ...
```

当前 slot：

```python
slot_offset=0
```

下一 slot：

```python
slot_offset=1
```

`_prepare_native_stage_and_accel_obs_tensor_impl` 需要支持 world/local 分别写不同 slot：

```python
world_history_slot_offset: int = 0
local_history_slot_offset: int = 0
```

用于 finish 写 next actor obs 时：

```text
accel_stage local/world[k+1]:
  local_history_slot_offset = 1
  accel world 写 slot_offset = 1
```

`T+1 accel_stage.world_batch` 中 world 的 `slot_offset=1` 同时承担：

```text
非 done step 的下一步 actor world
非 done step 的 transition_next_world
rollout tail bootstrap-only world slot T
```

done env 不能使用 reset 后 `slot_offset=1` world 作为 terminal/time-limit `transition_next_world`。

done env 必须写：

```text
terminal_next_world[k]
terminal_next_world_mask[k] = True
```

## 修改 7：保留当前 step 可消费的 next accel fields

文件：`sagin_marl/env/structured_batch_env_core.py`

函数：`_runtime_tensor_finish_bw_and_prefetch_direct_impl`

`next accel fields` 指 BW finish 已经为下一步 actor_accel 写好的 stage fields。它可被下一步 begin 消费的条件是：

```text
env index set 相同
history slot 是下一步当前 slot
done env 已经写 reset 后 obs
not done env 已经写 post-step obs
```

当前逻辑设置 `prefetched_accel_stage_fields` 后又清空。应改为：

```python
runtime.main.prefetched_accel_stage = None
if bool(has_next_actor_slot):
    runtime.main.prefetched_accel_stage_fields = next_stage_fields
    runtime.main.accel_stage_fields = next_stage_fields
else:
    runtime.main.prefetched_accel_stage_fields = None
    runtime.main.next_stage_fields = None
```

GPU done-aware reset 是强制逻辑，因此 `has_next_actor_slot=True` 时 done env 的 next accel fields 也必须是 reset 后 obs，不允许存在 stale terminal obs。

不再引入“done env invalid、下一步 begin 再补救”的语义。reset 由 GPU BW finish 完成后，整批 next accel fields 都应可消费。

## 修改 8：区分 official rollout runtime 和 hot replay runtime

文件：`sagin_marl/env/structured_batch_env_core.py`

official training rollout runtime：

```text
record_rollout=True
使用 rollout 的 history ring
推进 history.cursor
提交给 build_rollout_views
```

hot replay / 验证 / profile runtime：

```text
独立 StructuredGpuRolloutRuntime
独立 history ring
独立 runtime.main buffers
独立 CUDA graph capture
record_rollout=False
不占用 official training rollout 的任何 slot
```

当前 `_runtime_step_publish_bw_obs` 和 `_runtime_step_finish_bw` 硬编码 `record_rollout=True`。修复方式不是让 hot replay 在 official runtime 上借 slot，而是把运行入口拆清楚：

```text
official rollout program:
  固定 record_rollout=True

hot replay program:
  使用独立 runtime
  固定 record_rollout=False
```

wrapper 函数可以保留 `record_rollout` 参数，但 official rollout program 与 hot replay program 的入口必须分开，且 runtime/workspace 必须独立。

禁止：

```text
在 official training rollout runtime 上把 training_history_slot 置 0
在 official training rollout runtime 上运行 record_rollout=False scratch step
hot replay 复用 official rollout 的 runtime.main / history ring / graph capture
```

允许：

```text
hot replay 创建自己的 runtime.main / history ring / graph capture
hot replay 在自己的 runtime 内使用 record_rollout=False
```

## 修改 9：reset 后覆盖 next accel fields

文件：

```text
sagin_marl/env/structured_batch_env_core.py
sagin_marl/rl/structured_train.py
sagin_marl/rl/structured_mappo.py
```

任何 reset 都必须满足：

```text
reset env e 后，不能继续使用 reset 前的 next accel fields
```

实现策略：

```text
GPU BW finish 直接把 reset obs 写到 slot k+1 对应 env rows
prefetched_accel_stage_fields 指向的 next fields 必须已经包含 reset 后 obs
```

现有 `refresh_prefetched_accel_env()` 只处理旧的 `_prefetched_accel_world_states` 缓存。native main-kernel 的：

```text
runtime.main.prefetched_accel_stage_fields
```

由 GPU BW finish 覆盖为 reset 后 fields，不再依赖 Python refresh。

## 修改 10：`begin_accel_obs` 消费 next accel fields 时校验一致性

文件：`sagin_marl/env/structured_batch_env_core.py`

函数：`_runtime_tensor_begin_accel_obs_impl`

当前 begin 只检查：

```python
_is_native_stage_fields(prefetched_fields)
and main.indices == selected_tuple
```

需要增加：

```text
next accel fields 对应的 history slot 是当前 slot
next accel fields 对应 env rows 均有效
reset 后的 env 没有 stale terminal fields
```

GPU BW finish 已经处理 done-aware reset 时，begin 不需要为 done env 重新 reset；它只校验当前 fields 是否匹配当前 slot/env set，然后直接读取 history slot k。

## 修改 11：return/bootstrap 保持 terminated/truncated 区分

文件：

```text
sagin_marl/rl/structured_buffer.py
sagin_marl/rl/structured_mappo.py
```

现有 return 逻辑应保持：

```python
if terminated:
    next_value = 0
elif truncated:
    next_value = truncated_bootstrap_values[idx]
else:
    next_value = next step value or rollout tail bootstrap
```

需要确保 native path 写出的 `transition_next_world[k]` 对三种情况都正确：

```text
terminated:
  可写 terminal post-step world
  return 不 bootstrap

truncated:
  写 time-limit post-step world
  return 使用 V(next_world)

rollout_tail 且未 done:
  写 post-step world
  return 使用 V(next_world)
```

`structured_buffer.py` 和 `structured_mappo.py` 中所有 `next_world_batch` 构造点改为按 `terminal_next_world_mask` 解析：

```text
mask=False:
  next_world = accel_stage.world_batch[k+1]

mask=True:
  next_world = terminal_next_world[k]
```

## 测试要求

### 1. rollout tail bootstrap 测试

构造 `rollout_env_steps < T_steps` 且无 episode done。

断言：

```text
rollout tail 的 transition_next_world 已写
bootstrap_view 包含每个 env 的 latest next_world
return 计算使用非零 bootstrap
```

### 2. time-limit truncated bootstrap 测试

构造当前 env 的 `t` 接近 `T_steps - 1`。

断言：

```text
truncated=True
terminated=False
transition_next_world[k] 已写
return 使用 timeout bootstrap
下一步 obs 来自 reset，不是 terminal 后自然推进 obs
```

### 3. collision terminated 测试

构造 collision。

断言：

```text
terminated=True
return bootstrap 为 0
reward 包含 collision penalty
下一步 obs 来自 reset，不是 collision 后自然推进 obs
```

### 4. energy depleted 测试

开启 `energy_enabled=True`，构造能量耗尽。

断言：

```text
terminated=True
return bootstrap 为 0
reward 包含 battery penalty
下一步 obs 来自 reset
```

### 5. next accel fields 不污染当前 slot

`finish step k + has_next_actor_slot` 后：

```text
history.accel_stage[k] 不变
history.accel_stage[k+1] 写入 next/reset obs
```

### 6. 中间 step 不重复重 prepare

无 done 的 rollout 中：

```text
step 0 begin 调用 prepare_accel_stage_local_obs
step 1..T-1 begin 不调用重 prepare
```

### 7. reset 后不消费 stale next accel fields

step k done/reset 后：

```text
begin step k+1 不读取 reset 前 prefetched_accel_stage_fields
```

## 性能验证

修复后诊断表应满足：

```text
无 done 的中间 step:
  begin_accel_obs 很轻
  不重复出现 prepare_accel_stage_local_obs
  重 env segment 主要是 accel_to_sat / sat_to_bw / finish_bw(+next accel)

rollout tail:
  finish_bw 仍写 transition_next_world
  不写 accel_stage future slot

done step:
  reset 由 GPU BW finish 完成
  但不能牺牲 correctness
```

## 最小修改清单

```text
[x] replay_step 支持 rollout_tail 显式参数
[x] replay_horizon / training loop 传入 rollout_tail 信息
[x] native runtime 增加 GPU reset rollout tapes
[x] BW finish 内实现 done-aware GPU reset
[x] structured_train.py 删除 native GPU rollout 的 defer_native_done_sync 正确性 gate
[x] BW finish 在 rollout_tail 时仍写 transition_next_world[k]
[x] 删除完整 bw_next_world_batch
[x] 引入 T+1 accel world storage、terminal_next_world、terminal_next_world_mask
[x] next actor accel obs 写 slot k+1，不覆盖 slot k
[x] 保留当前 step 可消费的 next accel fields，不保留 stale terminal fields
[x] reset 后由 GPU BW finish 覆盖 native main-kernel next accel fields
[x] record_rollout=False 只用于独立 hot replay runtime，且不推进该 runtime 的 history.cursor
[x] terminated/truncated/bootstrap 逻辑保持现有区别
[x] 增加 rollout_tail、truncated、terminated、reset、next accel fields 测试
```
