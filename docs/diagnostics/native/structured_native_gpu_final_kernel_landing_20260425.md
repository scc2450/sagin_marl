# Structured Native GPU 最终落地方案：减少真实 CUDA kernel 数和 PyTorch op 数

日期：2026-04-25  
范围：native GPU rollout 实现。  
目标：解决“每个 rollout env step 被多个串行段拆开，并且每段内部仍是大量 PyTorch CUDA tensor op / CUDA Graph replay”的问题。

本文不是诊断文档，也不是阶段性计划。本文定义一次到位的最终实现规格。后续实现必须按本文改，删除旧 native rollout 分支。

## 1. 最终结论

改完后的 native rollout 只允许三类存储语义：

```text
1. stage_fields
   runtime.main.accel_stage_field_buffers[2] / sat_stage_fields / bw_stage_fields。
   语义：当前 env step 的阶段状态和下一段 env kernel 的输入。
   accel stage 使用两套固定物理 buffer，由 runtime.main.accel_active_idx 选择 current。
   不保留 prefetched_accel_stage_* 字段和 bookkeeping 语义。

2. live actor I/O
   新增到 runtime.main 的固定地址 actor 输入输出 buffer。
   语义：三个 actor 的当步输入和输出，供 actor/env 热路径直接读写。
   accel actor 输入使用两套固定物理 live obs buffer，由同一个 accel_active_idx 选择 current：
     current_accel_live_obs = runtime.main.accel_live_obs_buffers[accel_active_idx]
     next_accel_live_obs = runtime.main.accel_live_obs_buffers[1 - accel_active_idx]

3. history ring
   现有 runtime.history。
   语义：PPO 训练日志和 per-step StructuredBatchStepResult 的 backing storage，只由 finish_commit_prepare_live_kernel 写入，不参与 actor/env 热路径通信。
   replay_horizon() 返回的 StructuredBatchStepResult 也是 begin_horizon(K) 时一次性绑定到 history/result ring slot 0..K-1 的 view。
   不存在每 step 重建 result view，也不存在单个 live result view 被 H 个 step 共享。
   这些 result 是 runtime-owned view，只保证到同一个 runtime 的下一次 begin_horizon() 前不被复用；外部长期保存必须显式 materialize，不在 hot path 自动 clone。
```

禁止新增 `scratch.sat_stage`、`scratch.bw_stage` 这类与 `sat_stage_fields`、`bw_stage_fields` 语义重复的存储。`stage_fields` 已经是 env 阶段状态。

最终每个 env step 的 env 侧真实 kernel launch 数固定为 3 个：

```text
accel_to_sat_live_kernel
sat_to_bw_live_kernel
finish_commit_prepare_live_kernel
```

step kernel 之外只保留一个初始化 kernel：

```text
prepare_initial_accel_live_kernel
```

`prepare_initial_accel_live_kernel` 只在 runtime 初始化、外部 reset 后的 horizon begin、以及每个 rollout horizon begin 调用。调用前固定设置 `runtime.main.accel_active_idx = 0`，kernel 写 `accel_live_obs_buffers[0] + accel_stage_field_buffers[0]`。标准 `replay_horizon()` 的最后一步带 `rollout_tail=True`，不准备下一步 actor 输入；下一次 horizon begin 必须再次调用 `prepare_initial_accel_live_kernel`。非 rollout-tail step 的 `finish_commit_prepare_live_kernel` 产出下一步 `accel_live_obs_buffers[nxt] + accel_stage_field_buffers[nxt]`，kernel 返回后只翻转 `accel_active_idx`，不得做 Python object host swap。

不能把这 3 个 step kernel 实现成 PyTorch 函数再靠 CUDA Graph replay。它们必须是真正的 CUDA/Triton/custom op kernel。本文以下称为 native CUDA kernel。

固定 3 个 env kernel launch 只约束 Python/PyTorch 调度边界，不代表允许 kernel 内部用单线程串行模拟旧逻辑。最终 native CUDA kernel 必须把 env 内部的 UAV / GU / SAT / candidate slot / obs feature / history row 写入等维度并行化；不能用 `blockDim=1` 或 `if (threadIdx.x != 0) return` 让每个 env 只有一个活跃 thread 完成主要计算。

所有由 `cfg` 决定的 feature/mode 条件必须在 rollout program build 或 workspace 初始化阶段解析成 typed static params、runtime 标量 mode code、固定 fallback tensor 或固定 output pointer。`replay_step()`、actor/env bridge、native binding launch 热路径不得读取 `cfg`、不得 `getattr(cfg, ...)`、不得按配置在 Python 中选择不同 tensor/op 路径。每 step 允许传入的调度标量只有 rollout cursor、random step、`accel_active_idx`、`rollout_tail` 这类运行时状态；done/reset 这类数据依赖条件必须留在 native kernel 内部。

`exec_source_by_stage` 的最终 native 支持范围固定为 `policy`、`zero`、`queue_aware`、`cluster_center_queue_aware`、`teacher`。这些 source 在 rollout program build 时解析成固定 producer 绑定和 `actor_source_mode_code`；live-action source 使用 mode code 0，zero 使用 mode code 1。不得在 step 内用 Python 字符串分支、`cfg` 读取或 PyTorch `index_fill_` 实现 zero/source producer。

## 2. 当前代码事实

### 2.1 外层 step 顺序

当前 native step 顺序在 `sagin_marl/env/structured_gpu_rollout_runtime.py`：

- `_current_or_begin_accel_obs()`：第 433 行。
- `replay_step()`：第 487 行开始。
- actor/env 顺序：第 505 到 563 行。

当前顺序是：

```text
begin/current accel obs
actor_accel
accel_to_sat
actor_sat
sat_to_bw
actor_bw
finish_bw
```

这个 actor 决策边界保留。本文不把三个 actor 合并，也不改变策略语义。

### 2.2 现有 stage_fields 已经是 env 阶段状态

当前 stage buffer 分配在 `sagin_marl/env/structured_batch_env_core.py`：

- `runtime.main.accel_stage_fields`：第 13373 行。
- `runtime.main.prefetched_accel_stage_out_fields`：第 13382 行。
- `runtime.main.sat_stage_fields`：第 13400 行。
- `runtime.main.bw_stage_fields`：第 13421 行。
- `runtime.main.bw_direct_input_fields`：第 13434 行。

这些对象保留为 env 阶段状态，不新增平行 scratch。

### 2.3 当前 actor/env 热路径把 history 当通信总线

当前 actor obs 从 history 读：

- `_runtime_training_history_local_state()` 在 `structured_batch_env_core.py` 第 14199 行。
- 其中 `accel/sat/bw` 都用 `index_select` 读 history，第 14209 到 14241 行。
- `_StructuredMAPPOGpuActorBridge._history_stage_local_state()` 在 `structured_mappo.py` 第 774 行。
- actor bridge 在 `write_accel_action()` 第 858 行、`write_sat_action()` 第 902 行、`write_bw_action()` 第 945 行又重新从 history `index_select`。

当前 actor action 写入 history：

- `structured_mappo.py` 第 872、916、959 行调用 actor。
- 调用时 `action_out=None`，`history_action_out=runtime.history.*.actions`。
- `structured_actor.py` 第 175 行 `_write_history_slot_tensor_out()` 使用 `index_copy_` 写 history。

当前 env 再从 history 读 action：

- accel action：`_apply_accel_stage_prepare_sat_obs_tensor_impl()` 在 `structured_batch_env_core.py` 第 8487 到 8498 行，`accel_action_t is None` 时从 history 读。
- sat action：`_apply_sat_selection_to_bw_stage_and_obs_tensor_impl()` 第 8965 到 8976 行，`selection_t is None` 时从 history 读。
- sat pair action：`_apply_sat_pair_to_bw_stage_and_obs_tensor_impl()` 第 9168 到 9179 行，`pair_idx_t is None` 时从 history 读。
- bw action：`_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl()` 第 11846 到 11864 行，`bw_action_t is None` 时从 history 读。

这些 history 读写在改完后的 native rollout 中全部删除。

### 2.4 当前 env 中间段直接写 history

当前 accel->sat 里会写 SAT history：

- `_apply_accel_stage_prepare_sat_obs_tensor_impl()` 第 8823 行调用 `_prepare_native_stage_and_sat_obs_tensor_impl()`。
- `_prepare_native_stage_and_sat_obs_tensor_impl()` 第 8349 行写 world history，第 8387 行写 local SAT history。

当前 sat->bw 里会写 BW history：

- `_apply_sat_pair_to_bw_stage_and_obs_tensor_impl()` 第 9206 行写 local BW history。
- 第 9275 行写 BW world history。

当前 finish_bw 里会写 BW reward/done/history/terminal/next accel：

- `_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl()` 第 11906 行调用 `_execute_bw_native_fused_step_tensor_impl()`。
- 第 12050 行调用 `_prepare_native_stage_and_accel_obs_tensor_impl()` 准备 next accel。
- 第 12123 行在 tail 场景下复制 terminal world。
- `finish_bw` 外层调用在第 18654 到 18781 行。

这些写 history 的 Python/PyTorch 函数全部从改完后的 native rollout 删除。history 只在 `finish_commit_prepare_live_kernel` 写。

### 2.5 PPO 读取 history 的现有语义

`sagin_marl/rl/structured_buffer.py` 中 `_build_rollout_views_from_native_training_ring()` 第 1188 行读取 native history：

- 当前 step stage world：第 1284 到 1304 行。
- next actor world：第 1305 到 1312 行，从 `accel_stage.world_batch` 的 `start_slot=1` 读取。
- terminal next world：第 1313 到 1325 行，用 `terminal_next_world_mask` 覆盖。
- stage local/action/logprob/value：第 1326 到 1383 行。

最终实现保留现有 PPO history ring prefix layout，保证 PPO reader 不需要改成 base-offset 或新 slot 语义。最终 custom kernel 直接写现有 PPO 所需 history 字段；reward parts 作为同一 flattened ring 语义下的新增诊断字段补齐。

PPO reader 当前按下面的公式构造 BW transition 的训练用 `next_world`：

```text
next_actor_world[k] = accel_stage.world_batch[k + 1]
training_next_world[k] =
  terminal_next_world_mask[k] ? terminal_next_world[k]
                              : next_actor_world[k]
```

这里的两个来源服务同一个语义：

```text
post_bw_world:
  BW action 执行完、reward/done 算完之后、reset 之前的真实后继世界。

runtime_next_world:
  reset 之后 runtime 准备给下一步 actor_accel 看的世界。
  done env: 来自 reset tape。
  non-done env: 等于 post_bw_world。

training_next_world:
  PPO/critic 对当前 BW transition 应该看到的 next_world。
  done env: post_bw_world。
  non-done env: runtime_next_world，也等于 post_bw_world。
```

因此 `terminal_next_world` 不是第二套 history record；它只在 `done_mask` 为 true 的 row 上保存 reset 前的真实后继，让 `training_next_world` 不被 reset 后状态污染。`accel_stage.world_batch[k + 1]` 保存下一步 actor_accel 会看到的 world。对非 done row，它同时也是训练用 next world；对 done row，它会被 `terminal_next_world_mask` 覆盖。

### 2.6 当前 cfg-derived 条件只做了一部分静态化

当前代码已经有一部分 cfg 解析前移：

- `_native_typed_domains_from_cfg()` 在 `structured_batch_env_core.py` 第 577 行开始构造 typed domains。
- workspace 初始化附近第 13289 到 13305 行把 flow proxy、fading、doppler 等配置写入 `runtime.main.*` 标量。
- 第 13461 行和第 13470 行附近已经分配了 fading unity、doppler zero 这类 fallback tensor。

但这还不是最终形态。当前 rollout program build 第 14015 和 14048 行仍读取 `bw_flow_proxy_base_action_mode`；finish wrapper 第 18573、18635、18646 行附近仍在 Python 中根据 reward/fading/doppler 状态选择参数和 tensor。改完后的 hot path 不能只是“不直接读 cfg，但读 runtime.main flag 再 Python 分支”；这些 cfg-derived 条件必须进一步变成 native binding 的 mode code、固定 pointer、固定 fallback tensor，由 kernel 内部统一处理。

## 3. 最终 per-step 数据流

最终 `replay_step()` 语义如下：

```text
cur = runtime.main.accel_active_idx
nxt = 1 - cur

accel_live_obs_buffers[cur] + accel_stage_field_buffers[cur]
  -> actor_accel
  -> live.accel_action / live.accel_old_logprob

live.accel_action + accel_stage_field_buffers[cur]
  -> accel_to_sat_live_kernel
  -> current_sat_stage_fields + current_live_sat_obs

current_live_sat_obs + current_sat_stage_fields
  -> actor_sat
  -> live.sat_action / live.sat_old_logprob

live.sat_action + current_sat_stage_fields
  -> sat_to_bw_live_kernel
  -> current_bw_stage_fields + runtime.main.bw_direct_input_fields + current_live_bw_obs

current_live_bw_obs + current_bw_stage_fields
  -> actor_bw
  -> live.bw_action / live.bw_ref_action / live.bw_old_logprob
     / live.bw_old_logprobs_per_agent / live.bw_old_logprobs_per_slot / live.bw_support_mask

live.bw_action + current_bw_stage_fields + runtime.main.bw_direct_input_fields
  -> finish_commit_prepare_live_kernel
  -> runtime tensor state
  -> history[slot k]
  -> runtime.result.step_result_views[step_index] as prebound view of history/result ring rows
  -> terminal_next_world[slot k]
  -> if rollout_tail:
       accel_stage.world_batch[slot k + 1] only
       no next accel stage/live obs, no accel_active_idx flip
     else:
       accel_stage_field_buffers[nxt] + accel_live_obs_buffers[nxt] for step k+1
       runtime.main.accel_active_idx = nxt
```

`finish_commit_prepare_live_kernel` 同时做五件事：

```text
1. finish current BW transition:
   执行 BW transition，得到 reward、terminated、truncated、post_bw_world。

2. commit current step:
   把当前 step 的 actor I/O、stage world/local、reward/done/aux/reward parts 写到 history[slot k]。
   local 来自 current live obs；world 来自 current stage_fields。

3. write training next override:
   done_mask = terminated | truncated。
   done row 把 post_bw_world 写入 terminal_next_world[slot k]，并把 terminal_next_world_mask[slot k] 置 true。
   non-done row 把 terminal_next_world_mask[slot k] 置 false。

4. update runtime after reset:
   done row 根据 reset tape 写 reset 后 runtime state。
   non-done row 根据 post_bw_world 写正常延续的 runtime state。

5. prepare next actor input:
   rollout_tail=False 时，基于 reset 后 runtime state 构造下一步 accel_stage_field_buffers[nxt] 和 accel_live_obs_buffers[nxt]，kernel 返回后只翻转 accel_active_idx。
   rollout_tail=True 时，不构造下一步 actor 输入，不翻转 accel_active_idx；只把 reset 后或正常延续后的 next actor world 写入 accel_stage.world_batch[slot k + 1]。
```

下一 slot 的 actor 输入来自 live buffers，不来自 history。`rollout_tail=True` 时没有下一 actor step，但 PPO reader 仍需要最后一个 BW transition 的候选 next actor world，所以 finish kernel 只补 `accel_stage.world_batch[slot k + 1]`，不写 local/action/logprob/reward。

## 4. 必须新增的 runtime.main live actor I/O 与 accel parity buffer

在 `StructuredGpuNativeMainKernelBuffers` 中新增以下字段。位置：`sagin_marl/env/structured_gpu_rollout_runtime.py` 第 650 行附近。

```python
accel_active_idx: int                         # 0 or 1
accel_stage_field_buffers: tuple[Any, Any] | None
accel_live_obs_buffers: tuple[StructuredGpuAccelObsView, StructuredGpuAccelObsView] | None
live_sat_obs: StructuredGpuSatObsView | None
live_bw_obs: StructuredGpuBwObsView | None

live_accel_action: torch.Tensor | None          # (num_envs, num_uav, 2), float32
live_sat_action: torch.Tensor | None            # (num_envs, num_uav), long
live_bw_action: torch.Tensor | None             # (num_envs, num_uav, users_obs_max), float32
live_bw_ref_action: torch.Tensor | None         # same as live_bw_action
live_bw_flow_proxy_override_action: torch.Tensor | None # same as live_bw_action

live_accel_old_logprob: torch.Tensor | None     # (num_envs,), float32
live_sat_old_logprob: torch.Tensor | None       # (num_envs,), float32
live_bw_old_logprob: torch.Tensor | None        # (num_envs,), float32

live_bw_old_logprobs_per_agent: torch.Tensor | None # (num_envs, num_uav), float32
live_bw_old_logprobs_per_slot: torch.Tensor | None  # same as live_bw_action
live_bw_support_mask: torch.Tensor | None           # same as live_bw_action

accel_actor_source_mode_code: int               # live-action=0, zero=1
sat_actor_source_mode_code: int                 # live-action=0, zero=1
bw_actor_source_mode_code: int                  # live-action=0, zero=1
```

当前 native actor bridge 没有 rollout critic value 输出。最终 commit kernel 按当前语义把 `accel_stage.values`、`sat_stage.values`、`bw_stage.values` 写 0。不要新增 live value buffer。

## 5. 分配位置和 shape

修改 `sagin_marl/env/structured_batch_env_core.py` 第 13274 到 13418 行附近。

当前代码创建：

```text
accel_action_template
accel_logprob_template
sat_pair_index_template
sat_logprob_template
bw_action_template
bw_ref_action_template
bw_logprob_template
accel_obs_view
sat_obs_view
bw_obs_view
```

最终改为：

```text
1. 这些 template 仍作为 history ring schema 传入 preallocate_native_main_kernel_training_ring_buffers。
2. 同时分配固定地址 live buffer；accel obs 分配两套，sat/bw obs 各一套。
3. actor/env 热路径只读写 runtime.main 的 live actor buffers，不读写 history。
4. accel stage fields 分配两套固定物理 buffer，和 accel_live_obs_buffers 共用 accel_active_idx。
```

具体 shape：

```text
row_count = num_envs * num_uav

accel_live_obs_buffers[0] / accel_live_obs_buffers[1]:
  每套都与当前 accel_obs_view 完全同 shape。
  active_idx 指向当前 actor_accel 输入；1 - active_idx 是非 rollout-tail finish 的 step k+1 输出目标。

live_sat_obs:
  与当前 sat_obs_view 完全同 shape。

live_bw_obs:
  与当前 bw_obs_view 完全同 shape。

live_accel_action:
  (num_envs, num_uav, 2), float32

live_sat_action:
  (num_envs, num_uav), long

live_bw_action / live_bw_ref_action:
  runtime.main.bw_action_shape = (num_envs, num_uav, users_obs_max), float32

live_bw_flow_proxy_override_action:
  runtime.main.bw_action_shape, float32
  只在 flow_proxy_base_action_mode_code == external_live_override 时被 kernel 读；其他 mode 下地址仍固定但语义无效。
  external_live_override 只有在 workspace 初始化时绑定固定 producer 或固定常量时才允许；否则 native rollout strict reject。

live_accel_old_logprob / live_sat_old_logprob / live_bw_old_logprob:
  (num_envs,), float32

live_bw_old_logprobs_per_agent:
  (num_envs, num_uav), float32

live_bw_old_logprobs_per_slot / live_bw_support_mask:
  runtime.main.bw_action_shape, float32

history.bw_stage.reward_parts:
  StructuredGpuRewardPartBuffers | None。
  每个字段都是 (history.capacity * num_envs,), float32。
  字段集合固定为 StructuredGpuRewardPartBuffers / _BW_REWARD_PART_KEYS。
  history.bw_stage.reward_part_tensors 是由 reward_parts dataclass 字段派生出的 dict[str, torch.Tensor] view，不能作为另一套物理存储。

actor_source_mode_code:
  host/runtime int scalar，rollout program build 时由 exec_source_by_stage 解析。
  live-action source=0, zero=1。
  `policy`、`teacher`、`queue_aware`、`cluster_center_queue_aware` 都是 live-action source；不在 step 内读取 cfg 或字符串。
```

optional history/result 字段的最终规则固定如下：

```text
danger_imitation_targets / danger_imitation_masks:
  只有 danger_imitation_enabled 且 buffer present 时分配 history tensor。
  未分配时，对应 StructuredBatchStepResult 和 StructuredStageTrainingBatch 字段都是 None。

bw_flow_proxy_scores / bw_flow_proxy_masks / bw_flow_proxy_deltas:
  只有 flow_proxy_enabled && flow_proxy_reward_mode_code > 0 且 buffer present 时分配 history tensor。
  未分配时，对应 StructuredBatchStepResult 和 StructuredStageTrainingBatch 字段都是 None。
  注意字段名：history/training stage 使用复数 bw_flow_proxy_masks，StructuredBatchStepResult 使用单数 bw_flow_proxy_mask。

reward_part_tensors:
  step result 侧保持现有可选语义；未暴露时是 None。
  StructuredRolloutViews 的 return/diagnostic reward_part_arrays 不使用 None，未暴露时使用空 dict {}。
```

## 6. Actor bridge 必须改成 live I/O

文件：`sagin_marl/rl/structured_mappo.py`。

删除 `_StructuredMAPPOGpuActorBridge._history_stage_local_state()` 在 native rollout 中的使用。不能保留任何从 history 读 actor obs 的 native rollout 分支。

`exec_source_by_stage` 在改完后的 native rollout 中只允许 `policy`、`zero`、`queue_aware`、`cluster_center_queue_aware`、`teacher`：

```text
policy:
  调用对应 actor，写 runtime.main.live_* action/logprob/support buffer。

teacher:
  调用冻结 teacher actor，写 runtime.main.live_* action/logprob/support buffer。
  与 policy 一样必须走 native strict actor contract；不能静默回到 eager PyTorch actor 调度。

queue_aware:
  不调用 actor。
  调用固定 native CUDA source producer 写 live action/logprob/support buffer。
  accel 使用 queue-aware baseline；SAT/BW 使用 queue-aware baseline。

cluster_center_queue_aware:
  不调用 actor。
  accel 调用固定 native CUDA cluster-center source producer，SAT/BW 使用 queue-aware native producer。

zero:
  不调用 actor。
  不调用 index_fill_ / zero_ / copy_ 写 history 或 live CUDA tensor。
  下游 native kernel 按 actor_source_mode_code 直接使用 zero 语义，并在 finish 写出等价 history。
```

zero source 的等价常量固定如下：

```text
accel zero:
  action = 0, old_logprob = 0

sat zero:
  pair index = 0, old_logprob = 0

bw zero:
  action = 0, ref_action = 0, old_logprob = 0
  old_logprobs_per_agent = 0
  old_logprobs_per_slot = 0
  support_mask = 0
```

actor bridge 在 rollout program build 时按 stage source 绑定成固定实现。`policy`/`teacher` 实现调用已捕获的 actor producer；`queue_aware`/`cluster_center_queue_aware` 实现调用 native CUDA source producer；`zero` 实现是 no-op，只让下游 native kernel 通过 `actor_source_mode_code` 使用常量语义。不得在每 step 里用字符串 source 或 `cfg` 做调度分支。

### 6.1 `write_accel_action`

当前问题位置：

- 第 858 行重新从 history 读 accel obs。
- 第 867 到 868 行用 `index_fill_` 清 history values。
- 第 872 到 883 行 actor 写 history action/logprob。

最终实现：

```python
accel_batch = accel_obs
learner.actor.act_accel_into(
    accel_batch,
    action_out=runtime.main.live_accel_action,
    logprob_out=runtime.main.live_accel_old_logprob,
    history_action_out=None,
    history_logprob_out=None,
    history_slot_t=None,
    history_env_row_ids_t=None,
    deterministic=deterministic,
    num_envs=num_envs,
    num_agents=self.num_agents,
)
```

上面是 `policy` source 的绑定实现。`teacher` source 写同一组 live buffer；`queue_aware` / `cluster_center_queue_aware` source 调用 native CUDA producer 写 live buffer；`zero` source 不调用 actor、不写 tensor。

删除 `_zero_history_stage_values()` 调用。values 由 `finish_commit_prepare_live_kernel` 写 0。

### 6.2 `write_sat_action`

当前问题位置：

- 第 902 行重新从 history 读 SAT obs。
- 第 911 到 912 行用 `index_fill_` 清 history values。
- 第 916 到 927 行 actor 写 history action/logprob。

最终实现：

```python
sat_batch = sat_obs
learner.actor.act_sat_pair_into(
    sat_batch,
    pair_index_out=runtime.main.live_sat_action,
    logprob_out=runtime.main.live_sat_old_logprob,
    history_action_out=None,
    history_logprob_out=None,
    history_slot_t=None,
    history_env_row_ids_t=None,
    deterministic=deterministic,
    num_envs=num_envs,
    num_agents=self.num_agents,
)
```

上面是 `policy` source 的绑定实现。`teacher` source 写同一组 live buffer；`queue_aware` / `cluster_center_queue_aware` source 调用 native CUDA producer 写 live buffer；`zero` source 不调用 actor、不写 tensor。

### 6.3 `write_bw_action`

当前问题位置：

- 第 945 行重新从 history 读 BW obs。
- 第 954 到 955 行用 `index_fill_` 清 history values。
- 第 959 到 978 行 actor 写 history action/logprob/support。

最终实现：

```python
bw_batch = bw_obs
learner.actor.act_bw_into(
    bw_batch,
    action_out=runtime.main.live_bw_action,
    ref_action_out=runtime.main.live_bw_ref_action,
    logprob_out=runtime.main.live_bw_old_logprob,
    logprob_per_agent_out=runtime.main.live_bw_old_logprobs_per_agent,
    logprob_per_slot_out=runtime.main.live_bw_old_logprobs_per_slot,
    support_mask_out=runtime.main.live_bw_support_mask,
    history_action_out=None,
    history_ref_action_out=None,
    history_logprob_out=None,
    history_logprob_per_agent_out=None,
    history_logprob_per_slot_out=None,
    history_support_mask_out=None,
    history_slot_t=None,
    history_env_row_ids_t=None,
    deterministic=deterministic,
    num_envs=num_envs,
    num_agents=self.num_agents,
)
```

上面是 `policy` source 的绑定实现。`teacher` source 写同一组 live buffer；`queue_aware` / `cluster_center_queue_aware` source 调用 native CUDA producer 写 live buffer；`zero` source 不调用 actor、不写 tensor。

`structured_actor.py` 中 `act_*_into()` 已支持 `action_out/logprob_out`。最终调用必须传 live out buffer，必须传 `history_* = None`。

## 7. Runtime step API 必须返回 live obs

文件：`sagin_marl/env/structured_batch_env_core.py`。

### 7.1 `_runtime_step_begin_accel_obs`

当前第 14102 到 14108 行调用 `_runtime_training_history_local_state()`。

最终改为：

```text
horizon begin、外部 reset 后的 horizon begin、或 runtime 初始化时调用 prepare_initial_accel_live_kernel。
调用前固定设置 runtime.main.accel_active_idx = 0。
step 内直接返回 runtime.main.accel_live_obs_buffers[runtime.main.accel_active_idx]。
上一 step 是 rollout_tail 时该 horizon 已结束；下一 horizon begin 重新调用 prepare_initial_accel_live_kernel。
episode 内 done reset 不调用 prepare_initial_accel_live_kernel，必须由 finish_commit_prepare_live_kernel 内部完成。
```

### 7.2 `_runtime_step_publish_sat_obs`

当前第 14117 到 14125 行：

```text
env kernel 写 history，然后 _runtime_training_history_local_state(stage_name="sat") 从 history 读 obs。
```

最终改为：

```text
_runtime_tensor_apply_accel_publish_sat_obs_impl 调用 accel_to_sat_live_kernel。
kernel 写 runtime.main.sat_stage_fields 和 runtime.main.live_sat_obs。
_runtime_step_publish_sat_obs 直接返回 runtime.main.live_sat_obs。
```

### 7.3 `_runtime_step_publish_bw_obs`

当前第 14133 到 14141 行：

```text
env kernel 写 history，然后 _runtime_training_history_local_state(stage_name="bw") 从 history 读 obs。
```

最终改为：

```text
_runtime_tensor_apply_sat_publish_bw_obs_impl 调用 sat_to_bw_live_kernel。
kernel 写 runtime.main.bw_stage_fields、runtime.main.bw_direct_input_fields 和 runtime.main.live_bw_obs。
_runtime_step_publish_bw_obs 直接返回 runtime.main.live_bw_obs。
```

### 7.4 `_runtime_step_finish_bw`

当前第 14153 行进入 `_runtime_tensor_finish_bw_and_prefetch_impl()`，最终会调用第 18654 行的 `bw_native_fused_step_next_accel_obs` wrapper。

最终改为：

```text
_runtime_step_finish_bw 调用 finish_commit_prepare_live_kernel。
kernel 写 history[slot]、per-step reward_part ring、runtime state，并按 rollout_tail 条件处理下一步 accel actor 输入。
函数返回 runtime.result.step_result_views[step_index]。
```

这里的 next live accel obs 是条件输出：

```text
rollout_tail=False:
  cur = runtime.main.accel_active_idx
  nxt = 1 - cur
  写 step k+1 的 accel_live_obs_buffers[nxt] 和 accel_stage_field_buffers[nxt]。
  kernel 返回后 runtime.main.accel_active_idx = nxt。

rollout_tail=True:
  不写 accel_live_obs_buffers[nxt]。
  不写 accel_stage_field_buffers[nxt]。
  不翻转 accel_active_idx。
  写 accel_stage.world_batch[slot k + 1] 这个 world-only bootstrap slot。
  下一 horizon begin 由 prepare_initial_accel_live_kernel 重建 actor 输入。
```

删除 `_finalize_native_main_kernel_training_step_result_view()` 在 native rollout 中的调用。当前第 14888 到 14930 行从 history 读 result view，这个读回必须消失。result view 在 `begin_horizon(K)` 时一次性绑定到 history/result ring 的 slot 0..K-1。

## 8. Native CUDA kernel 规格

新增 native CUDA 绑定文件：

```text
sagin_marl/env/native_cuda/__init__.py
sagin_marl/env/native_cuda/bindings.py
sagin_marl/env/native_cuda/kernels.cpp
sagin_marl/env/native_cuda/kernels.cu
```

`bindings.py` 使用 `torch.utils.cpp_extension.load` 加载 C++/CUDA extension。改完后的 native rollout 必须调用这些 binding。`structured_kernel_runtime.py` 的 CUDA Graph wrapper 不再用于 env kernel。

### 8.1 `prepare_initial_accel_live_kernel`

调用时机：runtime 初始化、外部 reset 后的 horizon begin、以及每个 rollout horizon begin 时一次。调用前必须设置 `runtime.main.accel_active_idx = 0`。

替代当前：

- `prepare_accel_stage_local_obs` wrapper。
- `_prepare_native_stage_and_accel_obs_tensor_impl()` 第 8000 行。

输入：

```text
runtime tensor state
runtime random current fading/doppler inputs
static ids/constants
runtime.main.accel_stage_field_buffers[0]
runtime.main.accel_live_obs_buffers[0]
```

输出：

```text
runtime.main.accel_stage_field_buffers[0]
runtime.main.accel_live_obs_buffers[0]
```

禁止输出：

```text
history
runtime.result
```

`prepare_initial_accel_live_kernel` 不处理 episode 内 done reset。episode 内 reset 只在 `finish_commit_prepare_live_kernel` 中根据 `done_mask` 和 reset tape 完成。

### 8.2 `accel_to_sat_live_kernel`

调用时机：actor_accel 之后。

替代当前：

- `_apply_accel_stage_prepare_sat_obs_tensor_impl()` 第 8422 行。
- `_prepare_native_stage_and_sat_obs_tensor_impl()` 的 history 写入行为。

输入：

```text
runtime.main.accel_actor_source_mode_code
runtime.main.live_accel_action when source mode == live-action
runtime.main.accel_stage_field_buffers[runtime.main.accel_active_idx]
runtime tensor state fields required by current accel->SAT logic
static ids/constants
```

输出：

```text
runtime.main.sat_stage_fields
runtime.main.live_sat_obs
runtime tensor state fields owned by accel phase:
  last_policy_accel
  last_exec_accel
  uav position/velocity preview fields required by later phases
```

禁止：

```text
不得读取 runtime.history.accel_stage.actions。
zero source 时不得读取 stale runtime.main.live_accel_action；kernel 内直接使用 accel action = 0。
不得写 runtime.history.sat_stage.*。
不得调用 _build_world_from_packed_specs_tensor_impl。
不得调用 _build_local_sat_obs_from_stage_tensor_impl。
```

### 8.3 `sat_to_bw_live_kernel`

调用时机：actor_sat 之后。

替代当前：

- `_apply_sat_pair_to_bw_stage_and_obs_tensor_impl()` 第 9114 行。
- `_apply_sat_selection_to_bw_stage_and_obs_tensor_impl()` 第 8913 行。

当前 native rollout 使用 pair action，所以改完后只保留 pair action ABI：

```text
runtime.main.live_sat_action shape = (num_envs, num_uav), dtype long。
```

删除 selection action 分支。`sat_selection_to_bw_stage_local_obs` 不进入改完后的 native rollout。

输入：

```text
runtime.main.sat_actor_source_mode_code
runtime.main.live_sat_action when source mode == live-action
runtime.main.sat_stage_fields
runtime.main.sat_subset_members_base
runtime tensor state current selection/counts
static ids/constants
```

输出：

```text
runtime.main.bw_stage_fields
runtime.main.bw_direct_input_fields
runtime.main.live_bw_obs
runtime tensor state fields required before BW:
  last_sat_selection_matrix candidate value
  last_sat_connection_counts candidate value
```

禁止：

```text
不得读取 runtime.history.sat_stage.actions。
zero source 时不得读取 stale runtime.main.live_sat_action；kernel 内直接使用 baseline 对齐的 SAT pair index = 0。这里的 live action 是 pair/subset index，不是解码后的卫星 ID 矩阵；解码后矩阵中的 -1 只表示该选择位为空。
不得写 runtime.history.bw_stage.*。
不得调用 _build_world_from_packed_specs_tensor_impl。
不得调用 _build_local_bw_obs_from_stage_tensor_impl。
不得逐字段 copy_ 填 runtime.main.bw_direct_input_fields；kernel 内直接写目标 buffer。
```

### 8.4 `finish_commit_prepare_live_kernel`

调用时机：actor_bw 之后，每 step 一次。

替代当前：

- `_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl()` 第 11708 行。
- `_execute_bw_native_fused_step_tensor_impl()` 的 PyTorch wrapper 调用。
- `_prepare_native_stage_and_accel_obs_tensor_impl()` 在 finish 中的 next accel prepare。
- `_finalize_native_main_kernel_training_step_result_view()` 第 14860 行。

输入：

```text
slot k = host/runtime rollout cursor scalar passed to binding
random_step = host/runtime random step scalar passed to binding
cur = runtime.main.accel_active_idx
nxt = 1 - cur
runtime.main.accel_live_obs_buffers[cur]
runtime.main.live_sat_obs
runtime.main.live_bw_obs
runtime.main.accel_actor_source_mode_code / sat_actor_source_mode_code / bw_actor_source_mode_code
runtime.main.live_accel_action / live_sat_action / live_bw_action when source mode == live-action
runtime.main.live_bw_ref_action when bw source mode == live-action
runtime.main.live_bw_flow_proxy_override_action fixed pointer, read only when flow_proxy_base_action_mode_code == external_live_override
runtime.main.live_*_old_logprob when source mode == live-action
runtime.main.live_bw_old_logprobs_per_agent when bw source mode == live-action
runtime.main.live_bw_old_logprobs_per_slot when bw source mode == live-action
runtime.main.live_bw_support_mask when bw source mode == live-action
runtime.main.accel_stage_field_buffers[cur]
runtime.main.sat_stage_fields
runtime.main.bw_stage_fields
runtime.main.bw_direct_input_fields

runtime tensor state:
  uav_pos / uav_vel / uav_energy / uav_queue
  gu_pos / gu_queue
  sat_queue / sat_pos / sat_vel
  prev_association / last_association / last_sat_selection_matrix / last_sat_connection_counts
  arrival_ref_bits_per_step / effective_task_arrival_rate / arrival_base_scale
  hotspot_active_idx / hotspot_subset_count / hotspot_member_mask
  traffic_reset_step / traffic_reset_ordinal / episode_idx
  gu_workload_ema / uav_workload_ema / sat_workload_ema
  last_gu_arrival_rate_vec / last_gu_arrival / last_gu_outflow
  last_gu_urgency_risk / last_gu_downstream_pressure / last_gu_service_gap
  last_gu_service_gap_risk / last_gu_deadline_slack / last_gu_deadline_risk / last_gu_deadline_age
  last_exec_accel / last_policy_accel / avoidance_eta_eff / last_avoidance_eta_exec
  doppler_residual
  prev_queue_sum_gu / prev_queue_sum_uav / prev_queue_sum_sat
  prev_q_norm_active / prev_gu_queue_vec / prev_uav_queue_vec / prev_sat_queue_vec
  t / global_step

runtime random / tape inputs:
  arrival_rollout_tape / arrival_rate_rollout_tape
  fading_gain_rollout_tape / runtime.random.fading_gain
  doppler_noise_rollout_tape / runtime.random.doppler_noise
  hotspot_active_rollout_tape / hotspot_mask_rollout_tape / hotspot_active_after_rollout_tape
  reset_followup_arrival_rollout_tape / reset_followup_arrival_rate_rollout_tape
  reset_followup_hotspot_active_after_rollout_tape
  reset_gu_pos_rollout_tape / reset_uav_pos_rollout_tape / reset_uav_vel_rollout_tape
  reset_gu_queue_rollout_tape / reset_uav_queue_rollout_tape / reset_sat_queue_rollout_tape
  reset_deadline_steps_rollout_tape / reset_doppler_residual_rollout_tape
  reset_effective_arrival_rate_rollout_tape / reset_arrival_base_scale_rollout_tape
  reset_arrival_rate_vec_rollout_tape / reset_arrival_ref_rollout_tape / reset_episode_idx_rollout_tape
  reset_hotspot_active_idx_rollout_tape / reset_hotspot_subset_count_rollout_tape / reset_hotspot_member_mask_rollout_tape
  reset_count

cfg-derived static/mode inputs:
  accel/sat/bw actor_source_mode_code
  traffic_model_code / hotspot_mode_code / arrival_feature flags
  fading_enabled / fading_source_code
  doppler_enabled / doppler_model_code / doppler cap/rho/sigma constants
  danger_imitation_enabled / danger_trigger_mode_code / safety threshold constants
  flow_proxy_enabled / flow_proxy_reward_mode_code / flow_proxy_base_action_mode_code
  reward_mode_code / reward normalization constants
  optional obs feature flags and fixed fallback tensor pointers

static/link/orbit inputs:
  orbit pos/vel lookup tables
  link transition override buffers and active mask
  candidate slot/env/uav/gu ids
  sat ids / sat subset metadata / offdiag and upper-pair masks
  BW reward/link constants

history ring field pointers
per-step reward_part ring field pointers
rollout_tail flag
```

finish ABI 不允许把 random/reset/traffic/doppler/fading/flow/danger 逻辑藏在 Python wrapper 里。上述 state group 要么作为 typed static params 传入，要么作为固定 runtime tensor pointer 传入；禁用功能必须通过 mode code 和固定 fallback tensor 在 kernel 内统一处理，例如 fading disabled 使用 unity fading、doppler disabled 使用 zero noise/zero residual 更新、flow proxy disabled 使用 mode code 0 并跳过对应写出。

finish ABI 的 state group 必须完整列入 binding 签名或等价的 packed runtime descriptor：

```text
current runtime state:
  position/velocity/energy/queues/association/selection/workload EMA/deadline/traffic/global counters

current random inputs:
  arrivals / arrival_rates / fading_gain / doppler_noise / current random step

rollout random tapes:
  arrival / arrival_rate / hotspot / fading / doppler

reset tapes and reset counters:
  reset positions / velocities / queues / deadline_steps / doppler_residual / arrival rates / hotspot state / episode_idx / reset_count

static channel/orbit/link inputs:
  orbit lookup tables / candidate ids / masks / bandwidth and reward constants / link override buffers

mode/static params:
  actor source / reward / traffic / hotspot / fading / doppler / danger / flow proxy modes and constants

outputs:
  runtime state after BW/reset / history slot rows / reward_part ring / terminal_next_world / optional flow and danger outputs
```

输出：

```text
runtime tensor state after BW/reset
history[slot k]:
  accel_stage local obs
  sat_stage local obs
  bw_stage local obs
  accel_stage world_batch
  sat_stage world_batch
  bw_stage world_batch
  accel/sat/bw actions
  accel/sat/bw old_logprobs
  bw ref action / per-agent logprob / per-slot logprob / support mask
  accel/sat/bw values = 0
  rewards / terminated / truncated
  bw_access_rewards / bw_weighted_workload_delta_rewards / bw_weighted_workload_level_rewards
  bw_gu_queue_level_rewards / bw_system_queue_level_rewards / bw_gu_service_queue_rewards
  history.bw_stage.reward_parts / reward_part_tensors for all _BW_REWARD_PART_KEYS
  danger_imitation_targets / danger_imitation_masks when buffers are present
  bw_flow_proxy_scores / bw_flow_proxy_masks / bw_flow_proxy_deltas when flow_proxy_enabled && flow_proxy_reward_mode_code > 0 and buffers are present
root history row fields:
  terminal_next_world[slot k]
  terminal_next_world_mask[slot k]
  if rollout_tail: accel_stage.world_batch[slot k + 1] only
if not rollout_tail:
  runtime.main.accel_stage_field_buffers[nxt] for step k+1
  runtime.main.accel_live_obs_buffers[nxt] for step k+1
runtime.random.fading_gain / runtime tensor doppler_residual follow-up buffers
runtime random reset_count / traffic / fading / doppler state after reset-aware update
```

kernel 内部顺序必须是：

```text
1. 读取 slot k、random_step、accel_active_idx，计算 cur/nxt；读取 current stage_fields、current live obs、actor source mode code、live-action source 的 live outputs。
2. 执行 BW transition。bw source 为 live-action 时使用 live_bw_action；bw source 为 zero 时直接使用 zero action。flow proxy base action 按 flow_proxy_base_action_mode_code 在 executed / deterministic / external_live_override 中选择，其中 bw source 为 zero 时 executed 与 deterministic 都解析为 zero action；external_live_override 只有绑定固定 producer 或固定常量时才允许。得到 post_bw_world、reward、terminated、truncated、reward parts、aux metrics。
3. 写完整 history[slot k]。
   local history 从 accel_live_obs_buffers[cur] / live_sat_obs / live_bw_obs 写出。
   world history 从 accel_stage_field_buffers[cur] / sat_stage_fields / bw_stage_fields 写出。
   live-action source 的 action/logprob/support 从 live output buffers 写出。
   zero source 的 action/logprob/support 写固定 zero 常量；SAT live/history action 写 baseline 对齐的 pair index = 0。解码后卫星 ID 矩阵里的 -1 只表示该选择位为空。
   values 写 0。
   写 rewards / terminated / truncated、top-level BW aux reward 字段、所有 history.bw_stage.reward_parts / reward_part_tensors。
   buffer present 时写 danger_imitation_targets / danger_imitation_masks。
   flow_proxy_enabled && flow_proxy_reward_mode_code > 0 且 buffer present 时写 bw_flow_proxy_scores / bw_flow_proxy_masks / bw_flow_proxy_deltas。
   done row 写 terminal_next_world[slot k] = post_bw_world，terminal_next_world_mask[slot k] = true。
   non-done row 写 terminal_next_world_mask[slot k] = false。
4. 根据 done_mask 执行 done-aware reset，写 runtime state。`runtime.result.step_result_views[step_index]` 已经是 history/result ring 的 view，不需要 kernel 另写一份 live result。
5. rollout_tail=False 时，基于 reset 后 runtime state 构造 step k+1 的 accel_stage_field_buffers[nxt] 和 accel_live_obs_buffers[nxt]；kernel 返回后外层只写 accel_active_idx = nxt。
6. rollout_tail=True 时，不写 accel_stage_field_buffers[nxt]，不写 accel_live_obs_buffers[nxt]，不翻转 accel_active_idx；只写 accel_stage.world_batch[slot k + 1] = step k+1 actor world。
```

kernel 返回后的外层 bookkeeping 只更新 runtime host scalar 状态：history cursor 变为 `slot k + 1`，当前 active slot 清空，random step 递增；`rollout_tail=False` 时把 `accel_active_idx` 设为 `nxt`。这些标量不得通过每步 `aten::fill_`、`aten::copy_`、`index_put` 写 CUDA tensor 来实现；如果 native binding 内部仍需要 device scalar，必须由 native 侧维护或由 launch scalar 参数替代。

`finish_commit_prepare_live_kernel` 不得原地覆盖 current_accel_stage_fields 或 current_accel_live_obs。它们既是 step k history 的 source，又在非 rollout-tail step 需要产出 step k+1 actor_accel 输入。普通 CUDA kernel 没有 block 间全局 barrier，所以不能让同一 buffer 同时承担 current source 和 next output。

解决方式不是新增 scratch，也不是 Python object host swap，而是使用两套固定物理 buffer 加 `accel_active_idx`：

```text
cur = runtime.main.accel_active_idx
nxt = 1 - cur

current_accel_stage_fields = runtime.main.accel_stage_field_buffers[cur]
next_accel_stage_fields    = runtime.main.accel_stage_field_buffers[nxt]

current_accel_live_obs     = runtime.main.accel_live_obs_buffers[cur]
next_accel_live_obs        = runtime.main.accel_live_obs_buffers[nxt]
```

非 rollout-tail finish 只写 next_accel_stage_fields 和 next_accel_live_obs；kernel 返回后外层只翻转 `accel_active_idx`。两套物理 buffer 的 tensor 地址始终固定，不得通过 `runtime.main.accel_stage_fields = ...` 或类似 Python 引用替换表达 current/next。

禁止：

```text
不得读取 history action。
不得通过 PyTorch where/index_copy/index_put 写 history。
不得调用 _copy_training_world_rows_where_。
不得调用 _read_history_ring_tensor_rows_。
不得调用 _copy_tensor_out_。
不得原地覆盖 current_accel_stage_fields。
不得原地覆盖 current_accel_live_obs。
不得为 slot k + 1 写 local/action/logprob/reward。
rollout_tail=False 时不写 accel_stage.world_batch[slot k + 1]；该 world 由下一 step 作为当前 accel world 提交。
rollout_tail=True 时只写 accel_stage.world_batch[slot k + 1] 这个 world-only bootstrap slot。
```

### 8.5 不新增独立 bootstrap kernel

不得新增 horizon 末尾独立 kernel。rollout 末尾的 bootstrap next actor world 由最后一个 step 的 `finish_commit_prepare_live_kernel(rollout_tail=True)` 写入 `accel_stage.world_batch[slot k + 1]`。

这个写入只补 PPO reader 所需的 world-only slot，不是下一 step 的 history record：

```text
允许写:
  accel_stage.world_batch[slot k + 1]

禁止写:
  accel_stage local obs[slot k + 1]
  accel/sat/bw actions[slot k + 1]
  accel/sat/bw old_logprobs[slot k + 1]
  rewards/terminated/truncated[slot k + 1]
```

## 9. History 写入语义

每个 env step 只提交一次 logical history record：

```text
finish_commit_prepare_live_kernel(slot k) 写 history[slot k]
```

history 物理行必须按现有 flattened ring 语义计算：

```text
env_row(slot, env_id) =
  slot * num_envs + env_id

local_row(slot, env_id, uav_id) =
  slot * (num_envs * num_uav) + env_id * num_uav + uav_id

finish_commit_prepare_live_kernel(slot k) 写完整 history[slot k]。
rollout_tail=True 时额外只写 accel_stage.world_batch 的 env_row(k + 1, env_id)。
```

字段 row 类型固定如下：

```text
local_row:
  accel/sat/bw local obs fields

env_row:
  accel/sat/bw world_batch
  accel/sat/bw actions
  accel/sat/bw old_logprobs
  accel/sat/bw values
  rewards / terminated / truncated
  terminal_next_world / terminal_next_world_mask
  danger_imitation_targets / danger_imitation_masks when buffers are present
  bw_access_rewards / bw_weighted_workload_delta_rewards / bw_weighted_workload_level_rewards
  bw_gu_queue_level_rewards / bw_system_queue_level_rewards / bw_gu_service_queue_rewards
  history.bw_stage.reward_parts / reward_part_tensors for all _BW_REWARD_PART_KEYS
  bw_ref_actions / bw_old_logprobs_per_agent / bw_old_logprobs_per_slot / bw_support_masks
  bw_flow_proxy_scores / bw_flow_proxy_masks / bw_flow_proxy_deltas when flow_proxy_enabled && flow_proxy_reward_mode_code > 0 and buffers are present
```

上面写成 "when buffers are present" 的字段，指针为 `None` 时 kernel 不写，result/training view 也绑定为 `None`；不得在禁用配置下为了避免 `None` 临时分配一个 hot-path zero tensor。需要零值参与 return/diagnostic 统计的字段在 `StructuredRolloutViews` builder 冷路径里补零或补空 dict。

step k 的 finish kernel 写当前 step 的 PPO 日志。`rollout_tail=True` 时，它额外写 `accel_stage.world_batch[slot k + 1]`，因为 PPO reader 需要这个 world-only slot 作为最后一个 BW transition 的候选 next actor world。这个额外写入不包含 local/action/logprob/reward。

保留当前 PPO reader 语义：

```text
accel_stage.world_batch[0:H] 是当前 accel world。
accel_stage.world_batch[1:H+1] 是候选 next_actor_world。
terminal_next_world_mask[k] 为 true 时，用 terminal_next_world[k] 覆盖候选 next_actor_world[k]。
```

这三个字段共同表达一个训练语义：

```text
post_bw_world:
  BW transition 后、reset 前的真实后继。

runtime_next_world:
  reset 后或正常延续后的下一 actor world。

training_next_world[k]:
  done row: terminal_next_world[k] = post_bw_world。
  non-done row: accel_stage.world_batch[k + 1] = runtime_next_world = post_bw_world。
```

`finish_commit_prepare_live_kernel` 必须先生成 `post_bw_world`，再按 `done_mask` 写 terminal override，然后执行 reset 并生成 `runtime_next_world`。不能先 reset 再尝试恢复 terminal next world。

## 10. 必须删除的旧 native rollout 调用

以下函数不得在改完后的 native rollout step 中出现：

```text
_runtime_training_history_local_state
_StructuredMAPPOGpuActorBridge._history_stage_local_state
_zero_history_stage_values
_write_history_slot_tensor_out
_read_history_ring_tensor_rows_
_finalize_native_main_kernel_training_step_result_view
_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl
_prepare_native_stage_and_accel_obs_tensor_impl
_prepare_native_stage_and_sat_obs_tensor_impl
_build_local_accel_obs_from_stage_tensor_impl
_build_local_sat_obs_from_stage_tensor_impl
_build_local_bw_obs_from_stage_tensor_impl
_build_world_from_packed_specs_tensor_impl
```

这些 Python/PyTorch helper 在 native rollout 中的调用必须删除。native CUDA kernels 必须直接产出同一批 live/history/result 字段，不得通过旧 helper、history readback、index_select、copy_、index_put 实现。

不得保留旧 native rollout 分支。需要对照时使用 legacy backend 或改完后的 native rollout 实现，不在 native runtime 内部保留 history-as-bus 备用分支。

## 11. 外层 program 调度改法

文件：`sagin_marl/env/structured_gpu_rollout_runtime.py`。

当前 `_current_or_begin_accel_obs()` 第 438 到 459 行处理 prefetched accel 并返回 history local rows。

最终改为：

```text
1. runtime 初始化、外部 reset 后的 horizon begin、以及每个 rollout horizon begin 调用 prepare_initial_accel_live_kernel。
   调用前固定设置 runtime.main.accel_active_idx = 0。
2. begin_horizon(K) 把 runtime.history.cursor 重置为 0，确认 history.capacity >= K 且 accel_stage.world_batch capacity >= K + 1，并一次性绑定 runtime.result.step_result_views[0:K] 到 slot 0..K-1。
3. 第 i 个 step 使用 host/runtime step index 作为 slot k=i，不再通过每步 CUDA tensor fill 设置 training_history_slot，也不引入 per-horizon base offset。
4. _current_or_begin_accel_obs() 只返回 runtime.main.accel_live_obs_buffers[accel_active_idx]。
5. finish_commit_prepare_live_kernel 返回后，外层更新 history cursor / active slot host scalar，并递增 random step host scalar。
6. 非 rollout-tail finish 返回后只翻转 runtime.main.accel_active_idx。
7. rollout-tail finish 返回后不翻转；下一 horizon begin 重新 prepare。
8. 不消费 main.prefetched_accel_stage_fields。
9. 删除 prefetched_accel_stage_* bookkeeping 语义。
```

当前 `replay_step()` 第 505 到 563 行的 actor/env 顺序保留。`rollout_tail` 继续控制是否准备下一步 live actor 输入：

```text
rollout_tail=False:
  cur = runtime.main.accel_active_idx
  nxt = 1 - cur
  finish_commit_prepare_live_kernel 写 accel_live_obs_buffers[nxt] 和 accel_stage_field_buffers[nxt]。
  episode done row 使用 reset 后状态生成 accel_live_obs_buffers[nxt]。
  kernel 返回后 runtime.main.accel_active_idx = nxt。

rollout_tail=True:
  finish_commit_prepare_live_kernel 不写 accel_live_obs_buffers[nxt]。
  finish_commit_prepare_live_kernel 不写 accel_stage_field_buffers[nxt]。
  不翻转 accel_active_idx。
  finish_commit_prepare_live_kernel 仍执行 done-aware reset 并更新 runtime state。
  finish_commit_prepare_live_kernel 写 accel_stage.world_batch[slot k + 1] 这个 world-only bootstrap slot。
  下一 horizon begin 由 prepare_initial_accel_live_kernel 重建 actor 输入。
```

当前 `_runtime_tensor_finish_bw_and_prefetch_direct_impl()` 第 14934 到 14985 行负责设置 prefetch。最终删除 native rollout 中的 prefetch 字段更新：

```text
runtime.main.prefetched_accel_stage_fields
runtime.main.prefetched_accel_stage_indices
runtime.main.prefetched_accel_stage_storage_kind
runtime.main.prefetched_accel_stage_history_slot
runtime.main.prefetched_accel_stage_reset_safe
runtime.main.next_stage_fields
runtime.main.accel_stage_fields 作为可重绑定 current 对象的语义
```

最终使用 `runtime.main.accel_stage_field_buffers[2] + runtime.main.accel_live_obs_buffers[2] + runtime.main.accel_active_idx`。实现迁移时可以复用现有 `accel_stage_fields` 和 `prefetched_accel_stage_out_fields` 的分配代码，但最终 runtime 语义不得保留 prefetch 字段名、slot/storage/reset-safe bookkeeping，也不得通过 Python 引用重绑定来表达 current/next。

PPO 读取仍是现有 prefix 语义：horizon 完成后只读 slot `0..K-1`，bootstrap next actor world 读 `accel_stage.world_batch[1..K]` 并与 `terminal_next_world + terminal_next_world_mask` 合成。`begin_horizon(K)` 不是 PPO reader；它只负责重置本次 active history window、绑定 K 个 result view、准备 slot 0 的 actor 输入。

## 12. Result view 改法

当前 `runtime.result.batch_result` 会在 `_finalize_native_main_kernel_training_step_result_view()` 中从 history rows 读出。

最终：

```text
begin_horizon(K) 一次性绑定 runtime.result.step_result_views[0:K] 到本 horizon 的 slot 0..K-1。
每个 step_result_views[i] 都是 StructuredBatchStepResult。
每个 StructuredBatchStepResult 的 tensor 字段直接指向 slot i 对应的 history/result ring row view。
finish_commit_prepare_live_kernel 写 slot i 后，对应 step_result_views[i] 自动看到数据。
```

不得只绑定一个 `runtime.result.batch_result` live 对象并在每 step 覆盖。`replay_horizon()` 当前返回 `list[StructuredBatchStepResult]`；如果 H 个元素共享同一个 live object，horizon 结束后所有 step 都会显示最后一步结果。

不得引入 per-horizon base offset。现有 `StructuredNativeRolloutTrainingBatchView` 和 PPO reader 直接按 prefix 读取 history slot `0..K-1`；最终方案继续保持这个语义。`begin_horizon(K)` 必须把 `history.cursor` 重置为 0，本 horizon 第 i 个 env step 的 slot 就是 `i`。

result view 绑定规则固定为：

```text
runtime.result.step_result_views: list[StructuredBatchStepResult]
runtime.result.horizon_num_steps: int

step_result_views[i].team_rewards ->
  history.bw_stage.rewards.narrow(0, i * num_envs, num_envs)

step_result_views[i].terminated ->
  history.terminated.narrow(0, i * num_envs, num_envs)

step_result_views[i].truncated ->
  history.truncated.narrow(0, i * num_envs, num_envs)

step_result_views[i].danger_imitation_target ->
  history.accel_stage.danger_imitation_targets.narrow(0, i * num_envs, num_envs)

step_result_views[i].danger_imitation_mask ->
  history.accel_stage.danger_imitation_masks.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_access_rewards ->
  history.bw_stage.bw_access_rewards.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_weighted_workload_delta_rewards ->
  history.bw_stage.bw_weighted_workload_delta_rewards.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_weighted_workload_level_rewards ->
  history.bw_stage.bw_weighted_workload_level_rewards.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_gu_queue_level_rewards ->
  history.bw_stage.bw_gu_queue_level_rewards.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_system_queue_level_rewards ->
  history.bw_stage.bw_system_queue_level_rewards.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_gu_service_queue_rewards ->
  history.bw_stage.bw_gu_service_queue_rewards.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_flow_proxy_scores ->
  history.bw_stage.bw_flow_proxy_scores.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_flow_proxy_mask ->
  history.bw_stage.bw_flow_proxy_masks.narrow(0, i * num_envs, num_envs)

step_result_views[i].bw_flow_proxy_deltas ->
  history.bw_stage.bw_flow_proxy_deltas.narrow(0, i * num_envs, num_envs)

step_result_views[i].reward_part_tensors[key] ->
  history.bw_stage.reward_part_tensors[key].narrow(0, i * num_envs, num_envs)

step_result_views[i].reward_mode_active ->
  runtime.main.bw_reward_mode_active
```

上表中的 optional tensor 规则固定为：对应 history tensor 为 `None` 时，`StructuredBatchStepResult` 字段也绑定为 `None`。尤其 flow proxy 字段名必须按上面的 singular/plural 对齐：history 是 `bw_flow_proxy_masks`，result 是 `bw_flow_proxy_mask`。

为保证这些 view 是普通 tensor slice 而不是 `index_select` 临时 tensor，`begin_horizon(K)` 必须使用 contiguous slot block `0..K-1`。若 history 容量不足以容纳 `K` 个 transition slot 和 `K + 1` 个 accel world slot，必须在 horizon begin 前扩容；不得在一个 horizon 内发生 modulo wrap。

`begin_horizon(K)` 设置 `runtime.result.horizon_num_steps = K`，并把所有 `step_result_views[i].reward_mode_active` 设为同一个 `runtime.main.bw_reward_mode_active` 字符串。`reward_mode_active` 是 horizon-static Python 元数据，不是 kernel 输出 tensor，不需要 K 次 CUDA 写。

`reward_part_tensors` 必须一次做到位：在 `StructuredGpuRolloutBwTrainingStageBuffers` 增加

```python
reward_parts: StructuredGpuRewardPartBuffers | None
reward_part_tensors: dict[str, torch.Tensor] | None
```

其中 `reward_parts.<field>` 的物理 shape 是 `(capacity * num_envs,)`，dtype `float32`，field 集合就是 `StructuredGpuRewardPartBuffers` / `_BW_REWARD_PART_KEYS`。`reward_part_tensors` 只是 `{field_name: reward_parts.<field>}` 的 Python dict view，不是第二份存储。`finish_commit_prepare_live_kernel` 写 `slot k` 对应的 `narrow(k * num_envs, num_envs)`。不能继续只用 `runtime.result.reward_parts` 的单步 live buffers，否则 `StructuredBatchStepResult.reward_part_tensors` 仍会发生最后一步 alias。

`StructuredRolloutViews` 也要扩展 reward parts，作为 return/diagnostic 侧字段，不参与 PPO loss 的必要输入：

```python
StructuredReturnBatchView.reward_part_arrays: dict[str, np.ndarray]
StructuredStageReturnBatch.reward_part_arrays: dict[str, np.ndarray]
```

这两个字段当前不在 `sagin_marl/rl/structured_buffer.py` 的 dataclass 中，最终实现必须把 dataclass、native builder、record builder、empty view builder 一起扩展，不能只在 native path 上临时挂属性。

shape 采用与同一 return batch 的 BW reward 字段一致的语义：top-level `return_view.reward_part_arrays[key]` 是 interleaved stage array，shape `(3 * K * num_envs,)`，accel/sat stage 填 0，BW stage 填真实值；`return_view.stage_batches[2].reward_part_arrays[key]` 是 `(K * num_envs,)`。最终 native rollout 在暴露 reward parts 时必须填完整 dict，key 集合就是 `_BW_REWARD_PART_KEYS`；empty view、未暴露 reward parts 的 cold/debug path 使用空 dict `{}`，不使用 `None`。native builder、record builder、empty view builder 都要补齐这个字段，避免 native/legacy 对照和空 batch 构造走不同类型。如果某个外部诊断只需要 step result，可以直接读 `step_result_views[i].reward_part_tensors`，但长期保存仍要 materialize。

`replay_horizon()` 返回 `runtime.result.step_result_views[:K]`。`collect_step(horizon=1)` 返回其中唯一一个 view。PPO 提交仍使用 `num_steps=K` 和 history prefix slot `0..K-1`，不能再通过每步 `_bind_step_result_views()` 或 `_read_history_ring_tensor_rows_()` 构造结果。

`StructuredBatchStepResult` 的生命周期规则必须写进 runtime API：

```text
native replay_horizon() 返回 runtime-owned view。
这些 view 在同一个 runtime 的下一次 begin_horizon() 前有效。
下一次 begin_horizon() 会复用同一批 history/result storage；旧 results 不再保证不变。
hot path 不自动 clone，也不为长期保存做 CPU/GPU materialize。
```

`StructuredNativeRolloutTrainingBatchView` 同样是 history-backed view，不持有一份独立 rollout 数据。正式训练路径保留当前训练代码的顺序：`begin_horizon(K)` / collect 完成本 horizon 后，在下一次 `begin_horizon()` 复用 history 前调用 `buffer.build_rollout_views(...)`，再把得到的 `StructuredRolloutViews` 传给 update。这个 build 仍属于正式训练的 PPO view 构造，不要求 hot rollout path materialize；跨 horizon 保存只在外部长期持有结果的 helper 边界显式处理。

需要长期保存的外部 logging、eval、diagnostic 代码使用 cold API：

```python
runtime.materialize_step_result_view(view: StructuredBatchStepResult, *, device: str | torch.device = "cpu") -> StructuredBatchStepResult
runtime.materialize_horizon_step_results(
    views: Sequence[StructuredBatchStepResult] | None = None,
    *,
    device: str | torch.device = "cpu",
) -> list[StructuredBatchStepResult]
```

materialize API 对所有 tensor 字段执行 `detach().to(device).clone()`，包括 `reward_part_tensors` 中每个 tensor；`None` 保持 `None`；`reward_mode_active` 复制同一个字符串。默认 `device="cpu"` 面向日志和验证；需要 GPU 长期快照时允许 `device="cuda"`。这些 API 不得被 `replay_step()` / `replay_horizon()` / PPO hot path 自动调用。若某个外部 helper 明确承诺返回可跨 horizon 保存的结果，它必须在自己的边界调用 materialize；普通 runtime hot path 不处理这个问题。

## 13. 编译与调用契约

改完后的 env kernel 不再通过 `StructuredKernelRuntime.compile_kernel()` 包装。`structured_kernel_runtime.py` 只服务 actor compile/CUDA Graph，不能包 env native kernels。

新增 binding 必须提供这些 Python 函数：

```python
prepare_initial_accel_live(...)
accel_to_sat_live(...)
sat_to_bw_live(...)
finish_commit_prepare_live(...)
```

`_native_main_kernel_callable()` 不再为 env 段返回 PyTorch eager/compiled wrapper；对应 env 段直接调用 native_cuda bindings。

actor 侧最终必须 true native CUDA 化。`policy` / `teacher` source 的三个 actor head 不能在 rollout hot path 中继续执行 eager PyTorch/ATen tensor 调度，也不能通过 history bridge 做 `copy_` / `gather` / `index_select` / `index_put_` 通信。最终形态固定为 `actor_accel_live` / `actor_sat_pair_live` / `actor_bw_live` 这类固定地址 live actor I/O 上的 native CUDA actor inference kernel；它们必须直接读 `runtime.main.live_*_obs`，直接写 `runtime.main.live_*_action` / logprob / support buffer。`torch.compile(..., backend="cudagraphs")`、Python policy fallback、history-as-action-bus 都不能作为 official native rollout 路径。

actor stochastic 路径必须由 native RNG tape / counter RNG 驱动。每个 actor binding 在 cold path 固定导出 RNG seed；rollout step 只传 `rng_step` 和固定 actor ABI。Gaussian、categorical、Gamma/Dirichlet、Beta/stick-breaking 等采样都在 CUDA kernel 内用 Philox/counter RNG 生成，不能每 step 回到 PyTorch `sample()` / `rsample()`。验收要覆盖 deterministic 与 stochastic，两次相同 `(seed, rng_step, row, stream)` 必须复现，同 seed 不同 `rng_step` 必须改变随机样本；所有 actor 结构、BW `parameterization`、`score_model`、`loc_readout`、competition block 组合都必须能走同一套 native ABI。

如果保留 debug/experimental CUDA Graph direct-input capture，`actor_accel` 必须按 parity 捕获两套 graph，且该路径不得作为 official native rollout：

```text
actor_accel_graph[0] 绑定 runtime.main.accel_live_obs_buffers[0]
actor_accel_graph[1] 绑定 runtime.main.accel_live_obs_buffers[1]
```

每个 step 根据 `accel_active_idx` 选择对应 graph。不得用同一个 direct-input graph replay 动态切换两个不同 data_ptr；不得通过 Python object host swap 伪装 current buffer。

`actor_sat` 和 `actor_bw` 也必须绑定固定 live obs/action buffer；不得把 actor 结果先写到 history、临时 tensor 或 Python object，再由 env kernel 读取。actor native 化后的 profiler 验收范围包含 `actor_accel`、`actor_sat`、`actor_bw` segment 本身，不再只检查 actor/env bridge。

### 13.1 cfg-derived 条件落地规则

最终 hot path 中，以下函数族不得读取 `cfg`、`self._cfg` 或 `getattr(cfg, ...)`，也不得因为配置值不同而走不同 Python 调度分支：

```text
StructuredGpuRolloutRuntime.replay_step / replay_horizon
_current_or_begin_accel_obs
actor bridge write_* / act_*_into
accel_to_sat_live / sat_to_bw_live / finish_commit_prepare_live 的 Python binding wrapper
result view publish/finalize
```

配置只允许在 rollout program build、workspace 初始化、或 native static params 构造阶段解析一次。解析结果必须落到明确的最终形态：

```text
bool/int mode code:
  accel_actor_source_mode_code, sat_actor_source_mode_code, bw_actor_source_mode_code
  fading_enabled, doppler_enabled, danger_imitation_enabled, flow_proxy_enabled
  flow_proxy_reward_mode_code, flow_proxy_base_action_mode_code
  traffic_model_code, hotspot_mode_code, reward_mode_code

fixed tensor pointer:
  unity fading fallback
  zero doppler noise / zero doppler residual fallback
  zero flow proxy output or null output pointer with mode code 0
  live_bw_flow_proxy_override_action
  danger target/mask output pointers when buffers exist

static constants:
  safety thresholds, doppler cap/rho/sigma, traffic constants, reward constants
```

`flow_proxy_base_action_mode_code` 的最终语义固定为：

```text
executed:
  bw source 为 live-action 时使用 runtime.main.live_bw_action；bw source 为 zero 时使用 zero BW action。
deterministic:
  bw source 为 live-action 时使用 runtime.main.live_bw_ref_action；bw source 为 zero 时使用 zero BW action。
external_live_override:
  使用预绑定 runtime.main.live_bw_flow_proxy_override_action。
  只有当 rollout program build/workspace 初始化阶段同时绑定了固定 producer 或固定常量初始化时才允许进入 native rollout。
  如果没有固定 producer，必须 strict reject，不能在 step 内靠 Python kwargs、cfg 分支或临时 tensor 写 override。
```

`runtime.main.live_bw_flow_proxy_override_action` 必须始终有固定地址，shape 与 `runtime.main.bw_action_shape` 一致，dtype `float32`。当 `flow_proxy_base_action_mode_code != external_live_override` 时 kernel 不读取它；当 mode 是 `external_live_override` 时，固定 producer 必须在每个 finish 前按同一 stream 顺序写好这块 buffer，或者它必须是在 workspace 初始化时写好的固定常量。当前配置若只有 `executed` / `deterministic`，则只落地这两个 mode；不要为了“兼容”保留一个会读 stale buffer 的 override 分支。

不得从 history 读 flow proxy base action，不得在 step 内按 cfg 选择 `bw_ref_actions` 或 `bw.actions` 的 history row。数据依赖条件，例如 `done_mask`、每 env reset、terminal override、support mask、链路可用性，保留在 kernel 内。调度条件只保留 `rollout_tail` 这个每 step scalar。

### 13.2 native binding stream 契约

native CUDA binding 必须遵守 PyTorch 当前 CUDA stream：

```text
binding 入口从 at::cuda::getCurrentCUDAStream() 获取 stream，或由 Python 显式传入当前 torch.cuda.current_stream().cuda_stream。
所有 kernel launch 使用这个 stream。
不得隐式使用 default stream。
不得在 replay_step / replay_horizon / binding wrapper 中 cudaDeviceSynchronize、torch.cuda.synchronize 或做 host blocking wait。
不得在 hot step 内分配新的 CUDA tensor。
```

如果 binding 内部需要临时 workspace，必须在 workspace 初始化或 horizon begin 前分配为 runtime-owned buffer，并作为固定指针传入。

### 13.3 native kernel 并行契约

固定 3 个 env kernel launch 只约束外层调度边界，不代表允许 kernel 内部用单线程串行模拟旧 tensor 逻辑。native CUDA kernel 必须把 env 内部主要计算和大 tensor 写入并行化，包括 UAV / GU / SAT / candidate slot / obs feature / history row 等维度。

`queue_aware` / `cluster_center_queue_aware` source producer 也是 official native hot path 的一部分，必须使用 CUDA block/thread 并行实现 GU 权重归约、SAT visible/top-k/subset 评分、BW user 权重归约、cluster center 与 UAV assignment 等主要计算。不得保留一个 thread0 串行扫描所有 GU/SAT/user 的 source producer 作为最终实现。

允许使用 one env one block 作为 grid 组织方式，但 `blockDim.x` 不得固定为 1，也不得用 `if (threadIdx.x != 0) return` 让单个 thread 串行完成整个 env 的主要计算。每个 kernel 应使用合理的 block 线程数，例如 128 或 256 threads，并通过 thread / warp / block 级并行覆盖 env 内循环。对于超过单 block 合理工作量的张量维度，可以使用二维 grid、tiling、grid-stride loop 或多 block per env 的设计，但不得增加 PyTorch op，也不得改变每 env step 的 3 个 env kernel launch 契约。

仅允许极少量 env 级标量 bookkeeping 在单线程执行，例如写 marker、更新 per-env scalar flag、处理最终小规模分支。主要计算和大 tensor 写入不能集中在 thread 0。

## 14. 验收标准

### 14.1 代码级验收

native rollout 实现中：

```text
actor bridge 不调用 _history_stage_local_state。
actor act_*_into 不传 history_* 输出。
zero source 不调用 index_fill_ / zero_ / copy_ 写 history 或 live action；zero 语义只由 native kernel mode code 实现。
env 段不传 action_history_t。
env 段不传 *_history_out 给中间 stage builder。
finish 不调用 bw_native_fused_step_next_accel_obs wrapper。
finish 不调用 _finalize_native_main_kernel_training_step_result_view。
finish 不原地覆盖 current_accel_stage_fields / current_accel_live_obs。
非 rollout-tail finish 写 accel_stage_field_buffers[nxt] / accel_live_obs_buffers[nxt] 后只翻转 accel_active_idx。
actor_accel CUDA Graph 如启用，必须按 accel parity 使用两套 capture。
replay_step / actor bridge / env binding wrapper 不读取 cfg/self._cfg/getattr(cfg, ...)。
actor_source_mode_code / flow_proxy_base_action_mode_code / reward/fading/doppler/danger/traffic mode code 都在 rollout build 或 workspace 初始化阶段固定。
flow_proxy_base_action_mode_code == external_live_override 时必须有固定 producer/常量；否则 native rollout strict reject。
begin_horizon(K) 必须重置 history.cursor=0，step i 使用 slot i；不得增加 per-horizon base offset 或要求 PPO reader 使用 base offset。
正式训练保持 collect horizon -> buffer.build_rollout_views(...) -> update -> 下一次 begin_horizon 的顺序；replay_step/replay_horizon 不自动 materialize。
StructuredBatchStepResult 的 optional 字段按 history tensor 是否存在绑定为 tensor slice 或 None；flow proxy mask 字段名必须按 result 单数 / history 复数映射。
StructuredRolloutViews 的 reward_part_arrays 字段在 native builder、record builder、empty view builder 中都存在；未暴露时是 {}，不是 None。
slot k / random_step / active slot bookkeeping 不通过每步 CUDA tensor fill_/copy_/index_put 实现。
native binding 使用当前 PyTorch CUDA stream，不隐式 default stream，不同步。
native env kernels 不得用 blockDim=1/thread0-only 串行 kernel 作为最终实现；主要 env 内计算和大 tensor 写入必须多线程并行。
```

### 14.2 profile 验收

`kernel_segments.csv` 中 env 段只允许出现：

```text
prepare_initial_accel_live                    count = horizon-begin/external-reset/runtime-init 次数
accel_to_sat_live                             count = H
sat_to_bw_live                                count = H
finish_commit_prepare_live                    count = H
```

不得出现：

```text
prepare_accel_stage_local_obs
accel_prepare_sat_stage_local_obs
sat_pair_to_bw_stage_local_obs
sat_selection_to_bw_stage_local_obs
bw_native_fused_step_next_accel_obs
```

torch profiler 中 env native rollout 的以下 op 必须接近 0：

```text
aten::index
aten::index_put
aten::index_select
aten::gather
aten::scatter
aten::copy_ from env/history bridge
```

torch profiler 中 actor native rollout 的以下 op 也必须接近 0：

```text
aten::copy_
aten::gather
aten::index
aten::index_put
aten::index_select
aten::scatter
```

actor head 不再享有“网络内部仍可使用 PyTorch/compiled actor op”的例外。若某个 actor source 尚未 native CUDA 化，配置不能进入最终验收。

profile 不能只看 kernel launch 数。不得用 `blockDim=1` 或 thread0-only kernel 通过验收；env native kernels 的主要 CUDA work 必须来自多线程并行执行，而不是每 env 一个活跃 thread 的串行循环。

### 14.3 正确性验收

扩展现有 `validate_structured_fixed_seed_long_rollout` 验收：

```text
脚本：scripts/diagnostics/validation/validate_structured_long_rollout_acceptance.py
入口：sagin_marl.rl.structured_eval.validate_structured_fixed_seed_long_rollout

同一 seed、同一 horizon、同一 action replay：
改完后的 native rollout 生成的 StructuredRolloutViews
与 legacy backend 对照的以下字段逐项一致：

accel/sat/bw local_batch
accel/sat/bw world_batch
accel/sat/bw actions
accel/sat/bw old_logprobs
bw_ref_actions
bw_old_logprobs_per_agent
bw_old_logprobs_per_slot
bw_support_masks
danger_imitation_targets when buffers are present
danger_imitation_masks when buffers are present
bw_access_rewards
bw_weighted_workload_delta_rewards
bw_weighted_workload_level_rewards
bw_gu_queue_level_rewards
bw_system_queue_level_rewards
bw_gu_service_queue_rewards
bw_flow_proxy_scores when flow_proxy_enabled && flow_proxy_reward_mode_code > 0 and buffers are present
bw_flow_proxy_masks when flow_proxy_enabled && flow_proxy_reward_mode_code > 0 and buffers are present
bw_flow_proxy_deltas when flow_proxy_enabled && flow_proxy_reward_mode_code > 0 and buffers are present
rewards
terminated
truncated
terminal_next_world
terminal_next_world_mask
next_world
return_view.reward_part_arrays / return_view.stage_batches[2].reward_part_arrays
```

其中 `terminal_next_world` 和 `terminal_next_world_mask` 不是 `StructuredRolloutViews` 的字段；验收必须直接读取 `runtime.history.terminal_next_world` 和 `runtime.history.terminal_next_world_mask` 对照。`StructuredRolloutViews` 只暴露合成后的 `next_world`，也必须单独对照。

`reward_part_arrays` 的验收规则是：最终 native 暴露 reward parts 时，对照 `_BW_REWARD_PART_KEYS` 全量 dict 和每个数组值；未暴露 reward parts 的 cold/empty path，对照空 dict `{}`，不是 `None`。

同一个验收还必须逐 step 对照 `StructuredBatchStepResult` / `runtime.result.step_result_views[i]`，不能只看 horizon 结束后的 `StructuredRolloutViews`。至少直接校验：

```text
step_result_views[i].team_rewards
step_result_views[i].terminated / truncated
step_result_views[i].bw_access_rewards
step_result_views[i].bw_weighted_workload_delta_rewards
step_result_views[i].bw_weighted_workload_level_rewards
step_result_views[i].bw_gu_queue_level_rewards
step_result_views[i].bw_system_queue_level_rewards
step_result_views[i].bw_gu_service_queue_rewards
step_result_views[i].reward_part_tensors
step_result_views[i].danger_imitation_target / danger_imitation_mask when buffers are present
step_result_views[i].bw_flow_proxy_scores / bw_flow_proxy_mask / bw_flow_proxy_deltas when flow_proxy_enabled && flow_proxy_reward_mode_code > 0 and buffers are present
```

逐 step flow proxy 对照必须使用 `StructuredBatchStepResult.bw_flow_proxy_mask` 这个单数字段名；对应的 history/training view 字段是 `bw_flow_proxy_masks`。禁用或未分配 buffer 时，step result 三个 flow proxy 字段都必须是 `None`。

验收 seed 必须覆盖以下情况：

```text
无 done row
terminated row
truncated row
rollout tail row
episode done 与 rollout tail 同时发生的 row
```

14.3 matrix 是组合语义验收，不替代主配置长 rollout。主配置必须另跑完整 horizon 的 2 episode 长 rollout；matrix case 默认使用 `structured_acceptance_matrix_steps=8` 的短 horizon（可由配置显式覆盖），用于快速覆盖 source / flow / terminated / truncated / tail 组合，避免把 400-step 主配置重复跑十几遍。

验收 source mode 必须覆盖：

```text
accel/sat/bw 全 policy
accel/sat/bw 全 queue_aware
accel/sat/bw 全 cluster_center_queue_aware
accel/sat/bw 全 teacher（配置 teacher checkpoint 时）
accel zero + sat policy + bw policy
accel policy + sat zero + bw policy
accel policy + sat policy + bw zero
accel queue_aware + sat policy + bw queue_aware
accel cluster_center_queue_aware + sat policy + bw queue_aware
accel teacher + sat policy + bw queue_aware（配置 teacher checkpoint 时）
accel/sat/bw 全 zero
```

actor native 结构与 stochastic parity 必须覆盖：

```text
accel_policy / sat_pair_policy / bw_policy 三个 head 都走 native actor kernel
BW parameterization 全部已知模式
BW score_model=neural / lowdim_q_eta
BW loc_readout 全部已知模式
BW competition arch/layers/heads
同 seed + 同 rng_step 输出完全复现
同 seed + 不同 rng_step stochastic 输出发生变化且 logprob/action/support 有限
profiler actor_accel / actor_sat / actor_bw segment 不出现 forbidden ATen op
```

flow proxy mode 覆盖：

```text
flow_proxy disabled
flow_proxy enabled + executed base
flow_proxy enabled + deterministic base
flow_proxy enabled + external_live_override base only when a fixed override producer/constant is implemented
```

长 rollout 对照回放必须使用 baseline 记录的完整 native random tape，而不是只记录 arrival tape 或重新依赖两边 RNG 同步。每个 step trace 至少包含并在 `begin_horizon(K)` 前写回：

```text
arrival_tape
arrival_rate_tape
fading_gain_tape
doppler_noise_tape
```

这样 `prepare_initial_accel_live_kernel` 读取 step 0 的 fading/doppler，`finish_commit_prepare_live_kernel` 读取 step k+1 的 fading/doppler 时，native replay 与 baseline canonical payload 使用同一套随机输入。缺少任一字段必须视为验收 trace 不完整并失败，不能退回到重新采样或只比较 native/legacy 差值。

当 fading 或 doppler 功能关闭、runtime 没有实际 rollout tape tensor 时，对照 trace 仍必须写出等价固定语义：fading 为全 1 unity，doppler noise 为全 0。replay 只能接受这些固定 fallback 值，不能因为 tape tensor 不存在而重新采样。

数值容差：

```text
float32: atol=1e-5, rtol=1e-5
bool/long: exact match
```

### 14.4 性能验收

正式 benchmark 使用现有：

```text
artifacts/structured_native/training_system_cuda_8env_250step_*.json
runs/diagnostics/native_rollout_profile/*/step_segments.csv
runs/diagnostics/native_rollout_profile/*/kernel_segments.csv
```

验收目标：

```text
env step 内真实 env-side CUDA kernel launch 数从“大量 PyTorch op replay”降为 3。
finish_bw 不再是一个包含大量 PyTorch tensor op 的 wrapper。
actor_accel/actor_sat/actor_bw 中 history bridge 相关 index_select/index_copy 消失。
每个 env kernel 内部的主要工作由多线程并行承担，不允许 blockDim=1/thread0-only 串行实现伪装成 native CUDA。
```

## 15. 最终实施清单

按这个顺序一次提交：

```text
1. 新增 native_cuda extension 和 4 个 binding。
2. native CUDA kernels 必须按 env 内维度并行化实现；禁止 blockDim=1/thread0-only 串行 kernel 作为最终实现。
3. 在 StructuredGpuNativeMainKernelBuffers 增加 accel_stage_field_buffers[2]、accel_live_obs_buffers[2]、accel_active_idx、actor_source_mode_code、live_bw_flow_proxy_override_action，以及 sat/bw live actor I/O 字段。
4. 在 workspace 初始化处为两套 accel stage/live obs、单套 sat/bw live obs、live action/logprob/support、history.bw_stage.reward_parts per-step ring、K-step result views 分配或绑定固定地址 buffer；danger/flow proxy optional history tensor 不存在时保持 None，不临时分配 hot-path zero tensor。
5. 修改 actor bridge，只读传入 live obs；policy/teacher source 只写 live action/logprob；queue_aware/cluster_center_queue_aware source 只调用 native CUDA producer 写 live action/logprob/support；zero source no-op，不写 history 或 live tensor。
6. 修改 begin/publish/finish runtime API，返回 live obs，调用 native_cuda kernels。
7. 删除 native rollout prefetch bookkeeping，不保留 prefetched_accel_stage_* 字段语义，不用 Python object host swap 表达 current/next。
8. 修改 result view：begin_horizon(K) 重置 history.cursor=0，并一次性绑定 K 个 StructuredBatchStepResult view 到 slot 0..K-1，包含 reward_part_tensors 和 horizon-static reward_mode_active；optional field 按 history tensor 存在与否绑定为 slice 或 None；flow proxy mask 明确按 history.bw_flow_proxy_masks -> result.bw_flow_proxy_mask 映射；不保留单 live result alias，不引入 per-horizon base offset。
9. 在 finish_commit_prepare_live_kernel 内实现 rollout_tail world-only bootstrap slot 写入。
10. 在 finish_commit_prepare_live_kernel 内按 env_row/local_row 两套 stride 写 history。
11. 将 cfg-derived 条件和 exec_source_by_stage 解析到 typed static params / runtime mode code / fallback tensor，热路径不读 cfg、不按 cfg 做 Python 分支。
12. 将 slot k、history cursor、random step、active slot 改成 host/runtime scalar bookkeeping 或 native-owned scalar，不用每步 PyTorch CUDA fill/copy 更新。
13. native binding 统一使用当前 PyTorch CUDA stream，不同步，不在 hot step 内分配 CUDA tensor。
14. 删除旧 native rollout history bridge/helper 调用，不保留 history-as-bus 分支。
15. 增加 cold materialize API，供外部 logging/eval/diagnostic 跨 horizon 保存 StructuredBatchStepResult；replay_step/replay_horizon/PPO hot path 不自动 clone；正式训练继续在下一次 begin_horizon 前调用 buffer.build_rollout_views(...)。
16. 扩展 StructuredRolloutViews 的 return/diagnostic reward_part_arrays，并同步修改 native builder、record builder、empty view builder；字段类型固定为 dict，未暴露时是 {}，保持 PPO loss 不依赖该字段。
17. 对 external_live_override flow proxy base 要么绑定固定 producer/常量并验收，要么 strict reject，不保留会读 stale buffer 的分支。
18. 扩展 validate_structured_fixed_seed_long_rollout 字段级验收，加入 step_result_views 逐 step 对照、runtime.history.terminal_next_world / terminal_next_world_mask 直接对照、next_world 合成结果对照、reward parts dict 对照、完整 random tape 回放，并保留 profiler 验收脚本。
```

这就是最终落地形态：accel current/next 由两套固定物理 `stage_fields + live obs` 和 `accel_active_idx` 表达，sat/bw 使用单套 live/stage buffer，`history` 只由 native fused kernel 作为 PPO 日志写入。最终目标不是移动 Python 调度边界，而是把现在 wrapper 内部的大量 PyTorch CUDA tensor op 替换成固定数量的真实 native CUDA kernels。
