# Structured Native 单 GPU 现状诊断

日期：2026-04-25  
编码：UTF-8  
基线文档：`docs/structured_native_gpu_diagnosis_20260424.md`  
范围：当前 native rollout 主路径、runtime prefetch、GPU reset 后的训练热路径、当前 benchmark/profile 产物。

## 1. 结论

当前核心结论仍然是：

```text
rollout 每个 env step 在 GPU 上被拆成多个串行段，并且每段内部仍是大量 PyTorch CUDA tensor op。
```

但 2026-04-24 文档里的一个具体根因已经改变：当时怀疑的“`finish_bw` 里准备了 next accel obs，下一步 `begin_accel_obs` 又重新 prepare 一遍”现在基本已经修掉。当前 profile 里 `prepare_accel_stage_local_obs` 只出现 1 次，说明 rollout 第一拍之后的 accel obs 已经在消费 BW finish 预取结果。

所以当前状态应更新为：

1. `begin_accel_obs` 重复 prepare 不再是主因。
2. 吞吐仍未回到历史最快的 500+ env steps/s。
3. 当前主耗时集中在 `finish_bw`、`accel_to_sat` 和三个 actor 段。
4. `finish_bw` 仍然是最重段，因为它同时承担 BW transition、reward/done/reset、terminal world、post-step world、next accel obs、history 写入和随机 tape/traffic 状态推进。
5. CUDA Graph 仍只是 replay 一串 PyTorch tensor op，不是把这些 op 变成一个真正的大 fused CUDA kernel。

## 2. 当前诊断产物

普通 benchmark：

```text
artifacts/structured_native/training_system_cuda_8env_250step_after_record_rollout_name_cleanup_20260425.json
```

segment/kernel 表：

```text
runs/diagnostics/native_rollout_profile/20260425_after_record_rollout_name_cleanup/
```

torch profiler op 表和 trace：

```text
runs/diagnostics/native_rollout_profile/20260425_after_runtime_cleanup_torch_ops/
```

其中 `trace.json` 可以用 Perfetto 打开：

```text
https://ui.perfetto.dev/
```

## 3. 正式速度结果

这组是不带 profiler 的普通 benchmark，可和之前普通 benchmark 比较：

| 指标 | 当前值 |
|---|---:|
| `env_steps` | 2000 |
| `transition_samples` | 6000 |
| `env_steps_per_sec` | 353.59 |
| `samples_per_sec` | 1060.76 |
| `rollout_total_time_sec` | 5.5353 |
| `update_total_time_sec` | 0.1211 |
| `iteration_time_sec` | 5.6563 |
| `cpu_util_percent` | 6.14 |
| `gpu_util_percent` | 61.00 |
| `gpu_memory_util_percent` | 22.27 |

解释：

1. 时间仍主要在 rollout，PPO update 很小。
2. 当前 `gpu_util_percent=61%` 低于旧文档里的 94%，但这个指标是采样型上下文指标，不足以单独判断效率。
3. 吞吐从旧文档记录的 230.51 env steps/s 回升到 353.59 env steps/s，但仍未回到历史最快的 500+。

## 4. 当前每 step segment 表

来自：

```text
runs/diagnostics/native_rollout_profile/20260425_after_record_rollout_name_cleanup/step_segments.csv
```

| segment | count | avg ms/step | 占 total_step |
|---|---:|---:|---:|
| `begin_step` | 250 | 0.0226 | 0.10% |
| `begin_accel_obs` | 250 | 1.0247 | 4.49% |
| `actor_accel` | 250 | 2.5374 | 11.12% |
| `accel_to_sat` | 250 | 3.8928 | 17.07% |
| `actor_sat` | 250 | 1.8462 | 8.09% |
| `sat_to_bw` | 250 | 2.1368 | 9.37% |
| `actor_bw` | 250 | 2.2813 | 10.00% |
| `finish_bw` | 250 | 8.9019 | 39.03% |
| `finish_step` | 250 | 0.0078 | 0.03% |
| `total_step` | 250 | 22.8107 | 100.00% |

最重要的变化：

| 指标 | 2026-04-24 诊断 | 当前 |
|---|---:|---:|
| `begin_accel_obs` | 6.1799 ms/step | 1.0247 ms/step |
| `finish_bw` | 6.9827 ms/step | 8.9019 ms/step |
| `prepare_accel_stage_local_obs` 调用 | 每 step 可疑重复 | 只 1 次 |

`begin_accel_obs` 已经从主要热点退下来；当前最重段转移到了 `finish_bw`。

## 5. 当前 kernel 表

来自：

```text
runs/diagnostics/native_rollout_profile/20260425_after_record_rollout_name_cleanup/kernel_segments.csv
```

| kernel wrapper | count | calls/step | avg ms/step | 占 total_step |
|---|---:|---:|---:|---:|
| `bw_native_fused_step_next_accel_obs` | 250 | 1.000 | 8.6472 | 37.91% |
| `accel_prepare_sat_stage_local_obs` | 250 | 1.000 | 3.5518 | 15.57% |
| `sat_pair_to_bw_stage_local_obs` | 250 | 1.000 | 1.5442 | 6.77% |
| `prepare_accel_stage_local_obs` | 1 | 0.004 | 0.3366 | 1.48% |

这张表说明两个事实：

1. `prepare_accel_stage_local_obs` 只在 rollout 第一拍出现，说明 next accel prefetch 已生效。
2. `bw_native_fused_step_next_accel_obs` 名字里虽然有 `fused`，但它仍是 PyTorch compiled/CUDA graph wrapper 级别的融合，不是单个手写 CUDA kernel。它内部仍会展开为大量 CUDA tensor op。

## 6. torch profiler op 表

本轮 torch profiler 只用于诊断，吞吐不能和普通 benchmark 直接比较。该 run 输出：

```text
env_steps_per_sec = 271.85
torch_profiler_ops = 174
torch_profiler_segment_ops = 197
```

op 表显示当前 active window 内大量小 CUDA op：

| op 类别 | count | self cuda ms | 说明 |
|---|---:|---:|---|
| `native_segment::finish_bw` | 10 | 150.76 | 最重 segment |
| `native_segment::accel_to_sat` | 10 | 71.09 | 第二重 env segment |
| `native_segment::actor_accel` | 10 | 43.57 | actor compiled region + copy/gather |
| `native_segment::actor_bw` | 10 | 31.12 | actor compiled region + copy/gather |
| `native_segment::sat_to_bw` | 10 | 29.68 | env segment |
| `native_segment::actor_sat` | 10 | 25.16 | actor compiled region + copy/gather |
| index/index_put CUDA kernels | 2130 | 17.33 | 大量索引写 |
| vectorized mul CUDA kernels | 4040 | 14.66 | 大量 elementwise |
| elementwise mul CUDA kernels | 1830 | 10.76 | 大量 elementwise |
| clamp/compare/where kernels | 数千级 | 多个 3 到 10 ms 累计 | 小 op 很多 |
| `aten::copy_` / DtoD memcpy | 1750 | 4.46 | device-to-device copy 很多 |
| `aten::gather` | 650 | 2.16 | stage/local obs 构造大量 gather |

这正好支撑核心判断：GPU 忙并不等于吞吐高。当前很多时间消耗在大量很小的 tensor op、index/gather/copy/reduce/elementwise kernel 的串行 replay 上。

## 7. 代码路径分析

### 7.1 每个 step 的串行边界

固定调度在 `sagin_marl/env/structured_gpu_rollout_runtime.py`：

```text
_current_or_begin_accel_obs
actor_accel
accel_to_sat
actor_sat
sat_to_bw
actor_bw
finish_bw
```

代码位置：

```text
sagin_marl/env/structured_gpu_rollout_runtime.py:433
sagin_marl/env/structured_gpu_rollout_runtime.py:487
sagin_marl/env/structured_gpu_rollout_runtime.py:608
```

这些边界由 actor 决策依赖决定，不能简单把整个 env step 合成一个纯 env kernel。真正能优化的是：

1. 减少每个边界内的 PyTorch op 数量。
2. 降低 actor bridge 的 copy/index 写入。
3. 把确实稳定的大段逻辑下沉为更粗粒度 fused kernel。
4. 避免在训练 ring 和 runtime state 之间写重复数据。

### 7.2 prefetch 现在已经被消费

`_current_or_begin_accel_obs()` 现在会优先消费 `main.next_stage_fields`，并校验：

```text
prefetched_accel_stage_indices
prefetched_accel_stage_storage_kind
prefetched_accel_stage_history_slot
prefetched_accel_stage_reset_safe
```

这意味着中间 step 的 accel obs 不再重新走完整 `begin_accel_obs` prepare。代码位置：

```text
sagin_marl/env/structured_gpu_rollout_runtime.py:433
```

profile 中 `prepare_accel_stage_local_obs count=1` 与这段代码一致。

### 7.3 BW finish 当前承担太多职责

`_runtime_tensor_finish_bw_and_prefetch_direct_impl()` 只用 `rollout_tail` 决定是否写下一步 actor obs：

```text
write_next_actor_obs = not rollout_tail
```

代码位置：

```text
sagin_marl/env/structured_batch_env_core.py:14934
sagin_marl/env/structured_batch_env_core.py:14945
```

然后进入 `_execute_bw_stage_native_main_direct_out()`，其中 `run_post_step_world_prepare = True`，所以 BW finish 每步都会构造 post-step world。非 tail 还会写 next actor local obs 和保存 prefetch。代码位置：

```text
sagin_marl/env/structured_batch_env_core.py:18508
```

核心 wrapper 是：

```text
bw_native_fused_step_next_accel_obs
```

代码位置：

```text
sagin_marl/env/structured_batch_env_core.py:11708
```

这个 wrapper 内部会做：

1. 选择 rollout arrival/rate/fading/doppler tape。
2. 执行 BW transition。
3. 计算 reward、done、collision、energy depleted、timeout。
4. 写 terminal next world。
5. 对 done env 做 GPU reset。
6. 写 runtime tensor state。
7. 写 rollout history。
8. 构造 post-step world。
9. 非 tail 时写 next accel local obs。

因此它成为当前最大热点是合理的。

### 7.4 reset 原生化不是当前速度主因，但增加了 BW finish 复杂度

GPU done-aware reset 现在在 BW finish 内完成，相关输入包括 reset rollout tape、reset count、traffic reset step/ordinal、episode idx 等。代码位置：

```text
sagin_marl/env/structured_batch_env_core.py:10605
```

这条路径是正确性上必须的，因为 rollout 中间可能有 episode tail。它不应回退到 Python per-step reset。但它也使 `finish_bw` 变成一个同时处理 transition 和 reset 状态机的大段 tensor 程序。

### 7.5 PPO 读取语义已经是 terminal mask + next actor world

训练 view 构造里：

```text
next_actor_world = accel_stage.world_batch[k + 1]
terminal_next_world = history.terminal_next_world[k]
next_world = where(terminal_next_world_mask, terminal_next_world, next_actor_world)
```

代码位置：

```text
sagin_marl/rl/structured_buffer.py:1188
```

这说明 rollout tail / episode tail 的 next world 语义已经从存储层区分开，不是当前吞吐瓶颈。

## 8. 与 2026-04-24 诊断的差异

仍然成立：

1. 热点在 rollout，不在 PPO update。
2. 每个 env step 被 actor 决策拆成多段串行。
3. CUDA Graph replay 不等于真正的大 kernel fusion。
4. env 段内部仍有大量 PyTorch tensor op。
5. `finish_bw` 和 stage obs 构造是主要优化方向。

已经变化：

1. `record_rollout` 不再是调度开关。
2. hot replay 已经走独立 runtime/history/main workspace。
3. GPU reset 已经进入 BW finish，rollout 中间不再依赖 Python reset。
4. `prepare_accel_stage_local_obs` 不再每 step 重复执行，当前 count 为 1。
5. `begin_accel_obs` 不再是主热点。

需要更新旧文档的地方：

1. 旧文档把“重复 prepare next accel”列为最优先根因，现在应标为已修复。
2. 旧文档里的 `record_rollout`、`prepare_next_accel`、`has_next_actor_slot` 相关描述已过期。
3. 当前应把主热点改写为 `finish_bw` 内部的 BW transition/reset/post-step world/next local obs 大段 tensor op。

## 9. 下一步诊断建议

下一刀不应该再看外层 step 表，而应该进入 `finish_bw` 内部拆账。

建议做一张新的内部表，把 `bw_native_fused_step_next_accel_obs` 拆成这些子项：

| 子项 | 目的 |
|---|---|
| arrival/rate/fading/doppler tape 选择 | 看随机 tape 和 traffic reset followup 成本 |
| BW link/rate/queue transition | 看真正 BW 环境动力学成本 |
| reward/done/energy/collision | 看终止和 reward 成本 |
| terminal_next_world 写入 | 看 terminal storage 成本 |
| GPU reset where 写回 | 看 done-aware reset 成本 |
| post-step world 构造 | 看 world_batch[k+1] 成本 |
| next accel local obs 写入 | 看非 tail prefetch 成本 |
| history action/reward/done 写入 | 看训练 ring 写入成本 |

可以用两种方法：

1. 临时 A/B 开关：分别关闭或替换某个子功能，比较 `finish_bw` ms/step。
2. 内部 profiler 标记：在 eager/compiled 边界内给关键子函数加独立 callable 或 `record_function`，让 torch profiler 能分段归因。

## 10. 优化优先级

优先级 1：拆 `finish_bw` 内部成本。

目标不是马上删除逻辑，而是先知道 `8.6 到 8.8 ms/step` 里面哪一块最大。

优先级 2：减少 stage obs 构造里的 gather/index/copy。

`accel_to_sat` 仍有 3.55 ms/step 的 `accel_prepare_sat_stage_local_obs`，torch profiler 也显示 `aten::gather`、`index_select`、`index_put` 和 elementwise 很多。

优先级 3：减少 actor bridge 写 history 的 copy/index 开销。

三个 actor 段合计约：

```text
actor_accel + actor_sat + actor_bw = 6.665 ms/step
```

actor 本身是必须的，但 action/logprob/value 写入 history 的 copy/index/fill 仍可能有优化空间。

优先级 4：如果 PyTorch compile/CUDA Graph 仍无法把大量小 op 合并，再考虑更底层 fused kernel。

当前证据支持：继续只在 Python 层整理调度，收益会越来越小。真正的大收益大概率来自减少 tensor op 数量，或者把最重的局部逻辑下沉为更粗的 CUDA/Triton/custom op。

## 11. 当前一句话判断

是的，核心仍然是：

```text
每个 rollout env step 被拆成多个 actor/env 串行段；
每段内部仍是大量 PyTorch CUDA tensor op / CUDA Graph replay；
现在重复 begin accel prepare 已修掉，新的最大热点是 finish_bw 内部的大段 BW transition + reset + next-world/next-obs 构造。
```
