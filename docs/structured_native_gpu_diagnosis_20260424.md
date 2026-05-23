# Structured Native 单 GPU 主内核诊断记录

日期：2026-04-24  
编码：UTF-8  
范围：`sagin_marl/env/structured_batch_env_core.py`、`sagin_marl/env/structured_gpu_rollout_runtime.py`、native rollout 训练热路径、fixed-seed long-rollout 正确性测试、GPU 吞吐回退诊断。

## 1. 当前结论

这次诊断基本确认：现在 `gpu_util_percent` 可以很高，但 `env_steps_per_sec` 不高，主要不是 PPO update 慢，也不是 CPU 忙，而是 rollout 每个 env step 在 GPU 上被拆成多个串行段，并且每段内部仍是大量 PyTorch CUDA tensor op。CUDA Graph 只是在 replay 已捕获的 PyTorch op 序列，并不等于把它们变成一个真正的大 fused kernel。

更具体地说，当前热路径每个 step 至少包含：

| 阶段 | 性质 |
|---|---|
| `begin_accel_obs` | env 段，准备 accel stage/local obs |
| `actor_accel` | actor 前向和动作写入 |
| `accel_to_sat` | env 段，accel 动作后发布 sat obs |
| `actor_sat` | actor 前向和动作写入 |
| `sat_to_bw` | env 段，sat 动作后发布 bw obs |
| `actor_bw` | actor 前向和动作写入 |
| `finish_bw` | env 段，BW transition、reward、下一步状态、并尝试准备 next accel obs |

最值得优先处理的问题是：`finish_bw` 的 `bw_native_fused_step_next_accel_obs` 已经在内部调用了一次 `_prepare_native_stage_and_accel_obs_tensor_impl()` 来准备下一步 accel obs，但训练路径下一步进入 `begin_accel_obs` 时没有复用这份结果，又重新调用了 `prepare_accel_stage_local_obs`。这很可能造成每个 step 多付了一次完整的 accel prepare 成本。

这条结论的置信度较高，但仍建议用一个小 patch 做 A/B 实验来最终确认：让训练 `record_rollout=True` 路径安全复用 next accel prefetch，或者临时关闭 `finish_bw` 的 next-prepare，再比较普通 benchmark。

## 2. 术语和路径

### 2.1 native、legacy、reference

这里的几个词容易混：

| 名称 | 现在的含义 |
|---|---|
| `native` | 当前主路径，结构化环境的 batch tensor/native GPU 路径。训练时目标是走这个路径。 |
| `legacy` | 历史兼容名，不等于还有一套完整旧 Python 单环境实现可用于训练热路径。很多旧 adapter API 已被 native main-kernel 明确拒绝。 |
| `reference` | 正确性测试里的“参照/判据角色”，不是一个单独的官方 CUDA 主内核实现。当前 long-rollout 校验用 driver/stage 兼容路线作为参考，再用 action trace/random tape 驱动 native 路线对比。 |

相关代码位置：

| 文件 | 作用 |
|---|---|
| `sagin_marl/env/structured_batch_env_core.py` | native batch env core，大部分 tensor op 和 native segment 实现都在这里。 |
| `sagin_marl/env/structured_gpu_rollout_runtime.py` | native rollout runtime 的 step program 编排层，决定每个 step 如何串起 env segment 和 actor。 |
| `sagin_marl/env/structured_kernel_runtime.py` | PyTorch compile/CUDA Graph capture/replay 包装层。 |
| `sagin_marl/rl/structured_eval.py` | fixed-seed long-rollout parity 校验。 |
| `scripts/profile_native_rollout_step_segments.py` | 本次新增的独立诊断脚本，运行时 monkey-patch 计时，不修改主内核文件。 |

### 2.2 两个 backend 配置

| 配置 | 含义 | 默认值 |
|---|---|---|
| `structured_env_backend` | 语义实现路线，控制是 `native`、`legacy` 还是 `auto`。 | `native` |
| `structured_env_tensor_backend` | tensor 放在哪个设备上跑，控制 `cuda`、`cpu` 或 `auto`。 | `cuda` |

默认值在 `sagin_marl/env/config.py` 中：

```python
structured_env_backend: str = "native"
structured_env_tensor_backend: str = "cuda"
```

所以 config yaml 里没写 `structured_env_backend: native` 不代表不是 native。`load_config()` 会先构造 `SaginConfig()` 默认值，再应用 yaml 覆盖。

## 3. 正确性测试现在在测什么

`validate_structured_fixed_seed_long_rollout` 的用途是固定随机种子、固定策略、固定 action trace 和 random tape，对比 native 路线和 reference/legacy-compatible 路线的长 rollout 结果。

它主要比较：

| 对象 | 比较内容 |
|---|---|
| episode rows | 每个 episode 的 reward、长度、KPI 等聚合结果 |
| step traces | 每一步的关键状态、奖励、队列、动作相关 trace |
| summary | 汇总指标 |

这类测试的意义是“native 主内核没有偏离当前 reference 语义”。它不是证明物理模型一定正确，也不是和一个完全独立的旧环境实现做强对照。

之前它跑不起来，是因为 native official main-kernel 已经不允许调用 legacy adapter API，例如：

```text
official CUDA native main-kernel does not expose legacy adapter API export_runtime_state_batch()
```

这次已经修过，当前验收结果是：

```text
long_rollout_acceptance ... passed=1 exact_passed=1
all max diffs = 0
```

相关测试也通过过：

```powershell
.venv\Scripts\pytest.exe -q `
  tests\test_structured_mappo_rollout.py `
  tests\test_structured_batch_core_rollout.py `
  tests\test_structured_system_acceptance.py
```

结果：`47 passed in 18.83s`。

## 4. 训练和 benchmark 命令

### 4.1 直接 native 主内核训练

正式训练入口是 `scripts/train_structured.py`，参数名用下划线：

```powershell
.venv\Scripts\python.exe scripts\train_structured.py `
  --config configs\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml `
  --device cuda `
  --structured_env_tensor_backend cuda `
  --num_envs 8 `
  --vec_backend sync `
  --updates 400 `
  --rollout_env_steps 250 `
  --hidden_dim 256 `
  --embed_dim 64 `
  --run_id native_cuda_train
```

如果要明确写入 config，可写：

```yaml
structured_env_backend: native
structured_env_tensor_backend: cuda
```

### 4.2 当前用于速度复现的 benchmark

benchmark 脚本是 `scripts/bench_structured_training_system.py`，参数名用连字符：

```powershell
.venv\Scripts\python.exe scripts\bench_structured_training_system.py `
  --config configs\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml `
  --num-envs 8 `
  --num-updates 1 `
  --warmup-updates 1 `
  --rollout-env-steps 250 `
  --device cuda `
  --env-tensor-backend cuda `
  --hidden-dim 32 `
  --embed-dim 16 `
  --json-path runs\diagnostics\native_rollout_profile\20260424_current\.tmp_codex_gpu_bench.json
```

注意：普通 benchmark 的速度可以比较；cProfile、torch profiler、monkey-patch segment 表生成时的速度只能作为诊断上下文，不能和正式速度直接比较。

## 5. 已生成的诊断产物

产物统一整理在：

```text
runs/diagnostics/native_rollout_profile/
```

主要目录：

| 目录 | 内容 |
|---|---|
| `20260424_current/` | 普通 benchmark、cProfile、第一/第二张 segment 表。 |
| `20260424_torch_ops/` | torch profiler CUDA op 表，无 trace。 |
| `20260424_torch_ops_trace/` | torch profiler CUDA op 表和 `trace.json`。 |

`trace.json` 是 Chrome/Perfetto timeline trace，可以用 Perfetto 打开：

```text
https://ui.perfetto.dev/
```

这个 trace 主要用于看时间轴上 segment、CUDA Graph replay、CUDA kernel 之间的串行关系和间隔。

## 6. 总体速度结果

### 6.1 普通 benchmark

文件：

```text
runs/diagnostics/native_rollout_profile/20260424_current/.tmp_codex_gpu_bench.json
```

结果：

| 指标 | 值 |
|---|---:|
| `env_steps` | 2000 |
| `transition_samples` | 6000 |
| `env_steps_per_sec` | 230.51 |
| `samples_per_sec` | 691.52 |
| `rollout_total_time_sec` | 8.4181 |
| `update_total_time_sec` | 0.2584 |
| `iteration_time_sec` | 8.6765 |
| `cpu_util_percent` | 5.10 |
| `gpu_util_percent` | 94.00 |
| `gpu_memory_util_percent` | 22.51 |

这个结果说明主要时间在 rollout，不在 PPO update。`gpu_util_percent=94%` 只说明 GPU 采样时大部分时间非 idle，不说明每个 CUDA kernel 都高效，也不说明吞吐合理。

### 6.2 cProfile benchmark

文件：

```text
runs/diagnostics/native_rollout_profile/20260424_current/.tmp_codex_gpu_bench_profile_run.json
runs/diagnostics/native_rollout_profile/20260424_current/.tmp_codex_bench_profile.pstats
```

结果：

| 指标 | 值 |
|---|---:|
| `env_steps_per_sec` | 191.78 |
| `rollout_total_time_sec` | 10.0277 |
| `update_total_time_sec` | 0.4009 |
| `gpu_util_percent` | 93.00 |

cProfile 会引入额外开销，所以这个速度不能和普通 benchmark 直接比较。但函数排序很有用：

| 函数/位置 | 调用数 | cumulative time |
|---|---:|---:|
| `collect_env_horizon_native_tensor_policy` | 2 | 31.15s |
| `StructuredGpuNativeRuntimeStepProgram.replay_horizon` | 2 | 31.15s |
| `StructuredGpuNativeRuntimeStepProgram.replay_step` | 500 | 31.13s |
| `StructuredKernelRuntime._buffered_runner` | 2000 | 10.93s |
| `torch._C.CUDAGraph.replay` | 3497 | 7.83s |
| `write_accel_action` | 500 | 7.17s |

关键点：

1. 热点集中在 native rollout step。
2. `_buffered_runner` 调用数是 2000，正好对应 500 step × 4 个主要 env compiled segment。
3. `CUDAGraph.replay` 调用数 3497，说明除了 env segment，actor 或其他 compiled graph 也在频繁 replay。

## 7. 每 step segment 表

这张表来自：

```text
runs/diagnostics/native_rollout_profile/20260424_torch_ops_trace/step_segments.csv
```

该 run 同时启用了 torch profiler 和 trace，吞吐值本身只作为上下文，主要看占比：

| segment | calls | avg ms/step | 占 total_step |
|---|---:|---:|---:|
| `begin_accel_obs` | 250 | 6.1799 | 20.20% |
| `actor_accel` | 250 | 3.2067 | 10.48% |
| `accel_to_sat` | 250 | 4.8578 | 15.88% |
| `actor_sat` | 250 | 2.4564 | 8.03% |
| `sat_to_bw` | 250 | 2.9035 | 9.49% |
| `actor_bw` | 250 | 2.9033 | 9.49% |
| `finish_bw` | 250 | 6.9827 | 22.82% |
| `total_step` | 250 | 30.5984 | 100.00% |

同一诊断脚本的较早 monkey-patch run 绝对值更大，但排序一致：

| segment | avg ms/step | 占 total_step |
|---|---:|---:|
| `finish_bw` | 13.4589 | 31.98% |
| `begin_accel_obs` | 9.4141 | 22.37% |
| `accel_to_sat` | 9.0122 | 21.42% |
| `sat_to_bw` | 3.4661 | 8.24% |
| `actor_bw` | 2.4297 | 5.77% |
| `actor_accel` | 2.3098 | 5.49% |
| `actor_sat` | 1.8261 | 4.34% |
| `total_step` | 42.0796 | 100.00% |

结论：慢点主要在 env-side segment，尤其是 `finish_bw`、`begin_accel_obs`、`accel_to_sat`，actor 不是唯一主因。

## 8. project-level kernel segment 表

这张表来自：

```text
runs/diagnostics/native_rollout_profile/20260424_torch_ops_trace/kernel_segments.csv
```

| kernel segment | calls/step | avg ms/step | 占 total_step |
|---|---:|---:|---:|
| `bw_native_fused_step_next_accel_obs` | 1.0 | 6.3151 | 20.64% |
| `accel_prepare_sat_stage_local_obs` | 1.0 | 3.8594 | 12.61% |
| `prepare_accel_stage_local_obs` | 1.0 | 3.5501 | 11.60% |
| `sat_pair_to_bw_stage_local_obs` | 1.0 | 1.6793 | 5.49% |

较早 monkey-patch run：

| kernel segment | calls/step | avg ms/step | 占 total_step |
|---|---:|---:|---:|
| `bw_native_fused_step_next_accel_obs` | 1.0 | 13.0194 | 31.61% |
| `accel_prepare_sat_stage_local_obs` | 1.0 | 8.3930 | 20.38% |
| `prepare_accel_stage_local_obs` | 1.0 | 8.0920 | 19.65% |
| `sat_pair_to_bw_stage_local_obs` | 1.0 | 3.0013 | 7.29% |

关键点：不是外层有几十个 project-level segment，而是每 step 有 4 个主要 env compiled/CUDA Graph segment。真正的问题在于：

1. 这些 segment 串行排列，中间夹着 actor 决策。
2. 每个 segment 内部不是单个手写 CUDA kernel，而是很多 PyTorch CUDA op。
3. `prepare_accel_stage_local_obs` 每 step 都出现一次，而 `bw_native_fused_step_next_accel_obs` 内部又包含 next accel prepare，这是疑似重复计算。

## 9. torch profiler CUDA op 表

文件：

```text
runs/diagnostics/native_rollout_profile/20260424_torch_ops_trace/torch_profiler_ops.csv
runs/diagnostics/native_rollout_profile/20260424_torch_ops_trace/torch_profiler_segment_ops.csv
```

前几行里 `native_segment::...` 是我们加的 `record_function` 范围，不是实际 CUDA kernel：

| range | count | avg cuda us |
|---|---:|---:|
| `native_segment::begin_accel_obs` | 20 | 7351.66 |
| `native_segment::finish_bw` | 20 | 6105.21 |
| `native_segment::accel_to_sat` | 20 | 4908.31 |
| `native_segment::actor_accel` | 20 | 3770.01 |
| `native_segment::actor_bw` | 20 | 3321.27 |
| `native_segment::actor_sat` | 20 | 2966.79 |
| `native_segment::sat_to_bw` | 20 | 2763.75 |

实际 CUDA op 显示大量细碎 kernel：

| op 类型 | count | self cuda ms | 说明 |
|---|---:|---:|---|
| `index_put`/index elementwise kernel | 4820 | 21.07 | 大量索引写入 |
| vectorized elementwise `mul` | 7760 | 14.55 | 大量逐元素乘法 |
| vectorized elementwise `mul` 另一组 | 5740 | 10.86 | 同类小 kernel |
| elementwise `mul` | 3740 | 10.56 | 同类小 kernel |
| `clamp` | 4980 | 10.13 | 限幅 |
| scatter/gather elementwise kernel | 2600 | 8.85 | scatter/gather |
| direct copy kernel | 2920 | 7.89 | tensor copy |
| `Memcpy DtoD` | 3520 | 4.50 | device-to-device copy |
| `aten::copy_` | 3540 | 4.54 | copy 包装 |
| reduce `sum` | 1740 | 6.98 | reduction |
| `aten::gather` | 1700 | 3.53 | gather |
| `volta_sgemm_32x32...` | 820 | 3.67 | 小矩阵乘 |

这说明“GPU 94% busy 但吞吐不高”的原因不是 GPU 没活干，而是 GPU 一直在跑很多小而碎、索引密集、copy 密集、串行依赖强的 kernel。对 `num_envs=8` 这种小 batch，单个 kernel 的有效并行度不高，launch/replay/同步和内存访问开销会被放大。

## 10. 关键代码路径分析

### 10.1 外层 step program

位置：`sagin_marl/env/structured_gpu_rollout_runtime.py`

`StructuredGpuNativeRuntimeStepProgram.replay_step()` 的结构是：

```text
begin_accel_obs
actor_accel
accel_to_sat
actor_sat
sat_to_bw
actor_bw
finish_bw
```

这就是 segment 表里那几列的来源。

### 10.2 训练路径没有复用 next accel prefetch

位置：`sagin_marl/env/structured_gpu_rollout_runtime.py:409`

当前 `_current_or_begin_accel_obs()` 的逻辑：

```python
def _current_or_begin_accel_obs(self, *, record_rollout: bool) -> Any:
    if not bool(record_rollout):
        main = self.runtime.main
        if main.next_stage_fields is not None:
            ...
            accel_obs = self.executor._runtime_training_history_local_state(self.runtime, stage_name="accel")
            ...
            return accel_obs
    return self._begin(record_rollout=record_rollout)
```

也就是说，只有 `record_rollout=False` 的 hot replay 路径会尝试消费 `main.next_stage_fields`。正式训练收集 rollout 时 `record_rollout=True`，所以每一步都会走：

```python
return self._begin(record_rollout=True)
```

### 10.3 begin_accel_obs 会触发 standalone prepare

位置：`sagin_marl/env/structured_batch_env_core.py:13338`

`_runtime_tensor_begin_accel_obs_impl()` 开始时，如果 `record_rollout=True`，先调用：

```python
runtime.begin_native_main_kernel_training_write(num_envs=len(selected_indices))
```

然后检查：

```python
prefetched_fields = main.prefetched_accel_stage_fields
```

只有 `main.prefetched_accel_stage_fields` 存在时才复用。否则会调用：

```python
self._runtime_tensor_prepare_stage_and_obs(..., obs_kind="accel", record_rollout=record_rollout)
```

而 `_runtime_tensor_prepare_stage_and_obs()` 会调用 project-level segment：

```text
prepare_accel_stage_local_obs
```

这就是 kernel segment 表里每 step 1 次 `prepare_accel_stage_local_obs` 的来源。

### 10.4 finish_bw 已经包含 next accel prepare

位置：`sagin_marl/env/structured_batch_env_core.py:10941`

`_execute_bw_native_fused_step_next_accel_obs_out_tensor_impl()` 做完 BW transition 后，会再次调用：

```python
next_stage_t = _prepare_native_stage_and_accel_obs_tensor_impl(...)
```

也就是说 `bw_native_fused_step_next_accel_obs` 不是只做 BW step，它还会把下一步 accel stage/local obs 准备出来。

### 10.5 但 finish_bw 把可复用字段清掉了

位置：`sagin_marl/env/structured_batch_env_core.py:13864`

当前 `_runtime_tensor_finish_bw_and_prefetch_direct_impl()` 末尾逻辑大致是：

```python
runtime.main.prefetched_accel_stage_fields = None
if prepare_next_accel:
    runtime.main.prefetched_accel_stage_fields = next_stage_fields
if prepare_next_accel:
    runtime.main.accel_stage_fields = next_stage_fields
    runtime.main.prefetched_accel_stage_fields = None
    runtime.main.prefetched_accel_stage = None
runtime.advance_main_kernel_rollout_step()
```

这里的问题是：

1. `finish_bw` 先把 `prefetched_accel_stage_fields` 设置成 `next_stage_fields`。
2. 紧接着又把 `prefetched_accel_stage_fields` 清成 `None`。
3. 下一 step 的 `_runtime_tensor_begin_accel_obs_impl()` 只看 `main.prefetched_accel_stage_fields`，不看 `main.accel_stage_fields`。
4. 训练路径的 `_current_or_begin_accel_obs(record_rollout=True)` 也不会直接消费 `main.next_stage_fields`。

所以训练路径会重新跑 standalone `prepare_accel_stage_local_obs`。这和 kernel segment 表的现象完全吻合：每 step 同时有一次 `bw_native_fused_step_next_accel_obs` 和一次 `prepare_accel_stage_local_obs`。

## 11. 为什么 GPU busy 但 env/sec 低

可以把原因拆成四层。

### 11.1 GPU busy 不等于高效吞吐

`gpu_util_percent=94%` 是采样意义上的“GPU 非 idle”。如果 GPU 一直在跑大量小 kernel、copy、gather/scatter、index_put，它也会显示 busy，但每秒完成的 env step 不一定高。

### 11.2 串行边界多

当前 step 不能简单合成一个大 kernel，因为 actor 决策在中间：

```text
env accel obs -> actor accel -> env sat obs -> actor sat -> env bw obs -> actor bw -> env finish
```

每个 actor 的动作是后续 env segment 的输入，依赖链天然串行。

### 11.3 CUDA Graph replay 不是 kernel fusion

`StructuredKernelRuntime._buffered_runner()` 会 capture eager PyTorch 函数，然后 replay graph。这个机制减少 Python launch overhead，但 graph 里面仍然是原来那些 PyTorch CUDA op。

因此 profiler 看到的是很多 index/gather/scatter/copy/elementwise/reduce kernel，而不是少数几个真正融合后的大 kernel。

### 11.4 疑似重复的 accel prepare

`finish_bw` 内部已经做 next accel prepare；下一步 `begin_accel_obs` 又做 standalone prepare。这会额外增加每 step 的 GPU 工作量。

从 trace run 的 kernel segment 表看，standalone `prepare_accel_stage_local_obs` 自己占 `11.60%` 的 total step；从较早 monkey-patch run 看，它占 `19.65%`。如果能安全复用 next accel prepare，理论上可能直接拿回一块明显吞吐。

## 12. 下一步建议

### 12.1 第一优先级：验证并修复 prefetch 复用

目标：训练 `record_rollout=True` 路径中，`finish_bw` 已准备的 next accel stage/local obs 应该被下一步 `begin_accel_obs` 消费，而不是重新跑 `prepare_accel_stage_local_obs`。

修复时必须保留这些语义：

1. `begin_native_main_kernel_training_write()` 仍要为新 step 分配正确 history slot。
2. actor 读取的 accel obs 必须来自当前 step 对应的 history/local state。
3. history ring 中用于 PPO 的 obs/action/reward/done 行不能错位。
4. `validate_structured_fixed_seed_long_rollout` 必须继续 exact pass。

建议先做一个小实验 patch，目标是让 kernel segment 表中：

```text
prepare_accel_stage_local_obs calls_per_profiled_step
```

从当前 `1.0` 降到接近 `0.0` 或只在 rollout 初始化时出现一次。

### 12.2 两种实验路线

路线 A：真正复用 next prefetch。

优点：符合 `bw_native_fused_step_next_accel_obs` 的设计意图，理论收益最大。  
风险：要非常小心 history slot 和 local obs 行号。

路线 B：临时关闭训练路径里的 `prepare_next_accel`。

优点：实现简单，适合快速验证“重复 prepare 是否是大头”。  
风险：这不是最终最优设计，因为只是避免 finish_bw 里提前准备，下一步 begin 仍然要准备一次。

如果路线 B 速度明显变好，说明当前 next-prepare 是纯浪费；如果路线 A 速度更好，说明复用成功。

### 12.3 修复后的验收顺序

1. 跑 correctness：

```powershell
.venv\Scripts\python.exe scripts\validate_structured_long_rollout_acceptance.py `
  --config configs\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml `
  --episodes 1 `
  --num-envs 1 `
  --native-tensor-backend cuda
```

2. 跑测试集：

```powershell
.venv\Scripts\pytest.exe -q `
  tests\test_structured_mappo_rollout.py `
  tests\test_structured_batch_core_rollout.py `
  tests\test_structured_system_acceptance.py
```

3. 跑普通 benchmark，不开 profiler：

```powershell
.venv\Scripts\python.exe scripts\bench_structured_training_system.py `
  --config configs\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml `
  --num-envs 8 `
  --num-updates 1 `
  --warmup-updates 1 `
  --rollout-env-steps 250 `
  --device cuda `
  --env-tensor-backend cuda `
  --hidden-dim 32 `
  --embed-dim 16 `
  --json-path runs\diagnostics\native_rollout_profile\after_prefetch_fix\bench.json
```

4. 再跑 segment 诊断，确认 `prepare_accel_stage_local_obs` 次数下降：

```powershell
.venv\Scripts\python.exe scripts\profile_native_rollout_step_segments.py `
  --config configs\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml `
  --num-envs 8 `
  --num-updates 1 `
  --warmup-updates 1 `
  --rollout-env-steps 250 `
  --device cuda `
  --env-tensor-backend cuda `
  --hidden-dim 32 `
  --embed-dim 16 `
  --output-dir runs\diagnostics\native_rollout_profile\after_prefetch_fix
```

### 12.4 第二优先级：进入 prepare/finish 内部优化

如果 prefetch 复用后仍然慢，再看这些内部热点：

| 热点类型 | 可能方向 |
|---|---|
| `index_put`、scatter/gather | 减少动态索引写入，改成更规则的 buffer layout 或预计算索引。 |
| 大量 `copy_`/DtoD memcpy | 检查哪些 copy 是历史兼容或防御性 copy，能否用 view/固定 out buffer 替代。 |
| topk/argsort/where | 固定候选宽度、预筛选、减少每 step 全量排序。 |
| 多个 elementwise 小 kernel | 尝试 `torch.compile` fullgraph 或手写融合函数，但要先确认收益大于复杂度。 |
| 小 batch GPU 利用率 | 比较 `num_envs=8/16/32`，判断是否主要受 launch/fragment 开销限制。 |

## 13. 当前仓库状态提示

本轮新增文件：

```text
scripts/profile_native_rollout_step_segments.py
docs/structured_native_gpu_diagnosis_20260424.md
```

已经按用户要求把根目录下的临时产物移到：

```text
runs/diagnostics/native_rollout_profile/20260424_current/
```

因此 git 里会看到这些已跟踪临时文件表现为删除：

```text
.tmp_codex_bench_profile.pstats
.tmp_codex_formal_current.json
.tmp_codex_gpu_bench.json
.tmp_codex_gpu_bench_profile_run.json
```

这是有意整理产物，不是主代码删除。

## 14. 一句话版

现在最可疑、最值得下刀的点不是“GPU 没跑满”，而是“GPU 一直很忙地跑了太多碎片化、串行、索引/拷贝密集的活”，其中一个具体可修的浪费是：`finish_bw` 已经准备了下一步 accel obs，但训练路径下一步又重新准备了一遍。
