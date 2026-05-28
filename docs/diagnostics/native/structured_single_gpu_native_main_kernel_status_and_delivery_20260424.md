# 结构化环境单 GPU 原生主内核状态与交付文档（2026-04-24）
更新日期：2026-04-24

本文只回答四件事：

1. 当前 official CUDA native live path 的真实状态是什么
2. 什么才算“单 GPU 原生主内核彻底完成”
3. 当前代码是否已经满足这个完成标准
4. 在结构完成标准满足后，速度验收应如何进行

本文不把“某次测速结果看起来不错”当作结构完成，也不把“旧名字没了”“外层 API 收紧了”当作完成。判断只看 official CUDA native live path 的运行时结构。

---

## 1. 执行硬约束

后续实现与验收必须遵守这些硬约束：

1. 在单 GPU 原生主内核结构完成之前，禁止做速度验收
2. 禁止改一刀测一次速度
3. 禁止只改训练主路径，保留 eval / formal / reference 旧结构拖住 live core
4. 禁止停留在“先能跑”的过渡方案
5. 禁止为了兼容旧结构做局部修补
6. 禁止把阶段完成、回归通过、某次 GPU 比 CPU 快当作停止条件
7. 禁止把改名、删旧名、外层 API 收紧当作完成
8. 禁止继续包 wrapper 掩盖真实运行时边界
9. 禁止以不破坏 CPU/reference 测试为理由，在 official live core 保留旧分支
10. 每轮都必须按运行时结构反查，而不是只按名字反查

---

## 2. 当前结论

**截至当前代码状态，official CUDA native live path 已经满足本文第 7 节定义的结构完成标准。**

也就是说：

1. 它已经不再是“Python 串联旧阶段逻辑 + GPU 小张量”的过渡形态
2. 它已经收敛为“persistent GPU buffers + 固定 actor/env replay 边界”的 official live execution chain
3. 当前剩余工作不再是继续补 live core 结构，而是执行固定口径的速度验收并处理验收暴露出的纯性能问题

需要强调：

- 这个结论只针对 **official CUDA native live path**
- outer adapter / eval / reference / legacy 接口仍然存在，但它们已经不再反向污染 live core

---

## 3. 当前已经完成的结构事实

### 3.1 训练热路径里的 step result clone 已删除

official live path 中：

- `replay_horizon()` 不再保存 `_snapshot_batch_step_result(...)`
- `StructuredBatchStepResult` 只作为当前 step 的 live view
- 历史真值改为 runtime-owned history ring / `step_history_view`

这意味着：

- official live path 不再每步 clone `StructuredBatchStepResult`
- history / rollout 历史不再靠 result clone 承载

### 3.2 BW final 已经直接写 state / result / next accel / history

当前 official live path 中：

- `history_write` 不再是独立 env segment
- `_runtime_tensor_record_history_direct_impl()` 已退出 official live path
- history slot 写入已经并入 BW fused final segment

这意味着训练 steady-state 的 env 侧不再存在：

```text
BW final
-> 额外 history_write
```

而是：

```text
BW final
-> 直接写 state / result / next accel / history
```

### 3.3 actor 输出已经直接落到 runtime-owned canonical buffers

当前 official live policy path 中：

- `act_accel_into(...)`
- `act_sat_pair_into(...)`
- `act_bw_into(...)`

直接写入 runtime-owned persistent buffers：

- action buffers
- logprob buffers
- BW per-agent / per-slot / support auxiliary buffers

official live path 不再依赖 runtime 的 `write_accel_action / write_sat_action / write_bw_action` 做 host copy / bind。

### 3.4 训练 history payload 已迁入 runtime

当前训练录制路径中：

- `StructuredGpuRolloutHistoryPayload` 已删除
- `rollout_history_payload(...)` 已退出 official live path
- actor auxiliary / old logprob / support mask 直接写 runtime persistent buffers
- history ring 直接消费这些 runtime buffers

### 3.5 env steady-state 段数已经收敛到最小必要边界

当前 official live path 的 env steady-state 只剩三段：

1. `accel -> sat obs`
2. `sat -> bw obs`
3. `bw -> next accel / result / history`

actor 决策边界仍保留，这是不可消掉的语义边界；除此之外，env 侧不再多出额外 history segment。

### 3.6 `cfg/spec` 已退出 live 热段

当前官方 live 热段中：

- env 热段不再读 `self._cfg`
- actor bridge 热段不再读 `learner.cfg`
- `cfg/spec` 只在 rollout begin / build / strict contract 激活阶段生效

这个结论由结构测试直接约束，而不是靠人工口头判断。

### 3.7 `global_state_batch` 已经从 official live contract 外移

当前状态是：

- `global_state_batch` 不再属于 official strict env segment 集合
- 它不再出现在 official required compile name 集合中
- 在 strict live rollout 激活时，`get_global_state_batch()` 会被视为 legacy adapter API 并拒绝进入 live core

也就是说，`global_state_batch` 现在属于 outer adapter / non-live consumer，而不是 official live core 的一部分。

### 3.8 env / actor strict contract 不再依赖 YAML 手写

当前 official live path 中：

- env strict compiled / captured contract 自动激活
- actor strict compile contract 自动激活
- 正式 task YAML 不再承载主内核实现开关

这意味着 official live path 的实现契约已经从“任务 YAML 手写”迁回代码/运行时 profile。

---

## 4. 当前仍然存在、但已不属于 live core 的外层边界

下面这些东西仍然存在于仓库里，但它们不再属于 official live core：

1. `structured_eval.py` 中的 action replay / reference adapter
2. `get_global_state_batch()` 等 outer adapter API
3. legacy/reference/debug 所需的 materialization 边界
4. old `rl/mappo.py` 这类非 official live 训练路径

它们的存在不再构成“单 GPU 原生主内核未完成”的理由；判断标准只看 official live path 是否被这些外层边界反向污染。当前没有。

---

## 5. official live path 的最终形态

当前 official live path 应理解为：

```text
rollout begin
-> 预分配 persistent GPU buffers
-> 构建 strict actor/env callable

每个 step:
  accel actor replay
  -> 直接写 accel action buffer

  env segment B replay
  -> 直接写 SAT obs canonical buffers

  sat actor replay
  -> 直接写 SAT action buffer

  env segment C replay
  -> 直接写 BW obs canonical buffers

  bw actor replay
  -> 直接写 BW action / logprob / auxiliary buffers

  env segment D replay
  -> 一次写完：
     runtime state
     result tensors
     reward components
     next accel obs
     history ring
     rollout step / random cursor
```

这就是当前 official CUDA native live path 的最终交付形态。

---

## 6. 结构反查证据

本轮完成后，已通过的结构/正确性检查包括：

1. `tests/test_structured_system_acceptance.py::test_native_main_kernel_live_hosts_do_not_read_runtime_cfg`
2. `tests/test_structured_system_acceptance.py::test_native_main_kernel_actor_bridge_hot_methods_do_not_read_cfg`
3. `tests/test_structured_system_acceptance.py::test_cuda_native_main_kernel_rollout_begin_enforces_strict_graph_contract`
4. `tests/test_structured_system_acceptance.py::test_cuda_native_main_kernel_direct_path_does_not_call_legacy_stage_helpers`
5. `tests/test_structured_system_acceptance.py::test_cuda_native_main_kernel_hot_replay_does_not_read_static_spec`
6. `tests/test_structured_system_acceptance.py::test_cuda_native_main_kernel_hot_replay_does_not_allocate_live_tensors`
7. `tests/test_structured_system_acceptance.py::test_cuda_native_main_kernel_strict_rejects_partial_batch_compat_fallback`
8. `tests/test_structured_batch_core_rollout.py` 中两条 `global_state_batch` 非 live adapter 测试
9. `tests/test_structured_mappo_rollout.py`

同时已完成静态反查：

- official live path 不再出现 `_snapshot_batch_step_result`
- official live path 不再出现 `publish_history_inputs`
- official live path 不再出现 `bind_native_main_kernel_history_inputs`
- official live path 不再出现独立 `history_write` segment
- official strict env segment 集合中不再包含 `global_state_batch`

---

## 7. 完成标准

只有同时满足下面这些，才能说“单 GPU 原生主内核彻底完成”：

- [x] official CUDA native live path 不再每步 clone step result
- [x] BW final segment 直接写 state / result / next accel / history
- [x] 训练时不再存在额外 `history_write` segment
- [x] actor 输出直接成为 runtime canonical action / logprob / auxiliary buffers
- [x] host 侧不再存在 `write_*_action()` 的 copy / bind 热路径职责
- [x] env steady-state 段数收敛到 actor 决策所要求的最小边界
- [x] live core 不再依赖 `cfg/spec`、旧 stage 对象、legacy mirror、materialization 边界
- [x] official live path 只允许 strict compiled / captured contract，不允许 silent fallback

**结论：第 7 节完成标准当前已全部满足。**

---

## 8. 速度验收前提

因为第 7 节结构完成标准已经满足，速度验收现在才具有意义。

固定验收口径仍然是：

```text
num_envs = 8
rollout_env_steps = 250
GPU / CPU 分开跑
不并发 benchmark
```

测速只回答一件事：

> 结构完成后的 official live implementation 是否达到目标速度

它不再负责回答“主内核结构是不是已经完成”。

### 8.1 本轮验收结果

固定口径：

```text
config = phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml
num_envs = 8
num_updates = 1
warmup_updates = 1
rollout_env_steps = 250
hidden_dim = 32
embed_dim = 16
GPU / CPU 分开跑
```

结果：

1. GPU

```text
env_steps_per_sec = 423.17
samples_per_sec = 1269.52
rollout = 4.5672s
update = 0.1590s
iter = 4.7262s
cpu = 6.19%
gpu = 46.00%
gpu_mem = 24.98%
```

2. CPU

```text
env_steps_per_sec = 213.55
samples_per_sec = 640.65
rollout = 9.0871s
update = 0.2784s
iter = 9.3655s
cpu = 49.31%
```

结论：

- 结构完成后的 official live implementation 已完成同口径速度验收
- 当前 GPU `423.17 env_steps/s`，CPU `213.55 env_steps/s`
- 在该固定口径下，GPU 明显快于 CPU
