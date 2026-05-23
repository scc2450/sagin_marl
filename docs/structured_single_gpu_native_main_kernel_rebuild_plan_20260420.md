# 结构化环境单 GPU 原生主内核重建标准

更新日期：2026-04-23

本文定义 official CUDA train / eval / formal live path 的“单 GPU 原生主内核”完成形态、当前实现状态和验收方式。完成标准不是某次速度数字，也不是旧名字消失；完成标准必须看真实运行时是否仍由 Python 拼 stage、绑定 obs/result、组织临时 tensor record 或回落到兼容对象路径。

## 硬约束

唯一优先级：

1. 执行速度快。
2. 计算结果正确且稳定。

如果旧对象模型、接口边界、随机源、兼容缓存或 Python 调度会拖慢 official CUDA live path 或增加错误风险，应删除、隔离或 hard error，不能围绕旧结构做补丁。

在单 GPU 原生主内核结构完成前，不做速度验收。结构和正确性验证完成后，才允许按实际口径 `num_envs=8`、`rollout_env_steps=250` 分别运行 GPU/CPU 速度验收，且 GPU/CPU benchmark 不得同时跑。

## 非完成标准

以下不能作为完成证据：

1. 某个旧名字没有命中。
2. A/B/C/D callable 改名。
3. `stage_specs`、stage register、kernel cache 换了新类名。
4. 外部 API 看起来更窄。
5. formal exact 通过，但 live path 仍由 Python 拼 stage 或 tensor record。
6. CUDAGraph 没有 skip，但段输出仍由 Python 复制/绑定到下一段。
7. 某次 GPU 速度超过 CPU。

名称检查只能辅助定位；完成判断必须基于运行时行为和结构扫描。

## 当前完成形态

official CUDA strict full-batch direct path 已收敛为以下契约：

1. strict CUDA segment callable 在 rollout begin / graph build 阶段绑定静态语义；运行时 ABI 不接受 `cfg=cfg`，也不把 `cfg` 作为递归公共载体。
2. official env segment 使用 manual `torch.cuda.CUDAGraph` direct-input replay。capture/build 阶段保护 live persistent buffer：采样、warmup、capture 前后会 snapshot/restore direct-input side-effect buffers，避免首次 graph build 推进环境状态。
3. hot replay 只读稳定 input/action/random buffers，写 runtime-owned persistent state/obs/result/history buffers。输入 shape/dtype/device/data pointer 或非 tensor scalar 控制值变化会 hard error。
4. strict segment 递归 callee 静态扫描无 `.cpu()`、`.numpy()`、`.item()`、`.tolist()`、`np.*`、`channel.*`、`nonzero()`、`masked_select()`、`torch.empty/zeros/full/arange/eye/as_tensor/tensor/stack/cat` 等命中。
5. strict CUDA full-batch path 不能落入 stage group、stage view、dict/list spec、NumPy/PettingZoo mirror 或 compat materialization fallback；非 full-batch、非 direct、缺少 runtime tensor buffer 时 hard error。
6. prepare、accel->SAT、SAT->BW、BW step+next prepare 均以 fixed input/action/random/constants + fixed output buffers 的 captured/fused segment 方式执行。Python 负责 actor 决策边界和 replay 调用，不在段内或段后拼 stage、绑定 obs、组织 tensor record。
7. SAT->BW 直接产出 BW canonical input buffers：`assoc`、`prev_association`、`candidate_indices`、`candidate_mask`、`bw_valid_mask`、`sat_selection_matrix`、`active_sat_ids`、`gain_active`、`nu_eff_active`、`valid_flag_active`、`sat_pos`、`uav_ecef`、`uav_pos`、`uav_vel`、`gu_pos`。BW host 不再通过 `_stage_value`、临时 gather/clone/stack 组织这些输入。
8. BW fused step 直接写 state/result/reward-parts/link-transition/next-accel/history persistent buffers；segment 返回非 `None` 会 hard error。`StructuredBatchStepResult` 只是 result buffer view，不拥有当前 step 临时 tensor record。
9. BW link transition 与 BW result/reward parts 是 canonical tensor 输出。formal/reference/action replay 只作为校验消费者，消费同一份 canonical tape；旧 finalize 不再覆盖 official 主内核指标。
10. random/reset/arrival/fading/doppler 等输入由 native runtime tensor tape 持有；reference/formal 不再作为 official CUDA live path 的随机源。
11. rollout begin 预分配 result buffers、reward part buffers、BW link transition buffers、override buffers、next accel stage/obs buffers、flow proxy buffers、history ring buffers、固定 index/mask/template constants 和必要 scratch。
12. history ring 写入使用 persistent field buffers，不通过旧 world/stage object materialization。
13. direct live host 行为扫描无 `_stage_value`、`runtime.write_*obs_tensors`、`bind_step_result`、`write_step_result`、`StructuredBatchStepResult(...)` 构造、`_NativeBwStep*` record、`_native_world_from_stage_fields`、`write_fixed_rollout_history_slot`、运行时 factory/clone/stack/as_tensor。

## Segment Contract

每个 official CUDA env segment 必须满足：

1. 执行签名不接受 `cfg`，不读取 Python config。
2. 输入只能是 persistent buffers、action buffers、random tape cursor、frozen scalar closure、GPU constants。
3. 输出直接写 persistent output/result/state/history buffers。
4. 不返回临时 tensor record 给 Python 再复制/绑定。
5. 不调用 `channel.*`、NumPy、`lru_cache`、`.cpu()`、`.numpy()`、`.item()`、`.tolist()`。
6. hot replay 不调用 `torch.empty/zeros/full/arange/eye/as_tensor/tensor/stack/cat` 等 factory 创建 live buffer。
7. capture 失败、skip、fallback、输出结构变化必须 hard error。
8. index/mask/template 来自 rollout begin 预计算 GPU constants 或预分配 workspace。

## 验收检查

结构完成后必须先跑正确性和结构检查，再做速度验收。

已通过的正确性验证：

```powershell
.venv\Scripts\python.exe scripts\validate_structured_long_rollout_acceptance.py --config configs\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml --episodes 1 --num-envs 1 --native-tensor-backend cuda --json-path .tmp_formal_report_current.json
```

结果：

```text
passed=1 exact_passed=1
max_episode_diff reward=0 steps=0 drop=0 throughput=0 backlog=0
max_trace_diff reward=0 queue_total=0 backlog=0
```

已通过 graph-break 正确性验证：

```powershell
$env:TORCH_LOGS='graph_breaks'
.venv\Scripts\python.exe scripts\validate_structured_long_rollout_acceptance.py --config configs\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured.yaml --episodes 1 --num-envs 1 --native-tensor-backend cuda --json-path .tmp_formal_report_graphbreaks.json
Remove-Item Env:TORCH_LOGS
```

结果同样 `passed=1 exact_passed=1`，无 graph-break 输出。

已通过 targeted 回归：

```powershell
.venv\Scripts\pytest.exe -q tests\test_structured_batch_core_rollout.py tests\test_structured_mappo_rollout.py tests\test_structured_system_acceptance.py
```

结果：

```text
29 passed, 14 warnings
```

14 个 warning 均来自 `torch.jit.script_method` deprecation，不是本次主内核路径错误。

已通过结构扫描：

1. strict segment 递归 callee：65 个 reachable 函数，禁止项 0。
2. direct live host：禁止 `_stage_value`、旧 writer/binder、旧 step record、runtime factory 等命中 0。
3. 旧 `_NativeBwStep*` record 只在测试 guard 中出现，生产代码无定义/调用。

## 速度验收

速度验收只能在以上结构和正确性检查全部通过后执行。固定口径：

```text
num_envs=8
rollout_env_steps=250
device=cuda
```

GPU 和 CPU 必须分开运行，不得并发。速度结果只用于最终确认，不能替代本文定义的结构完成标准。
