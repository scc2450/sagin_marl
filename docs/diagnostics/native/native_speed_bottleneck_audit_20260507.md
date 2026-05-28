# native 真实速度瓶颈审计 2026-05-07

## 当前有效优化

### actor update microbatch 调到 1024

语义不变：仍然是同一个 PPO minibatch，内部拆 forward/backward 累积梯度，最后只 `optimizer.step()` 一次。

在 `3UAV/20GU/T=250/64env/BW-only PPO` 上：

| microbatch | iteration | rollout total | update total | actor train | env/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 26.55s | 16.86s | 9.69s | 8.14s | 603 |
| 4096 | 49.00s | 16.54s | 32.46s | 30.53s | 327 |
| 1024 | 20.93s | 16.90s | 4.04s | 2.53s | 764 |
| 512 | 21.74s | 17.15s | 4.59s | 2.96s | 736 |

结论：当前 6GB GPU 上 1024 是明显更好的默认值。4096 会触发显存/工作区压力，反而很慢。

## 已排除的方向

### runtime/stage snapshot copy 不是 finish 大头

临时关闭 runtime/stage snapshot 写入后，`finish_bw` 没有明显下降：

| 配置 | finish_bw | sat_to_bw | total_step |
| --- | ---: | ---: | ---: |
| 正常 | 约 22.2ms/step | 约 1.75-1.86ms/step | 约 40-42ms/step |
| 关闭 snapshot | 约 22.5ms/step | 约 1.66-1.71ms/step | 约 41ms/step |

结论：snapshot 看起来很肥，但不是当前主要耗时来源。不要优先做 snapshot 裁剪。

### access gain 去重不值得

尝试把 `refresh_stage_derived_parallel` 里的 GU-UAV access gain 从“candidate 算一次 + 全矩阵再算一次”改成“先算全矩阵，candidate 读取”。结果因为多了一次同步，`finish_bw` 反而略慢。

结论：这类小算力去重不是当前方向，已撤回。

## 剩余真实瓶颈

### 1. rollout collect 的 finish_commit_prepare_live

`all-policy` 分段 profile 里，`finish_bw` 稳定约 `22-24ms/step`，约占单步 50% 以上。

它不是 Python 小 op，而是一个大 CUDA kernel 内部偏串行：

- 每个 env 一个 block。
- 每步扫 `num_sat=144`。
- 每步做 access/backhaul/queue/reward。
- 每步准备下一步 accel stage。
- 每步写 critic world/local/history。

下一步如果要优化它，应该先加 finish 内部分段 profiler，分清：

- queue/reward transition。
- `refresh_stage_queue_derived_parallel`。
- `prepare_stage_from_state_parallel(next accel)`。
- `write_world_from_stage_parallel(next accel)`。
- `write_accel_obs_parallel`。

不要再靠猜直接改。

### 2. rollout prepare 随机 tape

真实训练中 `native_rollout_prepare_time_sec` 约 `3.2-3.4s/update`。

主要可疑点是 `_prepare_runtime_rollout_random_tapes(...)`：

- 初始准备 `structured_native_reset_tape_chunk_rows=8` 行 reset tape。
- sticky hotspot reset 仍有 CPU/GPU 往返和 per-env Python 逻辑。
- 目前 `ensure_native_rollout_reset_tape_capacity(...)` 只每 `chunk_rows` 步检查一次，不能简单把 reset rows 改成 1，否则多次 reset 会复用同一 reset row，语义不安全。

下一步优化方向：

- 把 reset tape 的 sticky hotspot/subset 生成向量化。
- 或实现真正安全的 on-demand reset row extension，再减少初始 reset rows。

### 3. native actor fused 仍不是全 fused

`actor_accel/sat/bw` 在 rollout 中仍各有数 ms。Torch profiler 显示 fused actor 路径实际由 `aten::addmm/bmm/copy_/mask` 等多个 ATen kernel 组成，不是单个大 CUDA kernel。

`host_linear` 改成 `aten::linear` 后有小幅改善，但这里不是当前最大头。真正大改需要把 actor 的投影/attention/MLP 做成更完整的 native fused kernel，风险和工作量都更高。

