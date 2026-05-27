# 环境与原生执行改动整理 2026-04-28

本文整理当前工作区未暂存改动中与环境、信道、单 GPU native 执行和 1UAV/2GU 校准相关的内容。文档使用 UTF-8 编码。

## 背景

这轮修改的出发点是 1UAV/2GU 最小 BW 问题在物理尺度上不稳定：GU-UAV access 链路误用了 UAV-SAT backhaul 的 30 GHz 载波，SNR 里混有固定 `+1e-12` 这类会改变链路预算的数值地板，access fading 用每秒 iid Rician gain 代表整秒信道也不合适。同时，新的单 GPU native 架构里 `sat=zero` 和 sticky hotspot 多环境 replay 有实现问题，导致 fixed-satellite + queue-aware BW 的校准结果不可信。

这轮目标不是“专门做 BW focus 配置”，而是把环境物理量和 native/legacy 两条路径先修到一致、可解释，再把 1UAV/2GU 临时配置调到能看出 BW 分配 leverage 的 regime。

## 核心改动

### 1. 拆分 access/backhaul 载波频率

新增配置字段：

```yaml
access_carrier_freq: 2.0e9
backhaul_carrier_freq: 3.0e10
```

含义：

- `access_carrier_freq` 用于 GU-UAV access 链路，默认 2 GHz。
- `backhaul_carrier_freq` 用于 UAV-SAT backhaul 链路，默认兼容旧 `carrier_freq`。
- 旧 `carrier_freq` 继续保留，但现在主要作为 `backhaul_carrier_freq` 的兼容 fallback。

涉及路径：

- `sagin_marl/env/config.py`
- `sagin_marl/env/channel.py`
- `sagin_marl/env/sagin_env.py`
- `sagin_marl/env/structured_batch_env_core.py`
- `sagin_marl/env/native_cuda/kernels.cu`

### 2. 加入 noise figure，并修正 SNR 语义

新增配置字段：

```yaml
access_noise_figure_db: 5.0
backhaul_noise_figure_db: 3.0
```

SNR 分母改为：

```text
noise_power = noise_density * bandwidth * 10^(NF/10)
```

并去掉物理链路分母里的固定 `+1e-12`。对于带宽为 0 或无效链路的情况，语义是直接输出 rate/SNR 为 0，而不是靠加一个固定瓦特量级的小数兜底。

涉及实现：

- numpy/legacy 路径：`channel.safe_snr_linear()`、`channel.snr_linear(..., noise_figure_db=...)`
- torch/native 路径：`_snr_linear_torch(..., noise_figure_db=...)`
- CUDA 路径：access/backhaul SNR 都使用 NF 线性因子

### 3. access fading 改为可配置，并新增 ergodic Rician

新增配置字段：

```yaml
access_fading_mode: ergodic_rician
access_rician_k_db: 10.0
access_ergodic_rician_quadrature_points: 16
```

支持模式：

- `large_scale`：只用大尺度信道，不乘小尺度 fading。
- `ergodic_rician`：在 spectral efficiency 处计算 `E_h[log2(1 + SNR * h)]`，不再每秒抽一个 iid fading gain 当整秒信道。
- `iid_rician`：保留旧式随机 Rician gain，用于需要显式快衰落采样的实验。

实现细节：

- numpy 路径使用 `channel.rician_ergodic_spectral_efficiency()`。
- torch/native 路径使用 `_rician_ergodic_spectral_efficiency_torch()`。
- CUDA 路径使用 `rician_ergodic_spectral_efficiency_device()`。
- 当前实现使用 Gauss-Hermite quadrature 计算复高斯 Rician 分量期望。

### 4. rain loss 与 backhaul 物理量同步到 native ABI

`rain_loss_enabled` 和雨衰参数被同步到 native typed params / CUDA ABI，使单 GPU native 路径和 legacy 环境对 backhaul loss 的处理一致。

涉及内容：

- `rain_rate_001_mmph`
- `rain_height_km`
- `rain_exceedance_pct`
- `rain_polarization_tilt_deg`
- `rain_lat_deg`

CUDA 中新增了 P.838/P.618 相关 rain attenuation 计算，和 Python/torch 路径保持同一语义。

## 数值 guard 重构

新增文件：

```text
sagin_marl/env/numeric_guards.py
```

目的不是“把 `1e-9/1e-12` 换个名字”，而是按语义区分防零方式：

- 配置归一化尺度：`normalize_scale()` / `positive_config_scale()`，用于 map size、queue capacity、速度尺度等必须为正的配置量。
- runtime ratio：`ratio_or_zero()`，用于诊断比例或运行时比例，分母为 0 时返回明确 fallback 0。
- reward denominator：`reward_ratio_denominator_scalar()`，用于 reward 定义里的 arrival/ref 分母，非正值直接报错。
- 几何分母：`geometry_denominator()`，用于距离、投影、elevation 等几何公式。
- log/prob 域：`log_ratio_argument()`、`relative_log_argument()`，用于数值域 clamp。
- 分布/归一化 fallback：`divide_or_default()`，用于 sum 为 0 时走显式默认值。

已经应用到：

- `sagin_marl/env/sagin_env.py`
- `sagin_marl/env/structured_batch_env_core.py`
- `sagin_marl/env/structured_driver.py`
- `sagin_marl/env/structured_stage_obs.py`
- `sagin_marl/env/channel.py`
- `sagin_marl/env/native_cuda/kernels.cu`

重点语义变化：

- reward 分母如果异常为 0，不再默默 floor，而是报错。
- 归一化尺度仍然会 clamp，因为这些是静态配置尺度。
- runtime 诊断比例在分母无效时返回 0。
- 物理链路不再用固定 `+1e-12` 当假噪声。

## native 单 GPU 路径修复

### 1. 修复 `sat=zero`

之前 native `sat_to_bw_live_kernel` 中：

```text
sat_source_mode == zero -> action = 0 -> subset 0
```

但 subset 0 是空集，所以 fixed-satellite 场景下 `sat=zero` 实际变成“不选卫星”，导致 backhaul 和 sat processed 都为 0。

现在改为：

- `sat=zero` 时每个 UAV 选择最近的有效可见卫星。
- 只填第一个 `sat_num_select` 槽，其余槽保持 `-1`。
- 行为与 legacy `fixed_satellite_strategy=True` 的 nearest visible satellite 语义对齐。

验证过：

```text
1 env × 1 step
accel=zero, sat=zero, bw=queue_aware
backhaul_sum ≈ 2.400864e6
sat_processed_sum ≈ 2.400864e6
```

### 2. 修复 sticky hotspot 多环境 sub-batch tape 选择

之前 `_select_runtime_tape_env_rows()` 默认所有 rollout tape 的 env 维都是 dim=1。但 sticky hotspot 的 `reset_followup_*_tape` 形状是：

```text
(steps, steps, env, ...)
```

env 维实际是 dim=2。多环境 sub-batch 时会把 env index 用到 dim=1 上，触发 CUDA：

```text
ScatterGatherKernel.cu / Indexing.cu index out of bounds
```

现在改为：

- 普通 rollout tape 仍按 `env_dim=1` 选择。
- `reset_followup_arrival_*` 和 `reset_followup_hotspot_active_after_*` 按 `env_dim=2` 选择。
- 增加 Python 侧边界检查，索引越界时先抛带 shape 的 `ValueError`，避免 CUDA device-side assert。

验证过：

```text
2 env × 1 step
accel=zero, sat=zero, bw=queue_aware
backhaul_sum/sat_processed_sum 正常非 0
```

### 3. native fading tape 语义调整

`ergodic_rician` 模式不需要每步随机 fading gain tape；只有 `access_fading_mode=iid_rician` 时才生成和使用随机 fading gain。这样避免把“快衰落的秒级随机抽样”混入当前想验证的带宽分配问题。

## 1UAV/2GU 临时配置调整

调整文件：

```text
configs/tmp/structured_bw_sanity_1uav_2gu_t10_stronggap_currentenv_competition_ppo.yaml
```

保留物理带宽缩放后的设置：

```yaml
b_acc: 3.0e6
b_backhaul_per_sat: 4.5e6
```

当前落点：

```yaml
task_arrival_rate: 8.0e6
queue_ref_gu_per_step: 1.60e7
queue_ref_uav_per_step: 1.60e7
queue_ref_sat_per_step: 1.60e7
queue_ref_sat_active_count: 1.0
queue_init_gu_steps: 0.0
sat_cpu_freq: 2.0e10
resource_scale_enabled: false
access_carrier_freq: 2.0e9
backhaul_carrier_freq: 3.0e10
access_noise_figure_db: 5.0
backhaul_noise_figure_db: 3.0
access_fading_mode: ergodic_rician
access_rician_k_db: 10.0
```

解释：

- 带宽不再用旧 `resource_scale` 自动缩放，而是显式写入 1UAV/2GU 小场景带宽。
- 预加载队列关闭，避免 T=10 的最小问题被初始 backlog 主导。
- `sat_cpu_freq=2.0e10` 使 SAT processing 约为 `20.0e6 bit/step`，高于当前总到达 `16.0e6 bit/step`，避免过早变成纯 SAT compute 瓶颈。
- `task_arrival_rate=8.0e6/GU` 是 sweep 后选出的点：queue-aware 与 uniform 有明显差距，但没有 drop。

校准结果：

```text
20 env × 10 step
accel=zero, sat=zero

queue_aware BW:
  reward_mean          = -0.619
  processed_ratio_mean = 0.976
  queue_total_mean     = 2.80e6
  gu_queue_mean        = 2.47e6
  sat_queue_mean       = 3.27e5
  drop_total           = 0

uniform BW:
  reward_mean          = -1.850
  processed_ratio_mean = 0.906
  queue_total_mean     = 7.45e6
  gu_queue_mean        = 7.45e6
  sat_queue_mean       = 0
  drop_total           = 0
```

结果文件：

```text
runs/analysis/queue_calibration_1uav2gu_scaled_20260428/native_tuned_queueaware_vs_uniform_summary.json
```

这个配置适合检查 BW actor 是否能学到“不同 GU 之间的带宽分配比例”。它不是最终正式大场景配置。

## RL/rollout 支持改动

文件：

```text
sagin_marl/rl/structured_mappo.py
```

改动：

- `bw_return_mode='bw_gae'` 仍要求 `target_mode='step_level'`。
- 但只有真的训练 actor 时，才要求 `train_accel=False, train_sat=False, train_bw=True`。
- fixed-policy rollout / calibration 可以 `train_bw=False`，不再因为配置里保留 `bw_gae` 而初始化失败。

原因：

`bw_gae` 是 return/advantage 计算语义；在 rollout-only、固定策略校准时不会更新 actor，因此不应该硬要求 `train_bw=True`。

## 当前未暂存文件清单

环境与执行主路径：

- `sagin_marl/env/channel.py`
- `sagin_marl/env/config.py`
- `sagin_marl/env/native_cuda/kernels.cu`
- `sagin_marl/env/numeric_guards.py`
- `sagin_marl/env/sagin_env.py`
- `sagin_marl/env/structured_batch_env_core.py`
- `sagin_marl/env/structured_driver.py`
- `sagin_marl/env/structured_stage_obs.py`

配置：

- `configs/tmp/structured_bw_sanity_1uav_2gu_t10_stronggap_currentenv_competition_ppo.yaml`

相关训练/rollout 支持：

- `sagin_marl/rl/structured_mappo.py`

实验脚本与产物残留：

- 删除：`scripts/solve_bw_1uav2gu_lp_oracle.py`
- 删除：`artifacts/bw_1uav2gu_lp_oracle_seed42.json`
- 删除：`artifacts/bw_1uav2gu_lp_oracle_seed43.json`
- 删除：`artifacts/bw_1uav2gu_lp_oracle_seed44.json`
- 新增未跟踪：`scripts/experiments/bw_tools/solve_bw_1uav2gu_flow_bcd.py`
- 新增未跟踪：`artifacts/bw_oracle/`

这些实验脚本/产物不是环境主路径的一部分，如果后续准备提交，建议单独决定是否保留或清理。

## 注意事项

- 当前 1UAV/2GU 配置是 diagnostic regime，不是最终论文/正式场景参数。
- `queue_ref_*` 已按当前 `task_arrival_rate * num_gu * tau0` 同步到 `1.60e7`，队列容量继续通过 `queue_max_*_steps` 表达“几步流量”。
- 若后续改变 `task_arrival_rate`，需要同步 `queue_ref_gu_per_step`、`queue_ref_uav_per_step`、`queue_ref_sat_per_step`。
- 如果想验证 actor 学习，不建议回到 `task_arrival_rate=1.2e6`，那一档 queue-aware 和 uniform 都会每步全清空，几乎没有 BW 分配信号。
- 如果把 arrival 继续升高到 `1.0e7/GU` 或以上，环境会明显变重，需要区分 access/BW 瓶颈和 SAT/backhaul downstream 瓶颈。
