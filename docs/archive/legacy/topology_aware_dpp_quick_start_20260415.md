## 拓扑感知 DPP 基线 - 快速使用指南

> 归档说明：本文来自 `lyapunov-dpp` 分支的拓扑感知 DPP baseline 实验资料。当前 `mac-python-rollout`/joint MC-GAE 主线未默认包含这些实现细节；使用前请先确认目标分支或把对应实现迁回当前分支。

### 背景
用户指出 Batch 1-3 的 DPP 实现仅在旧拓扑上评估候选加速度，忽视了位置变化带来的链路质量、可见性变化。  
**已修复**: 通过环境回调模式，现在支持完整的物理拓扑预测。

---

### 四个关键变更

#### 1️⃣ 新函数：完整拓扑预测
**文件**: `sagin_marl/rl/baselines.py` L356-417  
**函数**: `_predict_topology_after_accel()`

```python
# 调用方式（自动调用，无需手动）
topo = _predict_topology_after_accel(obs, cfg, accel_vec)
# 返回：{"rel_next", "eta", "sat_visible_mask", "sat_rate_per_user"}
```

**特点**:
- ✅ 自动降级：无回调时使用启发式  
- ✅ 完整支持：有回调时调用真实链路计算

---

#### 2️⃣ 改进的候选评估
**文件**: `sagin_marl/rl/baselines.py` L593-653  
**函数**: `_eval_candidate()` (内嵌于 `_dpp_one_agent`)

**改进**:
```python
# 旧：固定拓扑 + 距离惩罚
rel_next = _predict_users_rel_after_accel(obs, cfg, cand)
eta = obs["users"][:, 3]  # 用旧的 eta

# 新：完整拓扑 + 混合 eta + 更新 SAT 可见性
topo = _predict_topology_after_accel(obs, cfg, cand)
eta_blend = 0.4 * eta_obs + 0.6 * topo["eta"]
sat_sel[~topo["sat_visible_mask"]] = 0.0
```

---

#### 3️⃣ 环境回调接口
**文件**: `sagin_marl/rl/baselines.py` L654-780  
**函数**: `lyapunov_queue_aware_policy_step()`

**新参数**:
```python
def lyapunov_queue_aware_policy_step(
    obs_list,
    cfg,
    state=None,
    env_callbacks=None,  # 👈 新增
):
    # env_callbacks 是 Dict[str, Callable]:
    # - "compute_access_rates": (agent_id, accel, obs) -> (eta, rates)
    # - "check_sat_visibility": (agent_id, accel, obs) -> visible_mask
```

---

#### 4️⃣ evaluate.py 集成
**文件**: `scripts/evaluate.py` L75-130  
**函数**: `_build_lyapunov_env_callbacks()`

```python
# 自动创建回调（在 baseline==lyapunov 分支）
env_callbacks = _build_lyapunov_env_callbacks(env, cfg)

lyapunov_queue_aware_policy_step(
    obs_list, cfg,
    state=baseline_state,
    env_callbacks=env_callbacks,  # 自动传入
)
```

---

### 使用场景

#### 场景 A：直接运行评估（推荐）
```bash
python scripts/evaluate.py \
  --config configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml \
  --baseline lyapunov \
  --run_dir runs/lyapunov_aware \
  --num_episodes 50
```

✅ 自动启用拓扑感知（如果 env 可用）  
✅ 无需代码改动

---

#### 场景 B：手动 DPP 调用（自定义环境）
```python
from sagin_marl.rl.baselines import lyapunov_queue_aware_policy_step

# 定义自己的速率计算回调
def my_compute_access_rates(agent_id, accel_vec, obs):
    # 实现你的链路模型
    eta = compute_eta_based_on_position(obs["own"], accel_vec)
    rates = compute_rates(eta)
    return eta, rates

callbacks = {"compute_access_rates": my_compute_access_rates}

accel, bw, sat, state = lyapunov_queue_aware_policy_step(
    obs_list, cfg,
    env_callbacks=callbacks,
)
```

✅ 完全控制回调实现  
✅ 支持自定义链路模型

---

#### 场景 C：启发式模式（快速测试）
```python
# 不传入回调 → 自动降级
accel, bw, sat, state = lyapunov_queue_aware_policy_step(
    obs_list, cfg,
    env_callbacks=None,  # 或不传
)

# 等同于 Batch 1-3 的行为（距离惩罚 + 旧 eta）
```

✅ 性能同原始版本  
✅ 用于基准对比

---

### 配置参数（无变更）

所有现有 DPP 参数保持不变：

```yaml
# configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml
baseline:
  baseline_lyapunov_mode: dpp  # 启用 DPP 模式
  
  dpp_accel_num_candidates: 9
  dpp_accel_step_scale: 0.6
  dpp_gu_max_select: 6
  dpp_access_weight: 1.0
  dpp_backhaul_weight: 1.0
  # ...（其他参数）
```

---

### 测试与验证

```bash
# 运行烟测试（验证新功能）
python scripts/experiments/dpp/smoke_topology_aware_dpp.py

# 输出示例
# ✓ Topology prediction WITHOUT env_callbacks
# ✓ Topology prediction WITH mock env_callbacks  
# ✓ DPP Baseline with Topology
# ✓ DPP Baseline WITH env_callbacks
# ✓ ALL TESTS PASSED
```

---

### 常见问题

**Q1**: 性能会变慢吗？  
**A**: 约 5-10% 增加（回调开销）。可通过设置 `env_callbacks=None` 回到原速度。

**Q2**: 能否混用模式（urgency + dpp）？  
**A**: 是的，通过配置切换：
```yaml
baseline_lyapunov_mode: urgency  # 原始模式
# 或
baseline_lyapunov_mode: dpp      # 新的拓扑感知模式
```

**Q3**: 回调返回异常怎么办？  
**A**: 自动捕获异常 → 降级到启发式 → 继续运行（不会崩溃）

**Q4**: 怎样写自己的回调？  
**A**: 参见 `scripts/evaluate.py` 的 `_build_lyapunov_env_callbacks()` 示例

---

### 关键文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `sagin_marl/rl/baselines.py` | 356-417 + 593-653 + 654-780 | 核心拓扑实现 |
| `scripts/evaluate.py` | 75-130 | 环境回调生成 |
| `scripts/experiments/dpp/smoke_topology_aware_dpp.py` | 全部 | smoke 测试 |
| `docs/archive/legacy/topology_aware_dpp_summary_20260415.md` | 全部 | 详细文档 |

---

### 下一步建议

1. **立即**: 在 `lyapunov-dpp` 实验实现可用的分支上运行 `python scripts/experiments/dpp/smoke_topology_aware_dpp.py` 确保环境已就位
2. **评估**: 用新配置跑一次 baseline 对比，观察性能
3. **调优**: 如有需要，修改混合权重或回调实现
4. **论文**: 记录 DPP(启发式) vs DPP(完整拓扑) 的对比结果

---

### 快速命令参考

```bash
# 验证安装
python scripts/experiments/dpp/smoke_topology_aware_dpp.py

# 运行 DPP baseline（自动拓扑感知）
python scripts/evaluate.py --config configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml --baseline lyapunov

# 对比：urgency 模式
python scripts/evaluate.py --config configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml --baseline lyapunov -p baseline_lyapunov_mode=urgency

# 查看代码
grep -n "_predict_topology_after_accel\|env_callbacks" sagin_marl/rl/baselines.py
```

---

**更新日期**: 2026-04-15  
**状态**: ✅ 已完成，已测试，已文档化
