# 下一步应该改什么：把“队列惩罚”从平均线性变成“尾部阈值惩罚”（K 仍然=1，物理意义保留）

你担心 `q_norm_active` 小、平方会更小——没错，所以我们**不是对 q_norm 平方**，而是对**超过阈值的那部分**平方：

[
P_{\text{tail}} = w_{\text{tail}}\cdot \big[\max(q_norm-q_0,0)\big]^2
]

这样：

* 在安全区 (q_norm \le q_0)：惩罚≈0（不扰动）
* 在爆尾区 (q_norm \gg q_0)：惩罚迅速变大（专治 P95/P99）

### 2.1 阈值怎么选（直接用 eval_fixed）

用 fixed 的 P99：
✅ **`q0 = 0.0051`**（我建议先用 0.005，四舍五入）

### 2.2 权重怎么选（给你一个“不容易炸”的起点）

看 random 的 P95：0.0339。
此时 (q-q0 \approx 0.029)，平方约 0.00084。
如果我们希望在这种“明显危险但还没极端”的区域，惩罚贡献大概到 **0.01** 量级，则：

[
w \approx 0.01 / 0.00084 \approx 12
]

✅ 起步建议：**`omega_q_tail = 10`**（先保守点）

### 2.3 代码怎么改（只改 active 分支）

在你的 `_compute_reward()` 里，`use_active_queue_delta` 分支现在是：

```python
queue_term = q_norm_active
term_queue = -cfg.omega_q * queue_term
```

改成：

```python
# --- tail-risk queue penalty (active) ---
q0 = float(getattr(cfg, "q_norm_tail_q0", 0.005) or 0.005)  # fixed P99 ≈ 0.00506
x = max(q_norm_active - q0, 0.0)
queue_term = x * x   # quadratic on exceedance
term_queue = -float(getattr(cfg, "omega_q_tail", cfg.omega_q)) * queue_term
```

并把 `last_reward_parts["queue_pen"]` 保持为 `queue_term`，同时把 `q0` 也记录一下方便调参。

### 2.4 配置新增项（建议）

```yaml
q_norm_tail_q0: 0.005
omega_q_tail: 10.0
```

> 注意：你原来的 `omega_q=1` 不用动；我们是新增一个“尾部风险专用权重”。


# 4) 防撞层你现在这套“距离势场”为什么不好用？怎么改才更稳？

你现在的 repulsion：

```python
a_rep += eta * (1/dist - 1/d_alert) * unit(diff)
a = clip(a + a_rep, a_max)
```

常见问题是：**dist 小时 1/dist 爆炸 → a_rep 巨大 → 最后被 a_max clip → 方向抖动/推不开**。而且它只看距离，不看相对速度（加速度控制很容易“来不及刹”）。

## 4.1 立刻能提升稳定性的两处小改动（强烈建议先做）

### (A) 给 repulsion 自己加限幅（在 clip 之前）

```python
a_rep = np.clip(a_rep, -cfg.a_max, cfg.a_max)
a = a + a_rep
a = np.clip(a, -cfg.a_max, cfg.a_max)
```

### (B) 把 (1/dist - 1/d_alert) 换成“线性/平方”更平滑的形状

例如线性：

```python
# dist in (0, d_alert)
strength = (d_alert - dist) / max(d_alert - cfg.d_safe, 1e-6)   # 0..something
strength = np.clip(strength, 0.0, 1.0)
a_rep += cfg.avoidance_eta * strength * (diff / dist)
```

然后 `avoidance_eta` 的量纲就清晰了：它就是“最大推开加速度”。
✅ 建议你把 `avoidance_eta` 设成 **0.5~1.0 * a_max**（先别用 100 这种会必然触发 clip 的值）。

> 你现在 `avoidance_eta=100`，大概率是“几乎每次都被 clip 成 a_max”，所以效果看起来反而差。

## 4.2 解决“热点 vs 避碰矛盾”的更好方法：自适应安全权重（而不是固定增大 crash 惩罚）

共享 actor 下固定提高 crash 权重确实可能压死任务。更稳的是：

* 设目标碰撞率 `p_target`（例如每 400 步碰撞概率 < 5%）
* 若最近窗口碰撞率 > 目标：自动提高 `avoidance_eta` 或安全惩罚权重
* 若低于目标：自动降低

这相当于把“避碰”做成约束优化（拉格朗日乘子），不会一直把热点目标压死。

---

# 5) 追质心还要不要？

结合你的 eval：centroid 策略碰撞极多（16/20 早停），且队列尾部也远差于 fixed。说明 **“追质心 ≠ 更稳更小队列”**。

但你也说过：一开始完全没有 centroid，训练会崩（队列很快满溢）。因此建议是：

* **保留 centroid 作为早期 stabilizer**
* 但做两件事：

  1. **更快退火**（比如 120k 步太长了，可以缩短到 30k 或 50k）
  2. 退火的同时，让 **tail penalty + safety penalty 接棒**（这就是你说的交叉退火的意义）

---

# 6) 为什么 reward 一直下降？你该看什么指标？

只要 `centroid_eta` 在下降，`r_term_centroid` 就会下降，episode_reward 跟着下降是必然的（你图里就是这样）。这不是“训练变差”的证据。

从现在起你评估“训练是否进步”，建议用这 3 个稳定 KPI（逐 update 统计）：

1. `collision_rate`（每 N 步碰撞次数或 early stop 率）
2. `q_norm_active` 的 **P95/P99**（不要只看均值）
3. `queue_total_active` 的 **P95/P99**（bits）

你已经证明策略差异主要体现在尾部，这比均值更重要。

---



---

1. 不要并行读取 
2. 执行代码前先激活虚拟环境 
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```