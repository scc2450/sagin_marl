### Baseline 算法说明

#### lyapunov算法：非学习式基线

**Topology-aware One-Step Drift-Plus-Penalty Controller**
中文可以写：**拓扑感知的一步式 DPP 控制器**

假设有$i$个排队系统，队列积压用$Q_i$表示。变化方程：

$$Q_i(t+1)=\max\{Q_i(t)+a_i(t)-b_i(t),0\}$$

描述的是当前队列积压量收到新增数据速率和处理数据速率的影响。

$D_{\text{sys\_report}}=\frac{Q_{\text{gu,sum}} + Q_{\text{uav,sum}} + Q_{\text{sat,sum}}}
{\max(\text{sat\_processed\_bits}, \epsilon)}$

定义函数
$$L(t)=\frac12\left(
\sum_g Q_g^2(t)+
\sum_u Q_u^2(t)+
\sum_s Q_s^2(t)
\right)$$

定义lyapunov drift（李雅普诺夫漂移）
$$\Delta L(t)\doteq L(t+1)-L(t)
$$
化简：
$$\Delta L(t)\leq \underbrace{\frac12\sum^N_{i=1}(a_i(t)-b_i(t))^2}_{\doteq B(t)\leq B}+\sum^N_{i=1}Q_i(t)(a_i(t)-b_i(t))
$$
对于前项，假设存在常数B作为上界；后项为主要优化目标

对于额外的用于维护队列的成本函数，可以定义：

$$\min(\Delta L(t)+V⋅E[Cost(t)∣Q(t)])$$

为综合漂移+惩罚上界，通过最小化这个上界可以得到队列稳定且成本最小化的综合最优。其中V表示超参数


当前 step 拿到：

* 所有 ($Q_g, Q_u, Q_s$)
* 所有 UAV 当前位置
* GU 到各 UAV 的当前信道条件 / 路损
* 各 UAV 当前可见 SAT 集合
* 各 UAV 到可见 SAT 的当前回传率、是否多普勒超限
* 安全模块参数
* 每个 UAV 的最大服务 GU 数 ($K_u^{\max}$)
* 每个 UAV 的最大连接 SAT 数 ($M_u^{\max}$)

对每个候选 GU ($g \in \mathcal C_u$)，计算一个接入优先级：


$$\text{Urgency}(t)=P(t)+WQ_v(t)$$
其中：

压力函数为：
$$P(t+1)=\beta P(t)+(1-\beta)P_i(t)$$
虚拟队列函数：
$$Q_v(t+1)=Q_v(t)+P(t)-S(t)$$

根据紧急度函数对加速度和带宽策略进行分配。


评估

```bash
python scripts/evaluate.py --config configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml --run_dir runs/lyapunov --episodes 20 --baseline lyapunov
```
渲染
```bash
python scripts/render_episode.py --config configs/phase1_actions_curriculum_joint_3heads_fading_interference_lyapunov.yaml --run_dir runs/lyapunov --baseline lyapunov --episode_seed 82003 --out runs/lyapunov/episode_lyapunov_seed82003.gif --fps 10
```