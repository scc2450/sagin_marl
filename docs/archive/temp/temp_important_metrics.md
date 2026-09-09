明确原本的评价指标
---------------------
### A. 策略性能主指标

这是决定 policy 好坏的第一层。

我建议你主看这 5 个：

1. **outflow_arrival_ratio**
   越高越好，但不能靠异常短 episode 或冒险换出来。

2. **drop_ratio**
   越低越好。

3. **near_collision_ratio + collision + steps**
   这是安全门槛。旧 reward 虽然吞吐高，但全撞了，不能接受。

4. **gu_queue_mean / uav_queue_mean**
   因为 phase1 只有 accel 动作，最直接影响的是接入与前端中继层。

5. **terminated_early / termination_reason**
   有没有靠提前结束“刷指标”。

对你当前三组来说：

* old：吞吐和 drop 看着好，但安全不达标，所以不能用
* new：安全和前端队列明显更好，但 drop 偏高
* zero：是环境工作点基线，不是策略最优目标

---

### B. 链路分层健康指标

这是判断“策略是不是在把压力往后偷推”的关键。

你现在最该盯这 6 个：

1. **sat_drop_ratio**
2. **sat_incoming_arrival_ratio**
3. **sat_processed_arrival_ratio**
4. **sat_processed_incoming_ratio**
5. **sat_queue_drift_ratio**
6. **sat_queue_arrival_steps_mean**

这 6 个一起看，逻辑是：

* sat_incoming_arrival_ratio 高：说明你把很多流量推到后端了
* sat_processed_incoming_ratio < 1：说明 SAT 消化不过来
* sat_queue_drift_ratio > 0：说明 SAT backlog 还在积
* sat_drop_ratio 高：说明已经开始实打实地在 SAT 丢了
* sat_queue_arrival_steps_mean 高：说明后端 backlog 对“典型一步到达量”来说偏大

你现在 new reward 的典型形态正是这个组合。
所以它不是“纯粹更会通信了”，而是“更会把前端流量推向后端，但后端没完全接住”。

---

### C. 环境工作点指标

这是看环境是不是仍在可学区，不是直接给 policy 打分。

主看：

* all_layers_nonempty_step_fraction
* gu/uav/sat_queue_nonzero_step_fraction
* gu/uav/sat_queue_fill_fraction_mean
* gu/uav/sat_queue_drift_ratio

这里你要特别记住：

**这组指标不是越大越好，也不是越小越好。**
它们的作用是“解释训练发生在什么工作点上”。