我的目标不是让每层队列在 400 步里单调下降，而是让系统在持续到达业务下，三层队列长期维持在一个非零、非满的稳态工作区间；总出流大致匹配总到达，链路各层都持续有业务通过，从而体现系统持续通信能力。


思路我已经按你这次的修正改了，而且落到工具里了。现在不是只靠固定顺序判断，而是同时区分：
- 哪层 backlog 最重
- 哪层最接近自身 `max`
- 哪层先开始 drop
- 当前层该联调哪一对参数

这些都在 [diagnose_queue_regime.py#L428](/d:/研三上/毕设/sagin_marl/scripts/diagnose_queue_regime.py#L428) 之后加进去了；阈值和 CLI 参数在 [diagnose_queue_regime.py#L721](/d:/研三上/毕设/sagin_marl/scripts/diagnose_queue_regime.py#L721)。

现在三层各自都有对应你目标的指标了，不再只看 `active_queue_empty_step_fraction_mean`：
- 空载：`gu/uav/sat_empty_step_fraction_mean`
- 全程非空占比：`gu/uav/sat_nonempty_all_episode_share`
- 压力接近上限：`gu/uav/sat_buffer_fill_fraction_mean/p95`
- 丢弃：`gu/uav/sat_drop_ratio_mean`
- 持续有货：`gu/uav/sat_backlog_equiv_steps_mean/p95`

`active_queue_empty_step_fraction_mean` 我保留了，因为它是系统级护栏，回答的是“前两层这条在途链有没有被抽空”；更严格的三层联合指标现在是 `all_layers_nonempty_step_fraction_mean`，见 [diagnose_queue_regime.py#L327](/d:/研三上/毕设/sagin_marl/scripts/diagnose_queue_regime.py#L327)。

“目标阈值 + 打分函数”也已经做了。默认是：
- 全局：`active_empty <= 0.25`，`all_layers_nonempty >= 0.20`，`total_drop <= 0.05`，`outflow/arrival in [0.90, 1.10]`
- 每层：`empty <= 0.25`，`backlog >= 0.25` 步，`fill_p95 <= 0.20`，`drop <= 0.02`

这些都能按层单独覆盖。脚本现在会输出：
- `passes_thresholds`
- `failed_thresholds`
- `score_total`
- `recommended_tuning_pair`

当前主线 3-episode 验证在 [tmp_score_current_n3_summary.csv](/d:/研三上/毕设/sagin_marl/runs/queue_diag/tmp_score_current_n3_summary.csv)：
- `passes_thresholds = 0`
- `failed_thresholds = active_queue_empty_step_fraction, all_layers_nonempty_step_fraction, total_drop_fraction, gu_empty_step_fraction, uav_empty_step_fraction, uav_drop_fraction, sat_empty_step_fraction`
- `recommended_tuning_pair = b_acc + b_sat_total`

这直接回答了你的问题：`GU` 现在当然有自己的指标，而且阈值也是按层分开的。

我还按你的新思路做了一个把 `sat_cpu_freq` 纳入的候选验证：
[`tmp_score_uavpair_satcpu_n3_summary.csv`](/d:/研三上/毕设/sagin_marl/runs/queue_diag/tmp_score_uavpair_satcpu_n3_summary.csv)
参数是 `b_sat_total=1.30e6, b_acc=3.0e6, sat_cpu_freq=2.8e9`。结果没有更好：
- `score_total` 从 `27.499` 变成 `28.292`
- `sat_backlog_equiv_steps_mean` 掉到 `0.129`

这说明在这个点上，单纯把 `sat_cpu_freq` 往上提，会把末端抽得过轻，不是当前最优方向。

这次我没有回写主线配置，因为新工具已经能稳定告诉我们“哪个点过线、没过哪条、该调哪一对参数”，但我还没找到一个比当前主线明显更平衡的点。下一步最合理的是直接用这套阈值和分数去跑一个小网格：
- 固定 `task_arrival_rate=1.30e5`
- 扫 `b_acc x b_sat_total`
- 每个网格只补 2 个 `sat_cpu_freq` 近邻
- 用 `passes_thresholds + score_total + tuning_priority_layer` 排序

验证我已经跑过：
- `python -m py_compile scripts\diagnose_queue_regime.py`



**三层串联系统调参准则**

1. 先调**能力匹配**，后调**队列参数**。
2. 能力匹配按相邻两端看，不把三层完全独立拆开：

   * GU：`task_arrival_rate ↔ b_acc`
   * UAV：`b_acc ↔ b_sat_total`
   * SAT：`b_sat_total ↔ sat_cpu_freq`
3. 先把该层队列调到**长期不持续上升、也不持续衰减到空**，只在目标区间内波动。
4. 这一阶段优先看：`backlog_equiv_steps`、`empty_step_fraction`、`buffer_fill_fraction`、`drop_ratio`。
5. 当前问题在 **GU/SAT** 时，通常先改单边；问题在 **UAV** 时，通常要联调两边。
6. 当队列变化趋势已经合理后，再调 `queue_init_*`，作用是修正**开局太空、早期断流**。
7. 当能力匹配已基本合理、但仍因短时波动溢出时，再调 `queue_max_*`，作用是增加**缓冲容忍度**，不是替代能力提升。
8. 最后用系统级护栏收口：总 `drop` 不过高，`active/all_layers_nonempty` 不过差，三层都保持“非零、非满、可流动”。

你这句“**每一层变化量数值合适之后，才调初始化值和 max**”是合理的，应该保留。

按顺序是不是应该：
1.固定task_arrival_rate，调b_acc GU初始化队列量 和 GU队列max
1.1优先考虑队列变化量，让队列在400步内保持稳定，不一直增，更间接的是看空不空、到没到上限那些。
1.2 队列变化量合适之后，如果空的多，加初始化队列量，如果drop多，加队列max
2.固定task_arrival_rate b_acc和GU队列相关，调b_sat_total UAV初始化队列量和MAX
2.1 2.2类似
3.固定前面的，调sat_cpu_freq SAT初始化队列量 和 GU队列max
3.1 3.2类似
