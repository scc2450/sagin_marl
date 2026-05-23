只看你这两份 md，我的判断是：

**不像是“整个 BW 实现坏了”的问题。更像是：方法本身确实难学，但里面还夹着一个很窄的实现/接线问题。**
原因是你拿到的现象太“有结构”了：几何和参数化修复会带来方向一致的改善，Dirichlet 主线能做出稳定的 stochastic BW，all-policy 整体能赢 all-heuristic，但 BW 单头在 hybrid 里还是输 heuristic；后续四条路线也都是按各自理论预期失败，而不是随机崩坏。这样的 pattern 更像 credit、variance 和 readout 没对齐，不像一个大 bug。 

再加上你已经有相关回归测试通过、smoke run 正常、主实验吞吐稳定，这会进一步削弱“大范围实现错误”的可能。

但我会强烈怀疑还有一个**局部实现问题**，而且优先级很高：**deterministic readout / postprocess contract**。你这里同时出现了三个强信号：当前 noaux 主线同一 checkpoint 下 stochastic 很强、deterministic 很弱；offline ranking 指标能明显改善，但 deterministic holdout 不升反降；最小 decomposition 零步替换后 det/stoch 一起回退。与此同时，你又在另一组 matched eval 里见过 `dirichlet + per_simplex_dim` 的 deterministic 表现并不差。这更像“训练分布学到了一些东西，但部署读出没对齐”，不像 actor 本体完全没学到。 

第二个我会查的是 **`sample / log_prob / entropy / deterministic_readout` 在 mask 和 valid_count 变化下是不是严格同一个 contract**。你自己的总结已经说明，PPO 几何、`valid_count`、`per_simplex_dim` 和参数化都是病灶之一；这类地方最容易藏“不是大 bug、但足够把 BW 更新洗脏”的 silent mismatch。

第三个我会查的是 **`bw_counterfactual_credit_enabled` 的 critic 排序质量**，不是“有没有开这个开关”。你已经证明 shared advantage 确实很脏，但 v0 训练没兑现；这更像 baseline/critic 不够准，而不等于 counterfactual 方向错了。文献里这类现象也很常见：COMA 就是为单 agent counterfactual baseline 提出来的，但后续分析显示，多 agent policy gradient 的方差会随着 agent 数和其它 agent 的探索增大，COMA 也只能消掉一部分噪声。 ([arXiv][1])

另一个要明确说的是：**Dirichlet 这条线不该丢。** 对 simplex allocation，Dirichlet policy 本来就是更自然的分布族；已有工作直接报告它相对 Gaussian-softmax 更快、更稳、更鲁棒。你自己的结果也和这个方向一致。([arXiv][2]) 

我建议你先做 3 个快判，不要先改大训练：

1. **best-of-N readout test**
   取你当前最好 stochastic BW，在固定 accel/sat 的 panel states 上，每个 state 采 8 或 32 个 BW action，再用你现成的 local full-episode probe 或短程 proxy 选 best。
   如果 “同一个 policy 的 best-of-N” 明显逼近或超过 heuristic，而当前 deterministic readout 还很差，那几乎就能判定：**主要问题在 readout / deployment，不在 actor 本体**。多 action per state 的思路本来就有降方差依据，很适合你现成的 within-state BW re-eval 工具链。([Proceedings of Machine Learning Research][3])

2. **`old_log_prob` 回放一致性 test**
   从 rollout 缓存里抽样，把保存的 action、mask、valid_count、旧 policy 参数重新喂回去，检查 recompute 的 `old_log_prob` 能不能精确回放；再对 deterministic postprocess 后的 action 检查 simplex、mask 和 finite log_prob。
   这个 test 一旦不过，先别谈算法。

3. **critic rank-corr test**
   在 fixed partners 上，对同一个 state 的 K 个 BW 候选 action，同时算“真实 local return 排序”和“critic / `A_bw` 排序”。
   如果 rank correlation 很差，下一步就不该继续折腾 actor 头，而应该直接改 estimator。近两年的 DAE 和 GPAE 都是在干这件事：避免 MAPPO/GAE 给所有 agent 同一个 advantage，给出更显式的 per-agent credit。([Proceedings of Machine Learning Research][4])

至于“还有没有改 BW 的办法”，有，但我只会认真看这 3 条：

**第一条，最快也最实用：proposal + select + distill。**
不要再逼同一个 BW 头既负责 stochastic exploration，又天然给出好 deterministic readout。直接把当前 stochastic BW 当 proposal distribution。部署或离线 relabel 时，候选集放 `{heuristic, 当前 deterministic readout, N 个 policy samples, 少量局部扰动}`，再用快速 proxy 或短回滚选 best；最后把赢家 distill 成单独的 student/readout head。
这和你已经失败的 route 1 不一样，因为它**不再固定原 deterministic readout，也不只学 pairwise ranking**，而是直接学“在当前支持集里，哪个 action 真赢”。如果以后真要再碰 offline-to-online，我只会在这种有明确 action label 的版本上继续，而不是继续做 preference/pull。IQL 这类方法本来也更适合“先离线提取，再在线小步微调”的节奏。 ([arXiv][5])

**第二条，唯一还值得当新项目的：full autoregressive / sequential BW allocator。**
但必须是**全量版**，而不是你已经试过的 `support + epsilon tail` 轻量版；并且一定要带 **de-biased init**，最好再加 **随机或可学习的分配顺序**。
原因不是空想：CAOSD 和 PASPO 这两条 allocation 文献都表明，decomposition / autoregressive 在约束 allocation 上可以优于通用 constrained RL；PASPO 还明确指出 sequential sampling 会有初始顺序 bias，需要专门 de-biasing 才能避免早期探索被前面几个维度吃掉。你现在的 v0 只能否定那个轻量实现，不能否定 full sequential allocator。 ([arXiv][6])

**第三条，最像“方法上真正对症”的：per-agent advantage + many-action local rollouts。**
也就是：全局轨迹照常训练，但在少量 panel states 上，固定 accel/sat，只给 BW 采多个候选 action，用真实局部回报构造 within-state normalized advantage，再只更新 BW。
这比你做过的 counterfactual v0 更强，因为它不是把希望全押在一个 joint critic 上；也比 preference fitting 更直接，因为它直接进 actor loss。DAE、GPAE 和 optimal-baseline 这条线都在说明，共享 GAE/朴素 counterfactual 往往不够，显式 per-agent credit 才是重点。([Proceedings of Machine Learning Research][4])

我不建议你再往这些方向加时间了：AWR/V-MPO-lite 继续扫、固定 readout 的 preference/pull、以及不带 de-biasing 的轻量 decomposition。你自己的快判已经足够说明，它们不是当前阶段的高性价比路线。

所以最后一句话是：

**不像整个实现错了。**
**最像错的是 deterministic readout contract；其次是 BW 的 `old_log_prob` / mask contract；第三是 counterfactual critic 的排序质量。**
**还有路，但只剩一条快线（proposal+select+distill）和两条研究线（full autoregressive、per-agent advantage）。**

把这三段代码/函数贴出来最值钱：
`bw_deterministic_readout` 相关函数、BW 的 `log_prob` 回放路径、`bw_counterfactual_credit_enabled` 的 loss 片段。下一轮我就能把判断收紧到具体哪一块最可疑。

[1]: https://arxiv.org/abs/1705.08926 "https://arxiv.org/abs/1705.08926"
[2]: https://arxiv.org/abs/2201.08445 "https://arxiv.org/abs/2201.08445"
[3]: https://proceedings.mlr.press/v202/nauman23a/nauman23a.pdf "https://proceedings.mlr.press/v202/nauman23a/nauman23a.pdf"
[4]: https://proceedings.mlr.press/v162/li22w/li22w.pdf "https://proceedings.mlr.press/v162/li22w/li22w.pdf"
[5]: https://arxiv.org/abs/2110.06169 "https://arxiv.org/abs/2110.06169"
[6]: https://arxiv.org/pdf/2404.10683 "https://arxiv.org/pdf/2404.10683"
