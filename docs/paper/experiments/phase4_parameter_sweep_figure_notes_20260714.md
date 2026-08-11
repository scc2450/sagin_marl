# Phase 4 Parameter-Sweep Figure Notes

Generated: 2026-07-14.

Data source:

- `docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv`

Output figures:

- `docs/paper/manuscript/figures/perf_eval_offered_load_sensitivity_nohappo_20260714.pdf`
- `docs/paper/manuscript/figures/perf_eval_resource_capacity_sensitivity_nohappo_20260714.pdf`

Figure-source copies:

- `docs/paper/figure_sources/performance_evaluation_20260714/perf_eval_offered_load_sensitivity_nohappo_20260714.{pdf,png,svg}`
- `docs/paper/figure_sources/performance_evaluation_20260714/perf_eval_resource_capacity_sensitivity_nohappo_20260714.{pdf,png,svg}`

## Plotting Protocol

- Methods: STARS, STARS-GC, QCCS, Lyapunov, QBS, Uniform.
- HA-PPO is intentionally excluded from these sweep figures because its flat
  learned evaluation path was much slower and unstable under concurrent full
  sweep execution. HA-PPO remains available in the main source-scenario table.
- Error bars show standard deviation across the aggregate rows:
  - STARS and STARS-GC: 3 training seeds x 3 eval seed bases.
  - Fixed baselines: 3 eval seed bases.
- Offered-load x-axis uses the actual task arrival rate in Mbit/(GU slot).
- Resource x-axis uses the actual access bandwidth `B_U` in MHz. The sweep
  jointly sets `B_S = 5 B_U` MHz and `f_S = 25 B_U` Gbit/s.

## Claim Bullets

- Offered-load sensitivity: STARS is the best method across all tested arrival
  rates from 1.0 to 4.0 Mbit/(GU slot). Its margin is largest under light and
  nominal load, then narrows under overload as all methods become capacity
  limited.
- Resource-capacity sensitivity: STARS improves monotonically as the
  access/backhaul/service capacities increase and remains the strongest method
  at every tested point.
- STARS-GC has visible high variance across seeds, especially in reward and
  delay, supporting the critic-structure stability claim.
- QCCS and Lyapunov form the strongest non-learning references, but they remain
  below STARS in both sweeps.
- QBS and Uniform are much weaker under load pressure and resource scarcity,
  supporting the need for topology-aware scheduling rather than only bandwidth
  or static allocation.

## Caption Drafts

Offered-load sensitivity:

```latex
\caption{Sensitivity to task arrival intensity. The x-axis reports the actual
arrival rate in Mbit/(GU slot). STARS maintains the highest reward and processed
ratio while keeping lower drop ratio and system delay than the heuristic and
critic-ablation baselines across the tested load range.}
```

Resource-capacity sensitivity:

```latex
\caption{Sensitivity to joint access, backhaul, and satellite service capacity.
The x-axis reports the access bandwidth $B_U$ in MHz, with $B_S=5B_U$ MHz and
$f_S=25B_U$ Gbit/s. STARS benefits most from additional capacity and consistently
outperforms the fixed topology-aware and queue-aware schedulers.}
```
