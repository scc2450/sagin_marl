# Section 5 Table/Figure Placement and Claim Plan

Date: 2026-07-13

Status: fixed paper-facing drafting scaffold for Section 5.

Scope: Section 5 covers the controlled source scenario, checkpoint selection,
main held-out comparison, learning-side ablation, and runtime/resource cost.
Generalization, scale transfer, and larger-scale evidence belong in Section 6.

Planning labels such as `Table 5-P1` and `Fig. 5-F1` are local planning ids, not
final IEEE table or figure numbers.

## Evidence Anchors

- Main evidence freeze:
  `docs/paper/experiments/section5_performance_assets_20260713.md`
- Main selected-checkpoint table:
  `docs/paper/table_sources/phase4_formal_heldout_source_selected_main_20260713.csv`
- Selected-versus-final companion:
  `docs/paper/table_sources/phase4_formal_heldout_source_learned_selected_final_20260713.csv`
- Runtime/resource sources:
  `docs/paper/table_sources/phase4_runtime_resource_summary_20260713.csv`
  and
  `docs/paper/table_sources/phase4_formal_eval_wallclock_summary_20260713.csv`
- Figure sources:
  `docs/paper/figure_sources/phase4_section5_performance_20260713/`
- LaTeX-ready figures:
  `docs/paper/manuscript/figures/phase4_training_validation_curves_20260713.pdf`
  `docs/paper/manuscript/figures/phase4_main_performance_grouped_bars_20260713.pdf`
  `docs/paper/manuscript/figures/phase4_globalcritic_seed_level_20260713.pdf`

## Section 5 Opening

Role: state the evaluation questions before showing numbers.

Recommended questions:

1. Does the proposed structured scheduler improve held-out source-scenario
   performance over fixed scheduling heuristics?
2. Does the relational critic improve stability and performance over a global
   critic under the same training/evaluation interface?
3. Is a flat MAPPO-like learned adapter sufficient under the same hybrid masked
   SAGIN action interface?
4. What training/runtime cost is required to obtain the selected policies?

Claim boundary:

- This section validates source-scenario effectiveness and design necessity.
- It does not claim arbitrary scale generalization.
- It does not claim that MAPPO-like faithfully reproduces any specific external
  MAPPO, MADDPG, or SAGIN paper.

## 5.1 Evaluation Protocol

Role: make the benchmark auditable before any result appears.

| Planning id | Placement | Content | Source |
|---|---|---|---|
| `Table 5-P1` | End of 5.1 | Source scenario and training/evaluation protocol | `section5_performance_assets_20260713.md`; `experiment_protocol.md` |

`Table 5-P1` should include:

- scenario: `3uav20gu_t250`;
- policy mode: deterministic held-out evaluation;
- evaluation seed bases: `980000`, `981000`, `982000`;
- episodes per seed base: 64;
- total held-out episodes per learned training seed: 192;
- learned methods: RelCritic, GlobalCritic, MAPPO-like, each with 3 training
  seeds;
- fixed baselines: `cluster_center_queue_aware`, `maxweight_lyapunov`,
  `observable_cluster_queue_aware`, `queue_aware_bw`, `static_uniform`;
- checkpoint rule: selected checkpoint for main result, final checkpoint for
  companion/stability evidence;
- rollout scale: `num_envs=64`, `rollout_env_steps=250`, i.e. 16,000
  environment transitions per update.

Claim bullets:

- The evaluation uses held-out seed bases that are separate from checkpoint
  selection.
- The controlled source scenario keeps workload and system scale fixed, so
  differences can be attributed to policy and architecture choices.
- Reporting both selected and final checkpoints prevents early stopping from
  being mistaken for convergence proof.
- Update counts should be translated into environment-interaction scale when
  compared with papers that report episode counts.

Caveats:

- Do not present `3uav20gu_t250` as a generalization benchmark.
- Do not compare "hundreds of updates" to literature "thousands of episodes"
  without explaining vectorized update scale.

## 5.2 Baselines and Metrics

Role: define who we compare against and how good/bad is measured.

| Planning id | Placement | Content | Source |
|---|---|---|---|
| `Table 5-P2` | First half of 5.2 | Baseline taxonomy | `section5_performance_assets_20260713.md`; method/baseline notes |
| `Table 5-P3` | Second half of 5.2, or merged with `Table 5-P2` if space is tight | Metric definitions and directions | table-source metric columns and Section 3 objective |

`Table 5-P2` should classify:

- proposed method: RelCritic;
- critic ablation: GlobalCritic;
- learned flat adapter: MAPPO-like;
- strong fixed heuristic: `cluster_center_queue_aware`;
- queue/Lyapunov heuristic: `maxweight_lyapunov`;
- observable or partial-information heuristics:
  `observable_cluster_queue_aware`, `queue_aware_bw`, `static_uniform`.

`Table 5-P3` should define:

- Reward: larger is better; aggregate objective-facing score.
- Processed ratio: larger is better; task service effectiveness.
- Drop ratio: smaller is better; task loss/failure pressure.
- Backlog: smaller is better; queue pressure proxy.
- `D_sys`: smaller is better; system delay/queueing proxy used in logs.
- Collision: smaller is better; safety violation frequency.
- Runtime/resource cost: interpreted jointly with achieved performance.

Claim bullets:

- The fixed baselines are not strawmen; the strongest references include
  cluster-center queue-aware and MaxWeight/Lyapunov-style scheduling.
- Learned comparisons test two mechanism questions: relational critic versus
  global critic, and structured policy/critic versus a flat MAPPO-like adapter.
- Reward must be interpreted together with processed ratio, drop ratio, backlog,
  `D_sys`, and collision.

Caveats:

- MAPPO-like is an in-pipeline adapter baseline under the same hybrid masked
  action interface, not a faithful external-paper reproduction.
- Privileged or stronger-information baselines should be labeled honestly.

## 5.3 Training Dynamics and Checkpoint Selection

Role: explain how selected checkpoints are chosen and why final checkpoints are
not the center of the main table.

| Planning id | Placement | Content | Source |
|---|---|---|---|
| `Fig. 5-F1` | Early 5.3 | Training/checkpoint-validation curves with selected-best and final markers | `phase4_training_validation_curves_20260713.pdf` |
| `Table 5-P4` | End of 5.3 or appendix | Selected-versus-final learned-method companion table | `phase4_formal_heldout_source_learned_selected_final_20260713.csv` |

Claim bullets:

- Selected checkpoints are chosen by the predefined validation rule and then
  evaluated on held-out seed bases.
- RelCritic selected is the strongest and most stable source-scenario learned
  result in the current evidence.
- Final checkpoints are useful for stability analysis but should not replace the
  selected-checkpoint main result.
- MAPPO-like remains weak after completing three seeds, so its conclusion does
  not depend on an incomplete seed set.

Caveats:

- Do not write that patience stopping proves convergence.
- Do not hide final checkpoints; keep them as companion or appendix evidence.
- If page budget is tight, move `Table 5-P4` to appendix and summarize it in
  prose.

## 5.4 Main Held-Out Source-Scenario Results

Role: provide the central performance result for the controlled source scenario.

| Planning id | Placement | Content | Source |
|---|---|---|---|
| `Table 5-P5` | Start of 5.4 | Main selected-checkpoint held-out comparison, including fixed baselines | `phase4_formal_heldout_source_selected_main_20260713.csv` |
| `Fig. 5-F2` | After `Table 5-P5` | Grouped bars for Reward, Processed, Drop, and `D_sys` | `phase4_main_performance_grouped_bars_20260713.pdf` |

`Table 5-P5` should include RelCritic selected, GlobalCritic selected,
MAPPO-like selected, all five fixed baselines, Reward, Processed, Drop, Backlog,
`D_sys`, Collision, and mean +/- standard deviation where available.

Claim bullets:

- RelCritic is the strongest method in the source scenario across reward and
  network-facing metrics.
- RelCritic improves over the strongest fixed baselines, especially
  `cluster_center_queue_aware` and `maxweight_lyapunov`, rather than only over
  weak random/static baselines.
- The gain is visible in processed ratio, drop ratio, backlog, and `D_sys`, not
  only in reward.
- MAPPO-like underperforms both RelCritic and stronger fixed heuristics under
  the same held-out protocol, supporting the need for structured SAGIN policy
  design.

Caveats:

- Do not imply RelCritic dominates every possible heuristic family.
- Do not imply external MAPPO/MADDPG papers are beaten.
- Do not let `Fig. 5-F2` replace the main table; the table carries auditable
  values and the figure provides multi-metric readability.

## 5.5 Learning-Side Ablation and Baseline Boundary

Role: explain mechanism evidence: why the relational/structured design matters.

| Planning id | Placement | Content | Source |
|---|---|---|---|
| `Fig. 5-F3` | Middle of 5.5 | Seed-level RelCritic versus GlobalCritic display | `phase4_globalcritic_seed_level_20260713.pdf` |

Cross-references:

- Refer back to `Table 5-P5` for RelCritic, GlobalCritic, and MAPPO-like
  aggregate numbers.
- Refer back to `Table 5-P4` if discussing selected versus final stability.

Claim bullets:

- GlobalCritic has much larger training-seed variance than RelCritic, so a
  global-only critic can occasionally find a usable policy but is not robust
  under this protocol.
- RelCritic's advantage is paired with lower drop, lower backlog, and lower
  `D_sys`, not merely a higher mean reward.
- MAPPO-like tests whether a flat learned multi-agent baseline suffices after
  removing topology-aware actor/critic structure; the current answer is no under
  our hybrid masked interface.
- The ablation supports the Section 4 design claim that topology-aware
  relational value modeling matters for coordinated SAGIN scheduling.

Caveats:

- Do not describe GlobalCritic as completely failed; its high variance is the
  point.
- Do not describe MAPPO-like as "the" MAPPO result in the literature.
- Do not introduce Section 6 generalization results here.

## 5.6 Runtime and Resource Cost

Role: show that the performance gains have a reported computational cost.

| Planning id | Placement | Content | Source |
|---|---|---|---|
| `Table 5-P6` | End of 5.6 | Training wall-clock, selected update, final update, completed updates, rollout scale | `phase4_runtime_resource_summary_20260713.csv`; `phase4_formal_eval_wallclock_summary_20260713.csv` |

`Table 5-P6` should include method, training wall-clock hours, selected update,
final update, completed updates, `num_envs=64`, `rollout_env_steps=250`, and
optional formal held-out evaluation wall-clock if space permits.

Claim bullets:

- RelCritic requires more training time than the flatter learned variants, but
  the added cost is tied to clearly stronger held-out performance.
- Reporting selected and final updates lets readers estimate the actual
  interaction budget.
- The source protocol is not a tiny single-episode evaluation; each update
  corresponds to 16,000 environment transitions.

Caveats:

- Do not claim deployment-time feasibility beyond what inference/evaluation logs
  support.
- Do not compare wall-clock directly with literature unless hardware and
  vectorization details are aligned.

## Recommended Artifact Order

1. `Table 5-P1`: protocol.
2. `Table 5-P2`: baseline taxonomy.
3. `Table 5-P3`: metric definitions.
4. `Fig. 5-F1`: training/checkpoint validation.
5. `Table 5-P5`: main held-out source-scenario result.
6. `Fig. 5-F2`: multi-metric grouped bars.
7. `Fig. 5-F3`: GlobalCritic seed-level variance.
8. `Table 5-P6`: runtime/resource cost.
9. `Table 5-P4`: selected-versus-final companion, either after 5.3 or in
   appendix depending on page budget.

If page budget becomes tight:

1. Merge `Table 5-P2` and `Table 5-P3`.
2. Move `Table 5-P4` to appendix.
3. Keep `Table 5-P5`, `Fig. 5-F2`, and `Table 5-P6` in the main paper.
4. Keep `Fig. 5-F1` if checkpoint selection is questioned; otherwise compress
   it to prose plus appendix figure.

## Do-Not-Write List

- Do not lead Section 5 with the main numeric table before the protocol.
- Do not use reward-only discussion.
- Do not call MAPPO-like an external-paper reproduction.
- Do not make 6UAV/80GU claims in Section 5.
- Do not treat final checkpoints as the primary performance target.
- Do not hide high variance for GlobalCritic.
- Do not introduce untraceable rows that are not in the registered table
  sources.
