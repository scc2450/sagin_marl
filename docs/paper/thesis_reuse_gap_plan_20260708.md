# Thesis Reuse and Manuscript Gap Plan

Date: 2026-07-08

Source material:

- Thesis PDF: `thesis.pdf`
- Manuscript skeleton: `docs/paper/manuscript/`
- Paper structure guidance: `.local_guidance/phase4/manuscript_structure_20260707.md`
- Experiment protocol: `docs/paper/experiment_protocol.md`
- Evidence index: `docs/paper/evidence_index.md`
- Phase 2 and Phase 3 result guidance under `.local_guidance/`

This note is a first-pass paper-facing audit. It decides what can be reused from
the master's thesis, what must be rewritten or revalidated for a journal paper,
and what should be done first.

## 1. High-level judgment

The thesis is very useful as the system and method base, but it should not be
treated as the final paper result package.

What is reusable:

- The system model, notation scaffold, queue dynamics, task flow, mobility and
  visibility constraints.
- The method architecture narrative: CTDE, structured observations, masks,
  staged actor, critic, safety-aware execution, and queue-aware reward.
- The simulation parameter tables, after checking that they match the current
  configs used by the latest runs.
- The figure concepts, preferably redrawn in English rather than copied as
  screenshots.

What is not directly reusable as final journal evidence:

- The old thesis main-result numbers, because later work introduced stronger
  baselines, a stricter checkpoint-selection protocol, generalization tests, and
  6UAV/80GU evidence.
- Any claim that only compares against weak baselines such as static, random,
  or simple queue-aware policies.
- Any claim that does not distinguish training seeds from evaluation seeds.
- Any broad generalization claim beyond the tested same-scale perturbations,
  nearby scale-transfer settings, and the current 6UAV/80GU candidate setting.

## 2. Current submission posture

No final venue has been selected yet. The manuscript should therefore remain
venue-flexible while following the common expectations of IEEE Transactions
venues such as TCCN, TNSM, and TVT.

Current working posture:

- Use the generic IEEEtran transactions template already placed under
  `docs/paper/manuscript/`.
- Avoid journal-specific commands, cover-letter assumptions, or page-limit
  optimization until the target venue is selected.
- Build the paper around claims that can be traced to run directories,
  configs, checkpoints, seed bases, and commit hashes.
- Prefer communication-network framing over a generic RL framing: the paper is
  about dynamic SAGIN/UAV-satellite task scheduling under queue, topology,
  visibility, bandwidth, and safety constraints.
- Treat strong baselines, ablations, generalization, and runtime/resource cost
  as required paper evidence rather than optional extras.

## 3. Thesis material to migrate

| Thesis part | Paper destination | Reuse level | Required action |
|---|---|---|---|
| Chinese/English abstract motivation | Abstract and Introduction | Medium | Rewrite around SAGIN/UAV-satellite dynamic scheduling, not "using RL" alone. |
| Chapter 1 background and related work taxonomy | Introduction and Related Work | Medium | Keep taxonomy, update 2024-2026 related work and target-venue framing. |
| Chapter 2 network scenario | Section 3.1 | High | Translate and condense. Align with current notation. |
| Chapter 2 communication, computation, and queue model | Sections 3.2 and 3.5 | High | Reuse formulas after checking consistency with current implementation. |
| Chapter 2 mobility, visibility, and safety constraints | Section 3.3 and 3.4 | High | Keep constraints, clarify which are hard execution constraints. |
| Chapter 2 state, action, objective | Sections 3.4 and 3.5 | High | Reframe as problem formulation and metrics. |
| Chapter 3 CTDE framework and single-slot flow | Section 4.1 | High | Convert to a compact algorithm overview. |
| Chapter 3 structured observation and masks | Section 4.2 | High | Emphasize topology-aware representation and variable-scale compatibility. |
| Chapter 3 staged actor-critic | Sections 4.2 and 4.3 | High | Update return-target wording to match current MC/bootstrap evidence. |
| Chapter 3 safety correction | Section 4.4 | High | Keep as safety-aware execution, not as a post-hoc trick. |
| Chapter 4 experiment setup tables | Section 5.1 | Medium-high | Verify config parity before copying numbers. |
| Chapter 4 baseline table | Section 5.2 | Medium | Replace/extend with current strong baselines. |
| Chapter 4 convergence and performance plots | Sections 5 and 6 | Low-medium | Use only as legacy reference unless regenerated from current runs. |
| Chapter 5 future work | Discussion | Medium | Reuse limitations around business modeling, deployment, and robustness. |

## 4. Figure reuse plan

The thesis gives the right figure concepts, but paper figures should be redrawn
or regenerated in publication style.

| Figure concept | Paper use | Action |
|---|---|---|
| SAGIN/UAV-satellite scenario | System model figure | Redraw in English with GU/UAV/LEO/task-flow labels. |
| Task flow and queue evolution | System model figure or subfigure | Keep formulas consistent with Section 3. |
| Access/backhaul visibility geometry | System model or appendix | Use if visibility constraints need visual support. |
| Structured MARL framework | Method overview | Redraw as one compact CTDE scheduler diagram. |
| Single-slot scheduling flow | Method overview or algorithm box | Convert to a staged decision timeline. |
| Structured observation and masks | Method details | Useful for explaining variable entities and invalid-action masks. |
| Staged actor-critic architecture | Method details | Keep, but update labels for current return-target protocol. |
| Safety-aware action correction | Method details | Use to show hard feasibility enforcement. |
| Old result plots | Mostly legacy | Regenerate from current phase2/phase3 CSVs before inclusion. |

## 5. Current evidence that can support the paper

The current evidence is stronger than the thesis in the validation dimension.
The paper should be built around this newer evidence, not the old thesis result
tables.

### 5.1 Main/source setting

The current source setting remains `3UAV/20GU/T=250`, with selected learned
checkpoints chosen by source-scenario validation before held-out or zero-shot
evaluation.

Paper value:

- This is the cleanest source for the main method story.
- Phase 2 provides MC/bootstrap best-vs-final evidence and training-stability
  context.

Current caution:

- For return-target claims, report best and final where relevant.
- Do not write "bootstrap trains stably" without qualifying late-stage
  degradation in some seeds.

### 5.2 Same-scale perturbation evidence

Phase 3 records show same-scale zero-shot perturbations where learned policies
beat the strongest non-learning baselines across six scenario variants.

Paper value:

- Good candidate for the main generalization table.
- Supports the claim that the policy does not merely memorize one source
  topology/load setting.

Current caution:

- These are evaluation-seed robustness checks, not multi-training-seed
  robustness checks.

### 5.3 Nearby scale-transfer evidence

Phase 3 records show strict zero-shot loading and positive margins in nearby
scale-transfer settings: GU=10/30, UAV=2/4, and visible satellites=4/8.

Paper value:

- Strong candidate for Section 6.
- The GU=30 and UAV=2 boundary settings remain positive after eight evaluation
  seed bases, which is a useful robustness argument.

Current caution:

- This supports nearby scale transfer, not arbitrary scale generalization.
- The smallest margins should be discussed honestly rather than hidden.

### 5.4 6UAV/80GU candidate setting

Phase 3 has a larger `6UAV/80GU/T=250` candidate setting.

Current evidence:

- Baseline floor is about reward 33 under the formal baseline-only protocol.
- Bootstrap-GAE single training seed reaches about reward 35 under validation,
  held-out, and baseline-seed-aligned evaluation.
- MC under the current EV-gated protocol is weak, but the better explanation is
  effective actor-update sparsity, not "no training happened."

Paper value:

- Good candidate for a larger-scale extension result or discussion subsection.
- It shows the setting is trainable with the current infrastructure.

Current caution:

- It should not become the central claim until at least 1-2 more bootstrap
  training seeds are completed.
- MC needs a gate ablation before we make a strong return-target conclusion.

## 6. What is still needed for submission quality

| Need | Why it matters | Current status | Priority |
|---|---|---|---|
| Updated related work | TCCN/TNSM/TVT reviewers will expect recent SAGIN, UAV-MEC, LEO scheduling, MARL scheduling, and Lyapunov/DPP work. | Not yet paper-ready. | High |
| Main comparison table | Core performance claim against strongest baselines. | Evidence exists, needs table source and final metric selection. | High |
| Ablation table | Justifies staged actor, critic, return target, safety, and K. | Partly exists across phase2/history, needs paper-facing consolidation. | High |
| Generalization table | Supports same-scale and nearby scale-transfer claims. | Strong phase3 evidence exists. | High |
| Runtime/resource table | Needed for practicality and reproducibility. | Some benchmark/environment records exist, paper table not prepared. | Medium |
| 6UAV/80GU extra training seeds | Prevents single-seed overclaim on the larger setting. | Missing. | High if 6UAV/80GU is a major result |
| MC gate ablation on 6UAV/80GU | Separates return-target weakness from EV-gate update skipping. | Missing. | Medium-high |
| Figure redraws | Thesis figures are conceptually useful but not paper-ready. | Not started. | Medium |
| Artifact registry | Every table row needs run dir, config, checkpoint, seed base, and commit hash. | Partly documented. | High |
| Overleaf/local sync discipline | Avoids manuscript drift between local Git and collaboration copy. | Workflow exists. | Ongoing |

## 7. Recommended immediate order

### A. Do now without new experiments

1. Fill Section 3 from thesis Chapter 2.
   - Reuse the system model, queues, task arrivals, visibility, mobility,
     safety, state/action, and objective.
   - Keep it concise; Section 3 should define the problem, not explain the
     algorithm internals.

2. Fill Section 4 from thesis Chapter 3.
   - Reuse CTDE, structured observations, masks, staged actor, critic, and
     safety-aware execution.
   - Insert paper-safe wording for return targets: current evidence supports
     both MC and bootstrap analysis, but final claims need best/final reporting.

3. Build a paper-facing table-source registry.
   - Start from phase3 summary CSVs.
   - Create one clean source table for each planned manuscript table.
   - Record run directory, config, checkpoint, evaluation seed base, and commit
     hash whenever available.

4. Redraw the first two figures.
   - System scenario.
   - Method overview / staged scheduler.

5. Draft the Introduction around three claims.
   - Dynamic SAGIN/UAV-satellite task scheduling with queue pressure,
     visibility, and safety constraints.
   - Topology-aware structured MARL scheduler with staged decisions.
   - Evidence across strong baselines, perturbations, and nearby scale-transfer
     settings.

### B. Then run or regenerate small paper artifacts

1. Generate current main comparison table from the latest summary CSVs.
2. Generate generalization table for same-scale and scale-transfer settings.
3. Generate a compact phase2 best/final return-target table.
4. Generate runtime/resource table from available benchmark records.
5. Check which thesis experiment parameters differ from current configs.

### C. Run only targeted new experiments

1. Add 1-2 bootstrap-GAE training seeds for `6UAV/80GU/T=250`.
2. Run a short MC EV-gate ablation on `6UAV/80GU/T=250`.
3. Only after those, decide whether `6UAV/80GU` belongs in the main results,
   the generalization section, or the discussion/appendix.

## 8. Suggested manuscript role of each evidence block

| Manuscript section | Main content | Evidence status |
|---|---|---|
| Abstract | Problem, method, strongest verified claims | Draft after tables are fixed. |
| Introduction | Motivation, gap, contributions | Thesis reusable, needs stronger journal framing. |
| Related Work | SAGIN/UAV-MEC/LEO scheduling, MARL, queue control | Needs literature update. |
| System Model | Scenario, queues, visibility, constraints, objective | Thesis highly reusable. |
| Method | Structured staged scheduler, critic, safety execution | Thesis reusable with current protocol updates. |
| Experiments | Setup, baselines, main comparison, ablations | Needs table-source consolidation. |
| Generalization | Same-scale perturbation and nearby scale-transfer | Phase3 is strong enough for a candidate section. |
| Discussion | 6UAV/80GU, limitations, deployment, stability | Useful but should stay carefully qualified. |
| Conclusion | Summary of verified contributions | Draft last. |

## 9. Decision for the next writing sprint

The safest first writing sprint is:

1. Migrate Section 3 and Section 4 from the thesis into the IEEEtran manuscript.
2. In parallel, prepare table sources for Section 5 and Section 6 from current
   phase2/phase3 evidence.
3. Delay final Abstract, Introduction claims, and 6UAV/80GU positioning until
   the table-source registry is clean.

This order gives us quick manuscript progress while avoiding the classic trap:
writing a polished story before the paper-level evidence table is stable.
