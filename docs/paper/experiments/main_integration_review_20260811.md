# Main Integration Review

Date: 2026-08-11

This document is the working integration plan for reducing the current
experiment-heavy branch into reviewable units. It records the user's current
decisions and the proposed PR split before any merge into `main`.

## Current Decision Record

1. Split PRs are preferred. Do not merge
   `erik/phase4-learning-ablation-baseline` into `main` as one large PR.
2. `6UAV/80GU` should be treated as a high-load stress setting for now. Do not
   promote it as a clean large-scale generalization claim in the manuscript.
3. `docs/paper/` and similar manuscript/evidence assets do not all need to
   enter the long-lived `main` branch. They can remain on a paper branch because
   writing is primarily happening in Overleaf and local Git is mainly used for
   generated figures, table sources, and evidence hygiene.
4. Raw `runs/` directories should not be synchronized through GitHub. GitHub is
   the sync layer for code, configs, tracked docs, table sources, figure
   sources, and scripts.
5. Friday raw runs need a local archive index. Only selected CSV/JSON summaries
   and generated figure/table sources should be promoted into Git.

## Current Branch Shape

Local/friday/GitHub tracked branch:

```text
erik/phase4-learning-ablation-baseline
```

Latest synchronized commit at the time of this note:

```text
7bd2ec1 Consolidate phase3 generalization notes
```

Relative to `main`, the branch is integration-heavy:

| Metric | Approximate value |
|---|---:|
| Commits | 63 |
| Changed files | 909 |
| Major categories | configs, docs/paper, scripts, docs archive, core env/eval code, tests |

This is too large for a single high-quality PR.

## PR Split Proposal

### PR 1: Core Reorganization And Low-Risk Script Layout

Purpose: keep the project tree readable without changing training semantics.

Candidate content:

- script directory moves and import-path fixes;
- config/archive organization that is not paper-specific;
- docs explaining project/script/config layout;
- tests that only validate importability or unchanged entrypoint behavior.

Review focus:

- no accidental deletion of still-used scripts;
- no broken imports;
- no paper artifacts.

Minimum checks:

```bash
git diff --check
python -m py_compile <changed-python-files>
```

### PR 2: Core Evaluator/Baseline Runtime Fixes

Purpose: carry the runtime changes required by formal evaluation and strong
baselines without bundling manuscript assets.

Candidate content:

- `sagin_marl/env/native_cuda/*` baseline source support;
- structured evaluator changes;
- fixed-baseline/native source modes;
- focused tests such as `tests/test_baselines.py`.

Review focus:

- behavior parity for existing baselines;
- names and source modes are stable;
- no paper-only scripts or generated assets.

Minimum checks:

```bash
git diff --check
python -m py_compile <changed-python-files>
pytest tests/test_baselines.py
```

### PR 3: Phase3 Generalization Configs And Evidence Summaries

Purpose: keep Phase3 experiment definitions and paper-facing summaries
reviewable without mixing in Phase4 learning-ablation code.

Candidate content:

- `configs/experiments/phase3_generalization/*`;
- `scripts/experiments/phase3/*` if needed for reproducibility;
- `docs/paper/experiments/phase3_generalization_runbook.md`;
- Phase3 table sources under `docs/paper/table_sources/phase3_*` if we decide
  paper-facing evidence should be part of this branch.

Review focus:

- `6UAV/80GU` wording remains high-load stress setting;
- zero-shot checkpoint-selection rule is clear;
- table rows are traceable to run dirs and seed protocols.

Main-branch option:

- If `main` should stay code-only, keep this PR on a paper/evidence branch
  instead of merging into `main`.

### PR 4: Phase4 Learning-Ablation Code And Configs

Purpose: isolate the algorithm/evaluator changes used by RelCritic,
GlobalCritic, and MAPPO-like ablations.

Candidate content:

- Phase4 configs;
- checkpoint-eval early stopping and resume fixes;
- selected launch/evaluation scripts;
- focused tests for structured critic/action behavior.

Review focus:

- no generated paper figures in code PR;
- selected/final checkpoint semantics are stable;
- resume and early-stop state restoration are tested.

Minimum checks:

```bash
git diff --check
python -m py_compile <changed-python-files>
pytest tests/test_structured_action_modules.py tests/test_structured_critic_system_readout.py
```

### PR 5: Paper Branch / Manuscript Evidence Assets

Purpose: keep manuscript, Overleaf packaging, table sources, figure sources, and
paper runbooks together without forcing them into `main`.

Candidate content:

- `docs/paper/manuscript/*`;
- `docs/paper/package_overleaf.sh`;
- `docs/paper/sync_workflow.md`;
- `docs/paper/evidence_index.md`;
- `docs/paper/table_sources/*`;
- `docs/paper/figure_sources/*`;
- figure-generation scripts under `docs/paper/experiments/*`.

Review focus:

- paper numbers trace to table sources;
- raw `runs/` and checkpoints are not included;
- Overleaf source stays lightweight enough to upload/download by zip.

Preferred branch policy:

- Keep this as a paper branch unless we later decide `main` should contain the
  full manuscript workspace.

## Friday Run Archive Policy

Raw runs are evidence backing, not Git-managed source. They should be grouped by
purpose and indexed in-place on friday.

Current relevant roots:

| Root | Role |
|---|---|
| `/home/sgy/workspace/sagin_marl/runs/phase3` | Phase3 raw run archive |
| `/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs/phase4_learning_ablation` | Phase4 raw run archive |

Suggested run states:

| State | Meaning | Action |
|---|---|---|
| `keep` | directly supports a tracked table/figure source or selected checkpoint | preserve |
| `archive` | useful provenance but not active paper evidence | keep indexed, optionally compress later |
| `scratch` | smoke/debug/probe only | keep short term or delete after user approval |
| `failed` | failed launch needed only for debugging provenance | quarantine or delete after audit |
| `suspicious` | filesystem errors or unreadable dirs | do not delete casually; inspect disk health first |

Known caution:

- Two phase4 subdirectories returned `Input/output error` during `du`.
  They should not be used as paper evidence and should be inspected separately.

## Three-Way Sync Rule

Use GitHub as the canonical sync layer for tracked content:

```text
GitHub origin <-> local Mac clone <-> friday clone
```

Recommended workflow:

1. Make tracked changes on one machine.
2. Commit and push to `origin`.
3. On the other machine, use `git pull --ff-only`.
4. Keep raw run directories local to the run host.
5. Export selected evidence into tracked CSV/JSON/figure sources only after a
   run is selected for paper use.

## User Review Checklist

The user should review these points before we begin PR extraction:

1. Whether `docs/paper/` remains paper-branch-only.
2. Whether Phase3 table sources should live with paper assets or with Phase3
   config PR.
3. Whether `6UAV/80GU` is only a high-load stress note in current writing.
4. Whether friday raw run roots should be indexed only, compressed, or partly
   cleaned.
5. Which PR should be prepared first.

## Recommended Next Step

Start with a paper/evidence branch cleanup PR or draft PR, because it is
lowest-risk and does not affect runtime behavior. In parallel, prepare a
separate file-list audit for PR 2 so core evaluator/baseline changes can be
reviewed without manuscript noise.
