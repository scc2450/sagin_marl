# Phase4 Nominal Selected Rich Summaries

This directory stores per-run rich evaluation outputs for the nominal
source-scenario point used by the Section 5 queue and flow mechanism figures.

## Layout

- `nominal_load_x1p00/`: copied rich-summary tree for the formal parameter
  sweep's `load/x1p00` point.
- `nominal_load_x1p00/learned/`: selected-checkpoint evaluations for learned
  policies, organized by method, training seed, checkpoint label, and held-out
  evaluation seed base.
- `nominal_load_x1p00/fixed/`: fixed-baseline evaluations, organized by method
  and held-out evaluation seed base.

Each leaf evaluation directory keeps the summary JSON, episode-level CSV, and
`evaluate.log` for traceability.

## Naming Note

The original copied sweep directory was named `x1p00`, which means multiplier
1.00 in the `load/x1p00` formal parameter-sweep configuration. The local folder
is named `nominal_load_x1p00` to make the paper-facing evidence tree easier to
read while retaining the original experiment label.

## Consumers

`docs/paper/reproduction/aggregate_phase4_nominal_rich_metrics_20260715.py`
aggregates this tree into:

- `docs/paper/evidence_tables/phase4_nominal_selected_rich_raw_20260715.csv`
- `docs/paper/evidence_tables/phase4_nominal_selected_rich_aggregate_20260715.csv`

The resulting tables support the Section 5 queue-workload and task-flow
decomposition figures.
