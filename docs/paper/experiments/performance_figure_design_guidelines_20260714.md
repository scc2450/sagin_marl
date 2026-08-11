# Performance Figure Design Guidelines (2026-07-14)

This note freezes the first paper-facing figure style decisions for the
Performance Evaluation section.

## Palette

Use a colorblind-aware qualitative palette, inspired by the Okabe-Ito /
ColorBrewer-style categorical palettes. Keep the mapping stable across all
figures and tables.

| Paper label | Internal method | Color | Line/marker role |
|---|---|---:|---|
| STARS | RelCritic | `#0072B2` | solid line, circle |
| STARS-GC | GlobalCritic | `#D55E00` | dashed line, square |
| HA-PPO | MAPPO-like | `#CC79A7` | dash-dot line, triangle |
| QCCS | cluster_center_queue_aware | `#009E73` | solid line, diamond |
| Lyapunov | maxweight_lyapunov | `#E69F00` | dashed line, pentagon |
| QBS | queue_aware_bw / queue_aware | `#56B4E9` | dotted line, down-triangle |
| Uniform | static_uniform | `#666666` | dotted or thin line, x |

Do not rely on color alone. Use line style and marker shape in all multi-method
curves because IEEE figures may be printed in grayscale.

## Figure Types

### Section 5 manuscript layout

Use single-panel figure assets as the default unit. Combine panels in LaTeX with
`subfloat` only when a pair directly supports the same claim. Avoid compressing
four metrics into one slot for the main text; that format is useful for internal
screening but visually weak in IEEE two-column layout.

Keep manuscript-facing figures as PDF only. The current local plotting
environment is:

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/rl/bin/python
```

Use a single-column-oriented aspect ratio close to the figures in comparable
IEEE papers, rather than a very short strip plot. Current Section 5 panels use a
3.45 inch wide canvas with a height around 2.7 inches and explicit subplot
margins instead of tight bounding-box cropping. This leaves stable whitespace
around axis labels and legends.

Preserve the existing method-level visual identities across all plots: color,
line style, and marker are fixed by method.

Recommended main-text panels:

- checkpoint validation: raw checkpoint reward and best-so-far reward;
- offered-load sensitivity: reward, processed ratio, drop ratio, and queue
  workload if space permits;
- resource-capacity sensitivity: reward, processed ratio, system delay, and
  queue workload if space permits.

The queue-workload panels are preferred over standalone seed-spread ablation
plots in the current draft because they explain why the accumulated reward can
remain substantially higher even when episode-level processed/drop ratios are
close.

Keep redundant KPI panels in appendix/source figures, because the main table
already reports the complete metric set.

### Main comparison

Use a compact multi-metric table as the primary result carrier. A multi-panel
bar chart is acceptable as an overview, but it should not replace the table.
Report reward, processed ratio, drop ratio, backlog or delay, and collision if
needed.

### Training dynamics

Use line plots over training update or environment interactions. Mark selected
best checkpoints and final checkpoints explicitly. Do not claim convergence only
from early stopping.

### Ablation / seed stability

Use seed-level dot plots or line plots with error bars. For methods with high
variance, show individual training seeds instead of only mean bars.

### Sensitivity / sweep

Paper-facing sweeps should vary one semantically meaningful parameter on the
x-axis:

- task arrival intensity or offered load,
- number of ground users,
- number of UAVs,
- visible satellite count,
- access/backhaul/resource multiplier,
- delay deadline or queue/drop threshold if available.

Prefer line plots with markers and error bands. Avoid mixing unrelated scenario
names on the same x-axis in the main text. Categorical stress-test scenarios can
remain as appendix or internal diagnostic figures.

## Current Data Boundary

The formal phase4 source-scenario evidence supports:

- main held-out source comparison,
- training dynamics and checkpoint selection,
- STARS versus STARS-GC seed-level stability,
- HA-PPO fairness-boundary discussion.

The existing phase3 same-scale and scale-transfer categorical figures should be
treated as diagnostic drafts. They are not yet a unified formal STARS sweep.

The existing 6UAV/40GU resource multiplier data form a good continuous sweep
shape, but it is baseline-only in the registered source. It should not be used
as a proposed-method sensitivity figure unless current selected learned
checkpoints are re-evaluated under the same multiplier grid.

## Recommended Next Formal Sweeps

Minimal paper-facing sweep set:

1. Offered-load sweep: vary task arrival intensity with 5-7 points.
2. Resource-scarcity sweep: vary access/backhaul/resource multiplier with 5-7
   points.
3. Scale sweep: vary number of GUs or UAVs with at least 4 points.

For each point, evaluate STARS, STARS-GC, QCCS, Lyapunov, QBS, and Uniform
under the same held-out episode seeds. Keep HA-PPO as a source-scenario learned
baseline by default; if it is included in a sweep, run it as a separate serial
companion job rather than as part of the full concurrent sweep.

Minimum presentation requirements:

- x-axis must be numeric or naturally ordered,
- at least three independent evaluation seed bases per point,
- show mean with standard error or standard deviation,
- plot at least processed ratio, drop ratio, and system delay,
- keep reward as a companion panel rather than the only metric.
