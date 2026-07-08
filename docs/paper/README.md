# Paper workspace

This directory keeps the manuscript source, paper-facing notes, and reproducible
assets for the SAGIN-MARL journal submission.

## Layout

```text
docs/paper/
  manuscript/        IEEEtran LaTeX source that can be uploaded to Overleaf as a zip.
  figure_sources/    Scripts or data used to generate publication figures.
  table_sources/     Scripts or data used to generate publication tables.
  experiment_protocol.md
                     Paper-facing evaluation and reporting protocol.
  evidence_index.md  Claim-to-evidence map for manuscript statements.
  thesis_reuse_gap_plan_20260708.md
                     Audit of thesis reuse, evidence gaps, and first writing steps.
  package_overleaf.sh
                     Builds the source zip for Overleaf upload.
  sync_workflow.md   Free-plan Overleaf synchronization procedure.
```

## Working rule

Use the local repository as the canonical source for manuscript files that
Codex edits. Use Overleaf as the collaborative web editor and compiler. Because
the current Overleaf plan does not expose Git integration, sync with Overleaf by
uploading and downloading source zip files.

Do not put raw training outputs, checkpoints, or large run directories here.
Reference them from `runs/`, `.local_guidance/`, or the remote result index.

To prepare an upload package for Overleaf:

```bash
./docs/paper/package_overleaf.sh
```

Then upload `docs/paper/sagin_marl_ieee_transactions_manuscript.zip` to Overleaf.

The current manuscript skeleton follows the Phase 4 structure: Introduction,
Related Work, System Model, Method, Main Experiments, Generalization,
Discussion, and Conclusion.
