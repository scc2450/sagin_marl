# Free-plan Overleaf sync workflow

This workflow assumes Overleaf Git integration is unavailable.

## Source of truth

- Local canonical source: `docs/paper/manuscript/`
- Online collaboration surface: the Overleaf project using the generic IEEE Transactions template
- Reproducible figure and table sources: `docs/paper/figure_sources/` and `docs/paper/table_sources/`

## Local to Overleaf

Use this when Codex or local tools changed the manuscript.

1. Make sure `docs/paper/manuscript/` compiles locally or at least has no obvious missing files.
2. Create a zip from the contents of `docs/paper/manuscript/`, not from the parent directory.
3. Upload the zip or changed files to Overleaf.
4. Recompile in Overleaf.
5. If Overleaf reports missing package/file errors, fix them locally and repeat.

Suggested command:

```bash
./docs/paper/package_overleaf.sh
```

## Overleaf to local

Use this when a coauthor edited the Overleaf project directly.

1. In Overleaf, download the project source zip.
2. Unzip it to a temporary directory outside the repo.
3. Compare it against `docs/paper/manuscript/`.
4. Copy only intentional text, bibliography, figure, and table changes back into the local repo.
5. Commit the local manuscript changes before doing another local-to-Overleaf upload.

Suggested comparison shape:

```bash
diff -ru docs/paper/manuscript /tmp/overleaf_source
```

## Conflict rule

Avoid simultaneous large edits in both places. For structural edits, section
splits, figure/table regeneration, bibliography cleanup, or terminology-wide
changes, edit locally first and upload to Overleaf after review.

For short prose edits and comments, Overleaf is fine. Pull them back into local
source before the next Codex-assisted rewrite.
