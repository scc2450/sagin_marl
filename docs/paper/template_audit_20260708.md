# IEEEtran template audit

Date: 2026-07-08

Purpose: verify that `docs/paper/manuscript/` follows the generic IEEE
Transactions LaTeX template downloaded at:

```text
/Users/erik/Downloads/IEEE-Transactions-LaTeX2e-templates-and-instructions
```

## Checks

| Item | Result | Evidence |
|---|---|---|
| Document class | Pass | `\documentclass[lettersize,journal]{IEEEtran}` matches `bare_jrnl_new_sample4.tex`. |
| Class file | Pass | `docs/paper/manuscript/IEEEtran.cls` is byte-identical to the downloaded `IEEEtran.cls`; SHA-256 is `b0eb3567b81aec7fe98144a3ad283eeac2d31035bb19e0d9dcba7da190f18d9d`. |
| Front matter order | Pass | `\title`, `\author`, `\markboth`, `\maketitle`, `abstract`, `IEEEkeywords` follows the sample journal template. |
| TMLCN-specific commands | Pass | No active `ieeetmlcn`, `receiveddate`, `doiinfo`, `affil`, `corresp`, `authornote`, `OJlogo`, `new_logo`, or `tmlcncolor` commands remain in active manuscript files. |
| Section splitting | Pass | `\input{sections/...}` is standard LaTeX and does not change the IEEEtran class or front matter. |
| Bibliography | Pass | Uses `\bibliographystyle{IEEEtran}` and `\bibliography{refs}`, which is allowed by the template comments for BibTeX workflows. |
| Copyright line | Pass | `\IEEEpubid` is intentionally omitted for submission; the IEEE how-to says it is not necessary at the Transactions/Journals submission stage. |

## Deliberate differences from the sample

- The manuscript uses `amssymb` in addition to `amsmath,amsfonts`; this is a standard AMS package and is expected to be useful for equations.
- The manuscript keeps reusable notation in `macros.tex`; this is a normal LaTeX organization choice.
- The manuscript is split into section files under `sections/`; this is a normal LaTeX organization choice and should compile as long as the file names match.
- The sample's `\IEEEpubid` and `\IEEEpubidadjcol` are omitted because they are production/copyright placeholders, not required for initial submission.

## Local compile status

Local compilation was not run because this machine currently does not expose
`latexmk`, `pdflatex`, `tectonic`, or `chktex`. Overleaf should be used for the
first compile check.

If Overleaf reports an error, the first things to check are:

1. Main file is set to `main.tex`.
2. `IEEEtran.cls`, `macros.tex`, and `refs.bib` are in the project root.
3. All files referenced by `\input{sections/...}` exist under `sections/`.
