#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANUSCRIPT_DIR="${SCRIPT_DIR}/manuscript"
OUT_ZIP="${SCRIPT_DIR}/sagin_marl_tmlcn_manuscript.zip"

cd "${MANUSCRIPT_DIR}"
rm -f "${OUT_ZIP}"
zip -r "${OUT_ZIP}" . \
  -x ".gitignore" \
  -x "template_source/*" \
  -x "template_source/**" \
  -x "template_source/" \
  -x "*.aux" \
  -x "*.bbl" \
  -x "*.bcf" \
  -x "*.blg" \
  -x "*.fdb_latexmk" \
  -x "*.fls" \
  -x "*.log" \
  -x "*.out" \
  -x "*.run.xml" \
  -x "*.synctex.gz" \
  -x "*.toc" \
  -x "*.xdv"

echo "Wrote ${OUT_ZIP}"
