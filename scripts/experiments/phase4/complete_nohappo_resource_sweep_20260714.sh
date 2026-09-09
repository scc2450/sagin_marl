#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/sgy/workspace/sagin_marl_phase4_learning_ablation}
PYTHON=${PYTHON:-/home/sgy/workspace/sagin_marl/.venv/bin/python}
RUNNER=${RUNNER:-$REPO/scripts/experiments/phase4/run_formal_parameter_sweeps_20260714.py}
AGGREGATOR=${AGGREGATOR:-$REPO/scripts/analysis/phase4/aggregate_formal_parameter_sweeps_20260714.py}
ROOT=${ROOT:-runs/phase4_learning_ablation/3uav20gu_t250/formal_parameter_sweeps_20260714_full}
METHODS_FAST=${METHODS_FAST:-relcritic,globalcritic,cluster_center_queue_aware,maxweight_lyapunov,queue_aware_bw,static_uniform}

cd "$REPO"
mkdir -p "$ROOT/_queues" docs/paper/table_sources
LOG="$ROOT/_queues/complete_nohappo_resource_20260714.log"

count_nohappo_unique_ok() {
  python3 - "$ROOT" <<'PY'
import csv
import glob
import sys

root = sys.argv[1]
labels = set()
bad = []
for path in glob.glob(root + "/status*.csv"):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method_id") == "mappo_like":
                continue
            if row.get("event") not in ("end", "skip_existing"):
                continue
            if row.get("rc") == "0":
                labels.add(row.get("label"))
            else:
                bad.append(row.get("label"))
print(len(labels), len(bad))
PY
}

{
  echo "start $(date --iso-8601=seconds)"
  for attempt in $(seq 1 20); do
    read -r ok bad < <(count_nohappo_unique_ok)
    echo "$(date +%H:%M:%S) before_attempt=$attempt nonhap_unique_ok=$ok bad=$bad"
    if [[ "$ok" -ge 420 ]]; then
      break
    fi
    set +e
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" \
      --sweep resource \
      --gpu 0 \
      --methods "$METHODS_FAST" \
      --out-root "$ROOT" \
      --status-csv "$ROOT/status_gpu0_resource_fastpath_nohappo_serial_retry_${attempt}.csv"
    rc=$?
    set -e
    read -r ok_after bad_after < <(count_nohappo_unique_ok)
    echo "$(date +%H:%M:%S) after_attempt=$attempt rc=$rc nonhap_unique_ok=$ok_after bad=$bad_after"
    if [[ "$ok_after" -ge 420 ]]; then
      break
    fi
    sleep 5
  done

  status_args=()
  for f in "$ROOT"/status*.csv; do
    status_args+=(--status-csv "$f")
  done
  "$PYTHON" "$AGGREGATOR" "${status_args[@]}" \
    --exclude-methods mappo_like \
    --raw-out docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_raw_20260714.csv \
    --aggregate-out docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv
  read -r final_ok final_bad < <(count_nohappo_unique_ok)
  echo "done $(date --iso-8601=seconds) nonhap_unique_ok=$final_ok bad=$final_bad"
} >>"$LOG" 2>&1
