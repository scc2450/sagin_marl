#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/sgy/workspace/sagin_marl_phase4_learning_ablation}
PYTHON=${PYTHON:-/home/sgy/workspace/sagin_marl/.venv/bin/python}
RUNNER=${RUNNER:-$REPO/scripts/experiments/phase4/run_formal_parameter_sweeps_20260714.py}
AGGREGATOR=${AGGREGATOR:-$REPO/scripts/analysis/phase4/aggregate_formal_parameter_sweeps_20260714.py}
ROOT=${ROOT:-runs/phase4_learning_ablation/3uav20gu_t250/formal_parameter_sweeps_20260714_full}
METHODS_FAST=${METHODS_FAST:-relcritic,globalcritic,cluster_center_queue_aware,maxweight_lyapunov,queue_aware_bw,static_uniform}
EXCLUDE_METHODS=${EXCLUDE_METHODS:-mappo_like}
RAW_OUT=${RAW_OUT:-docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_raw_20260714.csv}
AGGREGATE_OUT=${AGGREGATE_OUT:-docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv}

cd "$REPO"
mkdir -p "$ROOT/_queues" docs/paper/table_sources
WATCH_LOG="$ROOT/_queues/watch_and_postprocess_20260714.log"

count_completed() {
  grep -hE '^(end|skip_existing),' "$ROOT"/status*.csv 2>/dev/null | cut -d, -f28 | grep -cx '0' || true
}

count_bad() {
  grep -hE '^(end|skip_existing),' "$ROOT"/status*.csv 2>/dev/null | cut -d, -f28 | grep -vc '^0$' || true
}

count_live() {
  pgrep -f 'run_formal_parameter_sweeps_20260714.py' | wc -l
}

{
  echo "watch_start $(date --iso-8601=seconds)"
  while true; do
    live=$(count_live)
    completed=$(count_completed)
    bad=$(count_bad)
    echo "$(date +%H:%M:%S) live_runners=$live completed_rc0=$completed bad=$bad"
    if [[ "$live" -eq 0 ]]; then
      break
    fi
    sleep 120
  done

  echo "retry_fastpath_start $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RUNNER" \
    --sweep load \
    --gpu 0 \
    --methods "$METHODS_FAST" \
    --out-root "$ROOT" \
    --status-csv "$ROOT/status_gpu0_load_fastpath_retry.csv"

  CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$RUNNER" \
    --sweep resource \
    --gpu 1 \
    --methods "$METHODS_FAST" \
    --out-root "$ROOT" \
    --status-csv "$ROOT/status_gpu1_resource_fastpath_retry.csv"

  status_args=()
  for f in "$ROOT"/status*.csv; do
    status_args+=(--status-csv "$f")
  done
  "$PYTHON" "$AGGREGATOR" "${status_args[@]}" \
    --exclude-methods "$EXCLUDE_METHODS" \
    --raw-out "$RAW_OUT" \
    --aggregate-out "$AGGREGATE_OUT"

  echo "watch_done $(date --iso-8601=seconds)"
  echo "completed_rc0=$(count_completed) bad=$(count_bad)"
} >>"$WATCH_LOG" 2>&1
