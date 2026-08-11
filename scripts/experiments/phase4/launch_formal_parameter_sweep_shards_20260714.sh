#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/sgy/workspace/sagin_marl_phase4_learning_ablation}
PYTHON=${PYTHON:-/home/sgy/workspace/sagin_marl/.venv/bin/python}
RUNNER=${RUNNER:-$REPO/scripts/experiments/phase4/run_formal_parameter_sweeps_20260714.py}
ROOT=${ROOT:-runs/phase4_learning_ablation/3uav20gu_t250/formal_parameter_sweeps_20260714_full}
METHODS_FAST=${METHODS_FAST:-relcritic,globalcritic,cluster_center_queue_aware,maxweight_lyapunov,queue_aware_bw,static_uniform}
INCLUDE_HAPPO=${INCLUDE_HAPPO:-0}

cd "$REPO"
mkdir -p "$ROOT/_queues"

is_running() {
  local marker=$1
  pgrep -f "$marker" >/dev/null 2>&1
}

launch_if_needed() {
  local name=$1
  local gpu=$2
  shift 2
  local log="$ROOT/_queues/${name}.log"
  local pid_file="$ROOT/_queues/${name}.pid"
  if is_running "$name" || { [[ -f "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; }; then
    printf 'already-running %s %s\n' "$name" "$(cat "$pid_file" 2>/dev/null || true)"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" "$RUNNER" "$@" >"$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf 'launched %s %s\n' "$name" "$pid"
}

launch_if_needed sweep_load_gpu0_fastpath_20260714 0 \
  --sweep load \
  --gpu 0 \
  --methods "$METHODS_FAST" \
  --out-root "$ROOT" \
  --status-csv "$ROOT/status_gpu0_load_fastpath.csv"

launch_if_needed sweep_resource_gpu1_fastpath_20260714 1 \
  --sweep resource \
  --gpu 1 \
  --methods "$METHODS_FAST" \
  --out-root "$ROOT" \
  --status-csv "$ROOT/status_gpu1_resource_fastpath.csv"

if [[ "$INCLUDE_HAPPO" == "1" ]]; then
  launch_if_needed sweep_load_gpu0_happo_serial_20260714 0 \
    --sweep load \
    --gpu 0 \
    --methods mappo_like \
    --out-root "$ROOT" \
    --status-csv "$ROOT/status_gpu0_load_happo_serial.csv"

  launch_if_needed sweep_resource_gpu1_happo_serial_20260714 1 \
    --sweep resource \
    --gpu 1 \
    --methods mappo_like \
    --out-root "$ROOT" \
    --status-csv "$ROOT/status_gpu1_resource_happo_serial.csv"
else
  printf 'skipped HA-PPO sweep shards; set INCLUDE_HAPPO=1 for a serial companion run\n'
fi
