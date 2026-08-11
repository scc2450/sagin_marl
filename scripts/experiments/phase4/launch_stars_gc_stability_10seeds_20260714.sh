#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/sgy/workspace/sagin_marl_phase4_learning_ablation}
PYTHON=${PYTHON:-/home/sgy/workspace/sagin_marl/.venv/bin/python}
ROOT=${ROOT:-runs/phase4_learning_ablation/3uav20gu_t250/stability_10seeds_20260714}

REL_CONFIG="configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_relational_critic.yaml"
GC_CONFIG="configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml"

GPU0_ITEMS=(
  "relational_critic:10331"
  "relational_critic:21893"
  "relational_critic:36467"
  "relational_critic:48761"
  "global_only_critic:10331"
  "global_only_critic:21893"
)

GPU1_ITEMS=(
  "relational_critic:59023"
  "relational_critic:64217"
  "relational_critic:87539"
  "global_only_critic:36467"
  "global_only_critic:48761"
  "global_only_critic:59023"
  "global_only_critic:64217"
  "global_only_critic:87539"
)

script_path() {
  cd "$(dirname "${BASH_SOURCE[0]}")"
  printf "%s/%s\n" "$(pwd)" "$(basename "${BASH_SOURCE[0]}")"
}

ensure_status_header() {
  local status_csv=$1
  mkdir -p "$(dirname "$status_csv")"
  if [[ ! -f "$status_csv" ]]; then
    printf "method,seed,gpu,status,run_dir,log_path,start_ts,end_ts,elapsed_sec,rc\n" > "$status_csv"
  fi
}

append_status() {
  local status_csv=$1
  shift
  printf "%s\n" "$*" >> "$status_csv"
}

already_done() {
  local status_csv=$1
  local label=$2
  local seed=$3

  [[ -f "$status_csv" ]] || return 1
  awk -F, -v label="$label" -v seed="$seed" '
    $1 == label && $2 == seed && $4 == "DONE" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$status_csv"
}

run_one() {
  local gpu=$1
  local item=$2
  local status_csv=$3

  local method=${item%%:*}
  local seed=${item##*:}
  local label config run_dir log_path start_ts start_epoch end_ts end_epoch elapsed rc

  case "$method" in
    relational_critic)
      label="STARS"
      config="$REL_CONFIG"
      ;;
    global_only_critic)
      label="STARS-GC"
      config="$GC_CONFIG"
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 2
      ;;
  esac

  if already_done "$status_csv" "$label" "$seed"; then
    echo "skip already DONE: $label seed $seed"
    return 0
  fi

  start_ts=$(date -Iseconds)
  start_epoch=$(date +%s)
  run_dir="$ROOT/$method/seed${seed}_$(date +%Y%m%d_%H%M%S)_bootstrapgae_nocompile"
  log_path="$ROOT/_logs/${method}_seed${seed}.log"
  mkdir -p "$(dirname "$log_path")" "$run_dir"

  append_status "$status_csv" "$label,$seed,$gpu,RUNNING,$run_dir,$log_path,$start_ts,,,"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" scripts/train_joint_mcgae.py \
    --config "$config" \
    --run_dir "$run_dir" \
    --device cuda \
    --num_envs 64 \
    --rollout_env_steps 250 \
    --return_target bootstrap_gae \
    --return_target_schedule fixed \
    --max_updates 700 \
    --seed "$seed" \
    --save_every 25 \
    --disable_torch_compile \
    > "$log_path" 2>&1
  rc=$?
  set -e

  end_ts=$(date -Iseconds)
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))

  if [[ "$rc" -eq 0 ]]; then
    append_status "$status_csv" "$label,$seed,$gpu,DONE,$run_dir,$log_path,$start_ts,$end_ts,$elapsed,$rc"
  else
    append_status "$status_csv" "$label,$seed,$gpu,FAILED,$run_dir,$log_path,$start_ts,$end_ts,$elapsed,$rc"
  fi

  return "$rc"
}

run_queue() {
  local gpu=$1
  local status_csv="$ROOT/status_gpu${gpu}.csv"
  local -a items

  cd "$REPO"
  mkdir -p "$ROOT/_logs" "$ROOT/_queues"
  ensure_status_header "$status_csv"

  if [[ "$gpu" == "0" ]]; then
    items=("${GPU0_ITEMS[@]}")
  elif [[ "$gpu" == "1" ]]; then
    items=("${GPU1_ITEMS[@]}")
  else
    echo "GPU must be 0 or 1, got $gpu" >&2
    exit 2
  fi

  local item
  for item in "${items[@]}"; do
    run_one "$gpu" "$item" "$status_csv" || true
  done
}

launch() {
  cd "$REPO"
  mkdir -p "$ROOT/_queues"

  local self
  self=$(script_path)

  nohup bash "$self" --queue 0 > "$ROOT/_queues/gpu0_queue.log" 2>&1 &
  local pid0=$!
  nohup bash "$self" --queue 1 > "$ROOT/_queues/gpu1_queue.log" 2>&1 &
  local pid1=$!

  {
    printf "root=%s\n" "$ROOT"
    printf "gpu0_pid=%s\n" "$pid0"
    printf "gpu1_pid=%s\n" "$pid1"
    printf "launched_at=%s\n" "$(date -Iseconds)"
  } > "$ROOT/_queues/launch_pids.txt"

  cat "$ROOT/_queues/launch_pids.txt"
}

case "${1:-launch}" in
  launch)
    launch
    ;;
  --queue)
    run_queue "${2:?missing gpu id}"
    ;;
  *)
    echo "Usage: $0 [launch|--queue 0|--queue 1]" >&2
    exit 2
    ;;
esac
