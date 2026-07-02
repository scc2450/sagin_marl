from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

METHODS = ("mc", "bootstrap_gae")

VALIDATION_FIELDS = [
    "method",
    "update",
    "checkpoint",
    "return_target",
    "train_sec",
    "eval_sec",
    "stop_reason_candidate",
    "best_reward_so_far",
    "best_update_so_far",
    "significant_best_reward_so_far",
    "significant_best_update_so_far",
    "patience_count",
    "degradation_count",
    "reward_sum",
    "processed_ratio_eval",
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "sat_overlap_eval",
    "collision_episode_fraction",
    "terminated_early",
    "episode_length",
    "outflow_arrival_ratio",
    "sat_processed_arrival_ratio",
    "drop_ratio",
    "episodes",
]

FINAL_FIELDS = [
    "method",
    "state",
    "stop_reason",
    "best_update_validation",
    "best_reward_validation",
    "final_update_validation",
    "final_reward_validation",
    "drop_best_to_final",
    "heldout_best_reward",
    "heldout_final_reward",
    "best_checkpoint",
    "final_checkpoint",
]


def _csv_cell(value: Any) -> str:
    return "" if value is None else str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name, "")) for name in fieldnames})


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    fieldnames = ["key", "value"]
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({"key": row["key"], "value": row["value"]})


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise TypeError(f"summary payload in {path} is not a dict")
    return summary


def _checkpoint_path(method_dir: Path, update: int) -> Path:
    return method_dir / f"checkpoint_update{int(update):04d}.pt"


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n$ " + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")
    return elapsed


def _train_segment(
    *,
    args: argparse.Namespace,
    repo: Path,
    env: dict[str, str],
    method: str,
    method_dir: Path,
    target_update: int,
    previous_update: int,
) -> float:
    ckpt = _checkpoint_path(method_dir, target_update)
    if ckpt.exists():
        print(f"[runner] train skip existing {ckpt}", flush=True)
        return 0.0
    cmd = [
        sys.executable,
        "scripts/train_joint_mcgae.py",
        "--config",
        args.config,
        "--run_dir",
        str(method_dir),
        "--updates",
        str(target_update),
        "--device",
        args.device,
        "--num_envs",
        str(args.num_envs),
        "--rollout_env_steps",
        str(args.rollout_env_steps),
        "--return_target",
        method,
        "--return_target_schedule",
        "fixed",
        "--critic_update_microbatch_size",
        str(args.critic_update_microbatch_size),
        "--save_every",
        str(args.interval_updates),
        "--seed",
        str(args.seed),
        "--torch_threads",
        str(args.torch_threads),
        "--structured_env_backend",
        "native",
        "--structured_env_tensor_backend",
        "cuda",
    ]
    if previous_update > 0:
        resume = _checkpoint_path(method_dir, previous_update)
        if not resume.exists():
            raise FileNotFoundError(f"resume checkpoint missing: {resume}")
        cmd.extend(["--resume", str(resume)])
    if args.disable_torch_compile:
        cmd.append("--disable_torch_compile")
    print(f"[runner] train method={method} target_update={target_update}", flush=True)
    return _run_command(
        cmd,
        cwd=repo,
        env=env,
        log_path=method_dir / "train_segments.log",
    )


def _eval_checkpoint(
    *,
    args: argparse.Namespace,
    repo: Path,
    env: dict[str, str],
    method: str,
    method_dir: Path,
    update: int,
    checkpoint: Path,
    episodes: int,
    num_envs: int,
    seed_base: int,
    label_prefix: str,
) -> tuple[float, dict[str, Any], Path]:
    label = f"{label_prefix}_{method}_u{int(update):04d}"
    out_dir = method_dir / label
    summary_path = out_dir / f"{label}_summary.json"
    if summary_path.exists():
        print(f"[runner] eval skip existing {summary_path}", flush=True)
        return 0.0, _load_summary(summary_path), out_dir
    cmd = [
        sys.executable,
        "scripts/evaluate_structured_mixed_heads_native.py",
        "--config",
        args.config,
        "--base_checkpoint",
        str(checkpoint),
        "--episodes",
        str(episodes),
        "--num_envs",
        str(num_envs),
        "--episode_seed_base",
        str(seed_base),
        "--policy_mode",
        "deterministic",
        "--device",
        args.device,
        "--exec_accel_source",
        "policy",
        "--exec_sat_source",
        "policy",
        "--exec_bw_source",
        "policy",
        "--out_dir",
        str(out_dir),
        "--label",
        label,
    ]
    print(f"[runner] eval method={method} update={update} episodes={episodes} seed={seed_base}", flush=True)
    elapsed = _run_command(
        cmd,
        cwd=repo,
        env=env,
        log_path=method_dir / "eval.log",
    )
    return elapsed, _load_summary(summary_path), out_dir


def _stop_state_from_rows(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    raw_best_reward = -float("inf")
    raw_best_update = 0
    sig_best_reward = -float("inf")
    sig_best_update = 0
    patience = 0
    degradation = 0
    stop_reason = "continue"
    for row in rows:
        update = int(row["update"])
        reward = float(row["reward_sum"])
        if reward > raw_best_reward:
            raw_best_reward = reward
            raw_best_update = update
        significant_margin = args.plateau_min_delta_rel * max(abs(sig_best_reward), 1.0) if sig_best_reward != -float("inf") else 0.0
        significant_improved = sig_best_reward == -float("inf") or reward > sig_best_reward + significant_margin
        if update < args.min_updates:
            if significant_improved:
                sig_best_reward = reward
                sig_best_update = update
            patience = 0
            degradation = 0
            stop_reason = "continue"
        else:
            if significant_improved:
                sig_best_reward = reward
                sig_best_update = update
                patience = 0
            else:
                patience += 1
            if raw_best_reward > -float("inf") and reward < raw_best_reward * (1.0 - args.degradation_drop_rel):
                degradation += 1
            else:
                degradation = 0
            if patience >= args.plateau_patience:
                stop_reason = "plateau_patience"
            elif degradation >= args.degradation_patience:
                stop_reason = "degradation_alarm"
            elif update >= args.hard_max_updates:
                stop_reason = "hard_max_updates"
            else:
                stop_reason = "continue"
        row["best_reward_so_far"] = raw_best_reward
        row["best_update_so_far"] = raw_best_update
        row["significant_best_reward_so_far"] = sig_best_reward
        row["significant_best_update_so_far"] = sig_best_update
        row["patience_count"] = patience
        row["degradation_count"] = degradation
        row["stop_reason_candidate"] = stop_reason
    return {
        "raw_best_reward": raw_best_reward,
        "raw_best_update": raw_best_update,
        "significant_best_reward": sig_best_reward,
        "significant_best_update": sig_best_update,
        "patience_count": patience,
        "degradation_count": degradation,
        "stop_reason": stop_reason,
    }


def _summary_row(
    *,
    method: str,
    state: str,
    stop: dict[str, Any],
    final_update: int,
    final_reward: float,
    heldout_best: dict[str, Any],
    heldout_final: dict[str, Any],
    best_checkpoint: Path,
    final_checkpoint: Path,
) -> dict[str, Any]:
    best_reward = float(stop["raw_best_reward"])
    return {
        "method": method,
        "state": state,
        "stop_reason": stop["stop_reason"],
        "best_update_validation": int(stop["raw_best_update"]),
        "best_reward_validation": best_reward,
        "final_update_validation": int(final_update),
        "final_reward_validation": float(final_reward),
        "drop_best_to_final": best_reward - float(final_reward),
        "heldout_best_reward": float(heldout_best.get("reward_sum", "nan")),
        "heldout_final_reward": float(heldout_final.get("reward_sum", "nan")),
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
    }


def run_method(args: argparse.Namespace, *, repo: Path, env: dict[str, str], run_root: Path, method: str) -> dict[str, Any]:
    method_dir = run_root / f"{method.replace('_gae', '')}_seed{args.seed}"
    method_dir.mkdir(parents=True, exist_ok=True)
    validation_csv = method_dir / "validation_summary.csv"
    rows: list[dict[str, Any]] = []
    previous_update = 0
    for update in range(args.interval_updates, args.hard_max_updates + 1, args.interval_updates):
        train_sec = _train_segment(
            args=args,
            repo=repo,
            env=env,
            method=method,
            method_dir=method_dir,
            target_update=update,
            previous_update=previous_update,
        )
        checkpoint = _checkpoint_path(method_dir, update)
        if not checkpoint.exists():
            raise FileNotFoundError(f"expected checkpoint not found after training: {checkpoint}")
        eval_sec, summary, _ = _eval_checkpoint(
            args=args,
            repo=repo,
            env=env,
            method=method,
            method_dir=method_dir,
            update=update,
            checkpoint=checkpoint,
            episodes=args.validation_episodes,
            num_envs=args.validation_num_envs,
            seed_base=args.validation_seed_base,
            label_prefix="validation",
        )
        row = {
            "method": "bootstrap" if method == "bootstrap_gae" else method,
            "update": update,
            "checkpoint": str(checkpoint),
            "return_target": method,
            "train_sec": round(train_sec, 3),
            "eval_sec": round(eval_sec, 3),
        }
        for key in VALIDATION_FIELDS:
            if key in summary:
                row[key] = summary[key]
        rows.append(row)
        stop = _stop_state_from_rows(rows, args=args)
        _write_csv(validation_csv, rows, VALIDATION_FIELDS)
        print(
            "[runner] validation "
            f"method={method} update={update} reward={float(row['reward_sum']):.3f} "
            f"best={float(stop['raw_best_reward']):.3f}@{int(stop['raw_best_update'])} "
            f"pat={int(stop['patience_count'])} deg={int(stop['degradation_count'])} "
            f"reason={stop['stop_reason']}",
            flush=True,
        )
        previous_update = update
        if update >= args.min_updates and stop["stop_reason"] != "continue":
            break
    if not rows:
        raise RuntimeError(f"no validation rows produced for {method}")
    stop = _stop_state_from_rows(rows, args=args)
    final_update = int(rows[-1]["update"])
    final_reward = float(rows[-1]["reward_sum"])
    best_update = int(stop["raw_best_update"])
    best_checkpoint = _checkpoint_path(method_dir, best_update)
    final_checkpoint = _checkpoint_path(method_dir, final_update)
    _, heldout_best, _ = _eval_checkpoint(
        args=args,
        repo=repo,
        env=env,
        method=method,
        method_dir=method_dir,
        update=best_update,
        checkpoint=best_checkpoint,
        episodes=args.heldout_episodes,
        num_envs=args.heldout_num_envs,
        seed_base=args.heldout_seed_base,
        label_prefix="heldout_best",
    )
    _, heldout_final, _ = _eval_checkpoint(
        args=args,
        repo=repo,
        env=env,
        method=method,
        method_dir=method_dir,
        update=final_update,
        checkpoint=final_checkpoint,
        episodes=args.heldout_episodes,
        num_envs=args.heldout_num_envs,
        seed_base=args.heldout_seed_base,
        label_prefix="heldout_final",
    )
    method_summary = _summary_row(
        method="bootstrap" if method == "bootstrap_gae" else method,
        state="complete",
        stop=stop,
        final_update=final_update,
        final_reward=final_reward,
        heldout_best=heldout_best,
        heldout_final=heldout_final,
        best_checkpoint=best_checkpoint,
        final_checkpoint=final_checkpoint,
    )
    (method_dir / "training_stop_summary.json").write_text(
        json.dumps(method_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return method_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmented 6UAV/80GU MC vs bootstrap-GAE trainer with validation stopping.")
    parser.add_argument("--config", default="configs/experiments/phase3_generalization/main_6uav80gu_t250.yaml")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--seed", type=int, default=45211)
    parser.add_argument("--methods", default="mc,bootstrap_gae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-env-steps", type=int, default=250)
    parser.add_argument("--critic-update-microbatch-size", type=int, default=1000)
    parser.add_argument("--interval-updates", type=int, default=25)
    parser.add_argument("--min-updates", type=int, default=300)
    parser.add_argument("--soft-max-updates", type=int, default=500)
    parser.add_argument("--hard-max-updates", type=int, default=700)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-num-envs", type=int, default=32)
    parser.add_argument("--validation-seed-base", type=int, default=910000)
    parser.add_argument("--heldout-episodes", type=int, default=128)
    parser.add_argument("--heldout-num-envs", type=int, default=64)
    parser.add_argument("--heldout-seed-base", type=int, default=930000)
    parser.add_argument("--plateau-patience", type=int, default=4)
    parser.add_argument("--plateau-min-delta-rel", type=float, default=0.01)
    parser.add_argument("--degradation-patience", type=int, default=3)
    parser.add_argument("--degradation-drop-rel", type=float, default=0.10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--disable-torch-compile", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root or f"runs/phase3/main_6uav80gu_return_targets_seed{args.seed}_{timestamp}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise ValueError(f"unsupported methods {bad}; expected subset of {METHODS}")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo))
    env.setdefault("MPLCONFIGDIR", "/tmp/sagin_marl_mpl")
    env.setdefault("SAGIN_MARL_NATIVE_CUDA_CACHE", str(Path.home() / ".cache" / "sagin_marl_native_cuda_cache"))
    env.setdefault("SAGIN_MARL_NATIVE_CUDA_MAX_JOBS", "1")
    env.setdefault("SAGIN_MARL_SKIP_PRUNE_GC", "1")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env.setdefault("OMP_NUM_THREADS", str(args.torch_threads))

    manifest_path = run_root / "manifest.csv"
    if not manifest_path.exists():
        for key, value in [
            ("created_at", datetime.now().isoformat(timespec="seconds")),
            ("git_commit", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()),
            ("git_branch", subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()),
            ("seed", args.seed),
            ("config", args.config),
            ("methods", ",".join(methods)),
            ("interval_updates", args.interval_updates),
            ("min_updates", args.min_updates),
            ("soft_max_updates", args.soft_max_updates),
            ("hard_max_updates", args.hard_max_updates),
            ("validation_episodes", args.validation_episodes),
            ("validation_seed", args.validation_seed_base),
            ("heldout_episodes", args.heldout_episodes),
            ("heldout_seed", args.heldout_seed_base),
            ("num_envs", args.num_envs),
            ("rollout_env_steps", args.rollout_env_steps),
            ("critic_update_microbatch_size", args.critic_update_microbatch_size),
            ("structured_env_backend", "native"),
            ("structured_env_tensor_backend", "cuda"),
            ("plateau_patience", args.plateau_patience),
            ("plateau_min_delta_rel", args.plateau_min_delta_rel),
            ("degradation_patience", args.degradation_patience),
            ("degradation_drop_rel", args.degradation_drop_rel),
            ("disable_torch_compile", args.disable_torch_compile),
        ]:
            _append_manifest(manifest_path, {"key": key, "value": value})

    summaries: list[dict[str, Any]] = []
    for method in methods:
        summaries.append(run_method(args, repo=repo, env=env, run_root=run_root, method=method))
        _write_csv(run_root / "aggregate_reward_summary.csv", summaries, FINAL_FIELDS)
    print(f"[runner] all done run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
