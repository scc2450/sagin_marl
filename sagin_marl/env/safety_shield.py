from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
from typing import Any

import numpy as np


@dataclass
class SafetyShieldResult:
    accel: np.ndarray
    status: str
    solver: str
    feasible: bool
    active: bool
    pair_count: int
    delta_norm_mean: float
    delta_norm_max: float
    min_margin_before: float
    min_margin_after: float
    min_distance_after: float
    objective: float


def _project_l2_ball(values: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    limit = max(float(max_norm), 0.0)
    if limit <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limit / np.maximum(norms, 1.0e-8))
    return (arr * scale).astype(np.float32, copy=False)


def _solver_name(value: Any) -> str:
    return str(value or "CLARABEL").strip().upper()


def _configure_cuclarabel_runtime(cfg: Any, solver_name: str) -> None:
    if _solver_name(solver_name) != "CUCLARABEL":
        return
    env_fields = (
        ("JULIA_DEPOT_PATH", "safety_shield_julia_depot"),
        ("PYTHON_JULIACALL_EXE", "safety_shield_julia_exe"),
        ("PYTHON_JULIACALL_PROJECT", "safety_shield_julia_project"),
    )
    for env_key, attr in env_fields:
        value = str(getattr(cfg, attr, "") or "").strip()
        if value:
            os.environ[env_key] = value


@contextmanager
def _solver_output_context(suppress: bool):
    if not suppress:
        yield
        return
    stdout_copy = None
    stderr_copy = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_copy = os.dup(1)
        stderr_copy = os.dup(2)
    except Exception:
        yield
        return
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if stdout_copy is not None:
            os.dup2(stdout_copy, 1)
            os.close(stdout_copy)
        if stderr_copy is not None:
            os.dup2(stderr_copy, 2)
            os.close(stderr_copy)


def _pair_direction(r: np.ndarray, w: np.ndarray) -> np.ndarray:
    r_arr = np.asarray(r, dtype=np.float64)
    r_norm = float(np.linalg.norm(r_arr))
    if r_norm > 1.0e-8:
        return r_arr / r_norm
    w_arr = np.asarray(w, dtype=np.float64)
    w_norm = float(np.linalg.norm(w_arr))
    if w_norm > 1.0e-8:
        return -w_arr / w_norm
    return np.array([1.0, 0.0], dtype=np.float64)


def _brake_margin_stats(
    pos: np.ndarray,
    vel: np.ndarray,
    accel: np.ndarray,
    *,
    tau: float,
    d_safe: float,
    buffer: float,
    a_safe: float,
) -> tuple[float, float]:
    num_uav = int(pos.shape[0])
    if num_uav < 2:
        return float("inf"), float("inf")
    min_margin = float("inf")
    min_dist = float("inf")
    for u in range(num_uav):
        for v in range(u + 1, num_uav):
            r = np.asarray(pos[u] - pos[v], dtype=np.float64)
            w = np.asarray(vel[u] - vel[v], dtype=np.float64)
            a_rel = np.asarray(accel[u] - accel[v], dtype=np.float64)
            n = _pair_direction(r, w)
            w_next = w + float(tau) * a_rel
            r_next = r + float(tau) * w_next
            dist_next = float(np.linalg.norm(r_next))
            c_next = max(0.0, -float(np.dot(n, w_next)))
            radial_next = float(np.dot(n, r_next))
            margin = radial_next - (float(d_safe) + float(buffer) + c_next * c_next / (2.0 * float(a_safe)))
            min_margin = min(min_margin, margin)
            min_dist = min(min_dist, dist_next)
    return min_margin, min_dist


def solve_brake_distance_shield(
    *,
    pos: np.ndarray,
    vel: np.ndarray,
    nominal_accel: np.ndarray,
    cfg: Any,
) -> SafetyShieldResult:
    """Solve a CPU CVXPY brake-distance safety shield for UAV accelerations."""

    accel_nom = _project_l2_ball(np.asarray(nominal_accel, dtype=np.float32), float(cfg.a_max))
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    num_uav = int(accel_nom.shape[0])
    tau = max(float(getattr(cfg, "tau0", 1.0) or 1.0), 1.0e-8)
    a_max = max(float(getattr(cfg, "a_max", 0.0) or 0.0), 0.0)
    v_max = max(float(getattr(cfg, "v_max", 0.0) or 0.0), 0.0)
    d_safe = max(float(getattr(cfg, "d_safe", 0.0) or 0.0), 0.0)
    rho = max(float(getattr(cfg, "safety_shield_brake_rho", 0.8) or 0.8), 1.0e-6)
    buffer = max(float(getattr(cfg, "safety_shield_distance_buffer", 0.0) or 0.0), 0.0)
    a_safe = max(float(getattr(cfg, "safety_shield_a_safe", 0.0) or 0.0), 0.0)
    if a_safe <= 0.0:
        a_safe = rho * 2.0 * max(a_max, 1.0e-8)
    a_safe = max(a_safe, 1.0e-8)
    pair_count = max(num_uav * (num_uav - 1) // 2, 0)
    min_before, _ = _brake_margin_stats(
        pos_arr,
        vel_arr,
        accel_nom,
        tau=tau,
        d_safe=d_safe,
        buffer=buffer,
        a_safe=a_safe,
    )

    if num_uav <= 1 or pair_count <= 0:
        return SafetyShieldResult(
            accel=accel_nom,
            status="skipped",
            solver="",
            feasible=True,
            active=False,
            pair_count=pair_count,
            delta_norm_mean=0.0,
            delta_norm_max=0.0,
            min_margin_before=min_before,
            min_margin_after=min_before,
            min_distance_after=float("inf"),
            objective=0.0,
        )

    preferred_solver = _solver_name(getattr(cfg, "safety_shield_solver", "CLARABEL"))
    if preferred_solver == "NATIVE_CUDA":
        raise RuntimeError(
            "safety_shield_solver=NATIVE_CUDA is only available in the structured native CUDA rollout. "
            "Use the native structured GPU path for fused in-kernel projection, or set "
            "safety_shield_solver to CLARABEL/CUCLARABEL for the Python CVXPY shield."
        )
    _configure_cuclarabel_runtime(cfg, preferred_solver)

    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "safety_shield_enabled=True requires cvxpy. Install it with "
            "`python -m pip install cvxpy` in the project virtual environment."
        ) from exc

    solver_candidates = list(dict.fromkeys([preferred_solver, "CLARABEL", "SCS"]))
    installed = {str(s).upper() for s in cp.installed_solvers()}
    verbose = bool(getattr(cfg, "safety_shield_verbose", False))

    def build_problem(*, relaxed: bool):
        a_var = cp.Variable((num_uav, 2))
        constraints = []
        relax_terms = []
        for u in range(num_uav):
            constraints.append(cp.norm(a_var[u, :], 2) <= a_max)
            speed_expr = cp.norm(vel_arr[u] + tau * a_var[u, :], 2)
            if relaxed:
                speed_slack = cp.Variable(nonneg=True)
                relax_terms.append(speed_slack)
                constraints.append(speed_expr <= v_max + speed_slack)
            else:
                constraints.append(speed_expr <= v_max)

        for u in range(num_uav):
            for v in range(u + 1, num_uav):
                r = pos_arr[u] - pos_arr[v]
                w = vel_arr[u] - vel_arr[v]
                n = _pair_direction(r, w)
                w_next = w + tau * (a_var[u, :] - a_var[v, :])
                r_next_proj = float(np.dot(n, r + tau * w)) + tau * tau * (n @ (a_var[u, :] - a_var[v, :]))
                c_next = cp.Variable(nonneg=True)
                constraints.append(c_next >= -(n @ w_next))
                lhs = float(d_safe + buffer) + cp.square(c_next) / (2.0 * a_safe)
                if relaxed:
                    pair_slack = cp.Variable(nonneg=True)
                    relax_terms.append(pair_slack)
                    constraints.append(lhs <= r_next_proj + pair_slack)
                else:
                    constraints.append(lhs <= r_next_proj)

        nominal_error = cp.sum_squares(a_var - accel_nom.astype(np.float64))
        if relaxed and relax_terms:
            slack_weight = max(float(getattr(cfg, "safety_shield_relax_slack_weight", 1.0e4) or 1.0e4), 1.0e-6)
            action_weight = max(float(getattr(cfg, "safety_shield_relax_action_weight", 1.0) or 1.0), 0.0)
            objective = cp.Minimize(slack_weight * cp.sum_squares(cp.hstack(relax_terms)) + action_weight * nominal_error)
        else:
            objective = cp.Minimize(nominal_error)
        return cp.Problem(objective, constraints), a_var

    def solve_problem(problem):
        status = "not_solved"
        solver_used = ""
        last_error: Exception | None = None
        for solver_name in solver_candidates:
            if solver_name not in installed:
                continue
            try:
                suppress_output = solver_name == "CUCLARABEL" and not verbose
                with _solver_output_context(suppress_output):
                    problem.solve(solver=solver_name, verbose=verbose, warm_start=True)
                status = str(problem.status)
                solver_used = solver_name
                if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                    break
            except Exception as exc:  # pragma: no cover - depends on solver availability
                last_error = exc
                status = f"{solver_name}_error"
                continue
        return status, solver_used, last_error

    def make_result(
        *,
        accel_value: np.ndarray,
        status: str,
        solver_used: str,
        feasible: bool,
        objective_value: float,
    ) -> SafetyShieldResult:
        accel = _project_l2_ball(np.asarray(accel_value, dtype=np.float32), a_max)
        delta_norms = np.linalg.norm(accel - accel_nom, axis=1) if num_uav > 0 else np.zeros((0,), dtype=np.float32)
        min_after, min_dist_after = _brake_margin_stats(
            pos_arr,
            vel_arr,
            accel,
            tau=tau,
            d_safe=d_safe,
            buffer=buffer,
            a_safe=a_safe,
        )
        return SafetyShieldResult(
            accel=accel,
            status=status,
            solver=solver_used,
            feasible=feasible,
            active=bool(np.any(delta_norms > 1.0e-6)),
            pair_count=pair_count,
            delta_norm_mean=float(np.mean(delta_norms)) if delta_norms.size else 0.0,
            delta_norm_max=float(np.max(delta_norms)) if delta_norms.size else 0.0,
            min_margin_before=min_before,
            min_margin_after=min_after,
            min_distance_after=min_dist_after,
            objective=objective_value,
        )

    problem, a_var = build_problem(relaxed=False)
    status, solver_used, last_error = solve_problem(problem)
    feasible = status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and a_var.value is not None
    if feasible:
        return make_result(
            accel_value=a_var.value,
            status=status,
            solver_used=solver_used,
            feasible=True,
            objective_value=float(problem.value) if problem.value is not None else float("nan"),
        )

    if bool(getattr(cfg, "safety_shield_relax_on_infeasible", True)):
        relaxed_problem, relaxed_var = build_problem(relaxed=True)
        relaxed_status, relaxed_solver, relaxed_error = solve_problem(relaxed_problem)
        relaxed_feasible = (
            relaxed_status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            and relaxed_var.value is not None
        )
        if relaxed_feasible:
            return make_result(
                accel_value=relaxed_var.value,
                status=f"relaxed_after_{status}",
                solver_used=relaxed_solver,
                feasible=False,
                objective_value=float(relaxed_problem.value) if relaxed_problem.value is not None else float("nan"),
            )
        if last_error is None:
            last_error = relaxed_error
        if not solver_used:
            solver_used = relaxed_solver
        status = f"{status};relaxed_{relaxed_status}"

    if not feasible:
        if not solver_used and last_error is not None:
            status = f"error:{type(last_error).__name__}"
        min_after, min_dist_after = _brake_margin_stats(
            pos_arr,
            vel_arr,
            accel_nom,
            tau=tau,
            d_safe=d_safe,
            buffer=buffer,
            a_safe=a_safe,
        )
        return SafetyShieldResult(
            accel=accel_nom,
            status=status,
            solver=solver_used,
            feasible=False,
            active=False,
            pair_count=pair_count,
            delta_norm_mean=0.0,
            delta_norm_max=0.0,
            min_margin_before=min_before,
            min_margin_after=min_after,
            min_distance_after=min_dist_after,
            objective=float("nan"),
        )
