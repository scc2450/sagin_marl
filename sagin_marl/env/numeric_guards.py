from __future__ import annotations

import numpy as np

# Semantic numerical guards for environment math. These values are intentionally
# tiny relative to the domain units; they prevent undefined arithmetic without
# acting as physical noise floors or hidden model parameters.
NORMALIZATION_DENOM_EPS = 1.0e-9
RUNTIME_RATIO_ZERO_TOL = 1.0e-9
LOG_RATIO_EPS = 1.0e-12
RELATIVE_LOG_EPS = 1.0e-6
GEOMETRY_DENOM_EPS = 1.0e-9
POSITIVE_COEFF_EPS = 1.0e-12
FREQUENCY_GHZ_EPS = 1.0e-6
ANGLE_RAD_EPS = 1.0e-6
TRIG_DENOM_EPS = 1.0e-6
MIN_RAIN_REDUCTION_FACTOR = 1.0e-3
PROBABILITY_EQUALITY_TOL = 1.0e-12


def positive_float(value: float, eps: float = NORMALIZATION_DENOM_EPS) -> float:
    return max(float(value), float(eps))


def positive_array(value, eps: float = NORMALIZATION_DENOM_EPS) -> np.ndarray:
    return np.maximum(np.asarray(value), float(eps))


def require_positive_float(value: float, *, name: str) -> float:
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value_f!r}.")
    return value_f


def require_positive_array(value, *, name: str, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    invalid = (~np.isfinite(arr)) | (arr <= 0.0)
    if np.any(invalid):
        bad = arr[invalid]
        sample = bad[: min(int(bad.size), 5)].tolist()
        raise ValueError(f"{name} must be positive and finite; invalid sample={sample}.")
    return arr


def positive_config_scale(value: float) -> float:
    """Guard static config scales that are required to be positive.

    Use this for feature normalization scales such as map size, queue capacity,
    max speed, or energy capacity. Do not use it for runtime ratios where a
    zero denominator has a semantic fallback.
    """
    return positive_float(value, NORMALIZATION_DENOM_EPS)


def normalize_scale(value: float) -> float:
    return positive_config_scale(value)


def reward_ratio_denominator(value) -> np.ndarray:
    """Validate reward-defining denominators such as arrival_ref_bits_per_step.

    Unlike diagnostic ratios, reward denominators are part of the objective
    definition. A non-positive value is a configuration/state error and should
    fail instead of being silently floored.
    """
    return require_positive_array(value, name="reward ratio denominator")


def reward_ratio_denominator_scalar(value: float, *, name: str = "reward ratio denominator") -> float:
    return require_positive_float(value, name=name)


def geometry_denominator(value) -> np.ndarray:
    return positive_array(value, GEOMETRY_DENOM_EPS)


def log_ratio_argument(value) -> np.ndarray:
    return positive_array(value, LOG_RATIO_EPS)


def relative_log_argument(value) -> np.ndarray:
    return positive_array(value, RELATIVE_LOG_EPS)


def log_argument(value) -> np.ndarray:
    return positive_array(value, RELATIVE_LOG_EPS)


def safe_divide(numerator, denominator, *, eps: float = NORMALIZATION_DENOM_EPS):
    return np.asarray(numerator) / positive_array(denominator, eps)


def divide_or_default(
    numerator,
    denominator,
    *,
    eps: float = NORMALIZATION_DENOM_EPS,
    default: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    numerator_arr, denominator_arr = np.broadcast_arrays(
        np.asarray(numerator, dtype=dtype),
        np.asarray(denominator, dtype=dtype),
    )
    out = np.full_like(numerator_arr, float(default), dtype=dtype)
    np.divide(numerator_arr, denominator_arr, out=out, where=np.abs(denominator_arr) > float(eps))
    return out


def scalar_divide_or_default(
    numerator: float,
    denominator: float,
    *,
    eps: float = NORMALIZATION_DENOM_EPS,
    default: float = 0.0,
) -> float:
    denominator_value = float(denominator)
    if abs(denominator_value) <= float(eps):
        return float(default)
    return float(numerator) / denominator_value


def ratio_or_zero(numerator, denominator, *, dtype=np.float32) -> np.ndarray:
    return divide_or_default(
        numerator,
        denominator,
        eps=RUNTIME_RATIO_ZERO_TOL,
        default=0.0,
        dtype=dtype,
    )


def scalar_ratio_or_zero(numerator: float, denominator: float) -> float:
    return scalar_divide_or_default(
        numerator,
        denominator,
        eps=RUNTIME_RATIO_ZERO_TOL,
        default=0.0,
    )
