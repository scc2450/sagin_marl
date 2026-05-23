from __future__ import annotations

from functools import lru_cache

import numpy as np

from .numeric_guards import (
    ANGLE_RAD_EPS,
    FREQUENCY_GHZ_EPS,
    GEOMETRY_DENOM_EPS,
    LOG_RATIO_EPS,
    MIN_RAIN_REDUCTION_FACTOR,
    POSITIVE_COEFF_EPS,
    PROBABILITY_EQUALITY_TOL,
    RELATIVE_LOG_EPS,
    TRIG_DENOM_EPS,
    geometry_denominator,
    log_argument,
    positive_float,
)


def los_probability(phi_deg: np.ndarray, a: float, b: float) -> np.ndarray:
    # Probabilistic LoS model (phi in degrees)
    return 1.0 / (1.0 + a * np.exp(-b * (phi_deg - a)))


def pathloss_db(d: np.ndarray, phi_rad: np.ndarray, cfg, *, carrier_freq_hz: float | None = None) -> np.ndarray:
    d = geometry_denominator(d)
    freq_hz = float(cfg.carrier_freq if carrier_freq_hz is None else carrier_freq_hz)
    pl_base = cfg.pathloss_const_db + 20.0 * np.log10(max(freq_hz, 1.0) / 1e9)
    pl_los = pl_base + cfg.xi_los + 20.0 * np.log10(d)
    pl_nlos = pl_base + cfg.xi_nlos + 20.0 * np.log10(d)
    if getattr(cfg, "pathloss_mode", "prob_los") == "free_space":
        return pl_los
    phi_deg = np.degrees(phi_rad)
    p_los = los_probability(phi_deg, cfg.los_a, cfg.los_b)
    return p_los * pl_los + (1.0 - p_los) * pl_nlos


def rician_power_gain(K: float, size, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    K = max(float(K), 0.0)
    s = np.sqrt(K / (K + 1.0))
    sigma = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    h_real = rng.normal(loc=s, scale=sigma, size=size)
    h_imag = rng.normal(loc=0.0, scale=sigma, size=size)
    return h_real**2 + h_imag**2


def rician_k_linear_from_config(cfg) -> float:
    k_db = getattr(cfg, "access_rician_k_db", None)
    if k_db is not None:
        return 10.0 ** (float(k_db) / 10.0)
    return max(float(getattr(cfg, "rician_K", 0.0) or 0.0), 0.0)


def access_fading_mode_from_config(cfg) -> str:
    mode = str(getattr(cfg, "access_fading_mode", "ergodic_rician") or "ergodic_rician").strip().lower()
    if mode in {"none", "off", "disabled"}:
        return "large_scale"
    if mode not in {"large_scale", "ergodic_rician", "iid_rician"}:
        return "ergodic_rician"
    return mode


def noise_figure_linear(noise_figure_db: float) -> float:
    return 10.0 ** (max(float(noise_figure_db), 0.0) / 10.0)


def safe_snr_linear(
    power: float,
    gain: np.ndarray,
    noise_density: float,
    bandwidth: float | np.ndarray,
    interference: float | np.ndarray = 0.0,
    *,
    noise_figure_db: float = 0.0,
) -> np.ndarray:
    signal = float(power) * np.asarray(gain, dtype=np.float64)
    bw = np.asarray(bandwidth, dtype=np.float64)
    denom = float(noise_density) * bw * noise_figure_linear(noise_figure_db) + np.asarray(interference, dtype=np.float64)
    signal, denom, bw = np.broadcast_arrays(signal, denom, bw)
    out = np.zeros_like(signal, dtype=np.float64)
    valid = (bw > 0.0) & (denom > 0.0)
    np.divide(signal, denom, out=out, where=valid)
    return out


@lru_cache(maxsize=16)
def _hermgauss_cached(points: int) -> tuple[np.ndarray, np.ndarray]:
    n = max(int(points), 1)
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return nodes.astype(np.float64), weights.astype(np.float64)


def rician_ergodic_spectral_efficiency(
    snr: np.ndarray,
    K: float,
    *,
    quadrature_points: int = 16,
) -> np.ndarray:
    """Return E_h[log2(1 + snr * h)] for unit-mean Rician power gain h."""
    snr_arr = np.asarray(snr, dtype=np.float64)
    k = max(float(K), 0.0)
    if quadrature_points <= 1:
        return spectral_efficiency(snr_arr)
    nodes, weights = _hermgauss_cached(int(quadrature_points))
    sigma = np.sqrt(1.0 / (2.0 * (k + 1.0)))
    mean_real = np.sqrt(k / (k + 1.0))
    real = mean_real + sigma * np.sqrt(2.0) * nodes[:, None]
    imag = sigma * np.sqrt(2.0) * nodes[None, :]
    power_gain = real * real + imag * imag
    quad_weights = (weights[:, None] * weights[None, :]) / np.pi
    se = np.log2(1.0 + np.maximum(snr_arr[..., None, None], 0.0) * power_gain)
    return np.sum(se * quad_weights, axis=(-2, -1))


def atmospheric_loss_db(theta_rad: np.ndarray, base_loss_db: float) -> np.ndarray:
    sin_el = np.maximum(np.sin(theta_rad), 1e-3)
    return base_loss_db / sin_el


def _p838_curve(log_f: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    out = np.zeros_like(log_f, dtype=np.float64)
    for ai, bi, ci in zip(a, b, c, strict=True):
        out += ai * np.exp(-((log_f - bi) / abs(ci)) ** 2)
    return out


def _rain_frequency_terms_from_logf(log_f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_kh = np.array([-5.33980, -0.35351, -0.23789, -0.94158], dtype=np.float64)
    b_kh = np.array([-0.10008, 1.26970, 0.86036, 0.64552], dtype=np.float64)
    c_kh = np.array([1.13098, 0.45400, 0.15354, 0.16817], dtype=np.float64)
    a_kv = np.array([-3.80595, -3.44965, -0.39902, 0.50167], dtype=np.float64)
    b_kv = np.array([0.56934, -0.22911, 0.73042, 1.07319], dtype=np.float64)
    c_kv = np.array([0.81061, 0.51059, 0.11899, 0.27195], dtype=np.float64)

    a_ah = np.array([-0.14318, 0.29591, 0.32177, -5.37610, 16.1721], dtype=np.float64)
    b_ah = np.array([1.82442, 0.77564, 0.63773, -0.96230, -3.29980], dtype=np.float64)
    c_ah = np.array([-0.55187, 0.19822, 0.13164, 1.47828, 3.43990], dtype=np.float64)
    a_av = np.array([-0.07771, 0.56727, -0.20238, -48.2991, 48.5833], dtype=np.float64)
    b_av = np.array([2.33840, 0.95545, 1.14520, 0.791669, 0.791459], dtype=np.float64)
    c_av = np.array([-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], dtype=np.float64)

    log10_kh = _p838_curve(log_f, a_kh, b_kh, c_kh) - 0.18961 * log_f + 0.71147
    log10_kv = _p838_curve(log_f, a_kv, b_kv, c_kv) - 0.16398 * log_f + 0.63297
    alpha_h = _p838_curve(log_f, a_ah, b_ah, c_ah) + 0.67849 * log_f - 1.95537
    alpha_v = _p838_curve(log_f, a_av, b_av, c_av) - 0.053739 * log_f + 0.83433
    return 10.0 ** log10_kh, 10.0 ** log10_kv, alpha_h, alpha_v


@lru_cache(maxsize=16)
def _rain_frequency_terms_scalar(carrier_freq_hz: float) -> tuple[float, float, float, float]:
    freq_ghz = positive_float(float(carrier_freq_hz) / 1e9, FREQUENCY_GHZ_EPS)
    log_f = np.array(np.log10(freq_ghz), dtype=np.float64)
    k_h, k_v, alpha_h, alpha_v = _rain_frequency_terms_from_logf(log_f)
    return float(k_h), float(k_v), float(alpha_h), float(alpha_v)


def rain_specific_attenuation_coefficients(
    carrier_freq_hz: float | np.ndarray,
    theta_rad: float | np.ndarray,
    polarization_tilt_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_rad, dtype=np.float64)
    if np.ndim(carrier_freq_hz) == 0:
        k_h, k_v, alpha_h, alpha_v = _rain_frequency_terms_scalar(float(carrier_freq_hz))
    else:
        freq_ghz = np.asarray(carrier_freq_hz, dtype=np.float64) / 1e9
        freq_ghz = np.maximum(freq_ghz, FREQUENCY_GHZ_EPS)
        log_f = np.log10(freq_ghz)
        k_h, k_v, alpha_h, alpha_v = _rain_frequency_terms_from_logf(log_f)
    tau = np.radians(float(polarization_tilt_deg))
    cos_term = np.cos(theta) ** 2 * np.cos(2.0 * tau)
    k = 0.5 * (k_h + k_v + (k_h - k_v) * cos_term)
    alpha = 0.5 * (
        k_h * alpha_h + k_v * alpha_v + (k_h * alpha_h - k_v * alpha_v) * cos_term
    ) / np.maximum(k, POSITIVE_COEFF_EPS)
    return k.astype(np.float64, copy=False), alpha.astype(np.float64, copy=False)


def rain_specific_attenuation_db_per_km(
    carrier_freq_hz: float | np.ndarray,
    theta_rad: float | np.ndarray,
    rain_rate_mmph: float,
    polarization_tilt_deg: float = 45.0,
) -> np.ndarray:
    rain_rate = max(float(rain_rate_mmph), 0.0)
    if rain_rate <= 0.0:
        return np.zeros_like(np.asarray(theta_rad, dtype=np.float64))
    k, alpha = rain_specific_attenuation_coefficients(
        carrier_freq_hz,
        theta_rad,
        polarization_tilt_deg=polarization_tilt_deg,
    )
    return k * (rain_rate ** alpha)


def rain_attenuation_db(
    theta_rad: np.ndarray,
    carrier_freq_hz: float,
    rain_rate_001_mmph: float,
    rain_height_km: float,
    station_height_km: float,
    latitude_deg: float,
    exceedance_pct: float = 0.1,
    polarization_tilt_deg: float = 45.0,
) -> np.ndarray:
    theta = np.asarray(theta_rad, dtype=np.float64)
    rain_rate = max(float(rain_rate_001_mmph), 0.0)
    if rain_rate <= 0.0 or rain_height_km <= station_height_km:
        return np.zeros_like(theta, dtype=np.float64)

    theta = np.clip(theta, ANGLE_RAD_EPS, None)
    theta_deg = np.degrees(theta)
    sin_el = np.maximum(np.sin(theta), TRIG_DENOM_EPS)
    cos_el = np.cos(theta)
    freq_ghz = positive_float(float(carrier_freq_hz) / 1e9, FREQUENCY_GHZ_EPS)

    gamma_r = rain_specific_attenuation_db_per_km(
        carrier_freq_hz,
        theta,
        rain_rate,
        polarization_tilt_deg=polarization_tilt_deg,
    )

    delta_h = max(float(rain_height_km) - float(station_height_km), 0.0)
    l_s = delta_h / sin_el
    l_g = l_s * cos_el

    r_001 = 1.0 / (
        1.0
        + 0.78 * np.sqrt(np.maximum(l_g * gamma_r / freq_ghz, 0.0))
        - 0.38 * (1.0 - np.exp(-2.0 * np.maximum(l_g, 0.0)))
    )
    r_001 = np.clip(r_001, MIN_RAIN_REDUCTION_FACTOR, None)

    zeta = np.arctan2(delta_h, geometry_denominator(l_g * r_001))
    l_r = np.where(
        zeta > theta,
        np.maximum(l_g, 0.0) * r_001 / np.maximum(cos_el, TRIG_DENOM_EPS),
        l_s,
    )

    chi = max(36.0 - abs(float(latitude_deg)), 0.0)
    v_001 = 1.0 / (
        1.0
        + np.sqrt(sin_el)
        * (
            31.0
            * (1.0 - np.exp(-theta_deg / (1.0 + chi)))
            * np.sqrt(np.maximum(l_r * gamma_r, 0.0))
            / (freq_ghz ** 2)
            - 0.45
        )
    )
    v_001 = np.clip(v_001, MIN_RAIN_REDUCTION_FACTOR, None)
    l_e = l_r * v_001
    a_001 = gamma_r * l_e

    p = min(max(float(exceedance_pct), 0.001), 5.0)
    if abs(p - 0.01) < PROBABILITY_EQUALITY_TOL:
        return a_001

    lat_abs = abs(float(latitude_deg))
    if p >= 1.0 or lat_abs >= 36.0:
        beta = np.zeros_like(theta, dtype=np.float64)
    else:
        beta = np.where(
            theta_deg >= 25.0,
            -0.005 * (lat_abs - 36.0),
            -0.005 * (lat_abs - 36.0) + 1.8 - 4.25 * sin_el,
        )
    exponent = -(
        0.655
        + 0.033 * np.log(p)
        - 0.045 * np.log(log_argument(a_001))
        - beta * (1.0 - p) * sin_el
    )
    return a_001 * ((p / 0.01) ** exponent)


def doppler_attenuation(nu: np.ndarray, subcarrier_spacing: float) -> np.ndarray:
    if subcarrier_spacing <= 0:
        return np.ones_like(nu, dtype=np.float32)
    return np.sinc(nu / subcarrier_spacing) ** 2


def snr_linear(
    power: float,
    gain: np.ndarray,
    noise_density: float,
    bandwidth: float,
    interference: float | np.ndarray = 0.0,
    *,
    noise_figure_db: float = 0.0,
) -> np.ndarray:
    return safe_snr_linear(
        power,
        gain,
        noise_density,
        bandwidth,
        interference,
        noise_figure_db=noise_figure_db,
    )


def spectral_efficiency(snr: np.ndarray) -> np.ndarray:
    return np.log2(1.0 + snr)


def quantize_array(
    value: np.ndarray | float,
    quantum: float,
    *,
    dtype=np.float32,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    quantum_value = float(quantum)
    if quantum_value <= 0.0:
        return arr.astype(dtype, copy=False)
    scaled = np.rint(arr / quantum_value)
    return np.asarray(scaled * quantum_value, dtype=dtype)
