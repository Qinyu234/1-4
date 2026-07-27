"""Orbit error base = position/velocity closure E_r, E_v over time.

PROMPT definition (P fixed first):

    R* = argmin_R Σ ||r_i(T) − R r_{P(i)}(0)||²
    E_r = that min;  E_v = Σ ||v_i(T) − R* v_{P(i)}(0)||²

Exponential normalize is a secondary envelope on those errors
(defaults from legacy Td dense fit until re-calibrated on E_r).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Placeholder envelope until calibrate_error_base re-fits on E_r(t).
DEFAULT_LAM = 23.5
DEFAULT_EPS0 = 1.64e-6
LN10 = math.log(10.0)


@dataclass(frozen=True)
class ExpErrorBase:
    """Reference exponential envelope err_base(t) = ε0 · exp(λ t)."""

    lam: float = DEFAULT_LAM
    eps0: float = DEFAULT_EPS0
    source: str = "placeholder_until_Er_calibration"

    def envelope(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        out = self.eps0 * np.exp(self.lam * t_arr)
        if np.isscalar(t):
            return float(out)
        return out

    def normalize(self, err: float | np.ndarray, t: float | np.ndarray) -> float | np.ndarray:
        env = self.envelope(t)
        err_arr = np.asarray(err, dtype=float)
        out = err_arr / np.maximum(env, 1e-300)
        if np.isscalar(err) and np.isscalar(t):
            return float(out)
        return out

    def log_excess(
        self, err: float | np.ndarray, t: float | np.ndarray
    ) -> float | np.ndarray:
        err_arr = np.asarray(err, dtype=float)
        t_arr = np.asarray(t, dtype=float)
        out = (
            np.log10(np.maximum(err_arr, 1e-300))
            - math.log10(max(self.eps0, 1e-300))
            - (self.lam / LN10) * t_arr
        )
        if np.isscalar(err) and np.isscalar(t):
            return float(out)
        return out


DEFAULT_ERROR_BASE = ExpErrorBase()


def normalize_error(
    err: float | np.ndarray,
    t: float | np.ndarray,
    base: ExpErrorBase | None = None,
) -> float | np.ndarray:
    return (base or DEFAULT_ERROR_BASE).normalize(err, t)


def log_excess_error(
    err: float | np.ndarray,
    t: float | np.ndarray,
    base: ExpErrorBase | None = None,
) -> float | np.ndarray:
    return (base or DEFAULT_ERROR_BASE).log_excess(err, t)


def closure_drift_summary(E_r: np.ndarray, E_v: np.ndarray, times: np.ndarray) -> dict:
    """Compact drift stats for shape/velocity scores vs time."""
    E_r = np.asarray(E_r, dtype=float)
    E_v = np.asarray(E_v, dtype=float)
    times = np.asarray(times, dtype=float)
    return {
        "E_r_0": float(E_r[0]),
        "E_v_0": float(E_v[0]),
        "E_r_final": float(E_r[-1]),
        "E_v_final": float(E_v[-1]),
        "E_r_max": float(np.nanmax(E_r)),
        "E_v_max": float(np.nanmax(E_v)),
        "E_r_ptp": float(np.nanmax(E_r) - np.nanmin(E_r)),
        "E_v_ptp": float(np.nanmax(E_v) - np.nanmin(E_v)),
        "t_span": float(times[-1] - times[0]),
    }
