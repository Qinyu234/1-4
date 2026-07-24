"""Angular Momentum Deficit (AMD) diagnostics — Laskar-style secular exchange proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.observe.elements_series import ElementSeries


@dataclass
class AMDSeries:
    times: np.ndarray
    labels: list[str]
    amd: np.ndarray  # (T, N_fairy)
    amd_total: np.ndarray  # (T,)


def circular_angular_momentum(mass: float, mu: float, a: float) -> float:
    """L_circ = m √(μ a) for a circular planar orbit about central mass."""
    if a <= 0.0:
        return 0.0
    return float(mass * np.sqrt(mu * a))


def amd_of_elements(
    mass: float,
    mu: float,
    a: float,
    e: float,
    i: float,
) -> float:
    """
    AMD = L_circ - L = m √(μ a) [ 1 - √(1-e²) cos i ].

    Measures how much angular momentum is 'spent' on eccentricity/inclination.
    """
    if a <= 0.0 or e >= 1.0:
        return float("nan")
    Lcirc = circular_angular_momentum(mass, mu, a)
    return float(Lcirc * (1.0 - np.sqrt(max(0.0, 1.0 - e * e)) * np.cos(i)))


def extract_amd_series(
    elements: ElementSeries,
    masses: np.ndarray,
    mu: float,
) -> AMDSeries:
    """Per-fairy AMD time series from osculating elements."""
    T, n = elements.a.shape
    if masses.shape[0] != n:
        raise ValueError("masses length must match number of fairies")
    amd = np.zeros((T, n), dtype=float)
    for j in range(n):
        m = float(masses[j])
        for t in range(T):
            amd[t, j] = amd_of_elements(
                m,
                mu,
                float(elements.a[t, j]),
                float(elements.e[t, j]),
                float(elements.i[t, j]),
            )
    return AMDSeries(
        times=elements.times.copy(),
        labels=list(elements.labels),
        amd=amd,
        amd_total=np.nansum(amd, axis=1),
    )
