"""Resonance-angle diagnostics for adjacent ladder pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.observe.elements_series import ElementSeries


@dataclass
class ResonanceSeries:
    """Resonance angles for adjacent fairy pairs with integer ratios (p:q)."""

    times: np.ndarray
    pair_labels: list[str]
    ratios: list[tuple[int, int]]  # (p, q) meaning p*n_inner ≈ q*n_outer
    angles: np.ndarray  # (T, n_pairs) — φ = p λ_inner − q λ_outer


def _mean_longitude(omega: np.ndarray, Omega: np.ndarray, M: np.ndarray) -> np.ndarray:
    return omega + Omega + M


def _nearest_integers(period_ratio: float) -> tuple[int, int]:
    """Approximate period_ratio ≈ p/q with small integers."""
    best = (3, 2)
    best_err = abs(period_ratio - 1.5)
    for q in range(1, 8):
        for p in range(q + 1, 12):
            err = abs(period_ratio - p / q)
            if err < best_err:
                best_err = err
                best = (p, q)
    return best


def resonance_angles(
    elements: ElementSeries,
    period_ratios: list[float] | None = None,
) -> ResonanceSeries:
    """
    Compute adjacent-pair resonance angles φ = p λ_i − q λ_{i+1}.

    Default period_ratios come from mean a via Kepler if not supplied.
    """
    T, n_f = elements.a.shape
    if n_f < 2:
        return ResonanceSeries(
            times=elements.times.copy(),
            pair_labels=[],
            ratios=[],
            angles=np.zeros((T, 0)),
        )

    if period_ratios is None:
        # Use time-averaged a to estimate period ratios
        a_mean = np.nanmean(elements.a, axis=0)
        periods = a_mean**1.5
        period_ratios = [float(periods[i + 1] / periods[i]) for i in range(n_f - 1)]

    ratios: list[tuple[int, int]] = []
    pair_labels: list[str] = []
    angles = np.zeros((T, n_f - 1))

    lam = _mean_longitude(elements.omega, elements.Omega, elements.M)

    for i, pr in enumerate(period_ratios):
        p, q = _nearest_integers(pr)
        ratios.append((p, q))
        pair_labels.append(f"{elements.labels[i]}:{elements.labels[i + 1]} ({p}:{q})")
        # For outer/inner period ≈ p/q, resonant angle uses p λ_inner − q λ_outer
        # Wait: if T_outer/T_inner ≈ p/q with p>q, then n_inner/n_outer ≈ p/q
        # so p n_outer ≈ q n_inner → φ = q λ_inner − p λ_outer
        # Common MMR notation for period ratio ≈ p:q (outer:inner wait)...
        # PROMPT says adjacent period ratios 3:2 meaning T2/T1 ≈ 3/2, so outer slower.
        # Resonance: (p+q) λ_inner - p λ_outer - ... For first-order p:q = 3:2,
        # φ = 3 λ_inner - 2 λ_outer is wrong; standard is φ = 2 λ_outer - 3 λ_inner for 3:2?
        # Kepler: n ∝ a^{-3/2}. If T_out/T_in = 3/2 then n_in/n_out = 3/2.
        # So 3 n_out ≈ 2 n_in → φ = 2 λ_in - 3 λ_out.
        angles[:, i] = np.unwrap(q * lam[:, i] - p * lam[:, i + 1])

    return ResonanceSeries(
        times=elements.times.copy(),
        pair_labels=pair_labels,
        ratios=ratios,
        angles=angles,
    )
