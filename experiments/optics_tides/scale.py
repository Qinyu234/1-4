"""Earth-scale radii for central-observer optics (geometric anchor).

Fix ``R_central / mean_fairy_sep = r_frac`` (default 0.02), then assign fairy
radii at the same uniform density implied by ``(M_c, R_c)``.
"""

from __future__ import annotations

import math

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed

# SI reference (documentation / optional physical maps later)
R_EARTH_M = 6_371_000.0
M_EARTH_KG = 5.972e24


def fairy_mean_separation(seed: OrbitSeed) -> float:
    r = np.asarray(seed.positions, dtype=float)
    n = int(r.shape[0])
    if n < 2:
        return 1.0
    s = 0.0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += float(np.linalg.norm(r[i] - r[j]))
            c += 1
    return s / max(c, 1)


def earth_scaled_radii(
    seed: OrbitSeed,
    M_c: float,
    *,
    r_frac: float = 0.02,
) -> dict:
    """
    Geometric Earth anchor: ``R_c = r_frac * mean_sep``.

    Same density for fairies: ``R_i = R_c * (m_i / M_c)^(1/3)``.
    """
    Mc = float(M_c)
    if Mc <= 0.0:
        raise ValueError("M_c must be positive for Earth-scaled radii")
    mean_sep = fairy_mean_separation(seed)
    R_c = float(r_frac) * mean_sep
    masses_f = np.asarray(seed.masses, dtype=float)
    R_f = R_c * (masses_f / Mc) ** (1.0 / 3.0)
    rho = 3.0 * Mc / (4.0 * math.pi * R_c**3)
    return {
        "mean_sep": mean_sep,
        "r_frac": float(r_frac),
        "M_c": Mc,
        "R_c": R_c,
        "R_fairies": R_f,
        "radii": np.concatenate([[R_c], R_f]),
        "masses": np.concatenate([[Mc], masses_f]),
        "rho": rho,
        "R_c_over_mean_sep": R_c / mean_sep,
        "observer_approx_ok": (R_c / mean_sep) <= 0.05,
    }
