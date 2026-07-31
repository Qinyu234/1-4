"""Equal-density radii + visual overlap (perp proxy vs full angular).

``logρ ∈ [-1, 1]`` (log10) ⇒ ``ρ = 10^{logρ} ∈ [0.1, 10]``.
Radii: ``R_i = (3 m_i / (4 π ρ))^(1/3)``.

Two APIs (do not conflate):

* ``optical_overlap_perp`` — valid **only** at gravitational close encounters
  (``|r_A−r_B|`` already small ⇒ ``r_A≈r_B``). Uses sky-plane ``|Δr_perp|``.
* ``optical_overlap_angular`` — any pair / any time (including distant LOS).

Observer default: star center (relative vectors from the central body).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fairy_orbit.engine.trajectory import Trajectory

LOG_RHO_MIN = -1.0
LOG_RHO_MAX = 1.0
DEFAULT_LOG_RHO = 0.0  # ρ = 1


def rho_from_log_rho(log_rho: float) -> float:
    """``ρ = 10^{logρ}`` with ``logρ ∈ [-1, 1]``."""
    x = float(log_rho)
    if x < LOG_RHO_MIN or x > LOG_RHO_MAX:
        raise ValueError(
            f"log_rho={x} outside [{LOG_RHO_MIN}, {LOG_RHO_MAX}]"
        )
    return float(10.0**x)


def radii_from_uniform_density(
    masses: np.ndarray | list[float],
    *,
    log_rho: float = DEFAULT_LOG_RHO,
) -> np.ndarray:
    """``R_i = (3 m_i / (4 π ρ))^(1/3)`` for common density ``ρ=10^{logρ}``."""
    rho = rho_from_log_rho(log_rho)
    m = np.asarray(masses, dtype=float).ravel()
    if np.any(m < 0):
        raise ValueError("masses must be non-negative")
    # R=0 for m=0
    out = np.zeros_like(m, dtype=float)
    pos = m > 0
    out[pos] = (3.0 * m[pos] / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return out


def delta_r_perp(
    r_a: np.ndarray,
    r_b: np.ndarray,
    *,
    observer: np.ndarray | None = None,
) -> float:
    """
    Perpendicular separation of A,B as seen from ``observer`` (default origin).

    ``Δr = r_A - r_B``, ``û ∝ (r_A+r_B)/2`` relative to observer,
    ``Δr_perp = Δr - (Δr·û)û``.
    """
    o = np.zeros(3) if observer is None else np.asarray(observer, dtype=float).ravel()
    ra = np.asarray(r_a, dtype=float).ravel() - o
    rb = np.asarray(r_b, dtype=float).ravel() - o
    dr = ra - rb
    mid = ra + rb
    mid_n = float(np.linalg.norm(mid))
    if mid_n < 1e-300:
        return float(np.linalg.norm(dr))
    uhat = mid / mid_n
    perp = dr - float(np.dot(dr, uhat)) * uhat
    return float(np.linalg.norm(perp))


def optical_overlap_perp(
    r_a: np.ndarray,
    r_b: np.ndarray,
    R_a: float,
    R_b: float,
    *,
    observer: np.ndarray | None = None,
) -> bool:
    """
    Encounter-conditioned visual overlap: ``|Δr_perp| < R_A + R_B``.

    Only valid when A,B are already in a gravitational close encounter
    (``|r_A−r_B|`` small so ``r_A≈r_B``). Do not use for distant LOS alignments.
    """
    return delta_r_perp(r_a, r_b, observer=observer) < float(R_a) + float(R_b)


def angular_separation(
    r_a: np.ndarray,
    r_b: np.ndarray,
    *,
    observer: np.ndarray | None = None,
) -> float:
    """Angle between lines of sight from observer to A and B (radians)."""
    o = np.zeros(3) if observer is None else np.asarray(observer, dtype=float).ravel()
    ra = np.asarray(r_a, dtype=float).ravel() - o
    rb = np.asarray(r_b, dtype=float).ravel() - o
    na = float(np.linalg.norm(ra))
    nb = float(np.linalg.norm(rb))
    if na < 1e-300 or nb < 1e-300:
        return float("nan")
    c = float(np.dot(ra, rb) / (na * nb))
    return float(math.acos(min(1.0, max(-1.0, c))))


def angular_radius(r: np.ndarray, R: float, *, observer: np.ndarray | None = None) -> float:
    """``α = arcsin(clip(R/|r-obs|))``; NaN if body engulfs observer."""
    o = np.zeros(3) if observer is None else np.asarray(observer, dtype=float).ravel()
    rr = np.asarray(r, dtype=float).ravel() - o
    d = float(np.linalg.norm(rr))
    Rf = float(R)
    if d <= Rf or d < 1e-300:
        return float("nan")
    return float(math.asin(min(1.0, Rf / d)))


def optical_overlap_angular(
    r_a: np.ndarray,
    r_b: np.ndarray,
    R_a: float,
    R_b: float,
    *,
    observer: np.ndarray | None = None,
) -> bool:
    """
    General sky-sphere overlap: ``θ < α_A + α_B``.

    Valid for any pair/time, including distant line-of-sight alignments.
    Returns False if either body engulfs the observer (undefined α).
    """
    theta = angular_separation(r_a, r_b, observer=observer)
    aA = angular_radius(r_a, R_a, observer=observer)
    aB = angular_radius(r_b, R_b, observer=observer)
    if not (math.isfinite(theta) and math.isfinite(aA) and math.isfinite(aB)):
        return False
    return theta < aA + aB


def observer_validity(
    R_central: float,
    a: float,
    *,
    max_ratio: float = 0.05,
) -> bool:
    """True if center-observer approx is OK: ``R_central / a ≤ max_ratio``."""
    aa = float(a)
    if aa <= 0.0:
        return False
    return float(R_central) / aa <= float(max_ratio)


def verify_visual_overlap(
    positions: np.ndarray,
    masses: np.ndarray | list[float],
    i: int,
    j: int,
    *,
    log_rho: float = DEFAULT_LOG_RHO,
    mode: Literal["angular", "perp"] = "angular",
    observer_index: int | None = None,
    observer: np.ndarray | None = None,
) -> bool:
    """
    Pairwise visual overlap at one snapshot.

    ``mode="angular"`` — general (any geometry).
    ``mode="perp"`` — encounter-conditioned proxy only.
    Observer: explicit ``observer`` vector, or ``observer_index`` body, else origin.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")
    R = radii_from_uniform_density(masses, log_rho=log_rho)
    if observer is not None:
        obs = np.asarray(observer, dtype=float).ravel()
    elif observer_index is not None:
        obs = pos[int(observer_index)]
    else:
        obs = None
    ra, rb = pos[int(i)], pos[int(j)]
    if mode == "perp":
        return optical_overlap_perp(ra, rb, float(R[i]), float(R[j]), observer=obs)
    if mode == "angular":
        return optical_overlap_angular(ra, rb, float(R[i]), float(R[j]), observer=obs)
    raise ValueError(f"unknown mode {mode!r}")


@dataclass(frozen=True)
class VisualOverlapHit:
    time: float
    i: int
    j: int
    theta: float
    alpha_sum: float
    distance_3d: float


def scan_visual_overlaps(
    trajectory: Trajectory,
    masses: np.ndarray | list[float] | None = None,
    *,
    log_rho: float = DEFAULT_LOG_RHO,
    central_index: int | None = 0,
    fairy_only: bool = True,
) -> list[VisualOverlapHit]:
    """
    Scan trajectory for **angular** sky overlaps (diagnostic; not soft loss).

    If ``central_index`` is set, observer is that body; fairies are all other
    indices when ``fairy_only``. If ``central_index is None``, observer is the
    origin (COM-frame free-N) and pairs are among all bodies.
    """
    m = (
        np.asarray(masses, dtype=float)
        if masses is not None
        else (
            np.asarray(trajectory.masses, dtype=float)
            if trajectory.masses is not None
            else None
        )
    )
    if m is None:
        raise ValueError("masses required (traj.masses missing)")
    R = radii_from_uniform_density(m, log_rho=log_rho)
    N = trajectory.n_bodies
    if central_index is None:
        idxs = list(range(N))
    elif fairy_only:
        idxs = [k for k in range(N) if k != int(central_index)]
    else:
        idxs = [k for k in range(N) if k != int(central_index)]

    hits: list[VisualOverlapHit] = []
    for t in range(len(trajectory)):
        pos = trajectory.positions[t]
        obs = pos[int(central_index)] if central_index is not None else None
        for a, i in enumerate(idxs):
            for j in idxs[a + 1 :]:
                if not optical_overlap_angular(
                    pos[i], pos[j], float(R[i]), float(R[j]), observer=obs
                ):
                    continue
                theta = angular_separation(pos[i], pos[j], observer=obs)
                aA = angular_radius(pos[i], float(R[i]), observer=obs)
                aB = angular_radius(pos[j], float(R[j]), observer=obs)
                hits.append(
                    VisualOverlapHit(
                        time=float(trajectory.times[t]),
                        i=i,
                        j=j,
                        theta=float(theta),
                        alpha_sum=float(aA + aB),
                        distance_3d=float(np.linalg.norm(pos[i] - pos[j])),
                    )
                )
    return hits


def soft_optics_deficit_perp(
    r_a: np.ndarray,
    r_b: np.ndarray,
    R_a: float,
    R_b: float,
    *,
    observer: np.ndarray | None = None,
) -> float:
    """``relu(|Δr_perp| - (R_A+R_B))`` — zero when optically overlapping (perp)."""
    d = delta_r_perp(r_a, r_b, observer=observer)
    return max(0.0, d - (float(R_a) + float(R_b)))
