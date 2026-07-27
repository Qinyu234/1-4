"""PROMPT §2.4.1: Rodrigues tetrahedron → ε_numerical(N) noise floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.amd import extract_amd_series
from fairy_orbit.observe.elements_series import extract_element_series


def pairwise_edge_lengths(positions: np.ndarray) -> np.ndarray:
    """Six edge lengths of a 4-body tetrahedron. positions shape (4, 3)."""
    pos = np.asarray(positions, dtype=float).reshape(4, 3)
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(float(np.linalg.norm(pos[i] - pos[j])))
    return np.asarray(edges, dtype=float)


def tetrahedron_shape_error(positions: np.ndarray) -> float:
    """
    Relative edge-length scatter of a regular tetrahedron.

    PROMPT §2.4.1: under Rodrigues point-group IC, positions remain regular-
    tetrahedron vertices at every time ⇒ theory error = 0. Any growth is
    pure numerical noise (ε_numerical).
    """
    edges = pairwise_edge_lengths(positions)
    mean = float(np.mean(edges))
    if mean < 1e-30:
        return float("nan")
    return float(np.std(edges) / mean)


def gram_matrix_unit(positions: np.ndarray) -> np.ndarray:
    """4×4 Gram matrix G_ij = r̂_i · r̂_j of fairy positions (shape (4, 3))."""
    pos = np.asarray(positions, dtype=float).reshape(4, 3)
    norms = np.linalg.norm(pos, axis=1)
    hats = pos / np.maximum(norms[:, None], 1e-30)
    return hats @ hats.T


def td_breaking(positions: np.ndarray) -> float:
    """
    Td symmetry-breaking measure from the unit Gram matrix.

    Ideal regular tetrahedron: G_ij = −1/3 for i ≠ j (and 1 on diagonal).

        D_Td = √[ Σ_{i≠j} (G_ij + 1/3)² ]

    = 0 exactly on the Td manifold; O(1) once the configuration leaves it.
    """
    G = gram_matrix_unit(positions)
    off = G + (1.0 / 3.0)
    np.fill_diagonal(off, 0.0)
    return float(np.sqrt(np.sum(off * off)))


def radius_equality_error(positions: np.ndarray) -> float:
    """std(|r_i|) / mean(|r_i|) — all four must share the same radial shell."""
    pos = np.asarray(positions, dtype=float).reshape(4, 3)
    radii = np.linalg.norm(pos, axis=1)
    mean = float(np.mean(radii))
    if mean < 1e-30:
        return float("nan")
    return float(np.std(radii) / mean)


@dataclass
class CalibrationSeries:
    times: np.ndarray
    orbit_index: np.ndarray  # N = t / T_ref
    shape_error: np.ndarray  # primary ε_numerical (regular-tetra edge CV)
    radius_error: np.ndarray
    amd_total: np.ndarray
    amd_drift: np.ndarray


def measure_calibration(
    traj: Trajectory,
    *,
    mu: float,
    fairy_masses: np.ndarray | None = None,
    period_ref: float | None = None,
) -> CalibrationSeries:
    """Build ε_numerical(N) from a Rodrigues same-(a,e) tetrahedron run."""
    if traj.n_bodies < 5:
        raise ValueError("need central + 4 fairies")

    # Prefer fairy-centric positions (subtract central) so COM drift of the
    # whole system does not fake shape error.
    central = traj.positions[:, 0:1, :]
    fairy_pos = traj.positions[:, 1:5, :] - central

    T = fairy_pos.shape[0]
    shape = np.empty(T, dtype=float)
    radius = np.empty(T, dtype=float)
    for k in range(T):
        shape[k] = tetrahedron_shape_error(fairy_pos[k])
        radius[k] = radius_equality_error(fairy_pos[k])

    elements = extract_element_series(traj, mu=mu)
    if fairy_masses is None:
        if traj.masses is not None and traj.masses.shape[0] == traj.n_bodies:
            fairy_masses = np.asarray(traj.masses[1:5], dtype=float)
        else:
            fairy_masses = np.ones(4, dtype=float)
    amd = extract_amd_series(elements, fairy_masses, mu=mu)
    amd0 = float(amd.amd_total[0]) if len(amd.amd_total) else 0.0
    amd_drift = np.abs(amd.amd_total - amd0)

    if period_ref is None or period_ref <= 0.0:
        a0 = float(elements.a[0, 0]) if elements.a.size else 1.0
        period_ref = float(2.0 * np.pi * np.sqrt(max(a0, 1e-30) ** 3 / mu))

    return CalibrationSeries(
        times=traj.times.copy(),
        orbit_index=traj.times / float(period_ref),
        shape_error=shape,
        radius_error=radius,
        amd_total=amd.amd_total.copy(),
        amd_drift=amd_drift,
    )


def epsilon_at_orbit(series: CalibrationSeries, n_orbit: float) -> dict[str, float]:
    """Interpolate noise floors at orbit index N."""
    if len(series.orbit_index) < 2:
        return {
            "n_orbit": float(n_orbit),
            "shape_error": float(series.shape_error[0]) if len(series.shape_error) else float("nan"),
            "amd_drift": float(series.amd_drift[0]) if len(series.amd_drift) else float("nan"),
        }
    return {
        "n_orbit": float(n_orbit),
        "shape_error": float(np.interp(n_orbit, series.orbit_index, series.shape_error)),
        "amd_drift": float(np.interp(n_orbit, series.orbit_index, series.amd_drift)),
    }
