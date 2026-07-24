"""Orbit quality metrics: permutation period, collision, energy drift."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from fairy_orbit.simulation.runner import Trajectory


@dataclass
class EvalResult:
    score: float
    permutation_error: float
    best_permutation: tuple[int, ...]
    collision_penalty: float
    energy_drift: float
    min_pair_distance: float
    distance_matrix_error: float = 0.0  # New: pairwise distance matrix recovery error


@dataclass
class EvaluatorConfig:
    d_min: float = 0.5
    collision_weight: float = 10.0
    energy_weight: float = 1.0
    energy_threshold: float = 1e-2
    fairy_indices: tuple[int, ...] = (1, 2, 3, 4)
    planet_index: int = 0
    use_distance_matrix: bool = False  # Use pairwise distance matrix recovery instead of permutation


def _pairwise_min_distance(positions: np.ndarray) -> float:
    """positions: (N, 3)"""
    n = positions.shape[0]
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[j] - positions[i]))
            if d < dmin:
                dmin = d
    return float(dmin)


def permutation_error(
    pos0: np.ndarray,
    posT: np.ndarray,
    fairy_indices: tuple[int, ...] = (1, 2, 3, 4),
) -> tuple[float, tuple[int, ...]]:
    """
    Min over sigma in S4 of sum_i ||X_i(T) - X_sigma(i)(0)|| for fairies.
    Returns (error, best_sigma) where sigma maps current fairy order index → initial index.
    """
    x0 = pos0[list(fairy_indices)]
    xT = posT[list(fairy_indices)]
    best_err = np.inf
    best_perm: tuple[int, ...] = tuple(range(len(fairy_indices)))
    for perm in permutations(range(len(fairy_indices))):
        err = 0.0
        for i, j in enumerate(perm):
            err += float(np.linalg.norm(xT[i] - x0[j]))
        if err < best_err:
            best_err = err
            best_perm = perm
    return float(best_err), best_perm


def collision_penalty(
    positions_series: np.ndarray,
    d_min: float,
) -> tuple[float, float]:
    """Return (penalty, global_min_distance) over the whole trajectory."""
    global_min = np.inf
    for t in range(positions_series.shape[0]):
        d = _pairwise_min_distance(positions_series[t])
        if d < global_min:
            global_min = d
    if global_min >= d_min:
        return 0.0, float(global_min)
    # Soft penalty growing as distance falls below d_min
    penalty = (d_min / max(global_min, 1e-12) - 1.0) ** 2
    return float(penalty), float(global_min)


def energy_drift(energies: np.ndarray) -> float:
    e0 = energies[0]
    return float(np.max(np.abs(energies - e0)) / max(abs(e0), 1e-30))


def distance_matrix_error(
    pos0: np.ndarray,
    posT: np.ndarray,
) -> float:
    """
    Compute pairwise distance matrix recovery error.
    L = Σ_{i<j} (D_ij(T) - D_ij(0))²
    This is continuous and permutation-invariant.
    """
    n = pos0.shape[0]
    error = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d0 = float(np.linalg.norm(pos0[i] - pos0[j]))
            dT = float(np.linalg.norm(posT[i] - posT[j]))
            error += (dT - d0) ** 2
    return float(error)


def evaluate(
    traj: Trajectory,
    config: EvaluatorConfig | None = None,
    period_index: int | None = None,
) -> EvalResult:
    """
    Score a trajectory. If period_index is None, use the final frame as T.
    """
    if config is None:
        config = EvaluatorConfig()
    if period_index is None:
        period_index = len(traj.times) - 1

    pos0 = traj.positions[0]
    posT = traj.positions[period_index]
    col_pen, dmin = collision_penalty(traj.positions, config.d_min)
    e_drift = energy_drift(traj.energies)

    # Extra energy penalty only above threshold
    e_term = max(0.0, e_drift - config.energy_threshold)

    if config.use_distance_matrix:
        # Use continuous distance matrix error instead of permutation
        dist_err = distance_matrix_error(pos0, posT)
        score = dist_err + config.collision_weight * col_pen + config.energy_weight * e_term
        perm_err = 0.0
        best_perm = tuple(range(len(config.fairy_indices)))
    else:
        # Use original permutation-based error
        perm_err, best_perm = permutation_error(pos0, posT, config.fairy_indices)
        score = perm_err + config.collision_weight * col_pen + config.energy_weight * e_term
        dist_err = 0.0

    return EvalResult(
        score=float(score),
        permutation_error=perm_err,
        best_permutation=best_perm,
        collision_penalty=col_pen,
        energy_drift=e_drift,
        min_pair_distance=dmin,
        distance_matrix_error=dist_err,
    )
