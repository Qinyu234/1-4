"""Tests for OrbitEvaluator (T6)."""

import numpy as np

from fairy_orbit.analysis.evaluator import (
    EvaluatorConfig,
    collision_penalty,
    energy_drift,
    evaluate,
    permutation_error,
)
from fairy_orbit.simulation.runner import Trajectory


def _toy_traj(positions: np.ndarray, energies: np.ndarray | None = None) -> Trajectory:
    t = positions.shape[0]
    if energies is None:
        energies = np.ones(t)
    return Trajectory(
        times=np.arange(t, dtype=float),
        positions=positions,
        velocities=np.zeros_like(positions),
        energies=energies,
        angular_momenta=np.zeros((t, 3)),
        labels=["Planet", "A", "B", "C", "D"],
    )


def test_permutation_identity_zero():
    pos0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    err, perm = permutation_error(pos0, pos0.copy())
    assert err == 0.0
    assert perm == (0, 1, 2, 3)


def test_permutation_cycle_detected():
    pos0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    # Cycle A->B->C->D->A among fairies (indices 1..4)
    posT = pos0.copy()
    posT[1] = pos0[4]  # A gets old D
    posT[2] = pos0[1]  # B gets old A
    posT[3] = pos0[2]  # C gets old B
    posT[4] = pos0[3]  # D gets old C
    err, _ = permutation_error(pos0, posT)
    np.testing.assert_allclose(err, 0.0, atol=1e-12)


def test_collision_penalty_triggers():
    positions = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        ]
    )
    pen, dmin = collision_penalty(positions, d_min=0.5)
    assert dmin < 0.5
    assert pen > 0.0


def test_energy_drift():
    energies = np.array([1.0, 1.0, 1.05])
    assert abs(energy_drift(energies) - 0.05) < 1e-12


def test_evaluate_combined():
    pos = np.zeros((2, 5, 3))
    pos[0] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ]
    pos[1] = pos[0].copy()
    traj = _toy_traj(pos, energies=np.array([1.0, 1.0]))
    result = evaluate(traj, EvaluatorConfig(d_min=0.1))
    assert result.score == 0.0
