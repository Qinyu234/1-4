"""Tests for SO(3) Kabsch and position/velocity closure errors E_r, E_v."""

from __future__ import annotations

import numpy as np
import pytest

from fairy_orbit.observe.closure import (
    E_r,
    E_v,
    best_closure_by_Er,
    closure_for_perm,
    kabsch_rotation,
    radial_order,
)


def test_kabsch_recovers_known_rotation():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(4, 3))
    R_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    P = (R_true @ Q.T).T
    R = kabsch_rotation(P, Q)
    assert np.allclose(R @ Q.T, P.T, atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_Er_zero_on_identity():
    X = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    ) / np.sqrt(3.0)
    R = np.eye(3)
    P = (0, 1, 2, 3)
    assert E_r(X, X, R, P) == pytest.approx(0.0, abs=1e-14)
    assert E_v(X, X, R, P) == pytest.approx(0.0, abs=1e-14)


def test_closure_for_perm_uses_same_R_for_Ev():
    rng = np.random.default_rng(2)
    X0 = rng.normal(size=(4, 3))
    V0 = rng.normal(size=(4, 3))
    perm = (0, 1, 2, 3)
    R_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    X = (R_true @ X0.T).T
    # Velocity deliberately not matching position R — Ev should still use R* from positions
    V = V0 + 0.1
    res = closure_for_perm(X, V, X0, V0, perm)
    assert res.E_r == pytest.approx(0.0, abs=1e-10)
    # Ev = Σ ||V_i − R* V0_i||² with R* ≈ R_true
    expected = E_v(V, V0, res.R, perm)
    assert res.E_v == pytest.approx(expected, abs=1e-14)
    assert res.E_v > 0.0


def test_best_closure_finds_permutation_and_rotation():
    rng = np.random.default_rng(1)
    X0 = rng.normal(size=(4, 3))
    perm = (1, 2, 3, 0)
    R_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    X = np.stack([(R_true @ X0[perm[i]]) for i in range(4)])
    V0 = rng.normal(size=(4, 3))
    V = np.stack([(R_true @ V0[perm[i]]) for i in range(4)])
    res = best_closure_by_Er(X, V, X0, V0)
    assert res.E_r == pytest.approx(0.0, abs=1e-10)
    assert res.E_v == pytest.approx(0.0, abs=1e-10)
    assert tuple(res.perm) == perm


def test_radial_order_sorted_by_radius():
    pos = np.array(
        [
            [0.0, 0.0, 0.0],  # central
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    order = radial_order(pos, central_index=0)
    assert order == (1, 2, 0, 3)


def test_closure_series_zero_at_t0_identity():
    from fairy_orbit.design.manifold import ManifoldParams, build_manifold_system
    from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
    from fairy_orbit.observe.closure import closure_series

    sys = build_manifold_system(ManifoldParams())
    traj = integrate(
        sys,
        t_end=2.0,
        n_outputs=40,
        config=ReboundConfig(stop_on_collision=False, min_dt=1e-6),
    )
    s = closure_series(traj, mode="identity")
    assert s.E_r[0] == pytest.approx(0.0, abs=1e-12)
    assert s.E_v[0] == pytest.approx(0.0, abs=1e-12)
    assert s.perm == (0, 1, 2, 3)
