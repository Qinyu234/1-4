"""Tests for Bayesian PEO soft choreography residual and staged escalate."""

from __future__ import annotations

import numpy as np
import pytest

from fairy_orbit.design.manifold import ManifoldParams
from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.bayes import (
    UNLOCK_STAGES,
    BayesSpace,
    _try_escalate,
    bayes_objective_from_peo,
    expand_bayes_space,
    soft_choreography_residual,
)
from fairy_orbit.observe.closure import cyclic_shift
from fairy_orbit.observe.peo import PEOFilterResult
from fairy_orbit.observe.search import FreeParams, SeedAnchors, to_manifold


def _traj_from_orders(orders: list[tuple[int, ...]]) -> Trajectory:
    n_steps = len(orders)
    n_bodies = 5
    pos = np.zeros((n_steps, n_bodies, 3))
    vel = np.zeros((n_steps, n_bodies, 3))
    for k, order in enumerate(orders):
        for rank, fairy_local in enumerate(order):
            pos[k, fairy_local + 1] = np.array([float(rank + 1), 0.0, 0.0])
    return Trajectory(
        times=np.linspace(0, 1, n_steps),
        positions=pos,
        velocities=vel,
        energies=np.zeros(n_steps),
        angular_momenta=np.zeros((n_steps, n_bodies, 3)),
        labels=["central", "T1", "T2", "T3", "T4"],
    )


def test_soft_choreo_zero_on_bcda():
    o0 = (0, 1, 2, 3)
    oT = cyclic_shift(o0, 1)
    traj = _traj_from_orders([o0, oT, oT])
    soft = soft_choreography_residual(traj)
    assert soft["frac_noncyclic"] == 0.0
    assert soft["identity_at_T"] == 0.0
    assert soft["best_k_mismatch"] == 0.0
    assert soft["soft_choreo"] == 0.0


def test_bayes_objective_choreography_uses_soft():
    o0 = (0, 1, 2, 3)
    traj = _traj_from_orders([o0, o0, o0])  # identity at T
    res = PEOFilterResult(
        status="choreography",
        traj=traj,
        closure=None,
        closure_rep=None,
        params=ManifoldParams(),
        summary={"status": "choreography", "reason": "identity_radial_at_T"},
    )
    loss, extras = bayes_objective_from_peo(res)
    assert loss > 100.0
    assert extras["identity_at_T"] == 1.0
    assert extras["soft_choreo"] > 0.0


def test_unlock_order_a_e_m_v():
    assert UNLOCK_STAGES[0] == ()
    assert UNLOCK_STAGES[1] == ("a2",)
    assert UNLOCK_STAGES[2] == ("a2", "e2")
    assert UNLOCK_STAGES[3] == ("a2", "e2", "M2")
    assert UNLOCK_STAGES[4] == ("a2", "e2", "M2", "v1x", "v1y", "v1z")


def test_expand_then_unlock():
    space = BayesSpace(a1=(0.05, 0.20))
    new_space, stage, expands, ev = _try_escalate(
        space=space,
        stage_idx=0,
        n_expands=0,
        max_expands=2,
        expand_grow=0.5,
        trial_index=10,
    )
    assert ev is not None
    assert "expand" in ev.action and "unlock" in ev.action
    assert expands == 1
    assert stage == 1
    assert new_space.a1[1] > space.a1[1]
    assert list(ev.detail["unlocked"]) == ["a2"]

    # Next stagnate: expand again + unlock e2
    space2, stage2, expands2, ev2 = _try_escalate(
        space=new_space,
        stage_idx=1,
        n_expands=1,
        max_expands=2,
        expand_grow=0.5,
        trial_index=20,
    )
    assert ev2 is not None
    assert stage2 == 2
    assert expands2 == 2
    assert list(ev2.detail["unlocked"]) == ["a2", "e2"]


def test_expand_bayes_space_clamps():
    space = BayesSpace(log_m=(-6.0, -2.0))
    wider, changed = expand_bayes_space(space, grow=0.5)
    assert "log_m" in changed
    assert wider.log_m[0] < space.log_m[0]


def test_to_manifold_carries_quad():
    free = FreeParams(0.15, 0.0, 1.0, 0.0, 0.0, 0.0, a2=0.02, M2=0.1, v1x=0.01)
    p = to_manifold(SeedAnchors(1e-3, 0.05), free)
    assert p.a2 == pytest.approx(0.02)
    assert p.M2 == pytest.approx(0.1)
    assert p.v1x == pytest.approx(0.01)
