"""Tests for radial choreography gate (Level 1 hard reject)."""

from __future__ import annotations

import numpy as np

from fairy_orbit.observe.closure import (
    choreography_gate,
    cyclic_shift,
    radial_choreography_shift,
)
from fairy_orbit.engine.trajectory import Trajectory


def _traj_from_orders(orders: list[tuple[int, ...]]) -> Trajectory:
    """Build fake trajectory where radial rank at k is orders[k] (fairy local idx)."""
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


def test_cyclic_shift_bcda():
    s0 = (0, 1, 2, 3)
    assert cyclic_shift(s0, 1) == (1, 2, 3, 0)
    assert radial_choreography_shift(s0, (1, 2, 3, 0)) == 1


def test_choreography_gate_rejects_identity_at_T():
    order_0 = (0, 2, 1, 3)
    traj = _traj_from_orders([order_0, order_0, order_0])
    gate = choreography_gate(traj, "fixed_radial")
    assert not gate.ok
    assert gate.reason == "identity_radial_at_T"


def test_choreography_gate_accepts_bcda_at_T():
    order_0 = (0, 2, 1, 3)
    order_T = cyclic_shift(order_0, 1)
    traj = _traj_from_orders([order_0, order_0, order_T])
    gate = choreography_gate(traj, "fixed_radial")
    assert gate.ok
    assert gate.shift_final == 1


def test_choreography_gate_rejects_non_cyclic_midpoint():
    order_0 = (0, 1, 2, 3)
    bad_mid = (1, 0, 2, 3)  # swap, not cyclic
    traj = _traj_from_orders([order_0, bad_mid, cyclic_shift(order_0, 1)])
    gate = choreography_gate(traj, "fixed_radial")
    assert not gate.ok
    assert "non_cyclic" in gate.reason
