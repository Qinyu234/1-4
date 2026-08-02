"""Brightness-swap segmentation tests."""

from __future__ import annotations

import numpy as np

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.engine.trajectory import Trajectory
from experiments.optics_tides.brightness_swap import (
    find_brightness_swaps,
    segment_by_farthest,
)


def _toy_traj() -> Trajectory:
    # central at 0; two fairies swap who is farthest
    T = 6
    times = np.arange(T, dtype=float)
    pos = np.zeros((T, 3, 3))
    # body0 central
    # body1 far on +x for t<3, then near
    # body2 near then far
    for t in range(T):
        if t < 3:
            pos[t, 1] = [2.0, 0.0, 0.0]
            pos[t, 2] = [0.5, 0.0, 0.0]
        else:
            pos[t, 1] = [0.5, 0.0, 0.0]
            pos[t, 2] = [2.0, 0.0, 0.0]
    vel = np.zeros_like(pos)
    eng = np.zeros(T)
    L = np.zeros((T, 3))
    return Trajectory(
        times=times,
        positions=pos,
        velocities=vel,
        energies=eng,
        angular_momenta=L,
        labels=["C", "A", "B"],
        masses=np.array([1.0, 0.1, 0.1]),
    )


def test_segment_by_farthest_two_runs() -> None:
    traj = _toy_traj()
    segs = segment_by_farthest(traj, central_index=0)
    assert len(segs) == 2
    assert segs[0].farthest_body == 1
    assert segs[1].farthest_body == 2


def test_brightness_swap_exists_per_segment() -> None:
    traj = _toy_traj()
    segs, swaps = find_brightness_swaps(traj, central_index=0, metric="angular")
    assert len(segs) == 2
    assert len(swaps) == 2
    assert swaps[0].body_i != swaps[0].body_j
