"""Joint gravity ∧ optical encounter annotation."""

from __future__ import annotations

import numpy as np

from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.encounters import find_encounters_optical
from fairy_orbit.observe.optical_encounter import (
    optical_overlap_angular,
    optical_overlap_perp,
    radii_from_uniform_density,
    scan_visual_overlaps,
)


def _toy_traj_with_close_pair() -> tuple[Trajectory, np.ndarray]:
    # central at 0, two fairies that approach each other at t=1
    times = np.array([0.0, 1.0, 2.0])
    # bodies: C, A, B
    pos = np.zeros((3, 3, 3))
    pos[0, 0] = [0, 0, 0]
    pos[0, 1] = [1.0, 0.2, 0]
    pos[0, 2] = [1.0, -0.2, 0]
    pos[1, 0] = [0, 0, 0]
    pos[1, 1] = [1.0, 0.02, 0]
    pos[1, 2] = [1.0, -0.02, 0]
    pos[2, 0] = [0, 0, 0]
    pos[2, 1] = [1.0, 0.3, 0]
    pos[2, 2] = [1.0, -0.3, 0]
    vel = np.zeros_like(pos)
    masses = np.array([10.0, 1.0, 1.0])
    traj = Trajectory(
        times=times,
        positions=pos,
        velocities=vel,
        energies=np.zeros(3),
        angular_momenta=np.zeros((3, 3)),
        labels=["C", "A", "B"],
        masses=masses,
    )
    return traj, masses


def test_annotate_sets_light_swap_when_perp_ok() -> None:
    traj, masses = _toy_traj_with_close_pair()
    evs = find_encounters_optical(traj, threshold=0.5, central_index=0, masses=masses)
    assert len(evs) >= 1
    mid = [e for e in evs if abs(e.time - 1.0) < 1e-9][0]
    assert mid.optical_ok_perp is True
    assert mid.light_swap is True
    assert mid.delta_r_perp is not None
    assert mid.optical_ok_angular is not None


def test_scan_visual_overlaps_finds_angular() -> None:
    traj, masses = _toy_traj_with_close_pair()
    hits = scan_visual_overlaps(traj, masses, log_rho=0.0, central_index=0)
    assert any(h.i == 1 and h.j == 2 for h in hits)


def test_perp_and_angular_apis_exported() -> None:
    R = radii_from_uniform_density([1.0, 1.0], log_rho=0.0)
    a = np.array([1.0, 0.01, 0.0])
    b = np.array([1.0, -0.01, 0.0])
    assert optical_overlap_perp(a, b, float(R[0]), float(R[1]))
    assert optical_overlap_angular(a, b, float(R[0]), float(R[1]))
