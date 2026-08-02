"""Equal-albedo photometry via starry; evenness = period variance."""

from __future__ import annotations

import numpy as np
import pytest

from fairy_orbit.engine.trajectory import Trajectory
from experiments.optics_tides.photometry import (
    flux_at_frame,
    period_variance_stats,
    rotate_sun_at_swap,
    starry_reflected_flux,
    sun_role_timeline,
)


def test_rotate_sun_transfers_across_pair() -> None:
    assert rotate_sun_at_swap(1, 1, 3) == 3
    assert rotate_sun_at_swap(2, 1, 3) == 2


def test_no_rotate_keeps_sun_fixed() -> None:
    class S:
        def __init__(self, frame: int, i: int, j: int) -> None:
            self.swap_frame = frame
            self.body_i = i
            self.body_j = j

    series = sun_role_timeline(
        5, [S(2, 1, 3)], initial_sun=1, role_rotate=False
    )
    assert list(series) == [1, 1, 1, 1, 1]


def test_period_variance_stats() -> None:
    s = period_variance_stats(np.array([1.0, 1.0, 1.0]))
    assert s["F_var"] == 0.0
    assert s["F_cv"] == 0.0


def test_starry_full_phase_scaling() -> None:
    pytest.importorskip("starry")
    # Body on +x, far sun along −x → full phase from Planet at origin
    L, alpha = starry_reflected_flux(
        0.1,
        0.25,
        np.array([2.0, 0.0, 0.0]),
        s_hat=np.array([-1.0, 0.0, 0.0]),
        illumination="far",
    )
    assert alpha < 1.0
    # (2/3)*A*R² at Φ≈1
    expect = (2.0 / 3.0) * 0.25 * (0.1**2)
    assert abs(L - expect) / expect < 1e-6


def test_equal_A_flux() -> None:
    pytest.importorskip("starry")
    pos = np.zeros((1, 5, 3))
    pos[0, 1] = [2, 0, 0]
    pos[0, 2] = [0, 1, 0]
    pos[0, 3] = [0, -1, 0]
    pos[0, 4] = [-1, 0, 0]
    traj = Trajectory(
        times=np.array([0.0]),
        positions=pos,
        velocities=np.zeros_like(pos),
        energies=np.zeros(1),
        angular_momenta=np.zeros((1, 3)),
        labels=["C", "S", "M1", "M2", "M3"],
        masses=np.ones(5),
    )
    f = flux_at_frame(
        traj,
        0,
        np.array([0.05, 0.2, 0.1, 0.1, 0.1]),
        sun_index=1,
        central_index=0,
        L_sun=1.0,
        A=0.25,
        illumination="far",
        occultation=False,
    )
    assert f.engine == "starry"
    assert abs(f.F_total - (f.F_sun + f.F_moons)) < 1e-12
    assert f.L_fairies[0] == 0.0  # Sun body: no reflected self-term
