"""Path A long-horizon residual (k orbital periods)."""

from __future__ import annotations

import numpy as np

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.observe.continuation import (
    attach_central_mass,
    symmetry_residual_vector,
)


def test_horizon_periods_3_and_4_shapes() -> None:
    seed = build_free_polygon_seed(4, seed_id="poly4_h", family="free_4")
    sys = attach_central_mass(seed, 1e-3)
    f0 = symmetry_residual_vector(
        sys, seed, seed.period, optics_soft=False, horizon_periods=0.0
    )
    f3 = symmetry_residual_vector(
        sys, seed, seed.period, optics_soft=False, horizon_periods=3.0
    )
    f4 = symmetry_residual_vector(
        sys, seed, seed.period, optics_soft=False, horizon_periods=4.0
    )
    assert f0.shape == f3.shape == f4.shape == (6 * seed.n_bodies,)
    assert np.all(np.isfinite(f3))
    assert np.all(np.isfinite(f4))


def test_horizon_default_path_a_is_four() -> None:
    from fairy_orbit.observe.continuation import DEFAULT_PATH_A_HORIZON_PERIODS

    assert DEFAULT_PATH_A_HORIZON_PERIODS == 4.0
