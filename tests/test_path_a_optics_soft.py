"""Path-A residual soft extras for gravity ∧ optical channels."""

from __future__ import annotations

import numpy as np

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.observe.continuation import (
    attach_central_mass,
    symmetry_residual_vector,
)


def test_optics_soft_appended_when_central_present() -> None:
    seed = build_free_polygon_seed(4, seed_id="poly4", family="free_4")
    sys0 = attach_central_mass(seed, 0.0)
    f0 = symmetry_residual_vector(sys0, seed, seed.period, optics_soft=True)
    # no central → no extras (length 6N = 24)
    assert f0.shape[0] == 6 * seed.n_bodies

    sys_m = attach_central_mass(seed, 1e-3)
    f_m = symmetry_residual_vector(
        sys_m, seed, seed.period, optics_soft=True, log_rho=0.0
    )
    assert f_m.shape[0] == 6 * seed.n_bodies + 2
    assert np.all(np.isfinite(f_m))


def test_optics_soft_can_disable() -> None:
    seed = build_free_polygon_seed(4, seed_id="poly4", family="free_4")
    sys_m = attach_central_mass(seed, 1e-3)
    f = symmetry_residual_vector(sys_m, seed, seed.period, optics_soft=False)
    assert f.shape[0] == 6 * seed.n_bodies
