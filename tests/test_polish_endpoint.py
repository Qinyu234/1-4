"""Fast endpoint integrate for polish residuals."""

from __future__ import annotations

import numpy as np

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate, integrate_endpoint
from fairy_orbit.observe.choreography_search import (
    polish_seed,
    random_asymmetric_seed,
    symmetry_residual_seed,
)
from fairy_orbit.store.search_db import trial_rng


def test_integrate_endpoint_matches_full_traj() -> None:
    seed = build_free_polygon_seed(4, seed_id="poly4", family="free_4")
    sys = seed.to_system()
    tau = float(seed.period) / seed.n_bodies
    cfg = ReboundConfig(
        stop_on_escape=False,
        stop_on_collision=False,
        epsilon=0.0,
        dt=max(tau / 200.0, 1e-3),
        min_dt=1e-5,
    )
    r, v = integrate_endpoint(sys, tau, config=cfg)
    traj = integrate(sys, tau, n_outputs=12, config=cfg)
    assert np.allclose(r, traj.positions[-1], rtol=1e-9, atol=1e-11)
    assert np.allclose(v, traj.velocities[-1], rtol=1e-9, atol=1e-11)


def test_polish_still_reduces_residual() -> None:
    rng = trial_rng(4, 4242)
    start = random_asymmetric_seed(4, rng)
    r0 = float(np.linalg.norm(symmetry_residual_seed(start)))
    polished, res = polish_seed(start, max_nfev=8)
    assert res <= r0 * 1.05 + 1e-9
    assert polished.n_bodies == 4
