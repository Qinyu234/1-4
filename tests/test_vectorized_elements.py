"""Vectorized (a,e,i) + AMD fast path matches per-body extraction."""

from __future__ import annotations

import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design.graded import GradedParams, build_planar90_system
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.amd import amd_total_from_aei, extract_amd_series
from fairy_orbit.observe.elements_series import (
    extract_aei_series,
    extract_element_series,
)


def _traj():
    params = GradedParams(rho0=1.0, delta_rho=0.15, v_t=0.9, k=0.05, m=1e-3)
    system = build_planar90_system(params, central_radius=0.0, fairy_radius=0.0)
    return integrate(
        system,
        t_end=8.0,
        n_outputs=60,
        config=ReboundConfig(epsilon=1e-9, min_dt=1e-6, stop_on_collision=False),
    )


def test_vectorized_aei_matches_perbody():
    traj = _traj()
    cfg = SystemConfig(mass_ratio=1e-3)
    a, e, i = extract_aei_series(traj, mu=cfg.mu)
    ref = extract_element_series(traj, mu=cfg.mu)
    assert np.allclose(a, ref.a, rtol=1e-9, atol=1e-9)
    assert np.allclose(e, ref.e, rtol=1e-9, atol=1e-9)
    assert np.allclose(i, ref.i, rtol=1e-9, atol=1e-9)


def test_vectorized_amd_matches_perbody():
    traj = _traj()
    cfg = SystemConfig(mass_ratio=1e-3)
    a, e, i = extract_aei_series(traj, mu=cfg.mu)
    masses = np.full(a.shape[1], cfg.fairy_mass)
    fast = amd_total_from_aei(a, e, i, masses, mu=cfg.mu)
    ref_elems = extract_element_series(traj, mu=cfg.mu)
    ref = extract_amd_series(ref_elems, masses, mu=cfg.mu).amd_total
    assert np.allclose(fast, ref, rtol=1e-9, atol=1e-12)
