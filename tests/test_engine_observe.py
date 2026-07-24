"""REBOUND engine and observe diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder
from fairy_orbit.engine import ReboundConfig, integrate
from fairy_orbit.engine.rebound_engine import REBOUND_AVAILABLE, compute_megno
from fairy_orbit.observe import diagnose, extract_element_series, find_encounters


pytestmark = pytest.mark.skipif(not REBOUND_AVAILABLE, reason="rebound not installed")


def test_integrate_records_trajectory():
    cfg = SystemConfig(mass_ratio=1e-6)
    system = build_orbital_ladder(cfg, LadderParams(eccentricity=0.05))
    traj = integrate(system, t_end=5.0, n_outputs=50, config=ReboundConfig())
    assert len(traj) == 50
    assert traj.positions.shape == (50, 5, 3)
    assert traj.status in {"success", "collision", "escape"}


def test_element_series_length_matches():
    cfg = SystemConfig(mass_ratio=1e-6)
    system = build_orbital_ladder(cfg, LadderParams(eccentricity=0.05))
    traj = integrate(system, t_end=5.0, n_outputs=40)
    el = extract_element_series(traj, mu=cfg.mu)
    assert el.a.shape == (len(traj), 4)
    assert el.e.shape == (len(traj), 4)
    assert len(el.labels) == 4


def test_megno_callable():
    cfg = SystemConfig(mass_ratio=1e-6)
    system = build_orbital_ladder(cfg, LadderParams(eccentricity=0.05))
    val = compute_megno(system, t_end=10.0)
    assert np.isfinite(val)


def test_diagnose_smoke():
    cfg = SystemConfig(mass_ratio=1e-6)
    params = LadderParams(eccentricity=0.1)
    system = build_orbital_ladder(cfg, params)
    d = diagnose(
        system,
        cfg,
        t_end=10.0,
        n_outputs=80,
        ladder=params,
        run_megno=False,
    )
    assert d.summary["n_samples"] == len(d.trajectory)
    assert "a_initial" in d.summary
    _ = find_encounters(d.trajectory, threshold=0.5)
