"""Planar alternating pro/retro ladder IC (PROMPT §2.4 stage 1)."""

from __future__ import annotations

import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, OrbitalElements, build_orbital_ladder
from fairy_orbit.design.ladder import ALTERNATING_SENSE, ladder_period_ratios


def test_planar_alternating_is_coplanar():
    cfg = SystemConfig(mass_ratio=1e-6)
    system = build_orbital_ladder(
        cfg, LadderParams(geometry="planar_alternating", eccentricity=0.12)
    )
    for body in system.bodies[1:]:
        assert abs(body.position[2]) < 1e-12
        assert abs(body.velocity[2]) < 1e-12


def test_planar_alternating_senses():
    cfg = SystemConfig(mass_ratio=1e-6)
    system = build_orbital_ladder(
        cfg, LadderParams(geometry="planar_alternating", eccentricity=0.1)
    )
    central = system.bodies[0]
    for body in system.bodies[1:]:
        r = body.position - central.position
        v = body.velocity - central.velocity
        hz = float(np.cross(r, v)[2])
        want = ALTERNATING_SENSE[body.name]
        assert np.sign(hz) == want
        el = OrbitalElements.from_state(r, v, cfg.mu)
        if want > 0:
            assert el.i < 0.05
        else:
            assert el.i > np.pi - 0.05


def test_planar_alternating_period_ratios():
    cfg = SystemConfig(mass_ratio=1e-6)
    params = LadderParams(geometry="planar_alternating", eccentricity=0.1, a_inner=1.0)
    system = build_orbital_ladder(cfg, params)
    ratios = ladder_period_ratios(system, cfg)
    for got, want in zip(ratios, params.period_ratios, strict=True):
        assert abs(got - want) / want < 0.02
