"""Orbital ladder IC and Kepler elements."""

from __future__ import annotations

import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import (
    LadderParams,
    OrbitalElements,
    build_orbital_ladder,
    orbital_period,
    tetrahedral_phase_offsets,
)
from fairy_orbit.design.ladder import ladder_period_ratios


def test_elements_roundtrip_circular():
    mu = 1.0
    el = OrbitalElements(a=1.0, e=0.0, M=0.3)
    r, v = el.to_state(mu)
    back = OrbitalElements.from_state(r, v, mu)
    assert abs(back.a - 1.0) < 1e-8
    assert abs(back.e) < 1e-8
    assert abs(orbital_period(1.0, mu) - 2 * np.pi) < 1e-10


def test_tetrahedral_phases_are_distinct():
    phases = tetrahedral_phase_offsets()
    vals = list(phases.values())
    assert len(vals) == 4
    assert len(set(np.round(vals, 8))) == 4


def test_ladder_period_ratios():
    cfg = SystemConfig(mass_ratio=1e-6)
    params = LadderParams(eccentricity=0.1, a_inner=1.0, tetrahedral=True)
    system = build_orbital_ladder(cfg, params)
    assert system.n == 5
    ratios = ladder_period_ratios(system, cfg)
    targets = list(params.period_ratios)
    assert len(ratios) == 3
    for got, want in zip(ratios, targets, strict=True):
        assert abs(got - want) / want < 0.02


def test_ladder_shared_eccentricity():
    cfg = SystemConfig(mass_ratio=1e-6)
    params = LadderParams(eccentricity=0.2, tetrahedral=True)
    system = build_orbital_ladder(cfg, params)
    mu = cfg.mu
    central = system.bodies[0]
    for body in system.bodies[1:]:
        el = OrbitalElements.from_state(
            body.position - central.position,
            body.velocity - central.velocity,
            mu,
        )
        assert abs(el.e - 0.2) < 1e-8


def test_tetrahedral_ladder_is_non_coplanar():
    """PROMPT §5: periapses along tetrahedron vertices — not four flat i=0 rings."""
    from fairy_orbit.design.tetrahedron import VERTICES

    cfg = SystemConfig(mass_ratio=1e-6)
    system = build_orbital_ladder(cfg, LadderParams(eccentricity=0.15, tetrahedral=True))
    central = system.bodies[0]
    omegas_node = []
    hs = []
    for body in system.bodies[1:]:
        r = body.position - central.position
        v = body.velocity - central.velocity
        el = OrbitalElements.from_state(r, v, cfg.mu)
        assert el.i > 0.5  # not equatorial
        omegas_node.append(el.Omega)
        hs.append(np.cross(r, v))
        rhat = body.position / np.linalg.norm(body.position)
        assert float(np.dot(rhat, VERTICES[body.name])) > 0.99
    # Distinct nodal longitudes → mutually asymmetric planes
    assert len(set(np.round(omegas_node, 6))) == 4
    # Angular-momentum directions are not all parallel
    hhat = np.stack([h / np.linalg.norm(h) for h in hs])
    dots = hhat @ hhat.T
    assert float(np.min(np.abs(dots))) < 0.95


def test_coplanar_legacy_mode_still_works():
    cfg = SystemConfig(mass_ratio=1e-6)
    system = build_orbital_ladder(
        cfg, LadderParams(eccentricity=0.1, tetrahedral=False, inclination=0.0)
    )
    for body in system.bodies[1:]:
        assert abs(body.position[2]) < 1e-10
