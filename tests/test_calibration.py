"""Calibration IC: Newton ↔ orbit synonymy for Rodrigues tetrahedron."""

from __future__ import annotations

import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, OrbitalElements, build_orbital_ladder
from fairy_orbit.design.tetrahedron import (
    VERTICES,
    calibration_tetrahedron_elements,
    calibration_tetrahedron_states,
    kepler_vr_vt_at_true_anomaly,
)
from fairy_orbit.observe.calibration import td_breaking, tetrahedron_shape_error


def test_regular_tetrahedron_shape_error_zero():
    verts = np.stack([VERTICES[n] for n in ("T1", "T2", "T3", "T4")])
    assert tetrahedron_shape_error(verts) < 1e-12


def test_td_breaking_zero_on_ideal_tetrahedron():
    verts = np.stack([VERTICES[n] for n in ("T1", "T2", "T3", "T4")])
    assert td_breaking(verts) < 1e-14
    # Angular perturbation (radial scale alone leaves r̂ unchanged)
    bad = verts.copy()
    bad[0] = bad[0] + 0.3 * (bad[1] - bad[0])
    assert td_breaking(bad) > 0.1


def test_kepler_vr_vt_synonym_of_ae_at_periapsis():
    """(a,e,f=0) ↔ (v_rad=0, v_tan=√(μ/p)(1+e)) — same numbers, two writings."""
    vr, vt = kepler_vr_vt_at_true_anomaly(1.0, 0.2, 1.0, 0.0)
    assert abs(vr) < 1e-14
    p = 1.0 * (1.0 - 0.04)
    assert abs(vt - np.sqrt(1.0 / p) * 1.2) < 1e-12


def test_newton_and_orbit_forms_are_roundtrip_synonyms():
    """
    Six DOF per fairy: (r,v) ⇔ OrbitalElements.from_state / to_state.

    Calibration builds Newton form; orbit form is the same state rewritten.
    """
    mu = 1.0
    a, e = 1.0, 0.15
    states = calibration_tetrahedron_states(a, e, mu)
    elements = calibration_tetrahedron_elements(a, e, mu)
    pos = np.stack([states[n][0] for n in ("T1", "T2", "T3", "T4")])
    assert tetrahedron_shape_error(pos) < 1e-12

    for name in ("T1", "T2", "T3", "T4"):
        r, v = states[name]
        el = elements[name]
        assert abs(el.a - a) < 1e-8
        assert abs(el.e - e) < 1e-8
        # from_state(r,v) matches the table; to_state recovers (r,v)
        el2 = OrbitalElements.from_state(r, v, mu)
        assert abs(el2.a - el.a) < 1e-8
        assert abs(el2.e - el.e) < 1e-8
        r2, v2 = el.to_state(mu)
        assert np.allclose(r2, r, atol=1e-8)
        assert np.allclose(v2, v, atol=1e-8)


def test_calibration_ladder_builds_rodrigues_newton_state():
    cfg = SystemConfig(mass_ratio=1e-4)
    system = build_orbital_ladder(
        cfg, LadderParams(geometry="calibration", eccentricity=0.1, a_inner=1.0)
    )
    pos = np.stack([b.position for b in system.bodies[1:]])
    assert tetrahedron_shape_error(pos) < 1e-12
    central = system.bodies[0]
    for body in system.bodies[1:]:
        el = OrbitalElements.from_state(
            body.position - central.position,
            body.velocity - central.velocity,
            cfg.mu,
        )
        assert abs(el.a - 1.0) < 1e-8
        assert abs(el.e - 0.1) < 1e-8
