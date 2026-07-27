"""Tests for state ↔ elements and polar ↔ Cartesian conversions."""

from __future__ import annotations

import numpy as np

from fairy_orbit.design.elements import (
    OrbitalElements,
    cartesian_to_polar,
    elements_to_state,
    kepler_polar_speeds,
    kepler_radius,
    orbital_period,
    polar_to_cartesian,
    state_to_elements,
)


def test_polar_cartesian_roundtrip():
    r, theta, vr, vt = 1.5, 0.7, 0.1, 0.9
    pos, vel = polar_to_cartesian(r, theta, vr, vt)
    r2, th2, vr2, vt2 = cartesian_to_polar(pos, vel)
    assert abs(r2 - r) < 1e-12
    assert abs(vr2 - vr) < 1e-12
    assert abs(vt2 - vt) < 1e-12
    # θ mod 2π
    assert abs(np.angle(np.exp(1j * (th2 - theta)))) < 1e-12
    assert abs(pos[2]) < 1e-15
    assert abs(vel[2]) < 1e-15


def test_polar_to_cartesian_unit_circle():
    pos, vel = polar_to_cartesian(1.0, np.pi / 2, 0.0, 1.0)
    assert np.allclose(pos, [0.0, 1.0, 0.0], atol=1e-12)
    # ê_θ at θ=π/2 is (-1, 0) → v = (-1, 0)
    assert np.allclose(vel, [-1.0, 0.0, 0.0], atol=1e-12)


def test_kepler_polar_at_periapsis():
    a, e, mu = 1.0, 0.2, 1.0
    vr, vt = kepler_polar_speeds(a, e, mu, 0.0)
    assert abs(vr) < 1e-14
    p = a * (1.0 - e * e)
    assert abs(vt - np.sqrt(mu / p) * (1.0 + e)) < 1e-12
    assert abs(kepler_radius(a, e, 0.0) - a * (1.0 - e)) < 1e-12


def test_state_elements_roundtrip_circular():
    mu = 1.0
    el = OrbitalElements(a=1.0, e=0.0, i=0.0, omega=0.0, Omega=0.0, M=0.3)
    r, v = elements_to_state(el, mu)
    back = state_to_elements(r, v, mu)
    assert abs(back.a - 1.0) < 1e-8
    assert abs(back.e) < 1e-8
    assert abs(orbital_period(1.0, mu) - 2 * np.pi) < 1e-10


def test_state_elements_roundtrip_eccentric_inclined():
    mu = 1.0
    el = OrbitalElements(a=1.2, e=0.25, i=0.4, omega=0.5, Omega=1.1, M=2.0)
    r, v = elements_to_state(el, mu)
    back = state_to_elements(r, v, mu)
    assert abs(back.a - el.a) < 1e-8
    assert abs(back.e - el.e) < 1e-8
    assert abs(back.i - el.i) < 1e-8
    assert abs(np.angle(np.exp(1j * (back.Omega - el.Omega)))) < 1e-7
    assert abs(np.angle(np.exp(1j * (back.omega - el.omega)))) < 1e-7
    assert abs(np.angle(np.exp(1j * (back.M - el.M)))) < 1e-7
    # Newton form recovers
    r2, v2 = elements_to_state(back, mu)
    assert np.allclose(r2, r, atol=1e-8)
    assert np.allclose(v2, v, atol=1e-8)


def test_class_methods_delegate_to_module_functions():
    mu = 1.0
    el = OrbitalElements(a=1.0, e=0.1, M=0.2)
    r1, v1 = el.to_state(mu)
    r2, v2 = elements_to_state(el, mu)
    assert np.allclose(r1, r2)
    assert np.allclose(v1, v2)
    assert abs(OrbitalElements.from_state(r1, v1, mu).a - state_to_elements(r1, v1, mu).a) < 1e-14


def test_polar_kepler_matches_elements_to_state_coplanar():
    """Coplanar ellipse: polar (r,f,v_r,v_t) → Cartesian equals elements_to_state."""
    mu = 1.0
    a, e, f = 1.0, 0.3, 0.8
    # At true anomaly f: ω=0, Ω=0, i=0 → M from f
    E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(f / 2), np.sqrt(1 + e) * np.cos(f / 2))
    M = E - e * np.sin(E)
    el = OrbitalElements(a=a, e=e, i=0.0, omega=0.0, Omega=0.0, M=float(M))
    r_el, v_el = elements_to_state(el, mu)

    r = kepler_radius(a, e, f)
    vr, vt = kepler_polar_speeds(a, e, mu, f)
    # With ω=Ω=0, true anomaly measured from +x ⇒ θ = f
    r_pol, v_pol = polar_to_cartesian(r, f, vr, vt)
    assert np.allclose(r_pol, r_el, atol=1e-8)
    assert np.allclose(v_pol, v_el, atol=1e-8)
