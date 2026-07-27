"""Td group-orbit reduction: r_i = ρ(t) R(t) q_i."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fairy_orbit.design.tetra_eff import (
    MU_EFF_FAIRY_COEFF,
    analytic_states_at_time,
    build_td_system,
    circular_omega,
    mu_eff,
    reduced_masses,
    rho_accel,
    rhodot_from_energy,
    states_at_rho_theta,
    td_orbit_from_ic,
)
from fairy_orbit.observe.calibration import td_breaking, tetrahedron_shape_error


def test_ABC_match_Td_geometry():
    m = 0.01
    A, B, C = reduced_masses(m)
    assert A == pytest.approx(4.0 * m)
    assert B == pytest.approx(8.0 * m / 3.0)
    assert C / A == pytest.approx(1.0 + MU_EFF_FAIRY_COEFF * m)
    assert mu_eff(m) == pytest.approx(C / A)


def test_circular_equilibrium_zero_radial_accel():
    m, rho = 1e-3, 1.0
    w = circular_omega(m, rho)
    orb = td_orbit_from_ic(m, rho, 0.0, w)
    assert rho_accel(orb, rho) == pytest.approx(0.0, abs=1e-12)


def test_vt_derived_from_m_and_rho_only():
    from fairy_orbit.design.tetra_eff import reduced_masses, vc_scale, vt_circular

    m, rho = 1e-3, 1.5
    A, B, C = reduced_masses(m)
    vt = vt_circular(m, rho)
    assert vt == pytest.approx(math.sqrt(2.0 * C / (3.0 * B * rho)))
    assert vt == pytest.approx(vc_scale(m, rho))


def test_beta_e_roundtrip_and_domain():
    from fairy_orbit.design.tetra_eff import (
        alpha_from_beta_e,
        e_min_for_beta,
        eccentricity_from_beta_alpha,
        is_valid_beta_e,
        polar_from_beta_e,
    )

    # circular at β=1: e=0 ⇒ α=π/2
    assert e_min_for_beta(1.0) == pytest.approx(0.0, abs=1e-14)
    assert is_valid_beta_e(1.0, 0.0)
    a = alpha_from_beta_e(1.0, 0.0)
    assert a == pytest.approx(math.pi / 2, abs=1e-12)
    assert eccentricity_from_beta_alpha(1.0, a) == pytest.approx(0.0, abs=1e-12)

    # β=0.5 cannot reach e=0 (e_min ≈ 0.75)
    assert e_min_for_beta(0.5) == pytest.approx(math.sqrt(0.5625), rel=1e-12)
    assert not is_valid_beta_e(0.5, 0.0)
    assert is_valid_beta_e(0.5, 0.8)

    m, beta, e = 1e-3, 1.0, 0.4
    vr, vt, alpha = polar_from_beta_e(m, beta, e)
    assert eccentricity_from_beta_alpha(beta, alpha) == pytest.approx(e, abs=1e-12)
    assert vt > 0.0
    assert abs(vr) > 0.0


def test_ic_is_regular_tetrahedron():
    m, rho = 1e-3, 1.2
    w = circular_omega(m, rho)
    orb = td_orbit_from_ic(m, rho, 0.0, w)
    st = states_at_rho_theta(orb, rho, 0.0, 0.0)
    pos = np.stack([st[n][0] for n in ("T1", "T2", "T3", "T4")])
    assert tetrahedron_shape_error(pos) == pytest.approx(0.0, abs=1e-14)
    assert td_breaking(pos) == pytest.approx(0.0, abs=1e-14)


def test_analytic_preserves_Td_over_time():
    """Group-orbit reference must keep D_Td = 0 by construction."""
    m, rho = 1e-3, 1.0
    w = 0.7 * circular_omega(m, rho)
    orb = td_orbit_from_ic(m, rho, 0.05, w)
    for t in (0.0, 0.5, 1.0, 2.0):
        st = analytic_states_at_time(orb, t)
        pos = np.stack([st[n][0] for n in ("T1", "T2", "T3", "T4")])
        assert td_breaking(pos) == pytest.approx(0.0, abs=1e-10)
        radii = np.linalg.norm(pos, axis=1)
        assert np.std(radii) / np.mean(radii) == pytest.approx(0.0, abs=1e-10)


def test_energy_inversion_matches_ic():
    m, rho = 1e-3, 1.0
    w = circular_omega(m, rho)
    orb = td_orbit_from_ic(m, rho, 0.1, w)
    v = rhodot_from_energy(orb, rho, sign=1.0)
    assert v == pytest.approx(0.1, rel=1e-12)


def test_pure_breathing_J0():
    m, rho = 1e-3, 1.0
    orb = td_orbit_from_ic(m, rho, 0.1, 0.0)
    assert orb.J == pytest.approx(0.0)
    st0 = analytic_states_at_time(orb, 0.0)
    st = analytic_states_at_time(orb, 0.3)
    pos = np.stack([st[n][0] for n in ("T1", "T2", "T3", "T4")])
    assert td_breaking(pos) == pytest.approx(0.0, abs=1e-12)
    q = pos / np.linalg.norm(pos, axis=1)[:, None]
    q0 = np.stack(
        [st0[n][0] / np.linalg.norm(st0[n][0]) for n in ("T1", "T2", "T3", "T4")]
    )
    assert np.allclose(q, q0, atol=1e-10)


def test_build_system_masses_and_J():
    m, rho = 2e-3, 0.8
    w = circular_omega(m, rho)
    sys, orb = build_td_system(m, rho, 0.0, w)
    assert sys.n == 5
    assert sys.bodies[0].mass == pytest.approx(1.0)
    for b in sys.bodies[1:]:
        assert b.mass == pytest.approx(m)
    # total fairy angular momentum ≈ J n̂
    L = np.zeros(3)
    for b in sys.bodies[1:]:
        L += m * np.cross(b.position, b.velocity)
    assert np.linalg.norm(L) == pytest.approx(abs(orb.J), rel=1e-9)
