"""Manifold: linear/quad polys + Td-symmetric velocity kicks + COM frame."""

from __future__ import annotations

import numpy as np
import pytest

from fairy_orbit.core.body import com_position, com_velocity
from fairy_orbit.design.manifold import (
    ManifoldParams,
    apply_symmetric_velocity_kick,
    build_manifold_system,
    elements_for_index,
    from_error_seed,
    poly_linear,
    poly_quad,
)
from fairy_orbit.design.tetrahedron import FAIRY_ORDER, VERTICES, rotations_from_T1


def test_theta_roundtrip():
    p = ManifoldParams(
        a0=1.0,
        a1=0.1,
        a2=0.01,
        e0=0.05,
        e1=0.01,
        e2=-0.001,
        M0=0.0,
        M1=0.3,
        M2=0.05,
        vx=0.01,
        vy=-0.02,
        vz=0.0,
        v1x=0.002,
        v1y=0.0,
        v1z=-0.001,
        mu_mass=1e-3,
    )
    assert ManifoldParams.from_theta(p.as_theta()) == p
    assert len(p.as_theta()) == 16


def test_theta_legacy_10():
    legacy = (1.0, 0.1, 0.05, 0.01, 0.0, 0.3, 0.01, -0.02, 0.0, 1e-3)
    p = ManifoldParams.from_theta(legacy)
    assert p.a2 == 0.0 and p.e2 == 0.0 and p.M2 == 0.0
    assert p.v1x == 0.0 and p.a1 == pytest.approx(0.1)


def test_linear_a_ladder():
    assert poly_linear(1.0, 0.1, 0) == pytest.approx(1.0)
    assert poly_linear(1.0, 0.1, 1) == pytest.approx(1.1)
    assert poly_linear(1.0, 0.1, 3) == pytest.approx(1.3)
    p = ManifoldParams(a0=1.0, a1=0.1, e0=0.05, e1=0.0, M0=0.0, M1=0.0)
    for i in range(4):
        assert elements_for_index(p, i).a == pytest.approx(1.0 + 0.1 * i)


def test_quad_a_ladder():
    assert poly_quad(1.0, 0.1, 0.02, 2) == pytest.approx(1.0 + 0.2 + 0.08)
    p = ManifoldParams(a0=1.0, a1=0.1, a2=0.02, e0=0.05, e1=0.0, M0=0.0, M1=0.0)
    for i in range(4):
        assert elements_for_index(p, i).a == pytest.approx(poly_quad(1.0, 0.1, 0.02, i))


def test_symmetric_kick_uses_rodrigues():
    base = {n: np.zeros(3) for n in FAIRY_ORDER}
    dv = np.array([0.02, -0.01, 0.03])
    out = apply_symmetric_velocity_kick(base, dv)
    rots = rotations_from_T1()
    for name in FAIRY_ORDER:
        assert np.allclose(out[name], rots[name] @ dv, atol=1e-14)


def test_symmetric_kick_with_v1():
    base = {n: np.zeros(3) for n in FAIRY_ORDER}
    dv = np.array([0.02, 0.0, 0.0])
    dv1 = np.array([0.01, 0.0, 0.0])
    out = apply_symmetric_velocity_kick(base, dv, delta_v1_T1=dv1)
    rots = rotations_from_T1()
    for i, name in enumerate(FAIRY_ORDER):
        expect = rots[name] @ (dv + float(i) * dv1)
        assert np.allclose(out[name], expect, atol=1e-14)


def test_build_system_in_com_frame():
    sys = build_manifold_system(
        ManifoldParams(a1=0.15, e0=0.05, M1=1.0, vx=0.02, vy=0.01, vz=-0.01, mu_mass=1e-3)
    )
    assert np.linalg.norm(com_position(sys)) < 1e-12
    assert np.linalg.norm(com_velocity(sys)) < 1e-12


def test_from_error_seed_anchors():
    p = from_error_seed(1e-3, 0.05, a1=0.12, e1=0.01, M1=2.0, vx=0.01, a2=0.01)
    assert p.a0 == 1.0 and p.M0 == 0.0 and p.e0 == pytest.approx(0.05)
    assert p.vx == pytest.approx(0.01)
    assert p.a2 == pytest.approx(0.01)


def test_build_on_td_rays():
    sys = build_manifold_system(ManifoldParams(e0=0.05, a1=0.1, M1=0.0), com_frame=False)
    for i, name in enumerate(FAIRY_ORDER):
        r = sys.bodies[i + 1].position
        q = r / np.linalg.norm(r)
        q_ref = VERTICES[name] / np.linalg.norm(VERTICES[name])
        assert np.allclose(q, q_ref, atol=1e-12)
