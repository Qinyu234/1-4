"""Exponential error base + graded planar/stereo IC."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fairy_orbit.design.graded import (
    GradedParams,
    build_planar90_system,
    build_stereo_graded_system,
    graded_radii,
    graded_vr,
)
from fairy_orbit.observe.error_base import (
    DEFAULT_ERROR_BASE,
    ExpErrorBase,
    log_excess_error,
    normalize_error,
)


def test_exp_normalize_on_base_is_one():
    base = ExpErrorBase(lam=2.0, eps0=1e-4)
    t = 1.5
    err = base.envelope(t)
    assert normalize_error(err, t, base) == pytest.approx(1.0)
    assert log_excess_error(err, t, base) == pytest.approx(0.0, abs=1e-12)


def test_exp_normalize_array_and_default():
    t = np.array([0.0, 0.1, 0.2])
    err = DEFAULT_ERROR_BASE.envelope(t) * 10.0
    hat = normalize_error(err, t)
    assert np.allclose(hat, 10.0)
    excess = log_excess_error(err, t)
    assert np.allclose(excess, 1.0)  # one decade above base


def test_graded_radii_and_vr():
    rhos = graded_radii(1.0, 0.2)
    assert rhos == pytest.approx([0.7, 0.9, 1.1, 1.3])
    vrs = graded_vr(0.1)
    assert vrs == pytest.approx([-0.15, -0.05, 0.05, 0.15])
    with pytest.raises(ValueError):
        graded_radii(1.0, 1.0)  # ρ0 - 1.5 Δρ = -0.5


def test_planar90_alternating_and_planar():
    params = GradedParams(rho0=1.0, delta_rho=0.1, v_t=0.8, k=0.05, m=1e-3)
    sys = build_planar90_system(params)
    assert sys.n == 5
    # all z=0
    for b in sys.bodies[1:]:
        assert b.position[2] == pytest.approx(0.0)
        assert b.velocity[2] == pytest.approx(0.0)
        assert b.mass == pytest.approx(1e-3)
    # radii ordered
    rhos = [float(np.linalg.norm(b.position)) for b in sys.bodies[1:]]
    assert rhos == sorted(rhos)
    assert rhos == pytest.approx(graded_radii(1.0, 0.1))
    # alternating Lz signs (T1/T3 +, T2/T4 −)
    Lz = []
    for b in sys.bodies[1:]:
        L = np.cross(b.position, b.velocity)
        Lz.append(L[2])
    assert Lz[0] > 0 and Lz[2] > 0
    assert Lz[1] < 0 and Lz[3] < 0


def test_stereo_graded_on_td_directions():
    params = GradedParams(rho0=1.0, delta_rho=0.05, v_t=1.0, k=0.0, m=1e-3)
    sys = build_stereo_graded_system(params)
    assert sys.n == 5
    pos = np.stack([b.position for b in sys.bodies[1:]])
    # directions match tetrahedron unit vertices (up to radius)
    from fairy_orbit.design.tetrahedron import FAIRY_ORDER, VERTICES

    for i, name in enumerate(FAIRY_ORDER):
        q = pos[i] / np.linalg.norm(pos[i])
        assert np.allclose(q, VERTICES[name] / np.linalg.norm(VERTICES[name]), atol=1e-14)
    # equal |v_t| when k=0: |v| should be ~vt (radial component 0)
    speeds = [float(np.linalg.norm(b.velocity)) for b in sys.bodies[1:]]
    assert speeds == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=1e-12)
