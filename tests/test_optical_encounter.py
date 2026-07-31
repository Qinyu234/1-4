"""Unit tests for equal-density optical encounter geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fairy_orbit.observe.optical_encounter import (
    LOG_RHO_MAX,
    LOG_RHO_MIN,
    angular_separation,
    delta_r_perp,
    observer_validity,
    optical_overlap_angular,
    optical_overlap_perp,
    radii_from_uniform_density,
    rho_from_log_rho,
    soft_optics_deficit_perp,
    verify_visual_overlap,
)


def test_log_rho_bounds() -> None:
    assert rho_from_log_rho(0.0) == pytest.approx(1.0)
    assert rho_from_log_rho(-1.0) == pytest.approx(0.1)
    assert rho_from_log_rho(1.0) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        rho_from_log_rho(LOG_RHO_MIN - 1e-6)
    with pytest.raises(ValueError):
        rho_from_log_rho(LOG_RHO_MAX + 1e-6)


def test_radius_scales_with_mass_and_rho() -> None:
    R1 = radii_from_uniform_density([1.0], log_rho=0.0)[0]
    R8 = radii_from_uniform_density([8.0], log_rho=0.0)[0]
    assert R8 / R1 == pytest.approx(2.0)
    R_dense = radii_from_uniform_density([1.0], log_rho=1.0)[0]
    assert R_dense < R1


def test_delta_r_perp_identities() -> None:
    # identical → 0
    assert delta_r_perp([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    # pure radial (same LOS) → 0
    assert delta_r_perp([2, 0, 0], [3, 0, 0]) == pytest.approx(0.0, abs=1e-12)
    # pure tangential relative to mid-ray along x
    a = np.array([1.0, 0.5, 0.0])
    b = np.array([1.0, -0.5, 0.0])
    assert delta_r_perp(a, b) == pytest.approx(1.0, rel=1e-9)


def test_overlap_perp_threshold() -> None:
    a = np.array([1.0, 0.05, 0.0])
    b = np.array([1.0, -0.05, 0.0])
    # |Δr_perp| ≈ 0.1
    assert optical_overlap_perp(a, b, 0.06, 0.06)
    assert not optical_overlap_perp(a, b, 0.04, 0.04)
    assert soft_optics_deficit_perp(a, b, 0.04, 0.04) == pytest.approx(0.02, abs=1e-9)


def test_angular_overlap_distant_alignment() -> None:
    # Nearly collinear distant bodies: small θ, large 3D separation
    r_a = np.array([10.0, 0.0, 0.0])
    r_b = np.array([20.0, 0.05, 0.0])
    R = 0.5
    assert optical_overlap_angular(r_a, r_b, R, R)
    # Perp proxy also happens to be small here, but 3D distance is large —
    # angular is the right API for non-encounter verification.
    theta = angular_separation(r_a, r_b)
    assert theta < 0.01


def test_angular_true_but_perp_misuse_when_far() -> None:
    """
    Construct: small angular sep with large |Δr| (far alignment).
    Perp magnitude can still be moderate; ensure angular API fires and
    verify_visual_overlap(mode=angular) works without encounter precondition.
    """
    r_a = np.array([5.0, 0.0, 0.0])
    r_b = np.array([50.0, 0.2, 0.0])
    R_a, R_b = 0.3, 0.3
    assert optical_overlap_angular(r_a, r_b, R_a, R_b)
    # Large 3D separation ⇒ not a gravitational encounter
    assert float(np.linalg.norm(r_a - r_b)) > 40.0
    pos = np.stack([r_a, r_b])
    assert verify_visual_overlap(pos, [1.0, 1.0], 0, 1, log_rho=0.0, mode="angular")


def test_observer_validity() -> None:
    assert observer_validity(0.01, 1.0, max_ratio=0.05)
    assert not observer_validity(0.1, 1.0, max_ratio=0.05)


def test_engulfed_observer_angular_false() -> None:
    # Body A at origin engulfs observer at 0
    assert not optical_overlap_angular(
        np.zeros(3), np.array([2.0, 0, 0]), 0.5, 0.1
    )
