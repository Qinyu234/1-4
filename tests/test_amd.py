"""AMD helper tests."""

from fairy_orbit.observe.amd import amd_of_elements


def test_amd_zero_for_circular_planar():
    assert abs(amd_of_elements(1e-3, 1.0, 1.0, 0.0, 0.0)) < 1e-15


def test_amd_positive_for_eccentric():
    assert amd_of_elements(1e-3, 1.0, 1.0, 0.2, 0.0) > 0.0
