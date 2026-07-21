"""Tests for Rodrigues tetrahedron symmetry (T3)."""

import numpy as np

from fairy_orbit.icgen.tetrahedron import RA, RB, RC, RD, local_frame, tetra_rotation


def test_rotation_maps_A_to_targets():
    for target in (RB, RC, RD):
        R = tetra_rotation(RA, target)
        mapped = R @ RA
        np.testing.assert_allclose(mapped, target, atol=1e-12)


def test_rotation_is_so3():
    R = tetra_rotation(RA, RB)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_angle_to_radius_preserved():
    r_hat, t_hat, _ = local_frame(RA)
    a_hat = (0.3 * r_hat + 0.7 * t_hat)
    a_hat = a_hat / np.linalg.norm(a_hat)
    angle_A = float(np.arccos(np.clip(np.dot(a_hat, RA), -1.0, 1.0)))
    for target in (RB, RC, RD):
        R = tetra_rotation(RA, target)
        b_hat = R @ a_hat
        angle_X = float(np.arccos(np.clip(np.dot(b_hat, target), -1.0, 1.0)))
        np.testing.assert_allclose(angle_X, angle_A, atol=1e-12)


def test_dot_product_tetrahedron():
    np.testing.assert_allclose(np.dot(RA, RB), -1.0 / 3.0, atol=1e-12)
