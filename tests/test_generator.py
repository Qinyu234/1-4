"""Tests for IC generator (T4)."""

import numpy as np

from fairy_orbit.icgen.generator import generate_system, velocity_at_A
from fairy_orbit.icgen.tetrahedron import FAIRY_ORDER, RA, VERTICES, rotations_from_A
from fairy_orbit.physics.gravity import center_of_mass, total_momentum


def test_velocities_are_rodrigues_images():
    v_rad, v_tan = 0.1, 0.2
    v_A = velocity_at_A(v_rad, v_tan)
    system = generate_system(v_rad, v_tan, radius=20.0)
    rotations = rotations_from_A()
    # bodies: Planet, A, B, C, D
    for i, label in enumerate(FAIRY_ORDER, start=1):
        expected = rotations[label] @ v_A
        np.testing.assert_allclose(system.bodies[i].velocity, expected, atol=1e-12)


def test_positions_on_sphere():
    system = generate_system(0.1, 0.2, radius=20.0)
    for i, label in enumerate(FAIRY_ORDER, start=1):
        pos = system.bodies[i].position
        np.testing.assert_allclose(np.linalg.norm(pos), 20.0, atol=1e-12)
        np.testing.assert_allclose(pos / 20.0, VERTICES[label], atol=1e-12)


def test_total_momentum_cancelled():
    system = generate_system(0.15, 0.25, planet_mass=1.0, fairy_mass=0.01)
    P = total_momentum(system)
    np.testing.assert_allclose(P, 0.0, atol=1e-12)
