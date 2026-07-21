"""Tests for Body/System/gravity (T1)."""

import numpy as np

from fairy_orbit.physics.body import Body, System
from fairy_orbit.physics.gravity import (
    accelerations,
    potential_energy,
    total_energy,
)


def test_pairwise_force_antisymmetry():
    a = Body(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], name="a")
    b = Body(2.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], name="b")
    system = System([a, b], G=1.0)
    acc = accelerations(system)
    # m_a * a_a + m_b * a_b = 0 for mutual gravity
    force_sum = a.mass * acc[0] + b.mass * acc[1]
    np.testing.assert_allclose(force_sum, 0.0, atol=1e-12)


def test_potential_two_body():
    a = Body(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    b = Body(1.0, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    system = System([a, b], G=1.0)
    np.testing.assert_allclose(potential_energy(system), -0.5)


def test_total_energy_includes_kinetic():
    a = Body(1.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b = Body(1.0, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    system = System([a, b], G=1.0)
    # KE = 0.5, PE = -0.5
    np.testing.assert_allclose(total_energy(system), 0.0)
