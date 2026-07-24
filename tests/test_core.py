"""Core criteria and energy helpers."""

from __future__ import annotations

import numpy as np

from fairy_orbit.core import (
    Body,
    System,
    SystemConfig,
    check_collision,
    check_escape,
    specific_orbital_energy,
)
from fairy_orbit.core.body import total_energy


def test_specific_energy_bound_vs_escape():
    mu = 1.0
    # Circular at r=1: v=1, E = 0.5 - 1 = -0.5
    e_bound = specific_orbital_energy(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), mu)
    assert e_bound < 0.0
    # Fast: escape
    e_esc = specific_orbital_energy(np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]), mu)
    assert e_esc > 0.0


def test_collision_and_escape_criteria():
    central = Body(1.0, np.zeros(3), np.zeros(3), name="central", radius=0.1)
    fairy = Body(
        1e-4,
        np.array([0.15, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        name="T1",
        radius=0.1,
    )
    system = System([central, fairy], G=1.0)
    assert check_collision(system)

    esc = Body(
        1e-4,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        name="T1",
        radius=0.001,
    )
    system2 = System([central, esc], G=1.0)
    assert check_escape(system2)


def test_system_config_fairy_mass():
    cfg = SystemConfig(central_mass=2.0, mass_ratio=1e-3)
    assert abs(cfg.fairy_mass - 0.002) < 1e-15
    assert abs(cfg.mu - 2.0) < 1e-15


def test_total_energy_two_body():
    bodies = [
        Body(1.0, np.zeros(3), np.zeros(3), name="c"),
        Body(0.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), name="f"),
    ]
    # mass 0 fairy → PE = 0, KE = 0
    system = System(bodies, G=1.0)
    assert abs(total_energy(system)) < 1e-15
