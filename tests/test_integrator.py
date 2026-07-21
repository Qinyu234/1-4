"""Tests for Leapfrog vs Euler energy drift (T2)."""

import numpy as np

from fairy_orbit.physics.body import Body, System
from fairy_orbit.physics.gravity import total_energy
from fairy_orbit.physics.integrator import Euler, Leapfrog


def _circular_two_body(dt_factor: float = 0.01):
    """Unit circular orbit: M=1 at origin-ish reduced mass setup with fixed primary."""
    G = 1.0
    M = 1.0
    m = 1e-8  # test particle almost
    r = 1.0
    v = np.sqrt(G * M / r)
    primary = Body(M, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], name="M")
    secondary = Body(m, [r, 0.0, 0.0], [0.0, v, 0.0], name="m")
    system = System([primary, secondary], G=G)
    period = 2.0 * np.pi * np.sqrt(r**3 / (G * M))
    dt = period * dt_factor
    return system, period, dt


def _integrate(system: System, integrator, dt: float, n_steps: int) -> float:
    e0 = total_energy(system)
    for _ in range(n_steps):
        integrator.step(system, dt)
    e1 = total_energy(system)
    return abs(e1 - e0) / max(abs(e0), 1e-30)


def test_leapfrog_energy_better_than_euler():
    sys_lf, period, dt = _circular_two_body()
    sys_eu = sys_lf.copy()
    n_steps = int(10 * period / dt)  # 10 periods
    drift_lf = _integrate(sys_lf, Leapfrog(), dt, n_steps)
    drift_eu = _integrate(sys_eu, Euler(), dt, n_steps)
    assert drift_lf < drift_eu
    assert drift_lf < 1e-3
