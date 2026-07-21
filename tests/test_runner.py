"""Tests for SimulationRunner (T5)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.simulation.runner import run


def test_runner_records_energy_and_L():
    from fairy_orbit.icgen.tetrahedron import escape_speed

    vesc = escape_speed(1.0, 1.0, 20.0)
    # Near-escape tangential-dominated — avoid radial plunge / close encounters
    system = generate_system(0.1 * vesc, 0.95 * vesc, radius=20.0)
    period = orbital_period(1.0, 1.0, 20.0)
    traj = run(system, dt=period / 100, t_end=period, record_every=1)
    assert len(traj.times) > 2
    assert traj.positions.shape[1] == 5
    assert traj.energies.shape == traj.times.shape
    assert traj.angular_momenta.shape == (len(traj.times), 3)
    rel = abs(traj.energies[-1] - traj.energies[0]) / max(abs(traj.energies[0]), 1e-30)
    assert rel < 1e-2


def test_runner_supports_rebound_solver():
    try:
        import rebound  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        pytest.skip(f"rebound not available: {exc}")

    from fairy_orbit.icgen.tetrahedron import escape_speed

    vesc = escape_speed(1.0, 1.0, 20.0)
    system = generate_system(0.1 * vesc, 0.95 * vesc, radius=20.0)
    period = orbital_period(1.0, 1.0, 20.0)
    traj = run(system, dt=period / 100, t_end=period, record_every=1, solver_type="rebound")
    assert len(traj.times) > 2
    assert traj.positions.shape[1] == 5


def test_runner_uses_cache_when_available(tmp_path: Path):
    from fairy_orbit.icgen.tetrahedron import escape_speed

    vesc = escape_speed(1.0, 1.0, 20.0)
    system = generate_system(0.1 * vesc, 0.95 * vesc, radius=20.0)
    period = orbital_period(1.0, 1.0, 20.0)

    cached = tmp_path / "trajectory.sqlite"
    traj = run(system, dt=period / 100, t_end=period, record_every=1, cache_path=str(cached))
    assert traj.positions.shape[1] == 5
    assert (tmp_path / "trajectory.sqlite").exists()
