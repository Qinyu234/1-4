"""Trajectory runner with diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fairy_orbit.physics.body import System
from fairy_orbit.physics.gravity import angular_momentum, total_energy
from fairy_orbit.physics.integrator import Integrator, Leapfrog

from fairy_orbit.simulation.cache import (
    build_cache_key,
    ensure_cache_db,
    load_cached_trajectory,
)
from fairy_orbit.simulation.trajectory import Trajectory


def run(
    system: System,
    dt: float,
    t_end: float,
    integrator: Integrator | None = None,
    record_every: int = 1,
    solver_type: str = "own",
    cache_path: str | None = None,
) -> Trajectory:
    """Integrate system from t=0 to t_end inclusive of the initial state."""
    resolved_cache_path = cache_path or "orbit_library/targeted_cache"
    conn = ensure_cache_db(resolved_cache_path)
    cache_key = build_cache_key(system, dt, t_end, record_every, solver_type)
    cached = load_cached_trajectory(conn, cache_key)
    if cached is not None:
        return cached

    if solver_type == "rebound":
        from fairy_orbit.simulation.simulator import simulate

        return simulate(
            system,
            dt=dt,
            t_end=t_end,
            solver_type="rebound",
            record_every=record_every,
            cache_path=resolved_cache_path,
        )

    if integrator is None:
        integrator = Leapfrog()
    system = system.copy()

    n_steps = int(np.ceil(t_end / dt))
    times: list[float] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    energies: list[float] = []
    angular_momenta: list[np.ndarray] = []

    def record(t: float) -> None:
        times.append(t)
        positions.append(system.positions().copy())
        velocities.append(system.velocities().copy())
        energies.append(total_energy(system))
        angular_momenta.append(angular_momentum(system).copy())

    t = 0.0
    record(t)
    for step in range(1, n_steps + 1):
        integrator.step(system, dt)
        t = step * dt
        if step % record_every == 0 or step == n_steps:
            record(t)

    return Trajectory(
        times=np.asarray(times, dtype=float),
        positions=np.stack(positions, axis=0),
        velocities=np.stack(velocities, axis=0),
        energies=np.asarray(energies, dtype=float),
        angular_momenta=np.stack(angular_momenta, axis=0),
        labels=list(system.labels),
    )
