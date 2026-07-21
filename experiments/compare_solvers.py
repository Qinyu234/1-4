"""Compare custom integrator, scipy RK45, and REBOUND IAS15 for the same IC."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairy_orbit.analysis.evaluator import EvaluatorConfig, evaluate
from fairy_orbit.analysis.periodicity_evaluator import PeriodicityConfig, PeriodicityEvaluator

from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.physics.body import System
from fairy_orbit.physics.gravity import angular_momentum, total_energy
from fairy_orbit.physics.integrator import Leapfrog
from fairy_orbit.simulation.runner import run
from fairy_orbit.simulation.simulator import simulate
from fairy_orbit.simulation.trajectory import Trajectory


def _state(system: System) -> tuple[np.ndarray, np.ndarray]:
    return system.positions().reshape(-1), system.velocities().reshape(-1)


def _set_state(system: System, state: np.ndarray) -> None:
    positions = state[: system.n * 3].reshape(system.n, 3)
    velocities = state[system.n * 3 :].reshape(system.n, 3)
    system.set_state(positions, velocities)


def _rhs(_: float, state: np.ndarray, G: float, masses: np.ndarray) -> np.ndarray:
    n = len(masses)
    pos = state[: n * 3].reshape(n, 3)
    vel = state[n * 3 :].reshape(n, 3)
    acc = np.zeros_like(vel)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = pos[j] - pos[i]
            dist = np.linalg.norm(r)
            if dist < 1e-12:
                continue
            acc[i] += G * masses[j] * r / (dist**3)
    return np.concatenate([vel.reshape(-1), acc.reshape(-1)])


def run_rk45(system: System, t_end: float, dt: float) -> dict[str, Any]:
    n_steps = int(np.ceil(t_end / dt))
    t_eval = np.linspace(0.0, t_end, n_steps + 1)
    if t_eval[-1] < t_end:
        t_eval = np.append(t_eval, t_end)
    state0 = _state(system)
    masses = system.masses()
    sol = solve_ivp(
        fun=lambda t, y: _rhs(t, y, system.G, masses),
        t_span=(0.0, t_end),
        y0=np.concatenate([state0[0], state0[1]]),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"RK45 failed: {sol.message}")
    positions = sol.y[: system.n * 3].T.reshape(-1, system.n, 3)
    velocities = sol.y[system.n * 3 :].T.reshape(-1, system.n, 3)
    return {
        "times": sol.t,
        "positions": positions,
        "velocities": velocities,
        "energies": np.array([total_energy(_system_from_state(system, pos, vel)) for pos, vel in zip(positions, velocities)]),
        "angular_momenta": np.array([angular_momentum(_system_from_state(system, pos, vel)) for pos, vel in zip(positions, velocities)]),
    }


def _system_from_state(system: System, positions: np.ndarray, velocities: np.ndarray) -> System:
    sys_copy = system.copy()
    sys_copy.set_state(positions, velocities)
    return sys_copy


def run_custom(system: System, t_end: float, dt: float) -> dict[str, Any]:
    traj = run(system, dt=dt, t_end=t_end, record_every=1, solver_type="own")
    return {
        "times": traj.times,
        "positions": traj.positions,
        "velocities": traj.velocities,
        "energies": traj.energies,
        "angular_momenta": traj.angular_momenta,
    }


def run_rebound(system: System, t_end: float, dt: float) -> dict[str, Any]:
    traj = run(system, dt=dt, t_end=t_end, record_every=1, solver_type="rebound")
    return {
        "times": traj.times,
        "positions": traj.positions,
        "velocities": traj.velocities,
        "energies": traj.energies,
        "angular_momenta": traj.angular_momenta,
    }


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(np.linalg.norm(positions[j] - positions[i]))
    return np.array(out)


def metric_summary(name: str, result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    pos_ref = reference["positions"]
    pos = result["positions"]
    vel_ref = reference["velocities"]
    vel = result["velocities"]
    e_ref = reference["energies"]
    e = result["energies"]
    l_ref = reference["angular_momenta"]
    l = result["angular_momenta"]

    if pos.shape[0] != pos_ref.shape[0]:
        raise ValueError(f"{name}: trajectory length mismatch")

    traj_err = np.sqrt(np.mean((pos - pos_ref) ** 2))
    vel_err = np.sqrt(np.mean((vel - vel_ref) ** 2))
    dE = np.max(np.abs(e - e_ref))
    dL = np.max(np.linalg.norm(l - l_ref, axis=1))
    return {
        "name": name,
        "max_abs_energy_error": float(dE),
        "max_abs_angmom_error": float(dL),
        "trajectory_rmse": float(traj_err),
        "velocity_rmse": float(vel_err),
    }


def _as_trajectory(traj: Any, system: System) -> Trajectory:
    if isinstance(traj, Trajectory):
        return traj
    if isinstance(traj, dict):
        return Trajectory(
            times=np.asarray(traj["times"], dtype=float),
            positions=np.asarray(traj["positions"], dtype=float),
            velocities=np.asarray(traj["velocities"], dtype=float),
            energies=np.asarray(traj["energies"], dtype=float),
            angular_momenta=np.asarray(traj["angular_momenta"], dtype=float),
            labels=list(system.labels),
            G=system.G,
            masses=system.masses(),
        )
    raise TypeError(f"Unsupported trajectory type: {type(traj)!r}")


def score_breakdown(name: str, traj: Any, system: System, *, dt: float, t_end: float) -> dict[str, Any]:
    traj_obj = _as_trajectory(traj, system)
    eval_config = EvaluatorConfig()
    position_result = evaluate(traj_obj, config=eval_config, period_index=len(traj_obj.times) - 1)

    adaptive_config = None
    periodicity_config = PeriodicityConfig()
    periodicity_evaluator = PeriodicityEvaluator(periodicity_config)
    from fairy_orbit.simulation.adaptive_simulator import AdaptiveSimulator

    simulator = AdaptiveSimulator(adaptive_config)
    events = simulator.run(system, dt, t_end, record_every=1)
    initial_center = system.bodies[0].position.copy()
    final_center = system.bodies[0].position.copy()
    center_displacement = float(np.linalg.norm(final_center - initial_center))
    energy_drift_value = abs(traj_obj.energies[-1] - traj_obj.energies[0]) if len(traj_obj.energies) > 1 else 0.0
    periodicity_score = periodicity_evaluator.evaluate(
        events,
        center_displacement=center_displacement,
        energy_drift=energy_drift_value,
    )

    return {
        "name": name,
        "position": {
            "score": float(position_result.score),
            "permutation_error": float(position_result.permutation_error),
            "collision_penalty": float(position_result.collision_penalty),
            "energy_drift": float(position_result.energy_drift),
            "min_pair_distance": float(position_result.min_pair_distance),
            "distance_matrix_error": float(position_result.distance_matrix_error),
        },
        "event": {
            "total_score": float(periodicity_score.total_score),
            "time_variance": float(periodicity_score.time_variance),
            "event_sequence_error": float(periodicity_score.event_sequence_error),
            "event_count_difference": float(periodicity_score.event_count_difference),
            "center_motion_error": float(periodicity_score.center_motion_error),
            "energy_drift": float(periodicity_score.energy_drift),
        },
    }


def main() -> None:
    vesc = escape_speed(1.0, 1.0, 20.0)
    system = generate_system(0.1 * vesc, 0.95 * vesc, radius=20.0)
    period = orbital_period(1.0, 1.0, 20.0)
    T = 100.0
    dt = period / 100.0

    custom = run_custom(system.copy(), T, dt)
    rk45 = run_rk45(system.copy(), T, dt)
    rebound = run_rebound(system.copy(), T, dt)

    summary = {
        "T": T,
        "dt": dt,
        "custom_vs_rk45": metric_summary("custom_vs_rk45", custom, rk45),
        "rebound_vs_rk45": metric_summary("rebound_vs_rk45", rebound, rk45),
        "custom_vs_rebound": metric_summary("custom_vs_rebound", custom, rebound),
        "rk45_reference": {
            "initial_energy": float(rk45["energies"][0]),
            "initial_angmom": rk45["angular_momenta"][0].tolist(),
        },
    }

    summary["score_breakdowns"] = {
        "custom": score_breakdown("custom", custom, system, dt=dt, t_end=T),
        "rk45": score_breakdown("rk45", rk45, system, dt=dt, t_end=T),
        "rebound": score_breakdown("rebound", rebound, system, dt=dt, t_end=T),
    }

    out_path = Path("experiments/output/solver_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
