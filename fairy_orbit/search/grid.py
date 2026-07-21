"""2D grid search over (v_rad, v_tan)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from fairy_orbit.analysis.evaluator import EvalResult, EvaluatorConfig, evaluate
from fairy_orbit.analysis.periodicity_evaluator import PeriodicityConfig, PeriodicityEvaluator
from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.library.store import save_candidate
from fairy_orbit.simulation.adaptive_simulator import AdaptiveConfig, AdaptiveSimulator
from fairy_orbit.simulation.runner import Trajectory, run


@dataclass
class Candidate:
    v_rad: float
    v_tan: float
    score: float
    period: float
    metrics: dict[str, Any]
    initial_positions: list[list[float]]
    initial_velocities: list[list[float]]
    planet_mass: float
    fairy_mass: float
    radius: float
    G: float

    @property
    def mass_ratio(self) -> float:
        """k in fairy_mass = k * planet_mass."""
        return self.fairy_mass / self.planet_mass if self.planet_mass else float("nan")


def evaluate_params(
    v_rad: float,
    v_tan: float,
    *,
    planet_mass: float = 1.0,
    fairy_mass: float | None = None,
    mass_ratio: float | None = None,
    radius: float = 20.0,
    G: float = 1.0,
    n_periods: float = 5.0,
    steps_per_period: int = 200,
    eval_config: EvaluatorConfig | None = None,
    record_every: int = 1,
    t_scan_range: tuple[float, float] | None = None,
    t_scan_steps: int = 5,
    solver_type: str = "own",
    evaluation_mode: str = "position",
    adaptive_config: AdaptiveConfig | None = None,
    periodicity_config: PeriodicityConfig | None = None,
) -> tuple[Candidate, Trajectory, EvalResult]:
    """
    Evaluate one IC. Prefer mass_ratio k (outer = kM); fairy_mass overrides if given alone.
    
    If t_scan_range is provided (e.g., (0.8, 1.2)), scan around the orbital period to find
    the best evaluation time, avoiding phase mismatch.
    """
    if fairy_mass is None:
        k = 0.01 if mass_ratio is None else float(mass_ratio)
        fairy_mass = k * planet_mass
    system = generate_system(
        v_rad,
        v_tan,
        planet_mass=planet_mass,
        fairy_mass=fairy_mass,
        radius=radius,
        G=G,
    )
    period = orbital_period(G, planet_mass, radius)
    dt = period / steps_per_period
    t_end = n_periods * period
    if evaluation_mode == "event":
        adaptive_config = adaptive_config or AdaptiveConfig()
        periodicity_config = periodicity_config or PeriodicityConfig()
        simulator = AdaptiveSimulator(adaptive_config)
        events = simulator.run(system, dt, t_end, record_every=record_every)
        traj = Trajectory(
            times=np.asarray(simulator.times, dtype=float),
            positions=np.stack(simulator.positions_history, axis=0),
            velocities=np.stack(simulator.velocities_history, axis=0),
            energies=np.asarray(simulator.energies, dtype=float),
            angular_momenta=np.stack(simulator.angular_momenta, axis=0),
            labels=list(system.labels),
            G=system.G,
            masses=system.masses(),
        )
        initial_center = system.bodies[0].position.copy()
        final_center = system.bodies[0].position.copy()
        center_displacement = float(np.linalg.norm(final_center - initial_center))
        energy_drift_value = abs(traj.energies[-1] - traj.energies[0]) if len(traj.energies) > 1 else 0.0
        periodicity_score = PeriodicityEvaluator(periodicity_config).evaluate(
            events,
            center_displacement=center_displacement,
            energy_drift=energy_drift_value,
        )
        result = EvalResult(
            score=float(periodicity_score.total_score),
            permutation_error=0.0,
            best_permutation=tuple(range(4)),
            collision_penalty=0.0,
            energy_drift=float(energy_drift_value),
            min_pair_distance=float("inf"),
            distance_matrix_error=0.0,
        )
    else:
        traj = run(
            system,
            dt=dt,
            t_end=t_end,
            record_every=record_every,
            solver_type=solver_type,
        )

        # Determine evaluation time(s)
        if t_scan_range is not None and n_periods >= 1.0:
            # Scan around the expected period to avoid phase mismatch
            t_min, t_max = t_scan_range
            scan_times = np.linspace(t_min * period, t_max * period, t_scan_steps)
            best_result = None
            best_score = np.inf
            best_time = None
            
            for target_t in scan_times:
                if target_t > traj.times[-1]:
                    continue
                period_index = int(np.argmin(np.abs(traj.times - target_t)))
                result = evaluate(traj, config=eval_config, period_index=period_index)
                if result.score < best_score:
                    best_score = result.score
                    best_result = result
                    best_time = target_t
            
            result = best_result if best_result is not None else evaluate(
                traj, config=eval_config, period_index=len(traj.times) - 1
            )
        else:
            # Evaluate at one orbital period (or final if shorter)
            target_t = period if n_periods >= 1.0 else traj.times[-1]
            period_index = int(np.argmin(np.abs(traj.times - target_t)))
            result = evaluate(traj, config=eval_config, period_index=period_index)

    cand = Candidate(
        v_rad=float(v_rad),
        v_tan=float(v_tan),
        score=result.score,
        period=period,
        metrics={
            "permutation_error": result.permutation_error,
            "best_permutation": list(result.best_permutation),
            "collision_penalty": result.collision_penalty,
            "energy_drift": result.energy_drift,
            "min_pair_distance": result.min_pair_distance,
            "distance_matrix_error": result.distance_matrix_error,
            "event_timeline": [event.to_dict() for event in []],
        },
        initial_positions=system.positions().tolist(),
        initial_velocities=system.velocities().tolist(),
        planet_mass=planet_mass,
        fairy_mass=fairy_mass,
        radius=radius,
        G=G,
    )
    return cand, traj, result


def scan(
    v_rad_values: np.ndarray | list[float],
    v_tan_values: np.ndarray | list[float],
    *,
    save_top: int = 5,
    library_dir: str = "orbit_library",
    progress: Callable[[int, int, Candidate], None] | None = None,
    **eval_kwargs: Any,
) -> list[Candidate]:
    """Cartesian product grid scan; returns candidates sorted by score ascending."""
    candidates: list[Candidate] = []
    rads = list(v_rad_values)
    tans = list(v_tan_values)
    total = len(rads) * len(tans)
    done = 0
    for vr in rads:
        for vt in tans:
            cand, _, _ = evaluate_params(vr, vt, **eval_kwargs)
            candidates.append(cand)
            done += 1
            if progress is not None:
                progress(done, total, cand)

    candidates.sort(key=lambda c: c.score)
    for cand in candidates[:save_top]:
        vesc = float(np.sqrt(2 * cand.G * cand.planet_mass / cand.radius))
        payload = {
            "v_rad": cand.v_rad,
            "v_tan": cand.v_tan,
            "v_rad_over_vesc": cand.v_rad / vesc,
            "v_tan_over_vesc": cand.v_tan / vesc,
            "mass_ratio": cand.mass_ratio,
            "planet_mass": cand.planet_mass,
            "fairy_mass": cand.fairy_mass,
            "radius": cand.radius,
            "G": cand.G,
            "initial_position": cand.initial_positions,
            "initial_velocity": cand.initial_velocities,
            "period": cand.period,
            "score": cand.score,
            "metrics": cand.metrics,
        }
        save_candidate(payload, directory=library_dir)
    return candidates
