"""Bridge between existing optimizer and new event-driven system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.analysis.periodicity_evaluator import (
    PeriodicityConfig,
    PeriodicityEvaluator,
    PeriodicityScore,
)
from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.search.grid import Candidate
from fairy_orbit.simulation.adaptive_simulator import (
    AdaptiveConfig,
    AdaptiveSimulator,
)


@dataclass
class AdaptiveEvalResult:
    """Result from adaptive evaluation."""
    score: float
    periodicity_score: PeriodicityScore
    n_events: int
    total_time: float
    center_displacement: float
    energy_drift: float


def evaluate_adaptive(
    v_rad: float,
    v_tan: float,
    *,
    planet_mass: float = 1.0,
    fairy_mass: float | None = None,
    mass_ratio: float | None = None,
    radius: float = 20.0,
    G: float = 1.0,
    n_periods: float = 5.0,
    steps_per_period: int = 60,
    adaptive_config: AdaptiveConfig | None = None,
    periodicity_config: PeriodicityConfig | None = None,
) -> tuple[Candidate, AdaptiveEvalResult]:
    """
    Evaluate initial conditions using adaptive event-driven simulator.
    
    This function bridges the existing optimizer interface with the new
    event-driven simulation and periodicity evaluation system.
    
    Args:
        v_rad: Radial velocity component
        v_tan: Tangential velocity component
        planet_mass: Mass of central body
        fairy_mass: Mass of fairy bodies (overrides mass_ratio if provided)
        mass_ratio: Mass ratio k = fairy_mass / planet_mass
        radius: Initial orbital radius
        G: Gravitational constant
        n_periods: Number of orbital periods to simulate
        steps_per_period: Integration steps per period
        adaptive_config: Configuration for adaptive simulator
        periodicity_config: Configuration for periodicity evaluator
    
    Returns:
        (Candidate, AdaptiveEvalResult) tuple
    """
    # Determine fairy mass
    if fairy_mass is None:
        k = 0.01 if mass_ratio is None else float(mass_ratio)
        fairy_mass = k * planet_mass
    
    # Generate initial system
    system = generate_system(
        v_rad,
        v_tan,
        planet_mass=planet_mass,
        fairy_mass=fairy_mass,
        radius=radius,
        G=G,
    )
    
    # Calculate simulation parameters
    period = orbital_period(G, planet_mass, radius)
    dt = period / steps_per_period
    t_end = n_periods * period
    
    # Store initial state for center displacement calculation
    initial_center_pos = system.bodies[0].position.copy()
    initial_energy = _compute_total_energy(system)
    
    # Run adaptive simulation
    adaptive_config = adaptive_config or AdaptiveConfig()
    simulator = AdaptiveSimulator(adaptive_config)
    events = simulator.run(system, dt, t_end)
    
    # Calculate center displacement
    final_center_pos = system.bodies[0].position
    center_displacement = float(np.linalg.norm(final_center_pos - initial_center_pos))
    
    # Calculate energy drift
    final_energy = _compute_total_energy(system)
    energy_drift = abs(final_energy - initial_energy)
    
    # Evaluate periodicity
    periodicity_config = periodicity_config or PeriodicityConfig()
    evaluator = PeriodicityEvaluator(periodicity_config)
    periodicity_score = evaluator.evaluate(
        events,
        center_displacement=center_displacement,
        energy_drift=energy_drift,
    )
    
    # Create Candidate object (compatible with existing optimizer)
    cand = Candidate(
        v_rad=float(v_rad),
        v_tan=float(v_tan),
        score=periodicity_score.total_score,
        period=period,
        metrics={
            "n_events": len(events),
            "time_variance": periodicity_score.time_variance,
            "event_sequence_error": periodicity_score.event_sequence_error,
            "event_count_difference": periodicity_score.event_count_difference,
            "center_motion_error": periodicity_score.center_motion_error,
            "energy_drift": periodicity_score.energy_drift,
            "normalized_intervals": periodicity_score.normalized_intervals,
        },
        initial_positions=system.positions().tolist(),
        initial_velocities=system.velocities().tolist(),
        planet_mass=planet_mass,
        fairy_mass=fairy_mass,
        radius=radius,
        G=G,
    )
    
    # Create adaptive result
    adaptive_result = AdaptiveEvalResult(
        score=periodicity_score.total_score,
        periodicity_score=periodicity_score,
        n_events=len(events),
        total_time=t_end,
        center_displacement=center_displacement,
        energy_drift=energy_drift,
    )
    
    return cand, adaptive_result


def _compute_total_energy(system) -> float:
    """Compute total energy (kinetic + potential) of the system."""
    kinetic = 0.0
    potential = 0.0
    G = system.G
    n = len(system.bodies)
    
    for i in range(n):
        body_i = system.bodies[i]
        # Kinetic energy
        v_sq = np.dot(body_i.velocity, body_i.velocity)
        kinetic += 0.5 * body_i.mass * v_sq
        
        # Potential energy
        for j in range(i + 1, n):
            body_j = system.bodies[j]
            r = np.linalg.norm(body_i.position - body_j.position)
            if r > 1e-10:
                potential -= G * body_i.mass * body_j.mass / r
    
    return float(kinetic + potential)
