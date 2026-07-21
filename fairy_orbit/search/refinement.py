"""Local refinement module for improving promising candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize as optimize

from fairy_orbit.evaluation.simple_score import SimpleScoreConfig, SimpleScoreEvaluator
from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.search.candidate import Candidate
from fairy_orbit.simulation.constraints import ConstraintConfig, ConstraintFilter
from fairy_orbit.simulation.runner import run
from fairy_orbit.simulation.status import SimulationStatus


@dataclass
class RefinementConfig:
    """Configuration for local refinement."""
    maxiter: int = 40  # Maximum iterations for optimizer
    xtol: float = 1e-6  # Tolerance for parameter convergence
    ftol: float = 1e-8  # Tolerance for function convergence
    bounds_v_rad: tuple[float, float] = (0.0, 2.0)  # Bounds for v_rad (in units of v_esc)
    bounds_v_tan: tuple[float, float] = (0.0, 2.0)  # Bounds for v_tan (in units of v_esc)


class Refinement:
    """Local refinement using scipy.optimize.minimize."""
    
    def __init__(
        self,
        refinement_config: RefinementConfig | None = None,
        constraint_config: ConstraintConfig | None = None,
        score_config: SimpleScoreConfig | None = None,
    ):
        self.refinement_config = refinement_config or RefinementConfig()
        self.constraint_config = constraint_config or ConstraintConfig()
        self.score_config = score_config or SimpleScoreConfig()
        
        self.constraint_filter = ConstraintFilter(self.constraint_config)
        self.score_evaluator = SimpleScoreEvaluator(self.score_config)
    
    def refine_candidate(
        self,
        candidate: Candidate,
        n_periods: float = 2.0,
        steps_per_period: int = 60,
    ) -> Candidate:
        """
        Refine a candidate using local optimization.
        
        Args:
            candidate: Initial candidate to refine
            n_periods: Number of orbital periods for simulation
            steps_per_period: Integration steps per period
        
        Returns:
            Refined candidate
        """
        # Calculate escape velocity for bounds
        from fairy_orbit.icgen.tetrahedron import escape_speed
        vesc = escape_speed(candidate.G, candidate.planet_mass, candidate.radius)
        
        # Convert bounds to absolute velocities
        v_rad_bounds = (
            self.refinement_config.bounds_v_rad[0] * vesc,
            self.refinement_config.bounds_v_rad[1] * vesc,
        )
        v_tan_bounds = (
            self.refinement_config.bounds_v_tan[0] * vesc,
            self.refinement_config.bounds_v_tan[1] * vesc,
        )
        
        # Initial parameters (normalized by v_esc for better conditioning)
        x0 = np.array([
            candidate.v_rad / vesc,
            candidate.v_tan / vesc,
        ])
        
        # Define loss function
        def loss(x: np.ndarray) -> float:
            v_rad = float(x[0]) * vesc
            v_tan = float(x[1]) * vesc
            
            # Generate system
            system = generate_system(
                v_rad,
                v_tan,
                planet_mass=candidate.planet_mass,
                fairy_mass=candidate.fairy_mass,
                radius=candidate.radius,
                G=candidate.G,
            )
            
            # Check constraints
            status = self.constraint_filter.check_constraints(system)
            if status != SimulationStatus.SUCCESS:
                # Return large penalty for constraint violations
                return 1e6
            
            # Run simulation
            period = orbital_period(candidate.G, candidate.planet_mass, candidate.radius)
            dt = period / steps_per_period
            t_end = n_periods * period
            
            try:
                traj = run(system, dt=dt, t_end=t_end, record_every=1)
            except Exception:
                return 1e6
            
            # Evaluate score
            score_dict = self.score_evaluator.evaluate(traj)
            return score_dict["score"]
        
        # Run optimization
        result = optimize.minimize(
            loss,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": self.refinement_config.maxiter,
                "xatol": self.refinement_config.xtol,
                "fatol": self.refinement_config.ftol,
            },
        )
        
        # Create refined candidate
        refined_v_rad = float(result.x[0]) * vesc
        refined_v_tan = float(result.x[1]) * vesc
        
        # Run final simulation to get full trajectory and score
        system = generate_system(
            refined_v_rad,
            refined_v_tan,
            planet_mass=candidate.planet_mass,
            fairy_mass=candidate.fairy_mass,
            radius=candidate.radius,
            G=candidate.G,
        )
        
        period = orbital_period(candidate.G, candidate.planet_mass, candidate.radius)
        dt = period / steps_per_period
        t_end = n_periods * period
        
        traj = run(system, dt=dt, t_end=t_end, record_every=1)
        score_dict = self.score_evaluator.evaluate(traj)
        
        refined_candidate = Candidate(
            v_rad=refined_v_rad,
            v_tan=refined_v_tan,
            k=candidate.k,
            planet_mass=candidate.planet_mass,
            fairy_mass=candidate.fairy_mass,
            radius=candidate.radius,
            G=candidate.G,
            status=SimulationStatus.SUCCESS,
            score=score_dict["score"],
            integration_time=t_end,
            score_components=score_dict,
            positions=traj.positions,
            velocities=traj.velocities,
            times=traj.times,
            verified=False,  # Still needs verification
            metadata={
                "refinement_success": result.success,
                "refinement_iterations": result.nit,
                "initial_score": candidate.score,
                "refined_score": score_dict["score"],
            },
        )
        
        return refined_candidate
