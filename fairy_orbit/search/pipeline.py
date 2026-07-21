"""New search pipeline with constraint filtering, candidate archive, refinement, and verification."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from fairy_orbit.evaluation.geometry import GeometryConfig, GeometryEvaluator
from fairy_orbit.evaluation.periodicity import PeriodicityConfig, PeriodicityEvaluator
from fairy_orbit.evaluation.simple_score import SimpleScoreConfig, SimpleScoreEvaluator
from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.search.archive import CandidateArchive
from fairy_orbit.search.candidate import Candidate
from fairy_orbit.search.refinement import Refinement, RefinementConfig
from fairy_orbit.simulation.constraints import ConstraintConfig, ConstraintFilter
from fairy_orbit.simulation.hierarchical_simulator import HierarchicalConfig, HierarchicalSimulator
from fairy_orbit.simulation.runner import run
from fairy_orbit.simulation.status import SimulationStatus
from fairy_orbit.verification.verify import Verification, VerificationConfig


@dataclass
class SearchConfig:
    """Configuration for the search pipeline."""
    # Search parameters
    k_values: list[float] = None
    planet_mass: float = 1.0
    radius: float = 20.0
    G: float = 1.0
    
    # Grid search parameters
    alpha_grid: list[float] = None
    beta_grid: list[float] = None
    
    # Simulation parameters
    n_periods: float = 2.0
    steps_per_period: int = 60
    
    # Simulator choice
    use_hierarchical_simulator: bool = True  # Use event-driven hierarchical simulator
    hierarchical_config: HierarchicalConfig = None
    
    # Evaluation budget
    max_evaluations: int = 1000
    
    # Archive parameters
    archive_size: int = 100
    
    # Refinement parameters
    enable_refinement: bool = True
    refinement_config: RefinementConfig = None
    
    # Verification parameters
    enable_verification: bool = True
    verification_config: VerificationConfig = None
    
    # Constraint parameters
    constraint_config: ConstraintConfig = None
    
    # Score parameters
    score_config: SimpleScoreConfig = None
    geometry_config: GeometryConfig = None
    periodicity_config: PeriodicityConfig = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1.0]
        if self.alpha_grid is None:
            self.alpha_grid = list(np.linspace(0.0, 1.0, 5))
        if self.beta_grid is None:
            self.beta_grid = list(np.linspace(0.4, 1.2, 5))
        if self.hierarchical_config is None:
            self.hierarchical_config = HierarchicalConfig()
        if self.refinement_config is None:
            self.refinement_config = RefinementConfig()
        if self.verification_config is None:
            self.verification_config = VerificationConfig()
        if self.constraint_config is None:
            self.constraint_config = ConstraintConfig()
        if self.score_config is None:
            self.score_config = SimpleScoreConfig()
        if self.geometry_config is None:
            self.geometry_config = GeometryConfig()
        if self.periodicity_config is None:
            self.periodicity_config = PeriodicityConfig()


class SearchPipeline:
    """New search pipeline with constraint filtering and candidate management."""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        
        # Initialize components
        self.archive = CandidateArchive(max_size=config.archive_size)
        self.constraint_filter = ConstraintFilter(config.constraint_config)
        self.score_evaluator = SimpleScoreEvaluator(config.score_config)
        self.geometry_evaluator = GeometryEvaluator(config.geometry_config)
        self.periodicity_evaluator = PeriodicityEvaluator(config.periodicity_config)
        self.hierarchical_simulator = HierarchicalSimulator(config.hierarchical_config)
        self.refinement = Refinement(
            config.refinement_config,
            config.constraint_config,
            config.score_config,
        )
        self.verification = Verification(
            config.verification_config,
            config.constraint_config,
            config.score_config,
        )
        
        # Tracking
        self.evaluation_count = 0
        self.candidates_generated = 0
        self.candidates_refined = 0
        self.candidates_verified = 0
    
    def generate_initial_condition(
        self,
        k: float,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> tuple[float, float]:
        """
        Generate initial velocity conditions.
        
        Args:
            k: Mass ratio
            alpha: v_rad / v_esc (random if None)
            beta: v_tan / v_esc (random if None)
        
        Returns:
            (v_rad, v_tan) tuple
        """
        vesc = escape_speed(self.config.G, self.config.planet_mass, self.config.radius)
        
        if alpha is None:
            alpha = random.choice(self.config.alpha_grid)
        if beta is None:
            beta = random.choice(self.config.beta_grid)
        
        v_rad = alpha * vesc
        v_tan = beta * vesc
        
        return v_rad, v_tan
    
    def simulate_and_evaluate(
        self,
        v_rad: float,
        v_tan: float,
        k: float,
    ) -> Candidate | None:
        """
        Simulate and evaluate a single initial condition.
        
        Args:
            v_rad: Radial velocity
            v_tan: Tangential velocity
            k: Mass ratio
        
        Returns:
            Candidate if successful, None if constraint violation
        """
        # Generate system
        system = generate_system(
            v_rad,
            v_tan,
            planet_mass=self.config.planet_mass,
            fairy_mass=k * self.config.planet_mass,
            radius=self.config.radius,
            G=self.config.G,
        )
        
        # Check initial constraints
        status = self.constraint_filter.check_constraints(system)
        if status != SimulationStatus.SUCCESS:
            return None
        
        # Run simulation
        period = orbital_period(self.config.G, self.config.planet_mass, self.config.radius)
        t_end = self.config.n_periods * period
        
        try:
            if self.config.use_hierarchical_simulator:
                # Use event-driven hierarchical simulator
                final_system, timeline = self.hierarchical_simulator.simulate(system, t_end)
                
                # Evaluate using timeline
                periodicity_dict = self.periodicity_evaluator.evaluate_timeline(timeline)
                
                # For now, use periodicity score as primary score
                score_dict = {
                    "score": periodicity_dict["periodicity_score"],
                    "periodicity_score": periodicity_dict["periodicity_score"],
                    "sequence_error": periodicity_dict["sequence_error"],
                    "autocorr_period": periodicity_dict["autocorr_period"],
                    "autocorr_confidence": periodicity_dict["autocorr_confidence"],
                    "n_events": periodicity_dict["n_events"],
                }
                
                # Store timeline in metadata
                timeline_data = timeline.to_dict()
                
                # Create candidate with timeline data
                candidate = Candidate(
                    v_rad=v_rad,
                    v_tan=v_tan,
                    k=k,
                    planet_mass=self.config.planet_mass,
                    fairy_mass=k * self.config.planet_mass,
                    radius=self.config.radius,
                    G=self.config.G,
                    status=SimulationStatus.SUCCESS,
                    score=score_dict["score"],
                    integration_time=t_end,
                    score_components=score_dict,
                    positions=None,  # Hierarchical simulator doesn't store full trajectory
                    velocities=None,
                    times=None,
                    verified=False,
                    metadata={"timeline": timeline_data},
                )
            else:
                # Use traditional fixed-step simulator
                dt = period / self.config.steps_per_period
                traj = run(system, dt=dt, t_end=t_end, record_every=1)
                
                # Check constraints during simulation
                final_status = self.constraint_filter.check_constraints(system)
                if final_status != SimulationStatus.SUCCESS:
                    return None
                
                # Evaluate score
                score_dict = self.score_evaluator.evaluate(traj)
                
                # Also evaluate geometry
                geometry_dict = self.geometry_evaluator.evaluate(traj)
                score_dict.update(geometry_dict)
                
                # Create candidate
                candidate = Candidate(
                    v_rad=v_rad,
                    v_tan=v_tan,
                    k=k,
                    planet_mass=self.config.planet_mass,
                    fairy_mass=k * self.config.planet_mass,
                    radius=self.config.radius,
                    G=self.config.G,
                    status=SimulationStatus.SUCCESS,
                    score=score_dict["score"],
                    integration_time=t_end,
                    score_components=score_dict,
                    positions=traj.positions,
                    velocities=traj.velocities,
                    times=traj.times,
                    verified=False,
                )
        except Exception:
            return None
        
        self.evaluation_count += 1
        return candidate
    
    def search(self, progress_callback=None) -> CandidateArchive:
        """
        Run the full search pipeline.
        
        Args:
            progress_callback: Optional callback function for progress updates
        
        Returns:
            CandidateArchive with best candidates
        """
        if progress_callback is None:
            progress_callback = lambda msg: None
        
        progress_callback("Starting search pipeline")
        
        # Grid search over initial conditions
        for k in self.config.k_values:
            progress_callback(f"Searching k={k:.6f}")
            
            for alpha in self.config.alpha_grid:
                for beta in self.config.beta_grid:
                    if self.evaluation_count >= self.config.max_evaluations:
                        progress_callback(f"Reached evaluation budget: {self.evaluation_count}")
                        return self.archive
                    
                    v_rad, v_tan = self.generate_initial_condition(k, alpha, beta)
                    
                    # Simulate and evaluate
                    candidate = self.simulate_and_evaluate(v_rad, v_tan, k)
                    
                    if candidate is not None:
                        self.candidates_generated += 1
                        self.archive.add(candidate)
                        progress_callback(
                            f"  Generated candidate: score={candidate.score:.4f}, "
                            f"archive size={self.archive.size()}"
                        )
        
        progress_callback(f"Grid search complete: {self.evaluation_count} evaluations")
        
        # Refinement
        if self.config.enable_refinement:
            progress_callback("Starting refinement")
            
            # Get top candidates for refinement
            top_candidates = self.archive.get_best(n=10)
            
            for candidate in top_candidates:
                if self.evaluation_count >= self.config.max_evaluations:
                    break
                
                progress_callback(f"Refining candidate with score={candidate.score:.4f}")
                
                try:
                    refined = self.refinement.refine_candidate(
                        candidate,
                        n_periods=self.config.n_periods,
                        steps_per_period=self.config.steps_per_period,
                    )
                    self.archive.add(refined)
                    self.candidates_refined += 1
                    progress_callback(
                        f"  Refined: {candidate.score:.4f} -> {refined.score:.4f}"
                    )
                except Exception as e:
                    progress_callback(f"  Refinement failed: {e}")
        
        # Verification
        if self.config.enable_verification:
            progress_callback("Starting verification")
            
            # Get top candidates for verification
            top_candidates = self.archive.get_best(n=5)
            
            for candidate in top_candidates:
                progress_callback(f"Verifying candidate with score={candidate.score:.4f}")
                
                verified, results = self.verification.verify_candidate(candidate)
                
                if verified:
                    candidate.verified = True
                    candidate.metadata["verification"] = results
                    self.candidates_verified += 1
                    progress_callback(f"  Candidate verified!")
                else:
                    progress_callback(f"  Verification failed: {results.get('reason', 'unknown')}")
        
        progress_callback("Search pipeline complete")
        progress_callback(f"Statistics:")
        progress_callback(f"  Evaluations: {self.evaluation_count}")
        progress_callback(f"  Candidates generated: {self.candidates_generated}")
        progress_callback(f"  Candidates refined: {self.candidates_refined}")
        progress_callback(f"  Candidates verified: {self.candidates_verified}")
        progress_callback(f"  Archive size: {self.archive.size()}")
        
        return self.archive
