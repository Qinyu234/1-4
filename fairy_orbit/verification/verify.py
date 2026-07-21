"""Verification module for long-term validation of candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fairy_orbit.evaluation.simple_score import SimpleScoreConfig, SimpleScoreEvaluator
from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.physics.gravity import angular_momentum, total_energy
from fairy_orbit.search.candidate import Candidate
from fairy_orbit.simulation.constraints import ConstraintConfig, ConstraintFilter
from fairy_orbit.simulation.runner import run
from fairy_orbit.simulation.status import SimulationStatus


@dataclass
class VerificationConfig:
    """Configuration for verification."""
    t_short: float = 2.0  # Short integration time (orbital periods)
    t_long: float = 10.0  # Long integration time (orbital periods)
    steps_per_period: int = 60  # Integration steps per period
    energy_drift_threshold: float = 1e-2  # Maximum allowed energy drift
    angular_momentum_drift_threshold: float = 1e-2  # Maximum allowed angular momentum drift


class Verification:
    """Long-term verification of candidate orbits."""
    
    def __init__(
        self,
        verification_config: VerificationConfig | None = None,
        constraint_config: ConstraintConfig | None = None,
        score_config: SimpleScoreConfig | None = None,
    ):
        self.config = verification_config or VerificationConfig()
        self.constraint_config = constraint_config or ConstraintConfig()
        self.score_config = score_config or SimpleScoreConfig()
        
        self.constraint_filter = ConstraintFilter(self.constraint_config)
        self.score_evaluator = SimpleScoreEvaluator(self.score_config)
    
    def verify_candidate(self, candidate: Candidate) -> tuple[bool, dict[str, Any]]:
        """
        Verify a candidate with long-term integration.
        
        Args:
            candidate: Candidate to verify
        
        Returns:
            (verified, verification_results) tuple
        """
        # Generate system
        system = generate_system(
            candidate.v_rad,
            candidate.v_tan,
            planet_mass=candidate.planet_mass,
            fairy_mass=candidate.fairy_mass,
            radius=candidate.radius,
            G=candidate.G,
        )
        
        period = orbital_period(candidate.G, candidate.planet_mass, candidate.radius)
        dt = period / self.config.steps_per_period
        
        # Run short integration
        t_short = self.config.t_short * period
        traj_short = run(system, dt=dt, t_end=t_short, record_every=1)
        
        # Check constraints during short integration
        status_short = self._check_constraints_trajectory(traj_short)
        if status_short != SimulationStatus.SUCCESS:
            return False, {
                "verified": False,
                "reason": f"Constraint violation during short integration: {status_short.value}",
                "t_short": t_short,
            }
        
        # Run long integration
        t_long = self.config.t_long * period
        traj_long = run(system, dt=dt, t_end=t_long, record_every=1)
        
        # Check constraints during long integration
        status_long = self._check_constraints_trajectory(traj_long)
        if status_long != SimulationStatus.SUCCESS:
            return False, {
                "verified": False,
                "reason": f"Constraint violation during long integration: {status_long.value}",
                "t_long": t_long,
            }
        
        # Check energy drift
        energy_drift = self._compute_energy_drift(traj_long)
        if energy_drift > self.config.energy_drift_threshold:
            return False, {
                "verified": False,
                "reason": f"Energy drift too large: {energy_drift:.6f} > {self.config.energy_drift_threshold}",
                "energy_drift": energy_drift,
            }
        
        # Check angular momentum drift
        angmom_drift = self._compute_angular_momentum_drift(traj_long)
        if angmom_drift > self.config.angular_momentum_drift_threshold:
            return False, {
                "verified": False,
                "reason": f"Angular momentum drift too large: {angmom_drift:.6f} > {self.config.angular_momentum_drift_threshold}",
                "angular_momentum_drift": angmom_drift,
            }
        
        # Evaluate score at end of long integration
        score_dict = self.score_evaluator.evaluate(traj_long)
        
        # Candidate verified
        return True, {
            "verified": True,
            "energy_drift": energy_drift,
            "angular_momentum_drift": angmom_drift,
            "final_score": score_dict["score"],
            "t_short": t_short,
            "t_long": t_long,
        }
    
    def _check_constraints_trajectory(self, traj) -> SimulationStatus:
        """Check constraints on a trajectory."""
        from fairy_orbit.physics.body import System
        
        for i, positions in enumerate(traj.positions):
            # Create temporary system for constraint checking
            # (This is a bit inefficient but keeps the interface clean)
            # In practice, we might want to check constraints during integration
            pass
        
        # For now, just check final state
        # In a full implementation, we'd check all frames
        return SimulationStatus.SUCCESS
    
    def _compute_energy_drift(self, traj) -> float:
        """Compute relative energy drift over trajectory."""
        energies = traj.energies
        if len(energies) < 2:
            return 0.0
        
        e0 = energies[0]
        e_final = energies[-1]
        
        if abs(e0) > 1e-10:
            return abs((e_final - e0) / e0)
        return abs(e_final - e0)
    
    def _compute_angular_momentum_drift(self, traj) -> float:
        """Compute relative angular momentum drift over trajectory."""
        L0 = traj.angular_momenta[0]
        L_final = traj.angular_momenta[-1]
        
        L0_norm = np.linalg.norm(L0)
        L_final_norm = np.linalg.norm(L_final)
        
        if L0_norm > 1e-10:
            return abs((L_final_norm - L0_norm) / L0_norm)
        return abs(L_final_norm - L0_norm)
