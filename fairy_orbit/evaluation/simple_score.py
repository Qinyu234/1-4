"""Simplified score evaluator for ranking feasible candidates."""

from __future__ import annotations

from dataclasses import dataclass

from fairy_orbit.analysis.evaluator import distance_matrix_error, energy_drift
from fairy_orbit.simulation.trajectory import Trajectory


@dataclass
class SimpleScoreConfig:
    """Configuration for simplified scoring."""
    energy_weight: float = 1.0  # Weight for energy drift component


class SimpleScoreEvaluator:
    """
    Simplified score evaluator.
    
    Score = distance_matrix_error + energy_weight * energy_drift
    
    Only ranks feasible trajectories (collision/escape handled by constraint filter).
    """
    
    def __init__(self, config: SimpleScoreConfig | None = None):
        self.config = config or SimpleScoreConfig()
    
    def evaluate(
        self,
        traj: Trajectory,
        period_index: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate trajectory and return score components.
        
        Args:
            traj: Trajectory to evaluate
            period_index: Index to evaluate at (default: final frame)
        
        Returns:
            Dictionary with score components
        """
        if period_index is None:
            period_index = len(traj.times) - 1
        
        pos0 = traj.positions[0]
        posT = traj.positions[period_index]
        
        # Distance matrix error (continuous, permutation-invariant)
        dist_err = distance_matrix_error(pos0, posT, fairy_indices=(1, 2, 3, 4))
        
        # Energy drift
        e_drift = energy_drift(traj.energies)
        
        # Total score
        score = dist_err + self.config.energy_weight * e_drift
        
        return {
            "score": float(score),
            "distance_matrix_error": float(dist_err),
            "energy_drift": float(e_drift),
        }
