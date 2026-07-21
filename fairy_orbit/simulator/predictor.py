"""Event predictor for encounter detection using Kepler trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize as optimize

from fairy_orbit.simulator.kepler import KeplerOrbit


@dataclass
class EncounterPrediction:
    """Prediction of an encounter between two bodies."""
    time: float  # Predicted encounter time
    body_i: int  # Index of first body
    body_j: int  # Index of second body
    distance: float  # Predicted distance at encounter
    valid: bool = True  # Whether prediction is valid


@dataclass
class PredictorConfig:
    """Configuration for event predictor."""
    trigger_radius: float = 5.0  # Distance threshold for encounter prediction
    max_time_horizon: float = 1000.0  # Maximum time to look ahead
    time_tolerance: float = 1e-6  # Tolerance for root finding
    max_iterations: int = 50  # Maximum iterations for root finding


class EventPredictor:
    """Predict encounters between bodies using analytic Kepler trajectories."""
    
    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()
    
    def distance_function(
        self,
        t: float,
        orbit_i: KeplerOrbit,
        orbit_j: KeplerOrbit,
    ) -> float:
        """
        Compute distance between two bodies at time t using Kepler propagation.
        
        Args:
            t: Time to evaluate
            orbit_i: Kepler orbit of body i
            orbit_j: Kepler orbit of body j
        
        Returns:
            Distance between bodies at time t
        """
        pos_i = orbit_i.get_position_at_time(t)
        pos_j = orbit_j.get_position_at_time(t)
        return np.linalg.norm(pos_i - pos_j)
    
    def predict_encounter(
        self,
        orbit_i: KeplerOrbit,
        orbit_j: KeplerOrbit,
        current_time: float,
    ) -> EncounterPrediction:
        """
        Predict the next encounter between two bodies.
        
        Finds the earliest time t > current_time where distance(t) <= trigger_radius.
        
        Args:
            orbit_i: Kepler orbit of body i
            orbit_j: Kepler orbit of body j
            current_time: Current simulation time
        
        Returns:
            Encounter prediction
        """
        # Define function to find root of: distance(t) - trigger_radius = 0
        def distance_minus_trigger(t: float) -> float:
            return self.distance_function(t, orbit_i, orbit_j) - self.config.trigger_radius
        
        # Check current distance
        current_dist = self.distance_function(current_time, orbit_i, orbit_j)
        
        # If already within trigger radius, return immediate encounter
        if current_dist <= self.config.trigger_radius:
            return EncounterPrediction(
                time=current_time,
                body_i=0,  # Will be set by caller
                body_j=0,
                distance=current_dist,
                valid=True,
            )
        
        # Search for minimum distance in time horizon
        # Use golden section search or similar
        try:
            # First, find approximate minimum by sampling
            n_samples = 100
            t_samples = np.linspace(
                current_time,
                current_time + self.config.time_horizon,
                n_samples,
            )
            dist_samples = [self.distance_function(t, orbit_i, orbit_j) for t in t_samples]
            
            min_idx = int(np.argmin(dist_samples))
            min_dist = dist_samples[min_idx]
            min_t = t_samples[min_idx]
            
            # If minimum distance is still above trigger, no encounter predicted
            if min_dist > self.config.trigger_radius:
                return EncounterPrediction(
                    time=float('inf'),
                    body_i=0,
                    body_j=0,
                    distance=min_dist,
                    valid=False,
                )
            
            # Refine using local optimization around minimum
            result = optimize.minimize_scalar(
                lambda t: self.distance_function(t, orbit_i, orbit_j),
                bracket=(t_samples[max(0, min_idx - 5)], t_samples[min(n_samples - 1, min_idx + 5)]),
                method='brent',
                tol=self.config.time_tolerance,
            )
            
            if result.success and result.fun <= self.config.trigger_radius:
                return EncounterPrediction(
                    time=result.x,
                    body_i=0,
                    body_j=0,
                    distance=result.fun,
                    valid=True,
                )
            
            # Fallback: use the sampled minimum
            return EncounterPrediction(
                time=min_t,
                body_i=0,
                body_j=0,
                distance=min_dist,
                valid=min_dist <= self.config.trigger_radius,
            )
            
        except Exception:
            # If optimization fails, return invalid prediction
            return EncounterPrediction(
                time=float('inf'),
                body_i=0,
                body_j=0,
                distance=float('inf'),
                valid=False,
            )
    
    def predict_next_encounter(
        self,
        orbits: list[KeplerOrbit],
        current_time: float,
    ) -> EncounterPrediction:
        """
        Predict the earliest encounter among all body pairs.
        
        Args:
            orbits: List of Kepler orbits for all bodies
            current_time: Current simulation time
        
        Returns:
        Earliest encounter prediction
        """
        n = len(orbits)
        earliest_prediction = EncounterPrediction(
            time=float('inf'),
            body_i=0,
            body_j=0,
            distance=float('inf'),
            valid=False,
        )
        
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                prediction = self.predict_encounter(orbits[i], orbits[j], current_time)
                prediction.body_i = i
                prediction.body_j = j
                
                if prediction.valid and prediction.time < earliest_prediction.time:
                    earliest_prediction = prediction
        
        return earliest_prediction
