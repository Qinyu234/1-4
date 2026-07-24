"""Perturbation calculation and hierarchical transition logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from fairy_orbit.physics.body import Body, System


class DynamicMode(Enum):
    """Dynamic regime mode."""
    KEPLER = "kepler"  # Independent Kepler orbits (1+1 for each fairy)
    PAIR = "pair"  # 2+1 subsystem (planet + 2 fairies)
    TRIPLE = "triple"  # 3+1 subsystem (planet + 3 fairies)
    FULL = "full"  # 4+1 subsystem (planet + all 4 fairies)


@dataclass
class TransitionConfig:
    """Configuration for hierarchical transitions."""
    enter_threshold: float = 0.1  # η threshold to enter higher-order mode
    exit_threshold: float = 0.05  # η threshold to return to lower-order mode
    central_body_index: int = 0  # Index of central planet
    fairy_indices: tuple[int, ...] = (1, 2, 3, 4)  # Indices of fairy bodies


class PerturbationCalculator:
    """Calculate perturbation ratio η = |a_perturb| / |a_central|."""
    
    def __init__(self, config: TransitionConfig | None = None):
        self.config = config or TransitionConfig()
    
    def compute_central_acceleration(
        self,
        body: Body,
        central_body: Body,
        G: float,
    ) -> np.ndarray:
        """
        Compute acceleration due to central body gravity.
        
        Args:
            body: Body experiencing acceleration
            central_body: Central body
            G: Gravitational constant
        
        Returns:
            Acceleration vector from central body
        """
        r = central_body.position - body.position
        dist = np.linalg.norm(r)
        if dist < 1e-10:
            return np.zeros(3)
        return G * central_body.mass * r / dist**3
    
    def compute_perturbation_acceleration(
        self,
        body: Body,
        perturbing_bodies: list[Body],
        G: float,
    ) -> np.ndarray:
        """
        Compute acceleration due to perturbing bodies.
        
        Args:
            body: Body experiencing acceleration
            perturbing_bodies: List of perturbing bodies
            G: Gravitational constant
        
        Returns:
            Total acceleration from perturbing bodies
        """
        acc = np.zeros(3)
        for perturber in perturbing_bodies:
            r = perturber.position - body.position
            dist = np.linalg.norm(r)
            if dist > 1e-10:
                acc += G * perturber.mass * r / dist**3
        return acc
    
    def compute_perturbation_ratio(
        self,
        body: Body,
        central_body: Body,
        perturbing_bodies: list[Body],
        G: float,
    ) -> float:
        """
        Compute perturbation ratio η = |a_perturb| / |a_central|.
        
        Args:
            body: Body to evaluate
            central_body: Central body
            perturbing_bodies: List of perturbing bodies
            G: Gravitational constant
        
        Returns:
            Perturbation ratio η
        """
        a_central = self.compute_central_acceleration(body, central_body, G)
        a_perturb = self.compute_perturbation_acceleration(body, perturbing_bodies, G)
        
        a_central_mag = np.linalg.norm(a_central)
        a_perturb_mag = np.linalg.norm(a_perturb)
        
        if a_central_mag < 1e-10:
            return float('inf') if a_perturb_mag > 1e-10 else 0.0
        
        return a_perturb_mag / a_central_mag
    
    def compute_all_perturbation_ratios(
        self,
        system: System,
    ) -> dict[int, float]:
        """
        Compute perturbation ratio for all fairy bodies.
        
        Args:
            system: Current system state
        
        Returns:
            Dictionary mapping body index to perturbation ratio
        """
        central_body = system.bodies[self.config.central_body_index]
        ratios = {}
        
        for i in self.config.fairy_indices:
            body = system.bodies[i]
            # All other fairies are perturbers
            perturbers = [
                system.bodies[j]
                for j in self.config.fairy_indices
                if j != i
            ]
            ratios[i] = self.compute_perturbation_ratio(
                body, central_body, perturbers, system.G
            )
        
        return ratios


class TransitionManager:
    """Manage hierarchical transitions between dynamic modes."""
    
    def __init__(self, config: TransitionConfig | None = None):
        self.config = config or TransitionConfig()
        self.perturbation_calc = PerturbationCalculator(config)
        self.current_mode = DynamicMode.KEPLER
        self.active_participants: set[int] = set()  # Bodies in active subsystem
    
    def determine_mode(self, system: System) -> DynamicMode:
        """
        Determine appropriate dynamic mode based on perturbation ratios.
        
        Args:
            system: Current system state
        
        Returns:
            Appropriate dynamic mode
        """
        ratios = self.perturbation_calc.compute_all_perturbation_ratios(system)
        
        # Count bodies with significant perturbation
        significant = [
            i for i, eta in ratios.items()
            if eta >= self.config.enter_threshold
        ]
        
        n_significant = len(significant)
        
        if n_significant == 0:
            return DynamicMode.KEPLER
        elif n_significant == 1:
            # Single body with perturbation - treat as pair (planet + that body)
            return DynamicMode.PAIR
        elif n_significant == 2:
            # Two bodies with perturbation - triple subsystem
            return DynamicMode.TRIPLE
        else:
            # 3 or 4 bodies with perturbation - full system
            return DynamicMode.FULL
    
    def should_transition(self, system: System) -> bool:
        """
        Check if a mode transition is needed.
        
        Uses hysteresis: enter_threshold > exit_threshold.
        
        Args:
            system: Current system state
        
        Returns:
            True if transition needed, False otherwise
        """
        new_mode = self.determine_mode(system)
        return new_mode != self.current_mode
    
    def get_active_subsystem(
        self,
        system: System,
        mode: DynamicMode,
    ) -> tuple[int, ...]:
        """
        Get body indices for active subsystem in given mode.
        
        Args:
            system: Current system state
            mode: Dynamic mode
        
        Returns:
            Tuple of body indices in active subsystem
        """
        if mode == DynamicMode.KEPLER:
            return ()  # No active subsystem, all bodies in Kepler mode
        
        ratios = self.perturbation_calc.compute_all_perturbation_ratios(system)
        
        # Get bodies with significant perturbation
        significant = [
            i for i, eta in ratios.items()
            if eta >= self.config.exit_threshold
        ]
        
        # Always include central body
        participants = [self.config.central_body_index]
        
        if mode == DynamicMode.PAIR:
            # Add most perturbed body
            if significant:
                most_perturbed = max(significant, key=lambda i: ratios[i])
                participants.append(most_perturbed)
        elif mode == DynamicMode.TRIPLE:
            # Add 2 most perturbed bodies
            significant_sorted = sorted(significant, key=lambda i: ratios[i], reverse=True)
            participants.extend(significant_sorted[:2])
        else:  # FULL
            # Add all fairies
            participants.extend(self.config.fairy_indices)
        
        return tuple(participants)
    
    def transition_to(
        self,
        new_mode: DynamicMode,
        system: System,
    ) -> tuple[int, ...]:
        """
        Perform transition to new mode.
        
        Args:
            new_mode: New dynamic mode
            system: Current system state
        
        Returns:
            Tuple of body indices in new active subsystem
        """
        self.current_mode = new_mode
        self.active_participants = set(self.get_active_subsystem(system, new_mode))
        return tuple(self.active_participants)
