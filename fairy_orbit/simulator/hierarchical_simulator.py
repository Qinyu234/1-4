"""Event-driven hierarchical N-body simulator for Periodic Encounter Orbits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fairy_orbit.events.timeline import EventTimeline, EventType
from fairy_orbit.physics.body import System
from fairy_orbit.simulator.integrator import IntegratorConfig, SubsystemIntegrator
from fairy_orbit.simulator.kepler import KeplerOrbit
from fairy_orbit.simulator.predictor import EncounterPrediction, PredictorConfig, EventPredictor
from fairy_orbit.simulator.transition import DynamicMode, TransitionConfig, TransitionManager


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical simulator."""
    # Predictor config
    predictor_config: PredictorConfig = None
    # Transition config
    transition_config: TransitionConfig = None
    # Integrator config
    integrator_config: IntegratorConfig = None
    # Simulation limits
    max_time: float = 10000.0  # Maximum simulation time
    max_events: int = 1000  # Maximum number of events
    
    def __post_init__(self):
        if self.predictor_config is None:
            self.predictor_config = PredictorConfig()
        if self.transition_config is None:
            self.transition_config = TransitionConfig()
        if self.integrator_config is None:
            self.integrator_config = IntegratorConfig()


class HierarchicalSimulator:
    """
    Event-driven hierarchical N-body simulator.
    
    Execution loop:
    Generate Kepler Orbits → Predict Next Encounter → Jump to Predicted Event
    → Validate Perturbation → Switch Dynamic Regime → Integrate Active Subsystem
    → Return to Kepler
    """
    
    def __init__(self, config: HierarchicalConfig | None = None):
        self.config = config or HierarchicalConfig()
        
        # Initialize components
        self.predictor = EventPredictor(self.config.predictor_config)
        self.transition_manager = TransitionManager(self.config.transition_config)
        self.integrator = SubsystemIntegrator(self.config.integrator_config)
        
        # State
        self.current_time = 0.0
        self.event_count = 0
        self.timeline = EventTimeline()
        self.kepler_orbits: dict[int, KeplerOrbit] = {}
    
    def initialize_kepler_orbits(
        self,
        system: System,
        fairy_indices: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        """
        Initialize Kepler orbits for all fairy bodies.
        
        Args:
            system: Initial system state
            fairy_indices: Indices of fairy bodies
        """
        central_body = system.bodies[self.config.transition_config.central_body_index]
        
        for i in fairy_indices:
            body = system.bodies[i]
            orbit = KeplerOrbit(
                position=body.position.copy(),
                velocity=body.velocity.copy(),
                central_mass=central_body.mass,
                G=system.G,
                epoch=self.current_time,
            )
            self.kepler_orbits[i] = orbit
    
    def propagate_kepler_to_time(
        self,
        time: float,
        system: System,
    ) -> None:
        """
        Propagate all fairy bodies analytically using Kepler orbits.
        
        Args:
            time: Target time
            system: System to update
        """
        dt = time - self.current_time
        
        for i, orbit in self.kepler_orbits.items():
            new_pos, new_vel = orbit.propagate(dt)
            system.bodies[i].position = new_pos
            system.bodies[i].velocity = new_vel
            orbit.epoch = time
        
        self.current_time = time
    
    def validate_perturbation(
        self,
        system: System,
    ) -> tuple[float, bool]:
        """
        Validate perturbation ratio at current state.
        
        Args:
            system: Current system state
        
        Returns:
            (eta, should_enter_encounter) tuple
        """
        ratios = self.transition_manager.perturbation_calc.compute_all_perturbation_ratios(system)
        max_eta = max(ratios.values()) if ratios else 0.0
        
        should_enter = max_eta >= self.config.transition_config.enter_threshold
        return max_eta, should_enter
    
    def run_encounter(
        self,
        system: System,
        active_indices: tuple[int, ...],
    ) -> float:
        """
        Run numerical integration for active subsystem.
        
        Integrates until perturbation drops below exit threshold.
        
        Args:
            system: System to integrate
            active_indices: Indices of bodies in active subsystem
        
        Returns:
            Duration of encounter
        """
        start_time = self.current_time
        
        # Monitor function to check exit condition
        def monitor(sys: System, t: float) -> bool:
            ratios = self.transition_manager.perturbation_calc.compute_all_perturbation_ratios(sys)
            max_eta = max(ratios.values()) if ratios else 0.0
            return max_eta < self.config.transition_config.exit_threshold
        
        # Integrate with monitoring
        # Use a reasonable time limit for single encounter
        encounter_duration = 100.0  # Could be made configurable
        final_time, _ = self.integrator.integrate(
            system,
            active_indices,
            encounter_duration,
            monitor_callback=monitor,
        )
        
        self.current_time = final_time
        return final_time - start_time
    
    def refit_kepler_orbits(
        self,
        system: System,
    ) -> None:
        """
        Refit Kepler orbits after encounter.
        
        Args:
            system: Current system state
        """
        central_body = system.bodies[self.config.transition_config.central_body_index]
        
        for i in self.kepler_orbits.keys():
            body = system.bodies[i]
            self.kepler_orbits[i] = KeplerOrbit(
                position=body.position.copy(),
                velocity=body.velocity.copy(),
                central_mass=central_body.mass,
                G=system.G,
                epoch=self.current_time,
            )
    
    def simulate(
        self,
        system: System,
        t_end: float,
    ) -> tuple[System, EventTimeline]:
        """
        Run event-driven hierarchical simulation.
        
        Args:
            system: Initial system state
            t_end: End time for simulation
        
        Returns:
            (final_system, timeline) tuple
        """
        self.current_time = 0.0
        self.event_count = 0
        self.timeline = EventTimeline()
        
        # Initialize Kepler orbits
        fairy_indices = self.config.transition_config.fairy_indices
        self.initialize_kepler_orbits(system, fairy_indices)
        
        # Start in Kepler mode
        self.timeline.start_event(EventType.KEPLER_START, self.current_time, DynamicMode.KEPLER)
        
        # Main event loop
        while self.current_time < t_end and self.event_count < self.config.max_events:
            # Predict next encounter
            orbits = [self.kepler_orbits[i] for i in fairy_indices]
            prediction = self.predictor.predict_next_encounter(orbits, self.current_time)
            
            # If no encounter predicted, jump to t_end
            if not prediction.valid or prediction.time > t_end:
                self.propagate_kepler_to_time(t_end, system)
                self.timeline.end_event(t_end)
                break
            
            # Jump to predicted encounter time
            self.timeline.end_event(prediction.time)
            self.propagate_kepler_to_time(prediction.time, system)
            
            # Validate perturbation
            max_eta, should_enter = self.validate_perturbation(system)
            
            if not should_enter:
                # False prediction, continue Kepler
                self.timeline.add_instant_event(
                    EventType.TRANSITION,
                    self.current_time,
                    DynamicMode.KEPLER,
                    (),
                    {"rejected_prediction": True, "eta": max_eta},
                )
                self.timeline.start_event(EventType.KEPLER_START, self.current_time, DynamicMode.KEPLER)
                continue
            
            # Enter encounter mode
            new_mode = self.transition_manager.determine_mode(system)
            active_indices = self.transition_manager.get_active_subsystem(system, new_mode)
            
            self.timeline.start_event(
                EventType.ENCOUNTER_START,
                self.current_time,
                new_mode,
                active_indices,
            )
            
            # Run numerical integration
            encounter_duration = self.run_encounter(system, active_indices)
            
            # End encounter
            max_eta_after, _ = self.validate_perturbation(system)
            self.timeline.end_event(self.current_time, max_eta_after)
            
            # Refit Kepler orbits
            self.refit_kepler_orbits(system)
            
            # Return to Kepler mode
            self.timeline.start_event(EventType.KEPLER_START, self.current_time, DynamicMode.KEPLER)
            
            self.event_count += 1
        
        # Close final event
        if self.timeline.current_event is not None:
            self.timeline.end_event(self.current_time)
        
        return system, self.timeline
