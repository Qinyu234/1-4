"""Event timeline recording for hierarchical simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from fairy_orbit.simulator.transition import DynamicMode


class EventType(Enum):
    """Type of event."""
    KEPLER_START = "kepler_start"
    KEPLER_END = "kepler_end"
    ENCOUNTER_START = "encounter_start"
    ENCOUNTER_END = "encounter_end"
    TRANSITION = "transition"


@dataclass
class Event:
    """Single event record in the timeline."""
    event_type: EventType
    start_time: float
    end_time: float | None = None  # None for instantaneous events
    mode: DynamicMode | None = None
    participants: tuple[int, ...] = ()
    peak_eta: float | None = None  # Peak perturbation ratio during event
    metadata: dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def duration(self) -> float:
        """Get event duration."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "mode": self.mode.value if self.mode else None,
            "participants": list(self.participants),
            "peak_eta": self.peak_eta,
            "duration": self.duration(),
            "metadata": self.metadata,
        }


class EventTimeline:
    """Timeline of events from hierarchical simulation."""
    
    def __init__(self):
        self.events: list[Event] = []
        self.current_event: Event | None = None
    
    def start_event(
        self,
        event_type: EventType,
        time: float,
        mode: DynamicMode | None = None,
        participants: tuple[int, ...] = (),
    ) -> Event:
        """
        Start a new event.
        
        Args:
            event_type: Type of event
            time: Start time
            mode: Dynamic mode (if applicable)
            participants: Body indices involved
        
        Returns:
            The created event
        """
        # Close current event if open
        if self.current_event is not None:
            self.end_event(time)
        
        event = Event(
            event_type=event_type,
            start_time=time,
            mode=mode,
            participants=participants,
        )
        self.current_event = event
        self.events.append(event)
        return event
    
    def end_event(
        self,
        time: float,
        peak_eta: float | None = None,
    ) -> None:
        """
        End the current event.
        
        Args:
            time: End time
            peak_eta: Peak perturbation ratio during event
        """
        if self.current_event is not None:
            self.current_event.end_time = time
            self.current_event.peak_eta = peak_eta
            self.current_event = None
    
    def add_instant_event(
        self,
        event_type: EventType,
        time: float,
        mode: DynamicMode | None = None,
        participants: tuple[int, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an instantaneous event (start and end at same time).
        
        Args:
            event_type: Type of event
            time: Event time
            mode: Dynamic mode (if applicable)
            participants: Body indices involved
            metadata: Additional metadata
        """
        event = Event(
            event_type=event_type,
            start_time=time,
            end_time=time,
            mode=mode,
            participants=participants,
            metadata=metadata or {},
        )
        self.events.append(event)
    
    def get_events_by_type(self, event_type: EventType) -> list[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_events_by_mode(self, mode: DynamicMode) -> list[Event]:
        """Get all events with a specific dynamic mode."""
        return [e for e in self.events if e.mode == mode]
    
    def get_encounters(self) -> list[Event]:
        """Get all encounter events."""
        return self.get_events_by_type(EventType.ENCOUNTER_START)
    
    def get_total_time(self) -> float:
        """Get total simulation time."""
        if not self.events:
            return 0.0
        return max(e.end_time if e.end_time else e.start_time for e in self.events)
    
    def get_mode_duration(self, mode: DynamicMode) -> float:
        """Get total duration spent in a specific mode."""
        events = self.get_events_by_mode(mode)
        return sum(e.duration() for e in events)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get timeline statistics."""
        if not self.events:
            return {
                "total_events": 0,
                "total_time": 0.0,
                "n_encounters": 0,
                "n_transitions": 0,
                "mode_durations": {},
            }
        
        total_time = self.get_total_time()
        n_encounters = len(self.get_encounters())
        n_transitions = len(self.get_events_by_type(EventType.TRANSITION))
        
        mode_durations = {
            mode.value: self.get_mode_duration(mode)
            for mode in DynamicMode
        }
        
        return {
            "total_events": len(self.events),
            "total_time": total_time,
            "n_encounters": n_encounters,
            "n_transitions": n_transitions,
            "mode_durations": mode_durations,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert timeline to dictionary for serialization."""
        return {
            "events": [e.to_dict() for e in self.events],
            "statistics": self.get_statistics(),
        }
