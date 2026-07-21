"""Adaptive hierarchical N-body simulator with event detection."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

from fairy_orbit.physics.body import System

class InteractionMode(Enum):
    """Interaction mode for solver hierarchy."""
    NORMAL = "normal"  # Central-body dominated Kepler orbits
    PAIR = "2+1"       # 2+1: pair correction
    TRIPLE = "3+1"     # 3+1: numerical integration
    FULL = "4+1"       # 4+1: full N-body integration

class EventType(Enum):
    """Event type encoding."""
    PAIR = 1  # 2+1 interaction
    TRIPLE = 2  # 3+1 interaction
    FULL = 3  # 4+1 interaction

class Event:
    """Single event record."""
    def __init__(
        self,
        start_time: float,
        end_time: float,
        event_type: EventType | None = None,
        participants: tuple[int, ...] = (),
        type: str | None = None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.participants = participants
        if type is not None:
            self._type_str = type
            if type == "2+1":
                self.event_type = EventType.PAIR
            elif type == "3+1":
                self.event_type = EventType.TRIPLE
            else:
                self.event_type = EventType.FULL
        else:
            self.event_type = event_type
            if event_type == EventType.PAIR:
                self._type_str = "2+1"
            elif event_type == EventType.TRIPLE:
                self._type_str = "3+1"
            else:
                self._type_str = "4+1"

    @property
    def type(self) -> str:
        return self._type_str

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "type": self.type,
            "participants": list(self.participants),
        }

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive simulator."""
    influence_threshold: float = 1.5  # Distance threshold for close encounter
    event_tolerance: float = 1e-4  # Tolerance for event detection binary search
    max_iterations: int = 50  # Max iterations for binary search
    central_body_index: int = 0  # Index of central body
    outer_body_indices: tuple[int, ...] = (1, 2, 3, 4)  # Indices of outer bodies

class AdaptiveSimulator:
    """Adaptive hierarchical N-body simulator with event detection."""
    
    def __init__(self, config: AdaptiveConfig | None = None):
        self.config = config or AdaptiveConfig()
        self.events: list[Event] = []
        self.times: list[float] = []
        self.positions_history: list[np.ndarray] = []
        self.velocities_history: list[np.ndarray] = []
        self.energies: list[float] = []
        self.angular_momenta: list[np.ndarray] = []
    
    def determine_encounter_mode(
        self,
        positions: np.ndarray,
    ) -> tuple[str, tuple[int, ...]]:
        """Determine encounter mode based on distances between outer bodies."""
        threshold = self.config.influence_threshold
        n = positions.shape[0]
        
        # Calculate distances between fairies (indices 1 to n-1)
        dists = {}
        for i in range(1, n):
            for j in range(i + 1, n):
                dists[(i, j)] = np.linalg.norm(positions[i] - positions[j])
                
        if not dists:
            return "normal", ()
            
        closest_pair = min(dists.keys(), key=lambda k: dists[k])
        if dists[closest_pair] >= threshold:
            return "normal", ()
            
        participants = set(closest_pair)
        
        # Check third body
        for c in range(1, n):
            if c not in participants:
                if any(np.linalg.norm(positions[c] - positions[p]) < threshold for p in participants):
                    participants.add(c)
                    break
                    
        # Check fourth body
        for d in range(1, n):
            if d not in participants:
                if any(np.linalg.norm(positions[d] - positions[p]) < threshold for p in participants):
                    participants.add(d)
                    break
                    
        mode_map = {2: "2+1", 3: "3+1", 4: "4+1"}
        return mode_map[len(participants)], tuple(sorted(participants))

    def run(
        self,
        system: System,
        dt: float,
        t_end: float,
        record_every: int = 1,
    ) -> list[Event]:
        """Run adaptive simulation and return logged events."""
        self.events = []
        positions = system.positions().copy()
        velocities = system.velocities().copy()
        masses = system.masses()
        G = system.G
        
        t = 0.0
        active_event = None
        
        self.times = [t]
        self.positions_history = [positions.copy()]
        self.velocities_history = [velocities.copy()]
        self.energies = [_compute_energy_direct(positions, velocities, masses, G)]
        self.angular_momenta = [_compute_angular_momentum_direct(positions, velocities, masses)]
        
        n_steps = int(np.ceil(t_end / dt))
        
        for step in range(1, n_steps + 1):
            target_t = step * dt
            remaining_dt = target_t - t
            current_dt = remaining_dt
            
            while remaining_dt > 1e-12:
                mode, participants = self.determine_encounter_mode(positions)
                
                # Predict next step
                pos_new, vel_new = _leapfrog_step(positions, velocities, masses, G, mode, participants, current_dt)
                
                # Check overshoot
                is_overshoot = False
                if mode != "normal":
                    is_overshoot = _check_overshoot(positions, velocities, pos_new, vel_new, self.config.influence_threshold, participants)
                    
                if is_overshoot and current_dt > 1e-5:
                    current_dt /= 2.0
                else:
                    # Accept step
                    positions = pos_new
                    velocities = vel_new
                    t += current_dt
                    remaining_dt -= current_dt
                    current_dt = min(remaining_dt, dt)
            
            system.set_state(positions, velocities)
            
            # Event detection
            curr_mode, curr_participants = self.determine_encounter_mode(positions)
            
            if active_event is None:
                if curr_mode != "normal":
                    active_event = Event(
                        start_time=t - dt,
                        end_time=t,
                        participants=curr_participants,
                        type=curr_mode,
                    )
            else:
                if curr_mode == "normal":
                    active_event.end_time = t
                    self.events.append(active_event)
                    active_event = None
                elif curr_mode != active_event.type or curr_participants != active_event.participants:
                    active_event.end_time = t
                    self.events.append(active_event)
                    active_event = Event(
                        start_time=t - dt,
                        end_time=t,
                        participants=curr_participants,
                        type=curr_mode,
                    )
                else:
                    active_event.end_time = t
                    
            if step % record_every == 0 or step == n_steps:
                self.times.append(t)
                self.positions_history.append(positions.copy())
                self.velocities_history.append(velocities.copy())
                self.energies.append(_compute_energy_direct(positions, velocities, masses, G))
                self.angular_momenta.append(_compute_angular_momentum_direct(positions, velocities, masses))
                
        if active_event is not None:
            active_event.end_time = t
            self.events.append(active_event)
            
        return self.events

def _compute_energy_direct(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, G: float) -> float:
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    pe = 0.0
    n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist > 1e-12:
                pe -= G * masses[i] * masses[j] / dist
    return float(ke + pe)

def _compute_angular_momentum_direct(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    L = np.zeros(3)
    for i in range(len(masses)):
        L += masses[i] * np.cross(positions[i], velocities[i])
    return L

def _leapfrog_step(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, G: float, mode: str, participants: tuple[int, ...], dt: float) -> tuple[np.ndarray, np.ndarray]:
    acc = _get_accelerations(positions, masses, G, mode, participants)
    vel_half = velocities + 0.5 * dt * acc
    pos_new = positions + dt * vel_half
    acc_new = _get_accelerations(pos_new, masses, G, mode, participants)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new

def _get_accelerations(positions: np.ndarray, masses: np.ndarray, G: float, mode: str, participants: tuple[int, ...]) -> np.ndarray:
    n = len(masses)
    acc = np.zeros((n, 3))
    if mode == "normal":
        for i in range(1, n):
            r = positions[0] - positions[i]
            dist = np.linalg.norm(r)
            if dist > 1e-12:
                acc[i] = G * masses[0] * r / dist**3
    else:
        active = [0] + list(participants)
        for i in active:
            for j in active:
                if i != j:
                    r = positions[j] - positions[i]
                    dist = np.linalg.norm(r)
                    if dist > 1e-12:
                        acc[i] += G * masses[j] * r / dist**3
        for i in range(1, n):
            if i not in participants:
                r = positions[0] - positions[i]
                dist = np.linalg.norm(r)
                if dist > 1e-12:
                    acc[i] = G * masses[0] * r / dist**3
    return acc

def _check_overshoot(prev_pos: np.ndarray, prev_vel: np.ndarray, curr_pos: np.ndarray, curr_vel: np.ndarray, threshold: float, participants: tuple[int, ...]) -> bool:
    for i in participants:
        for j in participants:
            if i < j:
                r_prev = prev_pos[j] - prev_pos[i]
                v_prev = prev_vel[j] - prev_vel[i]
                r_curr = curr_pos[j] - curr_pos[i]
                v_curr = curr_vel[j] - curr_vel[i]
                
                d_prev = np.linalg.norm(r_prev)
                d_curr = np.linalg.norm(r_curr)
                
                if d_prev < threshold or d_curr < threshold:
                    dot_prev = np.dot(r_prev, v_prev)
                    dot_curr = np.dot(r_curr, v_curr)
                    if dot_prev < -1e-8 and dot_curr > 1e-8:
                        return True
    return False
