"""Periodicity evaluator for event sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.simulation.adaptive_simulator import Event, EventType


@dataclass
class PeriodicityScore:
    """Score components for periodicity evaluation."""
    total_score: float
    time_variance: float
    event_sequence_error: float
    event_count_difference: float
    center_motion_error: float
    energy_drift: float
    
    # Detailed breakdown
    cycle_intervals: list[float]
    normalized_intervals: list[float]
    event_sequences: list[list[int]]
    best_shift: int


@dataclass
class PeriodicityConfig:
    """Configuration for periodicity evaluation."""
    w_time: float = 1.0  # Weight for time variance
    w_event: float = 1.0  # Weight for event sequence error
    w_count: float = 1.0  # Weight for event count difference
    w_center: float = 1.0  # Weight for center motion error
    w_energy: float = 1.0  # Weight for energy drift
    
    min_cycles: int = 2  # Minimum cycles to evaluate
    center_threshold: float = 0.1  # Threshold for center motion penalty
    energy_threshold: float = 1e-2  # Threshold for energy drift penalty


class PeriodicityEvaluator:
    """Evaluate periodicity of event sequences."""
    
    def __init__(self, config: PeriodicityConfig | None = None):
        self.config = config or PeriodicityConfig()
    
    def normalize_event_sequence(
        self,
        events: list[Event],
    ) -> tuple[list[float], list[int]]:
        """
        Normalize event sequence.
        
        Returns:
        - Normalized time intervals (dt_hat)
        - Encoded event sequence
        """
        if len(events) < 2:
            return [], []
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(events)):
            dt = events[i].start_time - events[i-1].start_time
            intervals.append(dt)
        
        # Normalize by mean (remove time scaling)
        mean_dt = np.mean(intervals)
        if mean_dt > 1e-10:
            normalized_intervals = [dt / mean_dt for dt in intervals]
        else:
            normalized_intervals = intervals
        
        # Encode event sequence
        # Event type: PAIR=1, TRIPLE=2, FULL=3
        # Participants: encode as sorted tuple of body indices
        encoded_sequence = []
        for event in events:
            type_code = event.event_type.value
            # Encode participants as single integer (bitmask)
            participant_code = 0
            for p in event.participants:
                participant_code |= (1 << (p - 1))  # p-1 since bodies are 1-4
            encoded_sequence.append(type_code * 16 + participant_code)
        
        return normalized_intervals, encoded_sequence
    
    def split_into_cycles(
        self,
        events: list[Event],
        n_cycles: int,
    ) -> list[list[Event]]:
        """
        Split events into n_cycles based on time intervals.
        
        Simple approach: divide by equal time segments.
        """
        if len(events) == 0:
            return [[] for _ in range(n_cycles)]
        
        total_time = events[-1].end_time - events[0].start_time
        cycle_duration = total_time / n_cycles
        
        cycles: list[list[Event]] = [[] for _ in range(n_cycles)]
        
        for event in events:
            cycle_idx = int((event.start_time - events[0].start_time) / cycle_duration)
            cycle_idx = min(cycle_idx, n_cycles - 1)
            cycles[cycle_idx].append(event)
        
        return cycles
    
    def compute_time_variance(
        self,
        intervals: list[float],
    ) -> float:
        """Compute variance of normalized time intervals."""
        if len(intervals) < 2:
            return 0.0
        return float(np.var(intervals))
    
    def compute_event_sequence_error(
        self,
        cycles: list[list[int]],
    ) -> float:
        """
        Compare event sequences across cycles using cyclic shift invariant comparison.
        
        Uses convolution to find best cyclic shift and computes error.
        """
        if len(cycles) < 2:
            return 0.0
        
        # Compare consecutive cycles
        total_error = 0.0
        n_comparisons = 0
        
        for i in range(len(cycles) - 1):
            seq1 = cycles[i]
            seq2 = cycles[i + 1]
            
            if len(seq1) == 0 or len(seq2) == 0:
                total_error += 1.0  # Penalty for empty sequences
                n_comparisons += 1
                continue
            
            # Find best cyclic shift using convolution-like approach
            best_error = float('inf')
            
            for shift in range(len(seq2)):
                shifted_seq2 = seq2[shift:] + seq2[:shift]
                
                # Compute error (normalized by sequence length)
                min_len = min(len(seq1), len(shifted_seq2))
                error = 0.0
                for j in range(min_len):
                    if seq1[j] != shifted_seq2[j]:
                        error += 1.0
                
                # Penalty for length mismatch
                error += abs(len(seq1) - len(shifted_seq2))
                error /= max(len(seq1), len(shifted_seq2))
                
                best_error = min(best_error, error)
            
            total_error += best_error
            n_comparisons += 1
        
        return total_error / n_comparisons if n_comparisons > 0 else 0.0
    
    def compute_event_count_difference(
        self,
        cycles: list[list[Event]],
    ) -> float:
        """
        Compute penalty when number of events changes between cycles.
        """
        if len(cycles) < 2:
            return 0.0
        
        counts = [len(cycle) for cycle in cycles]
        
        if len(counts) == 0:
            return 0.0
        
        variance = np.var(counts)
        mean_count = np.mean(counts)
        
        if mean_count > 1e-10:
            return float(variance / mean_count)
        return 0.0
    
    def evaluate(
        self,
        events: list[Event],
        center_displacement: float = 0.0,
        energy_drift: float = 0.0,
    ) -> PeriodicityScore:
        """
        Evaluate periodicity of event sequence.
        
        Args:
            events: List of events from adaptive simulator
            center_displacement: Displacement of central body
            energy_drift: Total energy drift
        
        Returns:
            PeriodicityScore with all components
        """
        if len(events) < self.config.min_cycles * 2:
            # Not enough events, return high penalty
            return PeriodicityScore(
                total_score=1e6,
                time_variance=1e6,
                event_sequence_error=1e6,
                event_count_difference=1e6,
                center_motion_error=center_displacement,
                energy_drift=energy_drift,
                cycle_intervals=[],
                normalized_intervals=[],
                event_sequences=[],
                best_shift=0,
            )
        
        # Normalize event sequence
        normalized_intervals, encoded_sequence = self.normalize_event_sequence(events)
        
        # Split into cycles
        n_cycles = max(self.config.min_cycles, len(events) // 4)
        cycles = self.split_into_cycles(events, n_cycles)
        
        # Encode cycles for sequence comparison
        cycle_sequences = []
        for cycle in cycles:
            _, seq = self.normalize_event_sequence(cycle)
            cycle_sequences.append(seq)
        
        # Compute score components
        time_variance = self.compute_time_variance(normalized_intervals)
        event_sequence_error = self.compute_event_sequence_error(cycle_sequences)
        event_count_diff = self.compute_event_count_difference(cycles)
        
        # Center motion penalty
        center_error = max(0.0, center_displacement - self.config.center_threshold)
        
        # Energy drift penalty
        energy_error = max(0.0, energy_drift - self.config.energy_threshold)
        
        # Total score
        total_score = (
            self.config.w_time * time_variance +
            self.config.w_event * event_sequence_error +
            self.config.w_count * event_count_diff +
            self.config.w_center * center_error +
            self.config.w_energy * energy_error
        )
        
        return PeriodicityScore(
            total_score=total_score,
            time_variance=time_variance,
            event_sequence_error=event_sequence_error,
            event_count_difference=event_count_diff,
            center_motion_error=center_error,
            energy_drift=energy_error,
            cycle_intervals=normalized_intervals,
            normalized_intervals=normalized_intervals,
            event_sequences=cycle_sequences,
            best_shift=0,  # Could track best shift from sequence comparison
        )
