"""Periodicity score evaluator for detecting repeating encounter structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from fairy_orbit.events.timeline import Event, EventTimeline, EventType


@dataclass
class PeriodicityConfig:
    """Configuration for periodicity scoring."""
    min_cycles: int = 2  # Minimum cycles to evaluate
    autocorr_lag: int = 50  # Maximum lag for autocorrelation
    sequence_match_weight: float = 1.0  # Weight for sequence matching
    autocorr_weight: float = 1.0  # Weight for autocorrelation


class PeriodicityEvaluator:
    """
    Evaluate periodicity of event sequences and trajectories.
    
    Measures whether the trajectory or event timeline repeats.
    """
    
    def __init__(self, config: PeriodicityConfig | None = None):
        self.config = config or PeriodicityConfig()
    
    def encode_event_sequence(
        self,
        events: list[Event],
    ) -> list[int]:
        """
        Encode event sequence as integers for comparison.
        
        Args:
            events: List of events
        
        Returns:
            Encoded sequence
        """
        encoded = []
        for event in events:
            # Encode event type and mode
            type_code = {
                EventType.KEPLER_START: 1,
                EventType.KEPLER_END: 2,
                EventType.ENCOUNTER_START: 3,
                EventType.ENCOUNTER_END: 4,
                EventType.TRANSITION: 5,
            }.get(event.event_type, 0)
            
            # Encode mode
            mode_code = {
                None: 0,
                "kepler": 1,
                "pair": 2,
                "triple": 3,
                "full": 4,
            }.get(event.mode.value if event.mode else None, 0)
            
            # Encode participants as bitmask
            participant_code = 0
            for p in event.participants:
                participant_code |= (1 << p)
            
            encoded.append(type_code * 16 + mode_code * 4 + participant_code)
        
        return encoded
    
    def split_into_cycles(
        self,
        events: list[Event],
        n_cycles: int,
    ) -> list[list[Event]]:
        """
        Split events into n_cycles based on time.
        
        Args:
            events: List of events
            n_cycles: Number of cycles to split into
        
        Returns:
            List of event lists for each cycle
        """
        if not events:
            return [[] for _ in range(n_cycles)]
        
        total_time = max(e.end_time if e.end_time else e.start_time for e in events)
        cycle_duration = total_time / n_cycles
        
        cycles: list[list[Event]] = [[] for _ in range(n_cycles)]
        
        for event in events:
            cycle_idx = int(event.start_time / cycle_duration)
            cycle_idx = min(cycle_idx, n_cycles - 1)
            cycles[cycle_idx].append(event)
        
        return cycles
    
    def compute_sequence_similarity(
        self,
        seq1: list[int],
        seq2: list[int],
    ) -> float:
        """
        Compute similarity between two event sequences using cyclic matching.
        
        Args:
            seq1: First encoded sequence
            seq2: Second encoded sequence
        
        Returns:
            Similarity score (lower is better)
        """
        if not seq1 or not seq2:
            return 1.0
        
        # Try all cyclic shifts of seq2 to find best match
        min_error = float('inf')
        
        for shift in range(len(seq2)):
            shifted_seq2 = seq2[shift:] + seq2[:shift]
            
            # Compute error
            min_len = min(len(seq1), len(shifted_seq2))
            error = 0.0
            for i in range(min_len):
                if seq1[i] != shifted_seq2[i]:
                    error += 1.0
            
            # Penalty for length mismatch
            error += abs(len(seq1) - len(shifted_seq2))
            error /= max(len(seq1), len(shifted_seq2))
            
            min_error = min(min_error, error)
        
        return min_error
    
    def compute_autocorrelation(
        self,
        sequence: list[int],
        max_lag: int | None = None,
    ) -> np.ndarray:
        """
        Compute autocorrelation of event sequence.
        
        Args:
            sequence: Encoded event sequence
            max_lag: Maximum lag to compute
        
        Returns:
            Autocorrelation array
        """
        if max_lag is None:
            max_lag = self.config.autocorr_lag
        
        if len(sequence) < 2:
            return np.array([1.0])
        
        # Normalize sequence
        seq_array = np.array(sequence, dtype=float)
        seq_array = (seq_array - np.mean(seq_array)) / (np.std(seq_array) + 1e-10)
        
        # Compute autocorrelation
        autocorr = np.correlate(seq_array, seq_array, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]
        
        # Truncate to max_lag
        return autocorr[:max_lag + 1]
    
    def detect_period_from_autocorr(
        self,
        autocorr: np.ndarray,
    ) -> tuple[float, float]:
        """
        Detect dominant period from autocorrelation.
        
        Args:
            autocorr: Autocorrelation array
        
        Returns:
            (period, confidence) tuple
        """
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.5)
        peaks = peaks + 1  # Adjust for offset
        
        if len(peaks) == 0:
            return 0.0, 0.0
        
        # Find first significant peak (excluding lag 0)
        if len(peaks) > 0:
            period = float(peaks[0])
            confidence = float(autocorr[peaks[0]])
            return period, confidence
        
        return 0.0, 0.0
    
    def evaluate_timeline(
        self,
        timeline: EventTimeline,
    ) -> dict[str, Any]:
        """
        Evaluate periodicity of event timeline.
        
        Args:
            timeline: Event timeline to evaluate
        
        Returns:
            Dictionary with periodicity score components
        """
        events = timeline.events
        
        if len(events) < self.config.min_cycles * 2:
            return {
                "periodicity_score": 1e6,
                "sequence_error": 1e6,
                "autocorr_period": 0.0,
                "autocorr_confidence": 0.0,
                "n_events": len(events),
            }
        
        # Encode sequence
        encoded = self.encode_event_sequence(events)
        
        # Split into cycles
        n_cycles = max(self.config.min_cycles, len(events) // 4)
        cycles = self.split_into_cycles(events, n_cycles)
        
        # Compute sequence similarity between consecutive cycles
        sequence_errors = []
        for i in range(len(cycles) - 1):
            seq1 = self.encode_event_sequence(cycles[i])
            seq2 = self.encode_event_sequence(cycles[i + 1])
            error = self.compute_sequence_similarity(seq1, seq2)
            sequence_errors.append(error)
        
        avg_sequence_error = np.mean(sequence_errors) if sequence_errors else 1.0
        
        # Compute autocorrelation
        autocorr = self.compute_autocorrelation(encoded)
        period, confidence = self.detect_period_from_autocorr(autocorr)
        
        # Total score
        total_score = (
            self.config.sequence_match_weight * avg_sequence_error +
            self.config.autocorr_weight * (1.0 - confidence)
        )
        
        return {
            "periodicity_score": float(total_score),
            "sequence_error": float(avg_sequence_error),
            "autocorr_period": period,
            "autocorr_confidence": confidence,
            "n_events": len(events),
        }
    
    def evaluate_trajectory(
        self,
        positions: np.ndarray,
    ) -> dict[str, Any]:
        """
        Evaluate periodicity of trajectory using autocorrelation.
        
        Args:
            positions: Position array of shape (n_steps, n_bodies, 3)
        
        Returns:
            Dictionary with periodicity score components
        """
        # Use center of mass distance as signal
        com = np.mean(positions[:, 1:, :], axis=1)  # Fairy center of mass
        com_dist = np.linalg.norm(com, axis=1)
        
        if len(com_dist) < 2:
            return {
                "periodicity_score": 1e6,
                "autocorr_period": 0.0,
                "autocorr_confidence": 0.0,
            }
        
        # Normalize
        com_dist = (com_dist - np.mean(com_dist)) / (np.std(com_dist) + 1e-10)
        
        # Compute autocorrelation
        autocorr = self.compute_autocorrelation(com_dist.tolist())
        period, confidence = self.detect_period_from_autocorr(autocorr)
        
        return {
            "periodicity_score": float(1.0 - confidence),
            "autocorr_period": period,
            "autocorr_confidence": confidence,
        }
