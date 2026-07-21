"""Candidate archive for storing top N promising solutions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fairy_orbit.search.candidate import Candidate


class CandidateArchive:
    """Archive to store and manage top N candidates."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize candidate archive.
        
        Args:
            max_size: Maximum number of candidates to keep
        """
        self.max_size = max_size
        self.candidates: list[Candidate] = []
    
    def add(self, candidate: Candidate) -> bool:
        """
        Add a candidate to the archive.
        
        Only SUCCESS candidates are stored.
        Keeps only the top N candidates by score.
        
        Args:
            candidate: Candidate to add
        
        Returns:
            True if candidate was added, False otherwise
        """
        # Only store successful candidates
        if candidate.status.value != "success":
            return False
        
        # Add candidate
        self.candidates.append(candidate)
        
        # Sort by score and keep top N
        self.candidates.sort()
        
        if len(self.candidates) > self.max_size:
            self.candidates = self.candidates[:self.max_size]
        
        return True
    
    def get_best(self, n: int = 1) -> list[Candidate]:
        """
        Get the best n candidates.
        
        Args:
            n: Number of candidates to return
        
        Returns:
            List of best candidates (sorted by score)
        """
        return self.candidates[:n]
    
    def get_all(self) -> list[Candidate]:
        """Get all candidates in the archive."""
        return self.candidates.copy()
    
    def size(self) -> int:
        """Get current number of candidates in archive."""
        return len(self.candidates)
    
    def clear(self) -> None:
        """Clear all candidates from archive."""
        self.candidates.clear()
    
    def save(self, path: Path) -> None:
        """
        Save archive to JSON file.
        
        Args:
            path: Path to save file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "max_size": self.max_size,
            "candidates": [c.to_dict() for c in self.candidates],
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path) -> None:
        """
        Load archive from JSON file.
        
        Args:
            path: Path to load file
        """
        if not path.exists():
            return
        
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        self.max_size = data.get("max_size", 100)
        self.candidates = [
            Candidate.from_dict(c_data) 
            for c_data in data.get("candidates", [])
        ]
        
        # Sort by score
        self.candidates.sort()
    
    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the archive.
        
        Returns:
            Dictionary with archive statistics
        """
        if not self.candidates:
            return {
                "size": 0,
                "best_score": None,
                "worst_score": None,
                "mean_score": None,
                "verified_count": 0,
            }
        
        scores = [c.score for c in self.candidates]
        verified_count = sum(1 for c in self.candidates if c.verified)
        
        return {
            "size": len(self.candidates),
            "best_score": float(min(scores)),
            "worst_score": float(max(scores)),
            "mean_score": float(np.mean(scores)),
            "verified_count": verified_count,
        }
