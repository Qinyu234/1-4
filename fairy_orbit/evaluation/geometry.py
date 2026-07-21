"""Geometry score evaluator for tetrahedral symmetry preservation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.simulation.trajectory import Trajectory


@dataclass
class GeometryConfig:
    """Configuration for geometry scoring."""
    fairy_indices: tuple[int, ...] = (1, 2, 3, 4)  # Indices of fairy bodies
    distance_weight: float = 1.0  # Weight for distance matrix similarity
    symmetry_weight: float = 1.0  # Weight for rotational symmetry


class GeometryEvaluator:
    """
    Evaluate preservation of tetrahedral symmetry.
    
    Measures how well the fairy bodies maintain their tetrahedral configuration
    over the course of the trajectory.
    """
    
    def __init__(self, config: GeometryConfig | None = None):
        self.config = config or GeometryConfig()
    
    def compute_distance_matrix(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix for fairy bodies.
        
        Args:
            positions: Array of shape (n_bodies, 3)
        
        Returns:
            Distance matrix of shape (n_fairies, n_fairies)
        """
        fairy_pos = positions[list(self.config.fairy_indices)]
        n = len(fairy_pos)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(fairy_pos[i] - fairy_pos[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def distance_matrix_similarity(
        self,
        pos_initial: np.ndarray,
        pos_final: np.ndarray,
    ) -> float:
        """
        Compute similarity between initial and final distance matrices.
        
        Uses permutation-invariant comparison.
        
        Args:
            pos_initial: Initial positions
            pos_final: Final positions
        
        Returns:
            Similarity score (lower is better)
        """
        from fairy_orbit.analysis.evaluator import distance_matrix_error
        return distance_matrix_error(
            pos_initial,
            pos_final,
            fairy_indices=self.config.fairy_indices,
        )
    
    def compute_tetrahedral_volume(
        self,
        positions: np.ndarray,
    ) -> float:
        """
        Compute volume of tetrahedron formed by fairy bodies.
        
        Args:
            positions: Array of shape (n_bodies, 3)
        
        Returns:
            Tetrahedron volume
        """
        fairy_pos = positions[list(self.config.fairy_indices)]
        
        if len(fairy_pos) < 4:
            return 0.0
        
        # Use first 4 fairies for tetrahedron
        a, b, c, d = fairy_pos[:4]
        
        # Volume = |(b-a) · ((c-a) × (d-a))| / 6
        ab = b - a
        ac = c - a
        ad = d - a
        
        cross_ac_ad = np.cross(ac, ad)
        volume = abs(np.dot(ab, cross_ac_ad)) / 6.0
        
        return volume
    
    def volume_preservation(
        self,
        pos_initial: np.ndarray,
        pos_final: np.ndarray,
    ) -> float:
        """
        Measure preservation of tetrahedral volume.
        
        Args:
            pos_initial: Initial positions
            pos_final: Final positions
        
        Returns:
            Relative volume change (lower is better)
        """
        vol_initial = self.compute_tetrahedral_volume(pos_initial)
        vol_final = self.compute_tetrahedral_volume(pos_final)
        
        if vol_initial < 1e-10:
            return abs(vol_final)
        
        return abs(vol_final - vol_initial) / vol_initial
    
    def compute_center_of_mass(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute center of mass of fairy bodies.
        
        Args:
            positions: Array of shape (n_bodies, 3)
            masses: Array of masses
        
        Returns:
            Center of mass position
        """
        fairy_pos = positions[list(self.config.fairy_indices)]
        fairy_masses = masses[list(self.config.fairy_indices)]
        
        total_mass = np.sum(fairy_masses)
        if total_mass < 1e-10:
            return np.zeros(3)
        
        com = np.sum(fairy_pos * fairy_masses[:, np.newaxis], axis=0) / total_mass
        return com
    
    def center_of_mass_drift(
        self,
        trajectory: Trajectory,
    ) -> float:
        """
        Measure drift of fairy center of mass.
        
        Args:
            trajectory: Trajectory object
        
        Returns:
            Maximum displacement of center of mass
        """
        com_positions = []
        
        for positions in trajectory.positions:
            com = self.compute_center_of_mass(positions, trajectory.masses)
            com_positions.append(com)
        
        com_positions = np.array(com_positions)
        com_initial = com_positions[0]
        
        displacements = np.linalg.norm(com_positions - com_initial, axis=1)
        return float(np.max(displacements))
    
    def evaluate(
        self,
        trajectory: Trajectory,
        period_index: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate geometry preservation of trajectory.
        
        Args:
            trajectory: Trajectory to evaluate
            period_index: Index to evaluate at (default: final frame)
        
        Returns:
            Dictionary with geometry score components
        """
        if period_index is None:
            period_index = len(trajectory.times) - 1
        
        pos_initial = trajectory.positions[0]
        pos_final = trajectory.positions[period_index]
        
        # Distance matrix similarity
        dist_sim = self.distance_matrix_similarity(pos_initial, pos_final)
        
        # Volume preservation
        vol_pres = self.volume_preservation(pos_initial, pos_final)
        
        # Center of mass drift
        com_drift = self.center_of_mass_drift(trajectory)
        
        # Total geometry score
        total_score = (
            self.config.distance_weight * dist_sim +
            self.config.symmetry_weight * vol_pres +
            0.5 * com_drift
        )
        
        return {
            "geometry_score": float(total_score),
            "distance_matrix_similarity": float(dist_sim),
            "volume_preservation": float(vol_pres),
            "center_of_mass_drift": float(com_drift),
        }
