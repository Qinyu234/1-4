"""Orbital elements for Kepler dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitalElements:
    """
    Keplerian orbital elements.
    
    Standard orbital elements for 2-body problem.
    """
    a: float  # Semi-major axis
    e: float  # Eccentricity
    i: float  # Inclination (radians)
    omega: float  # Argument of periapsis (radians)
    Omega: float  # Longitude of ascending node (radians)
    M: float  # Mean anomaly (radians)
    
    @property
    def period(self, mu: float) -> float:
        """Orbital period."""
        if self.a <= 0:
            return float('inf')
        return 2 * np.pi * np.sqrt(self.a**3 / mu)
    
    def to_state(self, mu: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert orbital elements to state vectors (position, velocity).
        
        Args:
            mu: Gravitational parameter (G * M)
        
        Returns:
            (position, velocity) tuple
        """
        # Solve Kepler's equation for eccentric anomaly
        E = self._solve_kepler(self.M, self.e)
        
        # Position and velocity in perifocal frame
        r_mag = self.a * (1 - self.e * np.cos(E))
        
        x_perifocal = self.a * (np.cos(E) - self.e)
        y_perifocal = self.a * np.sqrt(1 - self.e**2) * np.sin(E)
        
        n = np.sqrt(mu / self.a**3)
        vx_perifocal = -n * self.a * np.sin(E) / (1 - self.e * np.cos(E))
        vy_perifocal = n * self.a * np.sqrt(1 - self.e**2) * np.cos(E) / (1 - self.e * np.cos(E))
        
        # Rotation matrices
        R3_Omega = np.array([
            [np.cos(self.Omega), -np.sin(self.Omega), 0],
            [np.sin(self.Omega), np.cos(self.Omega), 0],
            [0, 0, 1],
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(self.i), -np.sin(self.i)],
            [0, np.sin(self.i), np.cos(self.i)],
        ])
        
        R3_omega = np.array([
            [np.cos(self.omega), -np.sin(self.omega), 0],
            [np.sin(self.omega), np.cos(self.omega), 0],
            [0, 0, 1],
        ])
        
        # Combined rotation
        R = R3_Omega @ R1_i @ R3_omega
        
        # Transform to inertial frame
        r_perifocal = np.array([x_perifocal, y_perifocal, 0])
        v_perifocal = np.array([vx_perifocal, vy_perifocal, 0])
        
        position = R @ r_perifocal
        velocity = R @ v_perifocal
        
        return position, velocity
    
    @staticmethod
    def _solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 100) -> float:
        """Solve Kepler's equation M = E - e*sin(E) for E."""
        E = M if e < 0.8 else np.pi
        
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            df = 1 - e * np.cos(E)
            dE = f / df
            E -= dE
            if abs(dE) < tol:
                break
        
        return E
    
    @classmethod
    def from_state(
        cls,
        position: np.ndarray,
        velocity: np.ndarray,
        mu: float,
    ) -> OrbitalElements:
        """
        Convert state vectors to orbital elements.
        
        Args:
            position: Position vector
            velocity: Velocity vector
            mu: Gravitational parameter
        
        Returns:
            OrbitalElements object
        """
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        
        # Specific angular momentum
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h)
        
        # Eccentricity vector
        e_vec = (np.cross(velocity, h) / mu) - (position / r)
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = v**2 / 2 - mu / r
        if abs(energy) > 1e-10:
            a = -mu / (2 * energy)
        else:
            a = float('inf')
        
        # Inclination
        i = np.arccos(h[2] / h_mag) if h_mag > 1e-10 else 0.0
        
        # Node line
        n_vec = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n_vec)
        
        # Longitude of ascending node
        if n_mag > 1e-10:
            Omega = np.arccos(n_vec[0] / n_mag)
            if n_vec[1] < 0:
                Omega = 2 * np.pi - Omega
        else:
            Omega = 0.0
        
        # Argument of periapsis
        if n_mag > 1e-10 and e > 1e-10:
            omega = np.arccos(np.dot(n_vec, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0.0
        
        # True anomaly
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, position) / (e * r))
            if np.dot(position, velocity) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0.0
        
        # Mean anomaly
        if e < 1.0:
            E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
            M = E - e * np.sin(E)
        else:
            M = nu  # Approximation for parabolic
        
        return cls(a=a, e=e, i=i, omega=omega, Omega=Omega, M=M)
