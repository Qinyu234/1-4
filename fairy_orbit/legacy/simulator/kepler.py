"""Kepler orbit propagation using analytic solution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KeplerOrbit:
    """Kepler orbit parameters."""
    position: np.ndarray  # Current position (3,)
    velocity: np.ndarray  # Current velocity (3,)
    central_mass: float  # Mass of central body
    G: float  # Gravitational constant
    epoch: float = 0.0  # Current time
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
    
    def compute_orbital_elements(self) -> dict[str, float]:
        """Compute orbital elements from state vectors."""
        r = self.position
        v = self.velocity
        mu = self.G * self.central_mass
        
        # Specific angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Eccentricity vector
        e_vec = (np.cross(v, h) / mu) - (r / np.linalg.norm(r))
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        a = 1.0 / (2.0 / r_mag - v_mag**2 / mu)
        
        # Inclination
        i = np.arccos(h[2] / h_mag)
        
        return {
            "a": a,
            "e": e,
            "i": float(i),
            "h_mag": h_mag,
        }
    
    def propagate(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate orbit analytically by time dt.
        
        Uses universal variable formulation for robust propagation.
        
        Args:
            dt: Time step to propagate
        
        Returns:
            (new_position, new_velocity) tuple
        """
        r0 = self.position
        v0 = self.velocity
        mu = self.G * self.central_mass
        
        r0_mag = np.linalg.norm(r0)
        v0_mag = np.linalg.norm(v0)
        
        # Specific energy
        energy = v0_mag**2 / 2 - mu / r0_mag
        
        # Semi-major axis
        if abs(energy) > 1e-10:
            a = -mu / (2 * energy)
        else:
            # Parabolic case
            a = float('inf')
        
        # Universal variable formulation
        alpha = 1.0 / a if a != float('inf') else 0.0
        
        # Solve Kepler's equation using Newton-Raphson
        chi = np.sqrt(mu) * abs(dt) / r0_mag  # Initial guess
        
        for _ in range(50):
            z = alpha * chi**2
            
            # Stumpff functions
            if z > 1e-6:
                C = (1 - np.cos(np.sqrt(z))) / z
                S = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z)**3
            elif z < -1e-6:
                C = (1 - np.cosh(np.sqrt(-z))) / z
                S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt(-z)**3
            else:
                C = 0.5
                S = 1.0 / 6.0
            
            r = r0_mag * (1 - z * C) + np.dot(r0, v0) / np.sqrt(mu) * chi * (1 - z * S) + chi**2 * C
            
            # Universal Kepler equation
            f = r0_mag * chi * (1 - z * S) + np.dot(r0, v0) / np.sqrt(mu) * chi**2 * C + chi**3 * S - np.sqrt(mu) * dt
            
            # Derivative
            df = r0_mag * (1 - z * C) + np.dot(r0, v0) / np.sqrt(mu) * chi * (1 - z * S) + chi**2 * C
            
            if abs(df) < 1e-10:
                break
            
            chi_new = chi - f / df
            if abs(chi_new - chi) < 1e-10:
                chi = chi_new
                break
            chi = chi_new
        
        # Compute Lagrange coefficients
        z = alpha * chi**2
        if z > 1e-6:
            C = (1 - np.cos(np.sqrt(z))) / z
            S = (np.sqrt(z) - np.sin(np.sqrt(z))) / np.sqrt(z)**3
        elif z < -1e-6:
            C = (1 - np.cosh(np.sqrt(-z))) / z
            S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / np.sqrt(-z)**3
        else:
            C = 0.5
            S = 1.0 / 6.0
        
        f = 1 - chi**2 / r0_mag * C
        g = dt - chi**3 / np.sqrt(mu) * S
        
        r_new = f * r0 + g * v0
        r_new_mag = np.linalg.norm(r_new)
        
        g_dot = 1 - chi / r_new_mag * (1 - z * S)
        f_dot = np.sqrt(mu) / (r0_mag * r_new_mag) * chi * (z * S - 1)
        
        v_new = f_dot * r0 + g_dot * v0
        
        return r_new, v_new
    
    def get_position_at_time(self, t: float) -> np.ndarray:
        """
        Get position at absolute time t.
        
        Args:
            t: Absolute time
        
        Returns:
            Position at time t
        """
        dt = t - self.epoch
        pos, _ = self.propagate(dt)
        return pos
    
    def get_velocity_at_time(self, t: float) -> np.ndarray:
        """
        Get velocity at absolute time t.
        
        Args:
            t: Absolute time
        
        Returns:
            Velocity at time t
        """
        dt = t - self.epoch
        _, vel = self.propagate(dt)
        return vel
