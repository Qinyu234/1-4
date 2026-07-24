"""Keplerian orbital elements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def orbital_period(a: float, mu: float) -> float:
    if a <= 0.0:
        return float("inf")
    return float(2.0 * np.pi * np.sqrt(a**3 / mu))


@dataclass
class OrbitalElements:
    """Standard Keplerian orbital elements for the two-body problem."""

    a: float
    e: float
    i: float = 0.0
    omega: float = 0.0
    Omega: float = 0.0
    M: float = 0.0

    def period(self, mu: float) -> float:
        return orbital_period(self.a, mu)

    def to_state(self, mu: float) -> tuple[np.ndarray, np.ndarray]:
        E = self._solve_kepler(self.M, self.e)
        x_pf = self.a * (np.cos(E) - self.e)
        y_pf = self.a * np.sqrt(max(0.0, 1.0 - self.e**2)) * np.sin(E)
        n = np.sqrt(mu / self.a**3)
        denom = 1.0 - self.e * np.cos(E)
        vx_pf = -n * self.a * np.sin(E) / denom
        vy_pf = n * self.a * np.sqrt(max(0.0, 1.0 - self.e**2)) * np.cos(E) / denom

        cO, sO = np.cos(self.Omega), np.sin(self.Omega)
        ci, si = np.cos(self.i), np.sin(self.i)
        co, so = np.cos(self.omega), np.sin(self.omega)

        R3_Omega = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
        R1_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
        R3_omega = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])
        R = R3_Omega @ R1_i @ R3_omega

        position = R @ np.array([x_pf, y_pf, 0.0])
        velocity = R @ np.array([vx_pf, vy_pf, 0.0])
        return position, velocity

    @staticmethod
    def _solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 100) -> float:
        M = float(np.mod(M, 2.0 * np.pi))
        E = M if e < 0.8 else np.pi
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            dE = f / (1.0 - e * np.cos(E))
            E -= dE
            if abs(dE) < tol:
                break
        return float(E)

    @classmethod
    def from_state(
        cls,
        position: np.ndarray,
        velocity: np.ndarray,
        mu: float,
    ) -> OrbitalElements:
        position = np.asarray(position, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        r = float(np.linalg.norm(position))
        v = float(np.linalg.norm(velocity))

        h = np.cross(position, velocity)
        h_mag = float(np.linalg.norm(h))

        e_vec = (np.cross(velocity, h) / mu) - (position / r)
        e = float(np.linalg.norm(e_vec))

        energy = 0.5 * v * v - mu / r
        if abs(energy) > 1e-14:
            a = -mu / (2.0 * energy)
        else:
            a = float("inf")

        i = float(np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))) if h_mag > 1e-14 else 0.0

        n_vec = np.cross([0.0, 0.0, 1.0], h)
        n_mag = float(np.linalg.norm(n_vec))

        if n_mag > 1e-14:
            Omega = float(np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0)))
            if n_vec[1] < 0:
                Omega = 2.0 * np.pi - Omega
        else:
            Omega = 0.0

        if n_mag > 1e-14 and e > 1e-14:
            omega = float(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0)))
            if e_vec[2] < 0:
                omega = 2.0 * np.pi - omega
        elif e > 1e-14:
            omega = float(np.arctan2(e_vec[1], e_vec[0]))
            if omega < 0:
                omega += 2.0 * np.pi
        else:
            omega = 0.0

        if e > 1e-14:
            nu = float(np.arccos(np.clip(np.dot(e_vec, position) / (e * r), -1.0, 1.0)))
            if np.dot(position, velocity) < 0:
                nu = 2.0 * np.pi - nu
        else:
            if n_mag > 1e-14:
                nu = float(np.arccos(np.clip(np.dot(n_vec, position) / (n_mag * r), -1.0, 1.0)))
                if position[2] < 0:
                    nu = 2.0 * np.pi - nu
            else:
                nu = float(np.arctan2(position[1], position[0]))
                if nu < 0:
                    nu += 2.0 * np.pi

        if e < 1.0:
            E = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(nu / 2.0))
            M = E - e * np.sin(E)
        else:
            M = nu

        return cls(a=float(a), e=float(e), i=i, omega=float(omega), Omega=float(Omega), M=float(M))
