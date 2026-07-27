"""Keplerian orbital elements and coordinate conversions.

Newton (r, v) and orbit (a, e, i, Ω, ω, M) are synonymous 6-DOF writings.
Polar (r, θ, v_r, v_θ) ↔ Cartesian is the planar frame conversion used by
(v_rad, v_tan) constructions; Rodrigues is only a later special case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.core.body import cross3 as _cross3
from fairy_orbit.core.body import norm3 as _norm3


def orbital_period(a: float, mu: float) -> float:
    if a <= 0.0:
        return float("inf")
    return float(2.0 * np.pi * np.sqrt(a**3 / mu))


def _wrap_2pi(angle: float) -> float:
    return float(np.mod(angle, 2.0 * np.pi))


def _solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    M = _wrap_2pi(M)
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        dE = f / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return float(E)


# ---------------------------------------------------------------------------
# Polar ↔ Cartesian (planar)
# ---------------------------------------------------------------------------


def polar_to_cartesian(
    r: float,
    theta: float,
    v_rad: float = 0.0,
    v_tan: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Planar polar → Cartesian.

      x = r cos θ,  y = r sin θ,  z = 0
      v = v_rad ê_r + v_tan ê_θ
    """
    ct, st = np.cos(theta), np.sin(theta)
    position = np.array([r * ct, r * st, 0.0], dtype=float)
    # ê_r = (cos θ, sin θ), ê_θ = (-sin θ, cos θ)
    velocity = np.array(
        [v_rad * ct - v_tan * st, v_rad * st + v_tan * ct, 0.0],
        dtype=float,
    )
    return position, velocity


def cartesian_to_polar(
    position: np.ndarray,
    velocity: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Planar Cartesian → polar (r, θ, v_rad, v_tan).

    Uses the xy-plane projection; z components are ignored for the angles.
    """
    position = np.asarray(position, dtype=float).reshape(3)
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    x, y = float(position[0]), float(position[1])
    r = float(np.hypot(x, y))
    theta = float(np.arctan2(y, x))
    if r < 1e-15:
        return 0.0, theta, float(velocity[0]), float(velocity[1])
    ct, st = x / r, y / r
    v_rad = float(velocity[0] * ct + velocity[1] * st)
    v_tan = float(-velocity[0] * st + velocity[1] * ct)
    return r, _wrap_2pi(theta), v_rad, v_tan


def kepler_polar_speeds(
    a: float, e: float, mu: float, true_anomaly: float = 0.0
) -> tuple[float, float]:
    """Kepler (a, e, f) → polar speeds (v_rad, v_tan) in the orbital plane."""
    f = float(true_anomaly)
    h = np.sqrt(mu * a * max(0.0, 1.0 - e * e))
    v_rad = (mu / h) * e * np.sin(f)
    v_tan = (mu / h) * (1.0 + e * np.cos(f))
    return float(v_rad), float(v_tan)


def kepler_radius(a: float, e: float, true_anomaly: float = 0.0) -> float:
    """Polar radius from (a, e, f): r = a(1-e²)/(1+e cos f)."""
    f = float(true_anomaly)
    return float(a * (1.0 - e * e) / (1.0 + e * np.cos(f)))


# ---------------------------------------------------------------------------
# Orbit elements ↔ Newton state
# ---------------------------------------------------------------------------


@dataclass
class OrbitalElements:
    """Standard Keplerian orbital elements (two-body)."""

    a: float
    e: float
    i: float = 0.0
    omega: float = 0.0
    Omega: float = 0.0
    M: float = 0.0

    def period(self, mu: float) -> float:
        return orbital_period(self.a, mu)

    def to_state(self, mu: float) -> tuple[np.ndarray, np.ndarray]:
        return elements_to_state(self, mu)

    @classmethod
    def from_state(
        cls,
        position: np.ndarray,
        velocity: np.ndarray,
        mu: float,
    ) -> OrbitalElements:
        return state_to_elements(position, velocity, mu)


def elements_to_state(
    elements: OrbitalElements,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(a, e, i, Ω, ω, M) → inertial (r, v)."""
    E = _solve_kepler(elements.M, elements.e)
    x_pf = elements.a * (np.cos(E) - elements.e)
    y_pf = elements.a * np.sqrt(max(0.0, 1.0 - elements.e**2)) * np.sin(E)
    n = np.sqrt(mu / elements.a**3)
    denom = 1.0 - elements.e * np.cos(E)
    vx_pf = -n * elements.a * np.sin(E) / denom
    vy_pf = n * elements.a * np.sqrt(max(0.0, 1.0 - elements.e**2)) * np.cos(E) / denom

    cO, sO = np.cos(elements.Omega), np.sin(elements.Omega)
    ci, si = np.cos(elements.i), np.sin(elements.i)
    co, so = np.cos(elements.omega), np.sin(elements.omega)

    R3_Omega = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    R3_omega = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])
    R = R3_Omega @ R1_i @ R3_omega

    position = R @ np.array([x_pf, y_pf, 0.0])
    velocity = R @ np.array([vx_pf, vy_pf, 0.0])
    return position, velocity


def state_to_elements(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float,
) -> OrbitalElements:
    """Inertial (r, v) → (a, e, i, Ω, ω, M)."""
    position = np.asarray(position, dtype=float).reshape(3)
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    r = _norm3(position)
    v = _norm3(velocity)

    h = _cross3(position, velocity)
    h_mag = _norm3(h)

    e_vec = (_cross3(velocity, h) / mu) - (position / r)
    e = _norm3(e_vec)

    energy = 0.5 * v * v - mu / r
    if abs(energy) > 1e-14:
        a = -mu / (2.0 * energy)
    else:
        a = float("inf")

    i = float(np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))) if h_mag > 1e-14 else 0.0

    # n = ẑ × h = (−h_y, h_x, 0)
    n_vec = np.array([-h[1], h[0], 0.0], dtype=float)
    n_mag = _norm3(n_vec)

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
        # Stable half-angle formula (avoids tan singularity near f=±π)
        E = 2.0 * np.arctan2(
            np.sqrt(max(0.0, 1.0 - e)) * np.sin(nu / 2.0),
            np.sqrt(max(0.0, 1.0 + e)) * np.cos(nu / 2.0),
        )
        M = E - e * np.sin(E)
    else:
        M = nu

    return OrbitalElements(
        a=float(a),
        e=float(e),
        i=i,
        omega=float(omega),
        Omega=float(Omega),
        M=_wrap_2pi(float(M)),
    )
