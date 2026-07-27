"""Manifold generator on tetrahedron direction basis (PROMPT).

Index i = 0 is the nearest orbit. Linear polynomials:

    a_i = a0 + i a1
    e_i = e0 + i e1
    M_i = M0 + i M1

Velocity direction perturbation (Td-symmetric via Rodrigues from T1):

    δv_i = R_i · (vx, vy, vz)

with R_T1 = I and R_i = Rodrigues(q̂_T1 → q̂_i).

After construction the system is shifted into the inertial COM frame
(Σ m r = 0, Σ m v = 0).

Per Stage-A seed (m, e): a0=1, e0=e, M0=0, μ=m.
Free search: (a1, e1, M1, vx, vy, vz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.core.body import Body, System, to_com_inertial_frame
from fairy_orbit.design.elements import OrbitalElements
from fairy_orbit.design.tetrahedron import (
    FAIRY_ORDER,
    VERTICES,
    local_frame,
    rotations_from_T1,
)


@dataclass(frozen=True)
class ManifoldParams:
    """θ = (a0,a1, e0,e1, M0,M1, vx,vy,vz, μ)."""

    a0: float = 1.0
    a1: float = 0.15
    e0: float = 0.05
    e1: float = 0.0
    M0: float = 0.0
    M1: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    mu_mass: float = 1e-3

    def as_theta(self) -> tuple[float, ...]:
        return (
            self.a0,
            self.a1,
            self.e0,
            self.e1,
            self.M0,
            self.M1,
            self.vx,
            self.vy,
            self.vz,
            self.mu_mass,
        )

    @classmethod
    def from_theta(cls, theta: tuple[float, ...] | list[float]) -> ManifoldParams:
        if len(theta) != 10:
            raise ValueError("θ must have 10 components (a0,a1,e0,e1,M0,M1,vx,vy,vz,μ)")
        return cls(
            a0=float(theta[0]),
            a1=float(theta[1]),
            e0=float(theta[2]),
            e1=float(theta[3]),
            M0=float(theta[4]),
            M1=float(theta[5]),
            vx=float(theta[6]),
            vy=float(theta[7]),
            vz=float(theta[8]),
            mu_mass=float(theta[9]),
        )

    def delta_v_T1(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=float)


def poly_linear(q0: float, q1: float, i: int) -> float:
    """q_i = q0 + i q1."""
    return q0 + float(i) * q1


def from_error_seed(
    m: float,
    e: float,
    *,
    a1: float = 0.15,
    e1: float = 0.0,
    M1: float = 0.5,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> ManifoldParams:
    """Seed anchors: a0=1, e0=e, M0=0, μ=m; free a1,e1,M1,vx,vy,vz."""
    params = ManifoldParams(
        a0=1.0,
        a1=a1,
        e0=e,
        e1=e1,
        M0=0.0,
        M1=M1,
        vx=vx,
        vy=vy,
        vz=vz,
        mu_mass=m,
    )
    for i in range(4):
        elements_for_index(params, i)
    return params


def elements_for_index(params: ManifoldParams, i: int) -> OrbitalElements:
    a = poly_linear(params.a0, params.a1, i)
    e = poly_linear(params.e0, params.e1, i)
    M = poly_linear(params.M0, params.M1, i)
    if a <= 0.0:
        raise ValueError(f"a_{i}={a} must be positive")
    if not (0.0 <= e < 1.0):
        raise ValueError(f"e_{i}={e} must be in [0,1)")
    return OrbitalElements(a=a, e=e, i=0.0, omega=0.0, Omega=0.0, M=M)


def state_along_direction(
    elements: OrbitalElements,
    q_hat: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    from fairy_orbit.design.elements import _solve_kepler

    E = _solve_kepler(elements.M, elements.e)
    cos_E = np.cos(E)
    r_mag = elements.a * (1.0 - elements.e * cos_E)
    nu = 2.0 * np.arctan2(
        np.sqrt(max(0.0, 1.0 - elements.e)) * np.sin(E / 2.0),
        np.sqrt(max(0.0, 1.0 + elements.e)) * np.cos(E / 2.0),
    )
    p = elements.a * (1.0 - elements.e**2)
    h = np.sqrt(mu * p)
    vr = (mu / h) * elements.e * np.sin(nu)
    vt = h / r_mag

    r_hat, t_hat, _ = local_frame(q_hat)
    r = r_mag * r_hat
    v = vr * r_hat + vt * t_hat
    return r, v


def apply_symmetric_velocity_kick(
    velocities: dict[str, np.ndarray],
    delta_v_T1: np.ndarray,
    *,
    names: tuple[str, ...] = FAIRY_ORDER,
) -> dict[str, np.ndarray]:
    """
    Td-style symmetry: copy (vx,vy,vz) from T1 onto each fairy via Rodrigues R_i.

        v_i ← v_i + R_i · δv_T1
    """
    rots = rotations_from_T1()
    dv0 = np.asarray(delta_v_T1, dtype=float).reshape(3)
    out = {}
    for name in names:
        out[name] = np.asarray(velocities[name], dtype=float).reshape(3) + rots[name] @ dv0
    return out


def build_manifold_system(
    params: ManifoldParams | None = None,
    *,
    G: float = 1.0,
    central_mass: float = 1.0,
    names: tuple[str, ...] = FAIRY_ORDER,
    com_frame: bool = True,
) -> System:
    """Build X₀ in the central-body frame, then shift to inertial COM (default).

    Anchors/elements are set with the central body at the origin; afterwards
    `to_com_inertial_frame` removes translation only (rotation stays for R*).
    """
    params = params or ManifoldParams()
    mu = G * central_mass
    central = Body(
        mass=central_mass,
        position=np.zeros(3),
        velocity=np.zeros(3),
        name="central",
        radius=0.0,
    )
    m = params.mu_mass * central_mass
    vel_map: dict[str, np.ndarray] = {}
    pos_map: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        elems = elements_for_index(params, i)
        q = np.asarray(VERTICES[name], dtype=float)
        q = q / float(np.linalg.norm(q))
        r, v = state_along_direction(elems, q, mu)
        pos_map[name] = r
        vel_map[name] = v

    vel_map = apply_symmetric_velocity_kick(vel_map, params.delta_v_T1(), names=names)

    fairies = [
        Body(
            mass=m,
            position=pos_map[name],
            velocity=vel_map[name],
            name=name,
            radius=0.0,
        )
        for name in names
    ]
    system = System(bodies=[central, *fairies], G=G, labels=["central", *names])
    if com_frame:
        to_com_inertial_frame(system)
    return system
