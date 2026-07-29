"""Manifold generator on tetrahedron direction basis (PROMPT).

Index i = 0 is the nearest orbit. Polynomials:

    a_i = a0 + i a1 + i² a2
    e_i = e0 + i e1 + i² e2
    M_i = M0 + i M1 + i² M2

Velocity kick (Td-symmetric via Rodrigues from T1):

    δv_i = R_i · (v + i v₁)

with v=(vx,vy,vz), v₁=(v1x,v1y,v1z), R_T1=I.

After construction the system is shifted into the inertial COM frame.

Per Stage-A seed (m, e): a0=1, e0=e, M0=0, μ=m.
Search unlocks higher-order knobs in order a2 → e2 → M2 → (v1x,v1y,v1z).
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
    """θ includes linear + quadratic orbit polys and optional linear-in-i kick."""

    a0: float = 1.0
    a1: float = 0.15
    a2: float = 0.0
    e0: float = 0.05
    e1: float = 0.0
    e2: float = 0.0
    M0: float = 0.0
    M1: float = 0.5
    M2: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    v1x: float = 0.0
    v1y: float = 0.0
    v1z: float = 0.0
    mu_mass: float = 1e-3

    def as_theta(self) -> tuple[float, ...]:
        return (
            self.a0,
            self.a1,
            self.a2,
            self.e0,
            self.e1,
            self.e2,
            self.M0,
            self.M1,
            self.M2,
            self.vx,
            self.vy,
            self.vz,
            self.v1x,
            self.v1y,
            self.v1z,
            self.mu_mass,
        )

    @classmethod
    def from_theta(cls, theta: tuple[float, ...] | list[float]) -> ManifoldParams:
        t = [float(x) for x in theta]
        # Back-compat: old 10-vector (a0,a1,e0,e1,M0,M1,vx,vy,vz,μ)
        if len(t) == 10:
            return cls(
                a0=t[0],
                a1=t[1],
                e0=t[2],
                e1=t[3],
                M0=t[4],
                M1=t[5],
                vx=t[6],
                vy=t[7],
                vz=t[8],
                mu_mass=t[9],
            )
        if len(t) != 16:
            raise ValueError("θ must have 16 components (or legacy 10)")
        return cls(
            a0=t[0],
            a1=t[1],
            a2=t[2],
            e0=t[3],
            e1=t[4],
            e2=t[5],
            M0=t[6],
            M1=t[7],
            M2=t[8],
            vx=t[9],
            vy=t[10],
            vz=t[11],
            v1x=t[12],
            v1y=t[13],
            v1z=t[14],
            mu_mass=t[15],
        )

    def delta_v_T1(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=float)

    def delta_v1_T1(self) -> np.ndarray:
        return np.array([self.v1x, self.v1y, self.v1z], dtype=float)


def poly_linear(q0: float, q1: float, i: int) -> float:
    """q_i = q0 + i q1."""
    return q0 + float(i) * q1


def poly_quad(q0: float, q1: float, q2: float, i: int) -> float:
    """q_i = q0 + i q1 + i² q2."""
    ii = float(i)
    return q0 + ii * q1 + ii * ii * q2


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
    a2: float = 0.0,
    e2: float = 0.0,
    M2: float = 0.0,
    v1x: float = 0.0,
    v1y: float = 0.0,
    v1z: float = 0.0,
) -> ManifoldParams:
    """Seed anchors: a0=1, e0=e, M0=0, μ=m."""
    params = ManifoldParams(
        a0=1.0,
        a1=a1,
        a2=a2,
        e0=e,
        e1=e1,
        e2=e2,
        M0=0.0,
        M1=M1,
        M2=M2,
        vx=vx,
        vy=vy,
        vz=vz,
        v1x=v1x,
        v1y=v1y,
        v1z=v1z,
        mu_mass=m,
    )
    for i in range(4):
        elements_for_index(params, i)
    return params


def elements_for_index(params: ManifoldParams, i: int) -> OrbitalElements:
    """Build Kepler elements; soft-clip a>0 and e∈[0,1) (out-of-range is not an error)."""
    a = poly_quad(params.a0, params.a1, params.a2, i)
    e = poly_quad(params.e0, params.e1, params.e2, i)
    M = poly_quad(params.M0, params.M1, params.M2, i)
    a = max(float(a), 1e-6)
    e = float(np.clip(e, 0.0, 1.0 - 1e-9))
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
    delta_v1_T1: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Td-style symmetry:

        v_i ← v_i + R_i · (δv + i δv₁)
    """
    rots = rotations_from_T1()
    dv0 = np.asarray(delta_v_T1, dtype=float).reshape(3)
    dv1 = (
        np.zeros(3)
        if delta_v1_T1 is None
        else np.asarray(delta_v1_T1, dtype=float).reshape(3)
    )
    out = {}
    for i, name in enumerate(names):
        out[name] = (
            np.asarray(velocities[name], dtype=float).reshape(3)
            + rots[name] @ (dv0 + float(i) * dv1)
        )
    return out


def build_manifold_system(
    params: ManifoldParams | None = None,
    *,
    G: float = 1.0,
    central_mass: float = 1.0,
    names: tuple[str, ...] = FAIRY_ORDER,
    com_frame: bool = True,
) -> System:
    """Build X₀ in the central-body frame, then shift to inertial COM (default)."""
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

    vel_map = apply_symmetric_velocity_kick(
        vel_map,
        params.delta_v_T1(),
        names=names,
        delta_v1_T1=params.delta_v1_T1(),
    )

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
