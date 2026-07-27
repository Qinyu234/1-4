"""Low-dimensional graded IC families: planar 90° rays and Td-direction stereo.

Shared radius / radial-velocity ladder on index i = 0..3:

    ρ_i = ρ₀ + Δρ (i − 3/2)
    v_{r,i} = k (i − 3/2)

Planar (顺逆): four rays at 90° in the xy-plane, alternating sense on v_t.
Stereo: tetrahedron unit directions q̂_i, common spin axis n̂ for t̂_i.

Parameter space (both): (ρ₀, Δρ, v_t, k, m) — 5D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.core.body import Body, System
from fairy_orbit.design.ladder import ALTERNATING_SENSE
from fairy_orbit.design.tetrahedron import FAIRY_ORDER, VERTICES

# i − 3/2 for i = 0,1,2,3 → (−1.5, −0.5, +0.5, +1.5)
INDEX_OFFSETS: tuple[float, float, float, float] = (-1.5, -0.5, 0.5, 1.5)

# Planar unit rays at 0°, 90°, 180°, 270°.
PLANAR90_DIRS: dict[str, np.ndarray] = {
    "T1": np.array([1.0, 0.0, 0.0]),
    "T2": np.array([0.0, 1.0, 0.0]),
    "T3": np.array([-1.0, 0.0, 0.0]),
    "T4": np.array([0.0, -1.0, 0.0]),
}


@dataclass(frozen=True)
class GradedParams:
    """5D graded family: (ρ₀, Δρ, v_t, k, m)."""

    rho0: float = 1.0
    delta_rho: float = 0.0
    v_t: float = 1.0
    k: float = 0.0
    m: float = 1e-3


def index_offset(i: int) -> float:
    return float(INDEX_OFFSETS[i])


def graded_radii(rho0: float, delta_rho: float, n: int = 4) -> np.ndarray:
    """ρ_i = ρ₀ + Δρ (i − 3/2)."""
    if n != 4:
        raise ValueError("graded family is defined for 4 fairies")
    rhos = np.array([rho0 + delta_rho * index_offset(i) for i in range(n)], dtype=float)
    if np.any(rhos <= 0.0):
        raise ValueError(f"all radii must be positive; got {rhos}")
    return rhos


def graded_vr(k: float, n: int = 4) -> np.ndarray:
    """v_{r,i} = k (i − 3/2)."""
    return np.array([k * index_offset(i) for i in range(n)], dtype=float)


def tangential_hat(q_hat: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    """t̂ = n̂ × q̂ / |n̂ × q̂| (common-rotation sense)."""
    q = np.asarray(q_hat, dtype=float).reshape(3)
    n = np.asarray(n_hat, dtype=float).reshape(3)
    qn = float(np.linalg.norm(q))
    nn = float(np.linalg.norm(n))
    if qn < 1e-15 or nn < 1e-15:
        raise ValueError("q_hat and n_hat must be non-zero")
    q = q / qn
    n = n / nn
    t = np.cross(n, q)
    tn = float(np.linalg.norm(t))
    if tn < 1e-15:
        raise ValueError("n_hat parallel to q_hat; pick another spin axis")
    return t / tn


def graded_states(
    directions: dict[str, np.ndarray],
    params: GradedParams,
    *,
    n_hat: np.ndarray | None = None,
    alternating: bool = False,
    names: tuple[str, ...] = FAIRY_ORDER,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build (r_i, v_i) on given unit directions:

        r_i = ρ_i q̂_i
        v_i = v_t,i t̂_i + v_{r,i} q̂_i

    If alternating, v_t,i = ±v_t via ALTERNATING_SENSE (planar 顺逆).
    """
    if n_hat is None:
        n_hat = np.array([0.0, 0.0, 1.0])
    rhos = graded_radii(params.rho0, params.delta_rho)
    vrs = graded_vr(params.k)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(names):
        q = np.asarray(directions[name], dtype=float).reshape(3)
        q = q / float(np.linalg.norm(q))
        t_hat = tangential_hat(q, n_hat)
        sense = 1.0
        if alternating:
            sense = float(ALTERNATING_SENSE.get(name, +1))
        vt = sense * params.v_t
        r = rhos[i] * q
        v = vt * t_hat + vrs[i] * q
        out[name] = (r, v)
    return out


def build_graded_system(
    directions: dict[str, np.ndarray],
    params: GradedParams,
    *,
    n_hat: np.ndarray | None = None,
    alternating: bool = False,
    G: float = 1.0,
    central_mass: float = 1.0,
    central_radius: float = 0.0,
    fairy_radius: float = 0.0,
    names: tuple[str, ...] = FAIRY_ORDER,
) -> System:
    states = graded_states(
        directions, params, n_hat=n_hat, alternating=alternating, names=names
    )
    central = Body(
        mass=central_mass,
        position=np.zeros(3),
        velocity=np.zeros(3),
        name="central",
        radius=central_radius,
    )
    fairies = [
        Body(
            mass=params.m,
            position=states[name][0],
            velocity=states[name][1],
            name=name,
            radius=fairy_radius,
        )
        for name in names
    ]
    return System(bodies=[central, *fairies], G=G, labels=["central", *names])


def build_planar90_system(
    params: GradedParams,
    *,
    G: float = 1.0,
    central_mass: float = 1.0,
    central_radius: float = 0.0,
    fairy_radius: float = 0.0,
) -> System:
    """顺逆平面：90° 四射线 + 交替旋向。"""
    return build_graded_system(
        PLANAR90_DIRS,
        params,
        n_hat=np.array([0.0, 0.0, 1.0]),
        alternating=True,
        G=G,
        central_mass=central_mass,
        central_radius=central_radius,
        fairy_radius=fairy_radius,
    )


def build_stereo_graded_system(
    params: GradedParams,
    *,
    n_hat: np.ndarray | None = None,
    G: float = 1.0,
    central_mass: float = 1.0,
    central_radius: float = 0.0,
    fairy_radius: float = 0.0,
) -> System:
    """立体低维：四面体方向 + (ρ₀, Δρ, v_t, k, m)。"""
    dirs = {name: np.asarray(VERTICES[name], dtype=float) for name in FAIRY_ORDER}
    return build_graded_system(
        dirs,
        params,
        n_hat=n_hat,
        alternating=False,
        G=G,
        central_mass=central_mass,
        central_radius=central_radius,
        fairy_radius=fairy_radius,
    )
