"""Regular tetrahedron geometry for non-coplanar ladder phases (PROMPT §5)."""

from __future__ import annotations

import numpy as np

RA = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
RB = np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0)
RC = np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0)
RD = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0)

VERTICES = {"T1": RA, "T2": RB, "T3": RC, "T4": RD}
FAIRY_ORDER = ("T1", "T2", "T3", "T4")


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("zero vector")
    return v / n


def tetrahedral_phase_offsets() -> dict[str, float]:
    """
    Azimuthal angles of tetrahedron vertices (legacy 2D proxy).

    Prefer `kepler_state_along_vertex` for full 3D non-coplanar ICs.
    """
    offsets: dict[str, float] = {}
    for name, r in VERTICES.items():
        ang = float(np.arctan2(r[1], r[0]))
        if ang < 0.0:
            ang += 2.0 * np.pi
        offsets[name] = ang
    return offsets


def kepler_state_along_vertex(
    a: float,
    e: float,
    mu: float,
    vertex: np.ndarray,
    *,
    true_anomaly: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keplerian state with periapsis (f=0) along a tetrahedron vertex direction.

    Builds a non-coplanar ladder: each fairy's orbital plane contains its
    vertex vector, so the four planes are mutually asymmetric (PROMPT §5/§6 —
    not the old same-radius Rodrigues copy trap).
    """
    if not (0.0 <= e < 1.0):
        raise ValueError("need elliptical eccentricity in [0, 1)")
    if a <= 0.0:
        raise ValueError("a must be positive")

    u = _unit(vertex)
    # Orbital plane: span(u, t_hat). Stable tangent from a fixed reference.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(u, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t_hat = np.cross(u, ref)
    t_hat = _unit(t_hat)
    # Complete right-handed triad (u, q_hat, h_hat) with q_hat = h × u at periapsis
    h_hat = np.cross(u, t_hat)
    h_hat = _unit(h_hat)
    q_hat = np.cross(h_hat, u)  # in-plane, 90° from periapsis

    # Polar equation + perifocal velocity at true anomaly f
    f = float(true_anomaly)
    cf, sf = np.cos(f), np.sin(f)
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * cf)
    r = r_mag * (cf * u + sf * q_hat)

    # v = sqrt(mu/p) * (-sin f ê_r_peri + (e+cos f) ê_q) in perifocal basis
    # ê_r at periapsis frame: cos f û + sin f q̂ already used for position direction
    # Standard: vx_pf = -sqrt(mu/p) sin f, vy_pf = sqrt(mu/p) (e + cos f)
    speed = np.sqrt(mu / p)
    v = speed * (-sf * u + (e + cf) * q_hat)
    return r, v


def tetrahedral_ladder_states(
    axes: list[float],
    e: float,
    mu: float,
    names: tuple[str, ...] = FAIRY_ORDER,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One (r, v) per fairy on nested a's, periapsis along tetrahedron vertices."""
    if len(axes) != len(names):
        raise ValueError("axes and names length mismatch")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, a in zip(names, axes, strict=True):
        out[name] = kepler_state_along_vertex(a, e, mu, VERTICES[name])
    return out
