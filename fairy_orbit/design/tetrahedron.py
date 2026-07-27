"""Regular tetrahedron geometry + Rodrigues calibration IC (PROMPT §2.4.1 / §5)."""

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


def local_frame(r_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (radial, tangential, binormal) frame at r_hat."""
    r_hat = _unit(r_hat)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, r_hat))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t_hat = ref - np.dot(ref, r_hat) * r_hat
    t_hat = _unit(t_hat)
    n_hat = _unit(np.cross(r_hat, t_hat))
    return r_hat, t_hat, n_hat


def tetra_rotation(r_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """
    Rodrigues rotation mapping unit vector r_from onto r_to.

    Used to copy a local (v_rad, v_tan) velocity from T1 onto other vertices
    so the instantaneous configuration stays in the regular-tetrahedron orbit
    of the point group (PROMPT §2.4.1).
    """
    r_from = _unit(r_from)
    r_to = _unit(r_to)
    cos_theta = float(np.clip(np.dot(r_from, r_to), -1.0, 1.0))
    if cos_theta > 1.0 - 1e-14:
        return np.eye(3)
    if cos_theta < -1.0 + 1e-14:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(tmp, r_from))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(r_from, tmp))
        theta = np.pi
    else:
        axis = _unit(np.cross(r_from, r_to))
        theta = float(np.arccos(cos_theta))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    I = np.eye(3)
    return I * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * K


def rotations_from_T1() -> dict[str, np.ndarray]:
    """Identity for T1; Rodrigues maps RA → other tetrahedron vertices."""
    return {
        "T1": np.eye(3),
        "T2": tetra_rotation(RA, RB),
        "T3": tetra_rotation(RA, RC),
        "T4": tetra_rotation(RA, RD),
    }


def tetrahedral_phase_offsets() -> dict[str, float]:
    """Azimuthal angles of tetrahedron vertices (planar M-offset proxy)."""
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

    Used for nested-a tetrahedral_3d ladders (different planes). Not the
    calibration IC — calibration uses Rodrigues (v_rad, v_tan) copy.
    """
    if not (0.0 <= e < 1.0):
        raise ValueError("need elliptical eccentricity in [0, 1)")
    if a <= 0.0:
        raise ValueError("a must be positive")

    u = _unit(vertex)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(u, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t_hat = _unit(np.cross(u, ref))
    h_hat = _unit(np.cross(u, t_hat))
    q_hat = np.cross(h_hat, u)

    f = float(true_anomaly)
    cf, sf = np.cos(f), np.sin(f)
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * cf)
    r = r_mag * (cf * u + sf * q_hat)
    speed = np.sqrt(mu / p)
    v = speed * (-sf * u + (e + cf) * q_hat)
    return r, v


def kepler_vr_vt_at_true_anomaly(
    a: float, e: float, mu: float, true_anomaly: float = 0.0
) -> tuple[float, float]:
    """Alias of kepler_polar_speeds — (a,e,f) ↔ (v_rad, v_tan)."""
    from fairy_orbit.design.elements import kepler_polar_speeds

    return kepler_polar_speeds(a, e, mu, true_anomaly)


def calibration_tetrahedron_states(
    a: float,
    e: float,
    mu: float,
    names: tuple[str, ...] = FAIRY_ORDER,
    *,
    true_anomaly: float = 0.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    PROMPT §2.4.1 calibration IC in Newton form (r, v).

    Same physical state as the legacy recipe:
      1. shared (a,e,f)  ↔  (v_rad, v_tan)   [synonym, 2 of the 6 elements]
      2. T1 at r = r(f)·RA,  v = v_rad r̂ + v_tan t̂
      3. copy (r,v) to T2..T4 by Rodrigues maps RA→RX

    Orbital-element form of the same state is just OrbitalElements.from_state
    on each (r,v) — not a different IC.
    """
    if not (0.0 <= e < 1.0):
        raise ValueError("need elliptical eccentricity in [0, 1)")
    if a <= 0.0:
        raise ValueError("a must be positive")

    f = float(true_anomaly)
    from fairy_orbit.design.elements import kepler_polar_speeds, kepler_radius

    r_mag = kepler_radius(a, e, f)
    v_rad, v_tan = kepler_polar_speeds(a, e, mu, f)

    r_hat, t_hat, _ = local_frame(RA)
    v1 = v_rad * r_hat + v_tan * t_hat
    r1 = r_mag * RA

    rots = rotations_from_T1()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        R = rots[name]
        out[name] = (R @ r1, R @ v1)
    return out


def calibration_tetrahedron_elements(
    a: float,
    e: float,
    mu: float,
    names: tuple[str, ...] = FAIRY_ORDER,
    *,
    true_anomaly: float = 0.0,
) -> dict[str, "OrbitalElements"]:
    """Orbital-element synonym of `calibration_tetrahedron_states` (same 6-DOF)."""
    from fairy_orbit.design.elements import state_to_elements

    states = calibration_tetrahedron_states(
        a, e, mu, names=names, true_anomaly=true_anomaly
    )
    return {name: state_to_elements(r, v, mu) for name, (r, v) in states.items()}


def tetrahedral_ladder_states(
    axes: list[float],
    e: float,
    mu: float,
    names: tuple[str, ...] = FAIRY_ORDER,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Nested-a ladder: periapsis along tetrahedron vertices (asymmetric planes)."""
    if len(axes) != len(names):
        raise ValueError("axes and names length mismatch")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, a in zip(names, axes, strict=True):
        out[name] = kepler_state_along_vertex(a, e, mu, VERTICES[name])
    return out
