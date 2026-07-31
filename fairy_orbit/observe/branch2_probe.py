"""PROMPT §2.4 Branch-2 cheap existence probe.

Fix a verified free-4 (ABCD) choreography's (R, τ). Sample E with r_E>0 and
ask whether any IC shows a small residual for the *same* discrete symmetry

    x_E(τ) ≈ R · x_E(0),   v_E(τ) ≈ R · v_E(0)

before investing in continuation. ABCD are held as a *restricted* background
(their IC is re-loaded each trial; E is a light tracer so the field is nearly
the free choreography).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from fairy_orbit.core.body import Body, System, to_com_inertial_frame
from fairy_orbit.design.seeds import OrbitSeed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate_endpoint
from fairy_orbit.observe.choreography_verify import verify_choreography_Tn


@dataclass(frozen=True)
class Branch2ProbeHit:
    residual: float
    rho: float
    z: float
    speed: float


@dataclass(frozen=True)
class Branch2ProbeResult:
    n_samples: int
    tau: float
    angle: float
    axis: tuple[float, float, float]
    abcd_gate_ok: bool
    abcd_E_r_rel: float
    abcd_E_v_rel: float
    best_residual: float
    median_residual: float
    frac_below: dict[str, float]
    best: Branch2ProbeHit | None
    hopeful: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "tau": self.tau,
            "angle": self.angle,
            "axis": list(self.axis),
            "abcd_gate_ok": self.abcd_gate_ok,
            "abcd_E_r_rel": self.abcd_E_r_rel,
            "abcd_E_v_rel": self.abcd_E_v_rel,
            "best_residual": self.best_residual,
            "median_residual": self.median_residual,
            "frac_below": self.frac_below,
            "best": None
            if self.best is None
            else {
                "residual": self.best.residual,
                "rho": self.best.rho,
                "z": self.best.z,
                "speed": self.best.speed,
            },
            "hopeful": self.hopeful,
            "notes": self.notes,
        }


def _abcd_R_tau(seed4: OrbitSeed, *, shift: int = 1, atol_rel: float = 1e-6):
    gate = verify_choreography_Tn(
        seed4.to_system(),
        float(seed4.period),
        shift=shift,
        atol_rel=atol_rel,
        n_outputs=32,
    )
    tau = float(seed4.period) / seed4.n_bodies
    return gate, tau


def _sample_e_state(
    rng: np.random.Generator,
    axis: np.ndarray,
    *,
    theta: float,
    tau: float,
    rho_range: tuple[float, float] = (0.15, 1.8),
    z_scale: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Cylindrical IC about ``axis`` with rough counter-rotation ω≈-θ/τ."""
    a = np.asarray(axis, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-300)
    # orthonormal frame
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, tmp)
    e1 /= max(float(np.linalg.norm(e1)), 1e-300)
    e2 = np.cross(a, e1)
    rho = float(rng.uniform(*rho_range))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    z = float(rng.normal(0.0, z_scale))
    pos = rho * (math.cos(phi) * e1 + math.sin(phi) * e2) + z * a
    omega = -float(theta) / max(float(tau), 1e-300)
    # tangential for counter-rotation in the plane ⟂ axis
    tang = -math.sin(phi) * e1 + math.cos(phi) * e2
    speed = abs(omega) * rho * float(rng.uniform(0.6, 1.4))
    vz = float(rng.normal(0.0, 0.25))
    vel = speed * tang + vz * a
    return pos, vel, rho, z, speed


def branch2_existence_probe(
    seed4: OrbitSeed,
    *,
    n_samples: int = 64,
    m_e: float = 1e-6,
    shift: int = 1,
    seed: int = 0,
    hopeful_thresh: float = 0.15,
    frac_thresh: float = 0.35,
) -> Branch2ProbeResult:
    """
    Cheap Monte-Carlo probe for Branch-2 existence under ABCD's (R, τ).

    Returns ``hopeful=True`` if best residual is below ``hopeful_thresh`` *or*
    a non-trivial fraction of samples lands below ``frac_thresh``.
    """
    gate, tau = _abcd_R_tau(seed4, shift=shift, atol_rel=1e-5)
    axis = np.asarray(gate.axis, dtype=float)
    theta = float(gate.angle)
    R = np.asarray(gate.R, dtype=float)

    rng = np.random.default_rng(int(seed))
    residuals: list[float] = []
    best_hit: Branch2ProbeHit | None = None
    best_res = float("inf")

    cfg = ReboundConfig(
        stop_on_escape=False,
        stop_on_collision=False,
        epsilon=0.0,
        dt=max(tau / 200.0, 1e-3),
        min_dt=1e-5,
    )

    for _ in range(int(n_samples)):
        pos_e, vel_e, rho, z, speed = _sample_e_state(
            rng, axis, theta=theta, tau=tau
        )
        bodies = [
            Body(
                mass=float(seed4.masses[i]),
                position=np.asarray(seed4.positions[i], dtype=float).copy(),
                velocity=np.asarray(seed4.velocities[i], dtype=float).copy(),
                name=seed4.names[i] if i < len(seed4.names) else f"B{i}",
            )
            for i in range(seed4.n_bodies)
        ]
        bodies.append(
            Body(mass=float(m_e), position=pos_e, velocity=vel_e, name="E")
        )
        sys = System(bodies=bodies, G=float(seed4.G))
        to_com_inertial_frame(sys)
        # record E index after COM shift
        e_idx = len(sys.bodies) - 1
        r0 = sys.bodies[e_idx].position.copy()
        v0 = sys.bodies[e_idx].velocity.copy()
        r, v = integrate_endpoint(sys, tau, config=cfg)
        r_e, v_e = r[e_idx], v[e_idx]
        # residual under the *fixed* ABCD rotation R (not re-Kabsch on E alone)
        dr = r_e - R @ r0
        dv = v_e - R @ v0
        scale = max(float(np.linalg.norm(r0)), float(np.linalg.norm(v0)), 1e-300)
        res = float(np.linalg.norm(np.concatenate([dr, dv])) / scale)
        residuals.append(res)
        if res < best_res:
            best_res = res
            best_hit = Branch2ProbeHit(residual=res, rho=rho, z=z, speed=speed)

    arr = np.asarray(residuals, dtype=float)
    frac_below = {
        "0.5": float(np.mean(arr < 0.5)),
        "0.35": float(np.mean(arr < 0.35)),
        "0.2": float(np.mean(arr < 0.2)),
        "0.1": float(np.mean(arr < 0.1)),
    }
    hopeful = bool(
        best_res < float(hopeful_thresh) or frac_below["0.35"] >= 0.05
    )
    notes = (
        "hopeful: some tracer-E samples nearly obey ABCD's (R,τ)"
        if hopeful
        else "bleak for this ABCD family (R,τ) only — not a global Branch-2 kill; "
        "deprioritize Branch-2 until another family looks hopeful"
    )
    return Branch2ProbeResult(
        n_samples=int(n_samples),
        tau=float(tau),
        angle=theta,
        axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        abcd_gate_ok=bool(gate.ok),
        abcd_E_r_rel=float(gate.E_r_rel),
        abcd_E_v_rel=float(gate.E_v_rel),
        best_residual=float(best_res if np.isfinite(best_res) else 1e300),
        median_residual=float(np.median(arr)) if arr.size else float("nan"),
        frac_below=frac_below,
        best=best_hit,
        hopeful=hopeful,
        notes=notes,
    )
