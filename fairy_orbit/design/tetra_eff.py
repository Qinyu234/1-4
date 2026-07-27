"""Td group-orbit reduction: r_i(t) = ρ(t) R(t) q_i  (not Kepler ellipses).

N-body Lagrangian on the shape-preserving ansatz (central fixed at 0,
four equal fairy masses m, unit tetrahedron vertices q_i):

    T = ½ A ρ̇² + ½ B ρ² |Ω|²
    U = −C / ρ

with Ω = R^{-1} Ṙ ∈ so(3), and (G = M_central = 1)

    A = Σ m |q_i|²           = 4 m
    B = Σ m |q_i × n̂|²       = 8 m / 3     (axis-independent for Td)
    C = 4 m + 6 m² / √(8/3)  = 4 m (1 + 3√6/8 · m)

Angular momentum J = B ρ² ω conserved ⇒ ω = J / (B ρ²).

Scale equation:
    A ρ̈ = −C/ρ² + J² / (B ρ³)
         ⇒  ρ̈ = −μ_eff/ρ² + 3 J² / (32 m² ρ³)
    μ_eff = C/(4m) = 1 + (3√6/8) m

Orbit shape in the (ρ, θ) plane (θ cumulative rotation about Ĵ):
    dρ/dθ = (B ρ² / J) ρ̇
    with energy ½ A ρ̇² + V_eff(ρ) = E,  V_eff = −C/ρ + J²/(2 B ρ²)
    ⇒ θ(ρ) is generally an elliptic integral — not a Kepler ellipse.

State reconstruction:
    r_i(t) = ρ(t) exp(∫_0^t ω(τ) [n̂]_× dτ) q_i
    v_i(t) = ρ̇ R q_i + ρ ω n̂ × (R q_i)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fairy_orbit.design.tetrahedron import FAIRY_ORDER, VERTICES

# μ_eff fairy coefficient = 3√6/8 = (3/2)/√(8/3)
MU_EFF_FAIRY_COEFF = 3.0 * math.sqrt(6.0) / 8.0


def reduced_masses(m: float) -> tuple[float, float, float]:
    """Return (A, B, C) for equal-mass Td + fixed central (G=M=1)."""
    if m <= 0.0:
        raise ValueError("fairy mass m must be positive for Td reduction")
    A = 4.0 * m
    B = 8.0 * m / 3.0
    C = 4.0 * m * (1.0 + MU_EFF_FAIRY_COEFF * m)
    return A, B, C


def mu_eff(m: float, *, M: float = 1.0, G: float = 1.0) -> float:
    """C/(A) * (G M factors): one-body equivalent GM = G(M + α m)."""
    if m < 0.0:
        raise ValueError("fairy mass m must be non-negative")
    return float(G * (M + MU_EFF_FAIRY_COEFF * m))


def unit_vertices(names: tuple[str, ...] = FAIRY_ORDER) -> np.ndarray:
    """(4, 3) array of fixed Td unit vertices q_i."""
    return np.stack([VERTICES[n] for n in names], axis=0)


def hat_cross_matrix(n_hat: np.ndarray) -> np.ndarray:
    """so(3) hat map: [n]_×."""
    n = np.asarray(n_hat, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -n[2], n[1]],
            [n[2], 0.0, -n[0]],
            [-n[1], n[0], 0.0],
        ],
        dtype=float,
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues: exp([φ]_×) for rotation vector φ = θ n̂."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-15:
        return np.eye(3)
    K = hat_cross_matrix(phi / theta)
    return (
        np.eye(3)
        + math.sin(theta) * K
        + (1.0 - math.cos(theta)) * (K @ K)
    )


@dataclass(frozen=True)
class TdOrbit:
    """Constants of a Td group-orbit solution."""

    m: float
    A: float
    B: float
    C: float
    J: float  # |angular momentum| of the four fairies (body scalar)
    n_hat: np.ndarray  # unit rotation axis (inertial, fixed)
    rho0: float
    rhodot0: float
    E: float  # ½ A ρ̇² − C/ρ + J²/(2 B ρ²)

    @property
    def mu_eff(self) -> float:
        return self.C / self.A

    def omega(self, rho: float) -> float:
        """ω(ρ) = J / (B ρ²); 0 when J=0."""
        if abs(self.J) < 1e-30:
            return 0.0
        return float(self.J / (self.B * rho * rho))

    def v_eff(self, rho: float) -> float:
        """V_eff(ρ) = −C/ρ + J²/(2 B ρ²)."""
        return -self.C / rho + (self.J * self.J) / (2.0 * self.B * rho * rho)

    def bound_scale(self) -> bool:
        """True if E < 0 (cannot reach ρ → ∞ with V_eff → 0)."""
        return self.E < 0.0


def td_orbit_from_ic(
    m: float,
    rho0: float,
    rhodot0: float,
    omega0: float,
    n_hat: np.ndarray | None = None,
) -> TdOrbit:
    """
    Build TdOrbit from (m, ρ₀, ρ̇₀, ω₀, n̂).

    J = B ρ₀² ω₀.  Energy from the reduced Lagrangian.
    """
    if rho0 <= 0.0:
        raise ValueError("rho0 must be positive")
    A, B, C = reduced_masses(m)
    if n_hat is None:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = np.asarray(n_hat, dtype=float).reshape(3)
        nn = float(np.linalg.norm(n))
        if nn < 1e-15:
            raise ValueError("n_hat must be non-zero")
        n = n / nn
    J = B * rho0 * rho0 * float(omega0)
    E = 0.5 * A * rhodot0 * rhodot0 - C / rho0 + (J * J) / (2.0 * B * rho0 * rho0)
    return TdOrbit(
        m=float(m),
        A=A,
        B=B,
        C=C,
        J=float(J),
        n_hat=n,
        rho0=float(rho0),
        rhodot0=float(rhodot0),
        E=float(E),
    )


def circular_omega(m: float, rho: float) -> float:
    """ω for circular Td orbit: ρ̈=0 ⇒ J² = B C ρ ⇒ ω = √(C/(B ρ³))."""
    _A, B, C = reduced_masses(m)
    return float(math.sqrt(C / (B * rho**3)))


def vc_scale(m: float, rho: float = 1.0) -> float:
    """
    Circular speed scale v_c = √(μ_eff / ρ).  With ρ=1, v_c = √μ_eff.
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    return float(math.sqrt(mu_eff(m) / rho))


def polar_from_beta_alpha(
    m: float, beta: float, alpha: float, *, rho: float = 1.0
) -> tuple[float, float]:
    """
    Dimensionless IC: |v| = β v_c, α from radial axis (α=π/2 pure tangential).

        v_r = β √μ_eff cos α
        v_t = β √μ_eff sin α

    β<√2 bound; J ∝ v_t = β √μ_eff sin α.
    """
    vc = vc_scale(m, rho)
    v = beta * vc
    v_r = v * math.cos(alpha)
    v_t = v * math.sin(alpha)
    return float(v_r), float(v_t)


def energy_from_beta(m: float, beta: float, *, rho: float = 1.0) -> float:
    """E = μ_eff (β²/2 − 1) / ρ  (per the reduced one-body scale)."""
    return float(mu_eff(m) * (0.5 * beta * beta - 1.0) / rho)


def eccentricity_from_beta_alpha(beta: float, alpha: float) -> float:
    """
    Kepler-analogue eccentricity for the reduced (ρ, θ) problem at ρ0=1:

        e² = 1 − (2 − β²) β² sin²α

    Requires bound energy β² < 2 for a real elliptic-type e < 1.
    """
    s2 = math.sin(alpha) ** 2
    e2 = 1.0 - (2.0 - beta * beta) * (beta * beta) * s2
    if e2 < -1e-12:
        raise ValueError(f"imaginary eccentricity at β={beta}, α={alpha}")
    return float(math.sqrt(max(e2, 0.0)))


def e_min_for_beta(beta: float) -> float:
    """
    Smallest reachable e at fixed β (max |sin α|=1):

        e_min² = max(0, 1 − (2−β²)β²)
    """
    if beta * beta >= 2.0:
        return 1.0
    e2 = 1.0 - (2.0 - beta * beta) * (beta * beta)
    return float(math.sqrt(max(e2, 0.0)))


def is_valid_beta_e(beta: float, e: float) -> bool:
    """True if (β, e) maps to a real α with 0 < sin²α ≤ 1 and β² < 2."""
    if not (0.0 <= e < 1.0):
        return False
    if beta <= 0.0 or beta * beta >= 2.0:
        return False
    denom = (2.0 - beta * beta) * (beta * beta)
    if denom <= 0.0:
        return False
    s2 = (1.0 - e * e) / denom
    return 0.0 < s2 <= 1.0 + 1e-12


def sin_alpha_from_beta_e(beta: float, e: float) -> float:
    """
    Invert e² = 1 − (2−β²)β² sin²α:

        sin α = √( (1−e²) / ((2−β²) β²) )
    """
    if not is_valid_beta_e(beta, e):
        raise ValueError(
            f"(β, e)=({beta}, {e}) outside valid domain "
            f"(need e ≥ e_min(β)={e_min_for_beta(beta):.4g}, β<√2)"
        )
    denom = (2.0 - beta * beta) * (beta * beta)
    return float(math.sqrt((1.0 - e * e) / denom))


def alpha_from_beta_e(beta: float, e: float, *, vr_sign: float = 1.0) -> float:
    """α ∈ (0, π) with sin α > 0; sign(vr)=sign(vr_sign) via cos α."""
    s = sin_alpha_from_beta_e(beta, e)
    s = min(1.0, max(0.0, s))
    c = math.sqrt(max(0.0, 1.0 - s * s))
    if vr_sign < 0.0:
        c = -c
    return float(math.atan2(s, c))


def polar_from_beta_e(
    m: float,
    beta: float,
    e: float,
    *,
    rho: float = 1.0,
    vr_sign: float = 1.0,
) -> tuple[float, float, float]:
    """Return (v_r, v_t, α) from recommended search vars (m, β, e)."""
    alpha = alpha_from_beta_e(beta, e, vr_sign=vr_sign)
    vr, vt = polar_from_beta_alpha(m, beta, alpha, rho=rho)
    return vr, vt, alpha


def vt_from_omega(rho: float, omega: float) -> float:
    """
    RMS tangential speed implied by rigid Td spin.

    T_rot = ½ B ρ² ω² = ½ (4m) v_t,rms²  with B=8m/3
      ⇒ v_t,rms = ρ |ω| √(2/3).
    """
    return float(abs(rho * omega) * math.sqrt(2.0 / 3.0))


def vt_circular(m: float, rho: float) -> float:
    """Circular v_t,rms — equal to vc_scale(m, ρ) for Td masses."""
    return vt_from_omega(rho, circular_omega(m, rho))


def omega_from_vt(rho: float, v_t: float) -> float:
    """Invert v_t,rms = ρ|ω|√(2/3). Sign taken positive."""
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    return float(abs(v_t) / (rho * math.sqrt(2.0 / 3.0)))


def is_bound_polar(rho: float, v_r: float, v_t: float, m: float) -> bool:
    """v_r² + v_t² < 2 μ_eff / ρ  ⟺  β < √2."""
    return (v_r * v_r + v_t * v_t) < (2.0 * mu_eff(m) / rho)


def rho_accel(orbit: TdOrbit, rho: float) -> float:
    """ρ̈ = (−C/ρ² + J²/(B ρ³)) / A."""
    return (-orbit.C / (rho * rho) + (orbit.J * orbit.J) / (orbit.B * rho**3)) / orbit.A


def integrate_scale(
    orbit: TdOrbit,
    t_end: float,
    *,
    n_steps: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate ρ̈ = … with leapfrog; return (t, ρ, ρ̇, θ).

    θ̇ = ω = J/(B ρ²); θ(0)=0.  R(t) = exp(θ(t) [n̂]_×).
    """
    if t_end < 0.0:
        raise ValueError("t_end must be non-negative")
    n_steps = max(int(n_steps), 1)
    dt = t_end / n_steps if t_end > 0.0 else 0.0
    t = np.linspace(0.0, t_end, n_steps + 1)
    rho = np.empty(n_steps + 1)
    rhodot = np.empty(n_steps + 1)
    theta = np.empty(n_steps + 1)
    rho[0] = orbit.rho0
    rhodot[0] = orbit.rhodot0
    theta[0] = 0.0

    for k in range(n_steps):
        r = rho[k]
        v = rhodot[k]
        if r < 1e-12:
            rho[k:] = r
            rhodot[k:] = v
            theta[k:] = theta[k]
            break
        a = rho_accel(orbit, r)
        # leapfrog / velocity-Verlet
        v_half = v + 0.5 * dt * a
        r_new = r + dt * v_half
        if r_new < 1e-12:
            rho[k + 1] = max(r_new, 0.0)
            rhodot[k + 1] = v_half
            theta[k + 1] = theta[k] + dt * orbit.omega(max(r, 1e-12))
            rho[k + 2 :] = rho[k + 1]
            rhodot[k + 2 :] = rhodot[k + 1]
            theta[k + 2 :] = theta[k + 1]
            break
        a_new = rho_accel(orbit, r_new)
        v_new = v_half + 0.5 * dt * a_new
        omega_mid = orbit.omega(0.5 * (r + r_new))
        rho[k + 1] = r_new
        rhodot[k + 1] = v_new
        theta[k + 1] = theta[k] + dt * omega_mid
    return t, rho, rhodot, theta


def rotation_matrix(orbit: TdOrbit, theta: float) -> np.ndarray:
    """R(θ) = exp(θ [n̂]_×)."""
    return so3_exp(theta * orbit.n_hat)


def states_at_rho_theta(
    orbit: TdOrbit,
    rho: float,
    rhodot: float,
    theta: float,
    names: tuple[str, ...] = FAIRY_ORDER,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """r_i = ρ R q_i,  v_i = ρ̇ R q_i + ρ ω (n̂ × R q_i)."""
    R = rotation_matrix(orbit, theta)
    omega = orbit.omega(rho)
    wvec = omega * orbit.n_hat
    q = unit_vertices(names)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, name in enumerate(names):
        rq = R @ q[i]
        r = rho * rq
        v = rhodot * rq + rho * np.cross(wvec, rq)
        out[name] = (r, v)
    return out


def analytic_states_at_time(
    orbit: TdOrbit,
    t: float,
    *,
    n_steps: int | None = None,
    names: tuple[str, ...] = FAIRY_ORDER,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Integrate scale ODE to time t, then reconstruct Td states."""
    if t < 0.0:
        raise ValueError("t must be non-negative")
    if t == 0.0:
        return states_at_rho_theta(orbit, orbit.rho0, orbit.rhodot0, 0.0, names=names)
    # Adaptive-ish step count: ~200 steps per characteristic time √(ρ³/μ)
    if n_steps is None:
        t_char = math.sqrt(max(orbit.rho0, 1e-12) ** 3 / max(orbit.mu_eff, 1e-30))
        n_steps = max(int(200 * t / max(t_char, 1e-12)), 50)
    ts, rhos, rhodots, thetas = integrate_scale(orbit, t, n_steps=n_steps)
    return states_at_rho_theta(orbit, float(rhos[-1]), float(rhodots[-1]), float(thetas[-1]), names=names)


def build_td_system(
    m: float,
    rho0: float,
    rhodot0: float,
    omega0: float,
    *,
    n_hat: np.ndarray | None = None,
    G: float = 1.0,
    central_mass: float = 1.0,
    central_radius: float = 0.0,
    fairy_radius: float = 0.0,
):
    """Newtonian System on the Td group-orbit IC (radii default 0 → no collisions)."""
    from fairy_orbit.core.body import Body, System

    orbit = td_orbit_from_ic(m, rho0, rhodot0, omega0, n_hat=n_hat)
    states = states_at_rho_theta(orbit, rho0, rhodot0, 0.0)
    central = Body(
        mass=central_mass,
        position=np.zeros(3),
        velocity=np.zeros(3),
        name="central",
        radius=central_radius,
    )
    fairies = [
        Body(
            mass=m,
            position=states[name][0],
            velocity=states[name][1],
            name=name,
            radius=fairy_radius,
        )
        for name in FAIRY_ORDER
    ]
    bodies = [central, *fairies]
    return System(bodies=bodies, G=G, labels=[b.name for b in bodies]), orbit


# --- orbit shape ρ(θ) (time eliminated) ---------------------------------


def drho_dtheta(orbit: TdOrbit, rho: float, rhodot: float) -> float:
    """dρ/dθ = (B ρ² / J) ρ̇   (J≠0)."""
    if abs(orbit.J) < 1e-30:
        return float("inf") if rhodot != 0.0 else 0.0
    return (orbit.B * rho * rho / orbit.J) * rhodot


def rhodot_from_energy(orbit: TdOrbit, rho: float, *, sign: float = 1.0) -> float:
    """ρ̇² = (2/A)(E + C/ρ − J²/(2 B ρ²))."""
    inside = orbit.E + orbit.C / rho - (orbit.J * orbit.J) / (2.0 * orbit.B * rho * rho)
    if inside < 0.0:
        if inside > -1e-12:
            return 0.0
        raise ValueError("rho outside energetically allowed region")
    return float(sign * math.sqrt(2.0 * inside / orbit.A))
