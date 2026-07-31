"""Mass continuation stub (PROMPT §2.5 Path A).

Not black-box search: residual F(X,T; M_c) from Φ_T + §3.2 closure,
corrected by scipy least_squares (Newton-ish). Pseudo-arclength TBD.

Diagnostic: compare residual to M_c=0 seed (raw). First-order perturbation
baseline is TODO per PROMPT §2.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from fairy_orbit.core.body import Body, System, to_com_inertial_frame
from fairy_orbit.design.seeds import OrbitSeed, load_seed, save_seed, SEEDS_DIR
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.choreography_verify import (
    ChoreographyVerifyResult,
    cyclic_role_perm,
    verify_choreography_Tn,
)
from fairy_orbit.observe.closure import closure_for_perm


def attach_central_mass(seed: OrbitSeed, M_c: float) -> System:
    """
    Build COM-frame system: central mass at origin + seed bodies.

    At M_c=0 the central is a massless tracer (still present for bookkeeping);
    Newton forces ignore m=0 partners equivalently if we omit it — we keep a
    tiny floor only when M_c==0 to avoid zero-mass REBOUND issues? Prefer omit
    central when M_c==0 for exact free dynamics.
    """
    fairies = [
        Body(
            mass=float(seed.masses[i]),
            position=np.asarray(seed.positions[i], dtype=float).copy(),
            velocity=np.asarray(seed.velocities[i], dtype=float).copy(),
            name=seed.names[i] if i < len(seed.names) else f"B{i}",
        )
        for i in range(seed.n_bodies)
    ]
    if M_c <= 0.0:
        sys = System(bodies=fairies, G=float(seed.G))
    else:
        central = Body(
            mass=float(M_c),
            position=np.zeros(3),
            velocity=np.zeros(3),
            name="C",
        )
        sys = System(bodies=[central, *fairies], G=float(seed.G))
    to_com_inertial_frame(sys)
    return sys


def _fairy_slice(sys: System, seed: OrbitSeed) -> tuple[np.ndarray, np.ndarray]:
    """Positions/velocities of the N free bodies (skip central if present)."""
    if len(sys.bodies) == seed.n_bodies:
        return sys.positions(), sys.velocities()
    # central at index 0
    pos = np.stack([b.position for b in sys.bodies[1:]])
    vel = np.stack([b.velocity for b in sys.bodies[1:]])
    return pos, vel


def symmetry_residual_vector(
    sys: System,
    seed: OrbitSeed,
    period: float,
    *,
    shift: int = 1,
    n_outputs: int = 16,
    optics_soft: bool = True,
    log_rho: float = 0.0,
    d_target: float | None = None,
    w_gravity: float = 1.0,
    w_optics: float = 1.0,
    encounter_factor: float = 3.0,
) -> np.ndarray:
    """
    Flattened residual of §3.2 on the fairy subset after integrating τ=T/n.

    F = stack_i ( r_i(τ) - R r_{P(i)}(0),  v_i(τ) - R v_{P(i)}(0) ).

    When a central body is present and ``optics_soft``, also append soft
    extras: gravity close-approach attract + encounter-conditioned
    ``|Δr_perp|`` optical deficit (equal-density ``log_rho``).
    """
    from fairy_orbit.observe.optical_encounter import DEFAULT_LOG_RHO

    lr = float(log_rho) if log_rho is not None else DEFAULT_LOG_RHO

    n = seed.n_bodies
    tau = float(period) / n
    perm = cyclic_role_perm(n, shift=shift)
    r0, v0 = _fairy_slice(sys, seed)
    has_central = len(sys.bodies) > n
    cfg = ReboundConfig(
        stop_on_escape=False,
        stop_on_collision=False,
        # Fixed dt when a light central is present — IAS15 adaptive can stall
        # on near-symmetric 1+N configurations.
        epsilon=0.0,
        dt=2e-3,
        min_dt=1e-5,
    )

    extras: list[float] = []
    if optics_soft and has_central:
        n_out = max(int(n_outputs), 32)
        traj = integrate(sys, t_end=float(period), n_outputs=n_out, config=cfg)
        t_idx = int(np.argmin(np.abs(traj.times - tau)))
        r = traj.positions[t_idx, 1 : n + 1]
        v = traj.velocities[t_idx, 1 : n + 1]
        extras = list(
            _gravity_optics_soft_extras(
                traj,
                seed,
                log_rho=lr,
                d_target=d_target,
                w_gravity=float(w_gravity),
                w_optics=float(w_optics),
                encounter_factor=float(encounter_factor),
            )
        )
    else:
        traj = integrate(sys, t_end=tau, n_outputs=n_outputs, config=cfg)
        if traj.n_bodies == n:
            r = traj.positions[-1]
            v = traj.velocities[-1]
        else:
            r = traj.positions[-1, 1 : n + 1]
            v = traj.velocities[-1, 1 : n + 1]

    cl = closure_for_perm(r, v, r0, v0, perm)
    R = cl.R
    chunks = []
    for i, j in enumerate(perm):
        chunks.append(r[i] - R @ r0[j])
        chunks.append(v[i] - R @ v0[j])
    base = np.concatenate(chunks).astype(float)
    if not extras:
        return base
    return np.concatenate([base, np.asarray(extras, dtype=float)])


def _gravity_optics_soft_extras(
    traj,
    seed: OrbitSeed,
    *,
    log_rho: float,
    d_target: float | None,
    w_gravity: float,
    w_optics: float,
    encounter_factor: float,
    central_index: int = 0,
) -> tuple[float, float]:
    """Soft (grav_pen, optics_pen) from a period trajectory with central at 0."""
    from fairy_orbit.observe.optical_encounter import (
        radii_from_uniform_density,
        soft_optics_deficit_perp,
    )

    n = seed.n_bodies
    if traj.masses is not None and len(traj.masses) == n + 1:
        masses = np.asarray(traj.masses, dtype=float)
    else:
        m_c = float(traj.masses[central_index]) if traj.masses is not None else 1.0
        masses = np.concatenate([[m_c], np.asarray(seed.masses, dtype=float)])
    R = radii_from_uniform_density(masses, log_rho=log_rho)
    fairy_idx = [k for k in range(traj.n_bodies) if k != central_index]

    d_min = float("inf")
    best = (0, fairy_idx[0], fairy_idx[min(1, len(fairy_idx) - 1)])
    for t in range(len(traj)):
        pos = traj.positions[t]
        for a, i in enumerate(fairy_idx):
            for j in fairy_idx[a + 1 :]:
                d = float(np.linalg.norm(pos[i] - pos[j]))
                if d < d_min:
                    d_min = d
                    best = (t, i, j)

    R_f = R[fairy_idx]
    R_typ = float(np.mean(R_f)) if len(R_f) else 1e-3
    target = float(d_target) if d_target is not None else 4.0 * max(R_typ, 1e-6)
    grav_pen = float(w_gravity) * max(0.0, target - d_min) ** 2

    enc_thresh = float(encounter_factor) * target
    if d_min <= enc_thresh and len(fairy_idx) >= 2:
        t, i, j = best
        pos = traj.positions[t]
        obs = pos[central_index]
        deficit = soft_optics_deficit_perp(
            pos[i], pos[j], float(R[i]), float(R[j]), observer=obs
        )
        opt_pen = float(w_optics) * deficit**2
    else:
        opt_pen = 0.0
    return grav_pen, opt_pen


@dataclass
class ContinuationSmokeResult:
    M_c: float
    gate0: ChoreographyVerifyResult
    residual0_norm: float
    residual_mc_norm: float
    residual_corrected_norm: float | None
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "M_c": self.M_c,
            "gate0_ok": self.gate0.ok,
            "gate0": self.gate0.to_dict(),
            "residual0_norm": self.residual0_norm,
            "residual_mc_norm": self.residual_mc_norm,
            "residual_corrected_norm": self.residual_corrected_norm,
            "success": self.success,
            "message": self.message,
        }


def pack_fairy_state(sys: System, seed: OrbitSeed) -> np.ndarray:
    r, v = _fairy_slice(sys, seed)
    return np.concatenate([r.ravel(), v.ravel()])


def unpack_into_system(
    y: np.ndarray,
    seed: OrbitSeed,
    M_c: float,
    period: float,
) -> System:
    n = seed.n_bodies
    r = y[: 3 * n].reshape(n, 3)
    v = y[3 * n : 6 * n].reshape(n, 3)
    tmp = OrbitSeed(
        id=seed.id,
        family=seed.family,
        n_bodies=n,
        G=seed.G,
        masses=seed.masses,
        period=period,
        positions=r,
        velocities=v,
        names=seed.names,
        symmetry=seed.symmetry,
        source=seed.source,
        notes=seed.notes,
        central_index=None,
    )
    return attach_central_mass(tmp, M_c)


def mass_continuation_smoke(
    seed: OrbitSeed | None = None,
    *,
    M_c: float = 1e-4,
    shift: int = 1,
    atol_rel: float = 1e-6,
    correct: bool = True,
    max_nfev: int = 8,
) -> ContinuationSmokeResult:
    """
    Path A smoke: gate at M_c=0, evaluate residual at small M_c, optional LS corrector.
    """
    if seed is None:
        seed = load_seed(SEEDS_DIR / "free_4_square_re.json")

    sys0 = attach_central_mass(seed, 0.0)
    gate0 = verify_choreography_Tn(
        sys0, float(seed.period), shift=shift, atol_rel=atol_rel, n_outputs=32
    )
    if not gate0.ok:
        return ContinuationSmokeResult(
            M_c=M_c,
            gate0=gate0,
            residual0_norm=float("inf"),
            residual_mc_norm=float("inf"),
            residual_corrected_norm=None,
            success=False,
            message="§3.2 gate failed at M_c=0",
        )

    r0 = symmetry_residual_vector(sys0, seed, seed.period, shift=shift)
    n0 = float(np.linalg.norm(r0))

    sys_m = attach_central_mass(seed, M_c)
    r_m = symmetry_residual_vector(sys_m, seed, seed.period, shift=shift)
    n_m = float(np.linalg.norm(r_m))

    n_c: float | None = None
    msg = "no corrector"
    ok = gate0.ok

    if correct:
        y0 = pack_fairy_state(sys_m, seed)

        def fun(y: np.ndarray) -> np.ndarray:
            sys = unpack_into_system(y, seed, M_c, float(seed.period))
            return symmetry_residual_vector(sys, seed, seed.period, shift=shift)

        # TODO(PROMPT §2.5): diagnostic baseline = zero-order + first-order central
        # perturbation, not bare residual0 alone.
        try:
            sol = least_squares(
                fun, y0, method="lm", max_nfev=int(max_nfev), ftol=1e-8, xtol=1e-8
            )
            n_c = float(np.linalg.norm(sol.fun))
            ok = gate0.ok and bool(sol.success)
            msg = f"lm nfev={sol.nfev} cost={sol.cost:.3e}"
        except Exception as exc:  # pragma: no cover
            n_c = None
            ok = False
            msg = f"corrector failed: {exc}"

    return ContinuationSmokeResult(
        M_c=M_c,
        gate0=gate0,
        residual0_norm=n0,
        residual_mc_norm=n_m,
        residual_corrected_norm=n_c,
        success=bool(ok),
        message=msg,
    )


def correct_at_mass(
    seed: OrbitSeed,
    M_c: float,
    *,
    shift: int = 1,
    max_nfev: int = 10,
    period: float | None = None,
    optics_soft: bool = True,
    log_rho: float = 0.0,
) -> tuple[OrbitSeed, float, bool]:
    """Newton-ish corrector at fixed M_c; returns updated fairy seed + ||F||."""
    period = float(period if period is not None else seed.period)
    sys_m = attach_central_mass(seed, M_c)
    y0 = pack_fairy_state(sys_m, seed)

    def fun(y: np.ndarray) -> np.ndarray:
        sys = unpack_into_system(y, seed, M_c, period)
        return symmetry_residual_vector(
            sys,
            seed,
            period,
            shift=shift,
            optics_soft=optics_soft,
            log_rho=log_rho,
        )

    sol = least_squares(fun, y0, method="lm", max_nfev=max_nfev, ftol=1e-8, xtol=1e-8)
    sys_c = unpack_into_system(sol.x, seed, M_c, period)
    r, v = _fairy_slice(sys_c, seed)
    out = OrbitSeed(
        id=seed.id,
        family=seed.family,
        n_bodies=seed.n_bodies,
        G=seed.G,
        masses=seed.masses,
        period=period,
        positions=r,
        velocities=v,
        names=seed.names,
        symmetry=seed.symmetry,
        source=seed.source,
        notes=f"corrected M_c={M_c}",
        central_index=None,
    )
    return out, float(np.linalg.norm(sol.fun)), bool(sol.success)


def run_path_a_continuation(
    seed: OrbitSeed,
    *,
    wall_hours: float | None = None,
    M_c_max: float = 1.0,
    dM0: float = 1e-3,
    shift: int = 1,
    max_nfev: int = 10,
    res_tol: float = 1e-4,
    out_dir: Path | None = None,
    optics_soft: bool = True,
    log_rho: float = 0.0,
) -> dict[str, Any]:
    """
    Path A: raise M_c from 0 with adaptive step; each step LS-correct.
    On failure: halve dM (fold-lite); stop at wall (if set) or M_c_max.
    ``wall_hours=None`` / ``<=0`` means no wall clock limit.
    """
    import json
    import time

    out_dir = Path(out_dir or "experiments/output/continuation_n4")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "steps.jsonl"
    t_end = (
        None
        if wall_hours is None or float(wall_hours) <= 0
        else time.time() + float(wall_hours) * 3600.0
    )

    gate = verify_choreography_Tn(
        attach_central_mass(seed, 0.0), float(seed.period), shift=shift, atol_rel=1e-5
    )
    if not gate.ok:
        summary = {"ok": False, "reason": "gate_failed_Mc0", "gate": gate.to_dict()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    M_c = 0.0
    dM = float(dM0)
    current = seed
    steps = 0
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(
            json.dumps(
                {
                    "M_c": 0.0,
                    "residual": 0.0,
                    "ok": True,
                    "note": "start",
                    "optics_soft": optics_soft,
                    "log_rho": log_rho,
                }
            )
            + "\n"
        )
        while (t_end is None or time.time() < t_end) and M_c < M_c_max - 1e-15:
            target = min(M_c + dM, M_c_max)
            try:
                nxt, res_n, ok = correct_at_mass(
                    current,
                    target,
                    shift=shift,
                    max_nfev=max_nfev,
                    optics_soft=optics_soft,
                    log_rho=log_rho,
                )
            except Exception as exc:
                row = {"M_c": target, "error": str(exc), "dM": dM}
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                dM *= 0.5
                if dM < 1e-8:
                    break
                continue
            row = {
                "M_c": target,
                "residual": res_n,
                "ok": ok and res_n < res_tol,
                "dM": dM,
                "t_left_s": None if t_end is None else max(0.0, t_end - time.time()),
            }
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            steps += 1
            if ok and res_n < res_tol:
                M_c = target
                current = nxt
                save_seed(current, out_dir / f"state_Mc_{M_c:.6e}.json")
                dM = min(dM * 1.25, 0.05)
            else:
                dM *= 0.5
                if dM < 1e-8:
                    break

    summary = {
        "path": "A",
        "n": seed.n_bodies,
        "M_c_final": M_c,
        "steps": steps,
        "wall_hours": wall_hours,
        "out_dir": str(out_dir),
        "optics_soft": optics_soft,
        "log_rho": log_rho,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_seed(current, out_dir / "final.json")
    return summary


def scale_peripheral_masses(seed: OrbitSeed, mu: float) -> OrbitSeed:
    """Keep body 0 mass=1; set others to mu (PROMPT 5-body mass scan role)."""
    masses = list(seed.masses)
    if seed.n_bodies < 2:
        raise ValueError("need n>=2")
    masses[0] = 1.0
    for i in range(1, seed.n_bodies):
        masses[i] = float(mu)
    return OrbitSeed(
        id=seed.id,
        family=seed.family,
        n_bodies=seed.n_bodies,
        G=seed.G,
        masses=tuple(masses),
        period=seed.period,
        positions=seed.positions,
        velocities=seed.velocities,
        names=seed.names,
        symmetry=seed.symmetry,
        source=seed.source,
        notes=f"mu={mu}",
        central_index=0,
    )


def run_path_b_mass_scan(
    seed: OrbitSeed,
    *,
    wall_hours: float | None = None,
    mu_min: float = 1e-3,
    n_log_steps: int = 40,
    shift: int = 1,
    max_nfev: int = 10,
    res_tol: float = 1e-4,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Path B-style: fix one mass=1, sweep other masses μ in logspace downward.
    Uses free-5 choreography seed (no separate central); body 0 is the 'center' role.
    ``wall_hours=None`` / ``<=0`` means no wall clock limit.
    """
    import json
    import time

    out_dir = Path(out_dir or "experiments/output/continuation_n5")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "steps.jsonl"
    t_end = (
        None
        if wall_hours is None or float(wall_hours) <= 0
        else time.time() + float(wall_hours) * 3600.0
    )

    # At μ=1 equal-mass: require §3.2 gate
    eq = scale_peripheral_masses(seed, 1.0)
    # For free equal-mass verify, clear central_index semantics in verify
    eq_free = OrbitSeed(
        id=eq.id,
        family=eq.family,
        n_bodies=eq.n_bodies,
        G=eq.G,
        masses=eq.masses,
        period=eq.period,
        positions=eq.positions,
        velocities=eq.velocities,
        names=eq.names,
        symmetry=eq.symmetry,
        source=eq.source,
        notes=eq.notes,
        central_index=None,
    )
    gate = verify_choreography_Tn(
        eq_free.to_system(), float(eq_free.period), shift=shift, atol_rel=1e-5
    )
    if not gate.ok:
        summary = {"ok": False, "reason": "gate_failed_mu1", "gate": gate.to_dict()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    mus = np.logspace(0.0, np.log10(mu_min), int(n_log_steps))
    current = eq_free
    done = 0
    with log_path.open("a", encoding="utf-8") as logf:
        for mu in mus:
            if t_end is not None and time.time() >= t_end:
                break
            # Residual with unequal masses: still use same geometric §3.2 on all bodies
            # via M_c=0 attach (no extra central) + correct_at_mass with M_c=0 but
            # scaled masses in seed.
            scaled = OrbitSeed(
                id=current.id,
                family=current.family,
                n_bodies=current.n_bodies,
                G=current.G,
                masses=tuple(
                    1.0 if i == 0 else float(mu) for i in range(current.n_bodies)
                ),
                period=current.period,
                positions=current.positions,
                velocities=current.velocities,
                names=current.names,
                symmetry=current.symmetry,
                source=current.source,
                notes=f"mu={mu}",
                central_index=None,
            )
            try:
                nxt, res_n, ok = correct_at_mass(
                    scaled, 0.0, shift=shift, max_nfev=max_nfev
                )
                # restore masses on corrected geometry
                nxt = OrbitSeed(
                    id=nxt.id,
                    family=nxt.family,
                    n_bodies=nxt.n_bodies,
                    G=nxt.G,
                    masses=scaled.masses,
                    period=nxt.period,
                    positions=nxt.positions,
                    velocities=nxt.velocities,
                    names=nxt.names,
                    symmetry=nxt.symmetry,
                    source=nxt.source,
                    notes=scaled.notes,
                    central_index=None,
                )
            except Exception as exc:
                logf.write(json.dumps({"mu": float(mu), "error": str(exc)}) + "\n")
                logf.flush()
                break
            row = {
                "mu": float(mu),
                "residual": res_n,
                "ok": ok and res_n < res_tol,
                "t_left_s": None if t_end is None else max(0.0, t_end - time.time()),
            }
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            done += 1
            if ok and res_n < res_tol:
                current = nxt
                save_seed(current, out_dir / f"state_mu_{mu:.6e}.json")
            else:
                # keep last good; shrink by staying but continue scan (diagnostic)
                pass

    summary = {
        "path": "B",
        "n": seed.n_bodies,
        "steps": done,
        "wall_hours": wall_hours,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_seed(current, out_dir / "final.json")
    return summary
