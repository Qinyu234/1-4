"""Orbit seed catalogue for equal-mass continuation (Path A / Path B).

Canonical seeds live next to this package (committed JSON). Generated
artifacts go under ignored ``orbit_library/``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fairy_orbit.core.body import Body, System, com_position, com_velocity, to_com_inertial_frame

SEEDS_DIR = Path(__file__).resolve().parent
CATALOGUE_PATH = SEEDS_DIR / "catalogue.json"


@dataclass(frozen=True)
class OrbitSeed:
    id: str
    family: str
    n_bodies: int
    G: float
    masses: tuple[float, ...]
    period: float
    positions: np.ndarray  # (n, 3)
    velocities: np.ndarray  # (n, 3)
    names: tuple[str, ...]
    symmetry: str
    source: str
    notes: str = ""
    central_index: int | None = None
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "family": self.family,
            "n_bodies": self.n_bodies,
            "G": self.G,
            "masses": list(self.masses),
            "period": self.period,
            "positions": np.asarray(self.positions, dtype=float).tolist(),
            "velocities": np.asarray(self.velocities, dtype=float).tolist(),
            "names": list(self.names),
            "symmetry": self.symmetry,
            "source": self.source,
            "notes": self.notes,
            "central_index": self.central_index,
        }
        if self.verification is not None:
            d["verification"] = self.verification
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrbitSeed:
        return cls(
            id=str(data["id"]),
            family=str(data["family"]),
            n_bodies=int(data["n_bodies"]),
            G=float(data["G"]),
            masses=tuple(float(m) for m in data["masses"]),
            period=float(data["period"]),
            positions=np.asarray(data["positions"], dtype=float).reshape(-1, 3),
            velocities=np.asarray(data["velocities"], dtype=float).reshape(-1, 3),
            names=tuple(str(n) for n in data["names"]),
            symmetry=str(data["symmetry"]),
            source=str(data["source"]),
            notes=str(data.get("notes", "")),
            central_index=(
                None if data.get("central_index") is None else int(data["central_index"])
            ),
            verification=data.get("verification"),
        )

    def to_system(self) -> System:
        bodies = [
            Body(
                mass=float(self.masses[i]),
                position=np.asarray(self.positions[i], dtype=float),
                velocity=np.asarray(self.velocities[i], dtype=float),
                name=self.names[i] if i < len(self.names) else f"b{i}",
            )
            for i in range(self.n_bodies)
        ]
        return System(bodies=bodies, G=self.G, labels=[b.name for b in bodies])


def polygon_force_factor(n: int) -> float:
    """
    S_n = Σ_{k=1}^{n-1} 1 / (4 sin(π k / n))

    For equal masses m on a regular n-gon of circumradius R,
    ω² = G m S_n / R³  (relative equilibrium).
    """
    if n < 3:
        raise ValueError("n >= 3 required")
    s = 0.0
    for k in range(1, n):
        s += 1.0 / (4.0 * math.sin(math.pi * k / n))
    return float(s)


def regular_polygon_relative_equilibrium(
    n: int,
    *,
    mass: float = 1.0,
    radius: float = 1.0,
    G: float = 1.0,
    plane: str = "xy",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Equal-mass regular n-gon relative equilibrium in the inertial frame.

    Returns (positions (n,3), velocities (n,3), period T=2π/ω).
    """
    s_n = polygon_force_factor(n)
    omega = math.sqrt(G * mass * s_n / (radius**3))
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)
    for i in range(n):
        th = 2.0 * math.pi * i / n
        c, s = math.cos(th), math.sin(th)
        if plane == "xy":
            pos[i] = (radius * c, radius * s, 0.0)
            vel[i] = (-omega * radius * s, omega * radius * c, 0.0)
        else:
            raise ValueError("only plane='xy' supported")
    period = 2.0 * math.pi / omega
    return pos, vel, period


def build_free_polygon_seed(
    n: int,
    *,
    seed_id: str,
    family: str,
    mass: float = 1.0,
    radius: float = 1.0,
    G: float = 1.0,
) -> OrbitSeed:
    pos, vel, period = regular_polygon_relative_equilibrium(
        n, mass=mass, radius=radius, G=G
    )
    names = tuple(f"B{i+1}" for i in range(n))
    # COM should already be ~0; enforce exactly.
    sys = System(
        bodies=[
            Body(mass=mass, position=pos[i], velocity=vel[i], name=names[i])
            for i in range(n)
        ],
        G=G,
        labels=list(names),
    )
    to_com_inertial_frame(sys)
    pos2 = np.stack([b.position for b in sys.bodies])
    vel2 = np.stack([b.velocity for b in sys.bodies])
    return OrbitSeed(
        id=seed_id,
        family=family,
        n_bodies=n,
        G=G,
        masses=tuple(mass for _ in range(n)),
        period=period,
        positions=pos2,
        velocities=vel2,
        names=names,
        symmetry=f"regular_{n}_gon_relative_equilibrium",
        source="analytic: ω² = G m S_n / R³, S_n=Σ 1/(4 sin(πk/n))",
        notes="Path A free equal-mass seed (inertial period = one full rotation).",
        central_index=None,
    )


def build_hier_1plus4_manifold_seed(
    *,
    mu: float = 1e-3,
    e0: float = 0.05,
    a1: float = 0.15,
    M1: float = 0.5,
) -> OrbitSeed:
    from fairy_orbit.design.manifold import ManifoldParams, build_manifold_system

    params = ManifoldParams(a1=a1, e0=e0, M1=M1, mu_mass=mu)
    sys = build_manifold_system(params, com_frame=True)
    # Period of a0=1 circular Kepler around m_c=1: T=2π
    period = 2.0 * math.pi
    names = tuple(b.name for b in sys.bodies)
    masses = tuple(float(b.mass) for b in sys.bodies)
    pos = np.stack([b.position for b in sys.bodies])
    vel = np.stack([b.velocity for b in sys.bodies])
    return OrbitSeed(
        id="hier_1plus4_manifold",
        family="hier_1plus4",
        n_bodies=len(sys.bodies),
        G=float(sys.G),
        masses=masses,
        period=period,
        positions=pos,
        velocities=vel,
        names=names,
        symmetry="Td_rays_linear_poly",
        source="fairy_orbit.design.manifold.ManifoldParams default Path-B seed",
        notes="Not expected to be periodic; baseline hierarchical IC for μ-continuation.",
        central_index=0,
    )


def save_seed(seed: OrbitSeed, path: Path | None = None) -> Path:
    path = path or (SEEDS_DIR / f"{seed.id}.json")
    path.write_text(json.dumps(seed.to_dict(), indent=2), encoding="utf-8")
    return path


def load_seed(path: Path | str) -> OrbitSeed:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return OrbitSeed.from_dict(data)


def load_catalogue(path: Path | None = None) -> dict[str, Any]:
    path = path or CATALOGUE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def write_catalogue(entries: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or CATALOGUE_PATH
    payload = {
        "version": 1,
        "description": "Equal-mass continuation seed index (Path A free_*; Path B hier_*).",
        "seeds": entries,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def update_catalogue_verification(
    catalogue: dict[str, Any],
    seed_id: str,
    *,
    gate: Any,
    orbit_class: str | None = None,
) -> None:
    """
    Patch catalogue entry in-place with PROMPT §3.2 verification fields.

    ``gate`` is a ChoreographyVerifyResult (duck-typed).
    """
    for entry in catalogue.get("seeds", []):
        if entry.get("id") != seed_id:
            continue
        entry["perm"] = list(gate.perm)
        entry["perm_label"] = gate.perm_label
        entry["R_axis"] = list(gate.axis)
        entry["R_angle"] = float(gate.angle)
        entry["E_r_rel"] = float(gate.E_r_rel)
        entry["E_v_rel"] = float(gate.E_v_rel)
        entry["tau"] = float(gate.tau)
        entry["choreography_ok"] = bool(gate.ok)
        if gate.ok:
            entry["verified_claim"] = "prompt_3_2_Tn_rv"
            if orbit_class:
                entry["orbit_class"] = orbit_class
        else:
            entry["verified_claim"] = "failed_prompt_3_2"
            entry["orbit_class"] = "unverified"
        return
    raise KeyError(f"seed id {seed_id!r} not in catalogue")


def regenerate_canonical_seeds() -> list[OrbitSeed]:
    """Rebuild committed free_4 / free_5 / hier_1plus4 JSON + catalogue."""
    seeds = [
        build_free_polygon_seed(4, seed_id="free_4_square_re", family="free_4"),
        build_free_polygon_seed(5, seed_id="free_5_pentagon_re", family="free_5"),
        build_hier_1plus4_manifold_seed(),
    ]
    entries = []
    for s in seeds:
        save_seed(s)
        if s.family.startswith("free_"):
            orbit_class = "free_relative_equilibrium"
            verified_claim = "pending_prompt_3_2"
        elif s.family.startswith("hier_"):
            orbit_class = "hier_baseline_ic"
            verified_claim = "none_periodic_not_claimed"
        else:
            orbit_class = "unclassified"
            verified_claim = "unknown"
        entries.append(
            {
                "id": s.id,
                "family": s.family,
                "path": f"{s.id}.json",
                "n_bodies": s.n_bodies,
                "period": s.period,
                "symmetry": s.symmetry,
                "path_hint": "A" if s.family.startswith("free_") else "B",
                "orbit_class": orbit_class,
                "verified_claim": verified_claim,
            }
        )
    write_catalogue(entries)
    return seeds


def assert_com_frame(seed: OrbitSeed, *, atol: float = 1e-10) -> None:
    sys = seed.to_system()
    assert np.linalg.norm(com_position(sys)) < atol
    assert np.linalg.norm(com_velocity(sys)) < atol


def pairwise_distance_matrix(pos: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise distances as a flat vector (shape fingerprint)."""
    p = np.asarray(pos, dtype=float).reshape(-1, 3)
    n = p.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.linalg.norm(p[i] - p[j])))
    return np.asarray(out, dtype=float)


def shape_congruence_residual(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """
    Relative pairwise-distance mismatch (no absolute R).

    Zero iff the two labeled point clouds are congruent by some R∈SO(3)
    (and optional translation already removed in COM frame).
    """
    da = pairwise_distance_matrix(pos_a)
    db = pairwise_distance_matrix(pos_b)
    denom = max(float(np.linalg.norm(da)), 1e-300)
    return float(np.linalg.norm(da - db) / denom)


def role_shifted_cloud(cloud: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Cyclic role map on body index: slot i ← body (i+shift) mod n.

    For x_A(t)=R x_B(t+τ) (and the same for v), compare
    shape(cloud_t) to shape(role_shifted_cloud(cloud_{t+τ}, shift=1)).
    """
    p = np.asarray(cloud, dtype=float).reshape(-1, 3)
    n = p.shape[0]
    shift %= n
    idx = (np.arange(n) + shift) % n
    return p[idx]


# Back-compat name
def role_shifted_positions(pos: np.ndarray, shift: int = 1) -> np.ndarray:
    return role_shifted_cloud(pos, shift)


def verify_free_shape_congruence(
    seed: OrbitSeed,
    *,
    traj=None,
    shift: int = 1,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """
    Free equal-mass gate (no central body ⇒ no absolute R to write down).

    Conceptual relation: x_A(t) = R x_B(t+τ), v_A(t) = R v_B(t+τ), …
    Without a center there is no preferred frame for R, so verify the
    equivalent shape congruence on both positions and velocities:

        Shape(x_•(t)) ≅ Shape(x_{•+shift}(t+τ))
        Shape(v_•(t)) ≅ Shape(v_{•+shift}(t+τ))

    via pairwise ||·_i − ·_j|| residuals (R never appears as ground truth).
    Kabsch R* fitted from positions is applied to velocities only as a
    diagnostic cross-check.

    Default τ = T/n. If ``traj`` is None, only the instantaneous cyclic
    shape checks at the seed IC are reported.
    """
    from fairy_orbit.observe.closure import kabsch_rotation

    n = seed.n_bodies
    if seed.central_index is not None:
        return {"ok": False, "reason": "free shape check requires no central body"}

    pos0 = np.asarray(seed.positions, dtype=float)
    vel0 = np.asarray(seed.velocities, dtype=float)
    inst_r = shape_congruence_residual(pos0, role_shifted_cloud(pos0, shift))
    inst_v = shape_congruence_residual(vel0, role_shifted_cloud(vel0, shift))
    out: dict[str, Any] = {
        "formula": (
            "Shape(x_•(t))≅Shape(x_{•+k}(t+τ)), "
            "Shape(v_•(t))≅Shape(v_{•+k}(t+τ))  [no absolute R]"
        ),
        "shift": shift,
        "instant_shape_residual_r": inst_r,
        "instant_shape_residual_v": inst_v,
        "instant_shape_residual": inst_r,  # back-compat
        "tau": float(seed.period) / n,
        "period": float(seed.period),
    }

    if traj is None:
        out["ok"] = inst_r < atol and inst_v < atol
        out["note"] = "IC-only cyclic r/v shape; pass traj for time-lag τ check"
        return out

    times = np.asarray(traj.times, dtype=float)
    positions = np.asarray(traj.positions, dtype=float)
    velocities = np.asarray(traj.velocities, dtype=float)
    tau = float(seed.period) / n
    dt_tol = 0.55 * (times[1] - times[0] if len(times) > 1 else tau)
    res_r: list[float] = []
    res_v: list[float] = []
    kabsch_r: list[float] = []
    kabsch_v: list[float] = []
    for k, t in enumerate(times):
        t2 = t + tau
        if t2 > times[-1] + 1e-14:
            break
        j = int(np.argmin(np.abs(times - t2)))
        if abs(times[j] - t2) > dt_tol:
            continue
        pa = positions[k]
        pb = role_shifted_cloud(positions[j], shift)
        va = velocities[k]
        vb = role_shifted_cloud(velocities[j], shift)
        res_r.append(shape_congruence_residual(pa, pb))
        res_v.append(shape_congruence_residual(va, vb))
        R_fit = kabsch_rotation(pa, pb)
        diff_r = pa - (pb @ R_fit.T)
        diff_v = va - (vb @ R_fit.T)
        scale_r = max(float(np.linalg.norm(pa)), 1e-300)
        scale_v = max(float(np.linalg.norm(va)), 1e-300)
        kabsch_r.append(float(np.linalg.norm(diff_r) / scale_r))
        kabsch_v.append(float(np.linalg.norm(diff_v) / scale_v))

    max_r = max(res_r) if res_r else float("inf")
    max_v = max(res_v) if res_v else float("inf")
    out["n_samples"] = len(res_r)
    out["shape_residual_max_r"] = max_r
    out["shape_residual_max_v"] = max_v
    out["shape_residual_max"] = max_r  # back-compat
    out["shape_residual_mean_r"] = float(np.mean(res_r)) if res_r else float("inf")
    out["shape_residual_mean_v"] = float(np.mean(res_v)) if res_v else float("inf")
    out["kabsch_rel_max_r"] = max(kabsch_r) if kabsch_r else float("inf")
    out["kabsch_rel_max_v"] = max(kabsch_v) if kabsch_v else float("inf")
    out["kabsch_rel_max"] = out["kabsch_rel_max_r"]
    out["ok"] = bool(
        inst_r < atol
        and inst_v < atol
        and max_r < atol
        and max_v < atol
        and len(res_r) > 0
    )
    return out


def verify_seed_model(
    seed: OrbitSeed,
    *,
    traj=None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """
    Mandatory model check before using a catalogue seed.

    free_* (no center): r and v shape congruence under role lag τ — never an absolute R.
    hier_*: not a periodic claim; only notes baseline IC.
    """
    if seed.family.startswith("free_") or (
        seed.central_index is None
        and "relative_equilibrium" in seed.symmetry
    ):
        return verify_free_shape_congruence(seed, traj=traj, atol=atol)
    if seed.family.startswith("hier_"):
        return {
            "ok": True,
            "formula": "hier baseline (no free-shape claim)",
            "note": "Path-B IC; periodicity not required at store stage.",
        }
    return {"ok": False, "reason": f"no verifier for family={seed.family}"}


# Back-compat alias used by earlier verify script / tests.
def verify_seed_formulas(seed: OrbitSeed, *, rtol: float = 1e-9, traj=None) -> dict[str, Any]:
    return verify_seed_model(seed, traj=traj, atol=rtol)
