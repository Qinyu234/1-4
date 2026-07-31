"""Equal-mass choreography search (PROMPT construct path).

Multi-start polish of free-N IC for §3.2 residual. Orbits that *maintain*
a regular equal n-gon (rigid RE) are rejected; momentary polygonal shape is OK.

Trials persist in SQLite (resume + start/result fingerprint dedupe).
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

from fairy_orbit.core.body import Body, System, to_com_inertial_frame
from fairy_orbit.design.seeds import OrbitSeed, load_seed, save_seed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate_endpoint
from fairy_orbit.observe.choreography_verify import (
    accept_free_choreography,
    cyclic_role_perm,
    is_regular_equal_ngon,
)
from fairy_orbit.observe.closure import closure_for_perm
from fairy_orbit.observe.shape_families import shape_distance, shape_feature_vector
from fairy_orbit.store.search_db import (
    DEFAULT_SEARCH_DB_NAME,
    ChoreographySearchStore,
    seed_fingerprint,
    trial_rng,
)


def _pack(seed: OrbitSeed) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(seed.positions, dtype=float).ravel(),
            np.asarray(seed.velocities, dtype=float).ravel(),
        ]
    )


def _unpack(y: np.ndarray, template: OrbitSeed) -> OrbitSeed:
    n = template.n_bodies
    r = y[: 3 * n].reshape(n, 3)
    v = y[3 * n : 6 * n].reshape(n, 3)
    period = float(template.period)
    sys = System(
        bodies=[
            Body(
                mass=float(template.masses[i]),
                position=r[i].copy(),
                velocity=v[i].copy(),
                name=template.names[i] if i < len(template.names) else f"B{i}",
            )
            for i in range(n)
        ],
        G=float(template.G),
    )
    to_com_inertial_frame(sys)
    r2 = np.stack([b.position for b in sys.bodies])
    v2 = np.stack([b.velocity for b in sys.bodies])
    return OrbitSeed(
        id=template.id,
        family=template.family,
        n_bodies=n,
        G=template.G,
        masses=template.masses,
        period=period,
        positions=r2,
        velocities=v2,
        names=template.names,
        symmetry=template.symmetry,
        source="choreography_search_polish",
        notes="multi-start §3.2 polish",
        central_index=None,
    )


def _collision_penalty(pos: np.ndarray, floor: float = 1e-3) -> float:
    n = pos.shape[0]
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < floor:
                pen += (floor - d) ** 2 * 1e6
    return pen


def _maintained_ngon_soft_penalty(
    r0: np.ndarray,
    r_tau: np.ndarray,
    strength: float = 50.0,
) -> float:
    """Soft push away from rigid RE: regular at t=0 and still regular at T/n."""
    if is_regular_equal_ngon(r0, rtol=0.08) and is_regular_equal_ngon(
        r_tau, rtol=0.08
    ):
        return strength
    return 0.0


def random_asymmetric_seed(
    n: int,
    rng: np.random.Generator,
    *,
    G: float = 1.0,
    mass: float = 1.0,
    mode: str = "baseline",
) -> OrbitSeed:
    """Random planar IC (unequal radii/gaps preferred; polygonal snapshot OK).

    ``mode="away"`` uses wider radii/gap/speed/period ranges and small z so
    starts tend to leave the shape neighborhood of already-found families.
    """
    away = mode == "away"
    if away:
        gaps = rng.uniform(0.12, 2.4, size=n)
        radii = rng.uniform(0.2, 2.6, size=n)
        speed_lo, speed_hi = 0.25, 1.9
        vr_scale = 0.35
        z_scale = 0.12
        period = float(rng.uniform(3.0, 14.0))
        notes = "random planar start (away-family ranges)"
        source = "random_asymmetric_ic_away"
    else:
        gaps = rng.uniform(0.3, 1.7, size=n)
        radii = rng.uniform(0.4, 1.6, size=n)
        speed_lo, speed_hi = 0.4, 1.4
        vr_scale = 0.15
        z_scale = 0.0
        period = float(rng.uniform(4.0, 10.0))
        notes = "random planar start"
        source = "random_asymmetric_ic"

    gaps = gaps / gaps.sum() * 2.0 * math.pi
    angles = np.cumsum(gaps) - gaps[0]
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for i in range(n):
        c, s = math.cos(angles[i]), math.sin(angles[i])
        z = float(rng.normal(0.0, z_scale)) if z_scale > 0 else 0.0
        pos[i] = (radii[i] * c, radii[i] * s, z)
        speed = rng.uniform(speed_lo, speed_hi)
        vt = speed * (0.85 + 0.15 * rng.random())
        vr = rng.normal(0.0, vr_scale)
        vz = float(rng.normal(0.0, 0.5 * z_scale)) if z_scale > 0 else 0.0
        vel[i] = (-vt * s + vr * c, vt * c + vr * s, vz)

    names = tuple(f"B{i+1}" for i in range(n))
    family = f"free_{n}"
    sys = System(
        bodies=[
            Body(mass=mass, position=pos[i], velocity=vel[i], name=names[i])
            for i in range(n)
        ],
        G=G,
    )
    to_com_inertial_frame(sys)
    return OrbitSeed(
        id=f"rand_{n}",
        family=family,
        n_bodies=n,
        G=G,
        masses=tuple(mass for _ in range(n)),
        period=period,
        positions=np.stack([b.position for b in sys.bodies]),
        velocities=np.stack([b.velocity for b in sys.bodies]),
        names=names,
        symmetry="asymmetric_search",
        source=source,
        notes=notes,
        central_index=None,
    )


def accepted_shape_features(
    store: ChoreographySearchStore,
    n_bodies: int,
    *,
    max_residual: float | None = 1e-6,
    limit: int = 4000,
) -> list[np.ndarray]:
    """Shape features of currently accepted passes (for away-family sampling)."""
    feats: list[np.ndarray] = []
    for rec in store.list_passes(n_bodies, limit=limit, max_residual=max_residual):
        if rec.seed_json is None:
            continue
        try:
            feats.append(shape_feature_vector(OrbitSeed.from_dict(rec.seed_json)))
        except Exception:
            continue
    return feats


def min_distance_to_features(feat: np.ndarray, features: list[np.ndarray]) -> float:
    if not features:
        return float("inf")
    return min(shape_distance(feat, f) for f in features)


def sample_search_start(
    n: int,
    rng: np.random.Generator,
    family_feats: list[np.ndarray],
    *,
    away_prob: float,
    away_min_sep: float = 0.12,
    away_tries: int = 24,
) -> tuple[OrbitSeed, str, float]:
    """
    Draw a search IC; with probability ``away_prob`` prefer starts far from
    accepted shape families (rejection / best-effort).

    ``away_prob`` is normally driven by :class:`ResidualAnnealer` (not a fixed
    50/50 split).

    Returns ``(seed, start_mode, min_dist_to_families)``.
    """
    away_prob = float(np.clip(away_prob, 0.0, 1.0))
    want_away = bool(rng.random() < away_prob)
    if not want_away:
        seed = random_asymmetric_seed(n, rng, mode="baseline")
        d = min_distance_to_features(shape_feature_vector(seed), family_feats)
        return seed, "baseline", d

    if not family_feats:
        seed = random_asymmetric_seed(n, rng, mode="away")
        return seed, "away_no_families", float("inf")

    best: OrbitSeed | None = None
    best_d = -1.0
    for _ in range(max(1, int(away_tries))):
        cand = random_asymmetric_seed(n, rng, mode="away")
        d = min_distance_to_features(shape_feature_vector(cand), family_feats)
        if d >= float(away_min_sep):
            return cand, "away", d
        if d > best_d:
            best_d = d
            best = cand
    assert best is not None
    return best, "away_best_effort", best_d


@dataclass
class ResidualAnnealer:
    """Map residual convergence speed → probability of away-family starts.

    Early / still-improving search stays near baseline (mine current families).
    When the rolling best residual is already low *and* improvement stalls,
    raise ``away_prob`` so new starts leave known shape basins (anneal out).
    """

    window: int = 40
    warmup: int = 24
    low_residual: float = 1e-3
    very_low_residual: float = 1e-6
    stall_improve_factor: float = 1.25
    away_min: float = 0.05
    away_max: float = 0.9
    _residuals: deque[float] = field(default_factory=deque, repr=False)
    _best_curve: deque[float] = field(default_factory=deque, repr=False)
    best: float = field(default=float("inf"))
    n_obs: int = 0

    def __post_init__(self) -> None:
        self._residuals = deque(maxlen=int(self.window))
        self._best_curve = deque(maxlen=int(self.window))

    def seed_from_residuals(self, residuals: list[float]) -> None:
        for r in residuals:
            self.observe(r)

    def observe(self, residual: float | None) -> None:
        if residual is None or not np.isfinite(residual):
            return
        r = float(residual)
        self.n_obs += 1
        self._residuals.append(r)
        if r < self.best:
            self.best = r
        self._best_curve.append(self.best)

    def away_prob(self) -> float:
        if self.n_obs < int(self.warmup) or len(self._best_curve) < 4:
            return float(self.away_min)

        older = float(self._best_curve[0])
        newer = float(self._best_curve[-1])
        recent = list(self._residuals)
        med = float(np.median(recent)) if recent else newer

        # improvement factor over the window (>1 means best got better)
        improve = older / max(newer, 1e-300)
        stalled = improve < float(self.stall_improve_factor)
        rapidly_improving = improve >= 2.0

        score = 0.0
        if newer <= float(self.very_low_residual):
            score += 0.55
        elif newer <= float(self.low_residual):
            score += 0.35
        if stalled and newer <= float(self.low_residual):
            score += 0.35
        if med <= float(self.low_residual):
            score += 0.15
        if rapidly_improving and newer > float(self.very_low_residual):
            # still mining a productive basin — keep away low
            score *= 0.35

        return float(
            np.clip(
                self.away_min + score * (self.away_max - self.away_min),
                self.away_min,
                self.away_max,
            )
        )

    def status(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "best": None if not np.isfinite(self.best) else float(self.best),
            "away_prob": self.away_prob(),
            "window": int(self.window),
            "warmup": int(self.warmup),
        }


def symmetry_residual_seed(
    seed: OrbitSeed,
    *,
    shift: int = 1,
    n_outputs: int = 12,
) -> np.ndarray:
    """§3.2 residual at T/n. ``n_outputs`` kept for API compat; endpoint-only integrate."""
    del n_outputs  # endpoint path does not sample intermediates
    n = seed.n_bodies
    tau = float(seed.period) / n
    perm = cyclic_role_perm(n, shift=shift)
    r0 = np.asarray(seed.positions, dtype=float)
    v0 = np.asarray(seed.velocities, dtype=float)
    sys = seed.to_system()
    r, v = integrate_endpoint(
        sys,
        t_end=tau,
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(tau / 200.0, 1e-3),
            min_dt=1e-5,
        ),
    )
    cl = closure_for_perm(r, v, r0, v0, perm)
    R = cl.R
    chunks = []
    for i, j in enumerate(perm):
        chunks.append(r[i] - R @ r0[j])
        chunks.append(v[i] - R @ v0[j])
    extras = [
        _collision_penalty(r0),
        _maintained_ngon_soft_penalty(r0, r),
    ]
    return np.concatenate([np.concatenate(chunks).astype(float), np.asarray(extras)])


def polish_seed(
    seed: OrbitSeed,
    *,
    shift: int = 1,
    max_nfev: int = 12,
) -> tuple[OrbitSeed, float]:
    y0 = _pack(seed)

    def fun(y: np.ndarray) -> np.ndarray:
        return symmetry_residual_seed(_unpack(y, seed), shift=shift)

    sol = least_squares(
        fun, y0, method="trf", max_nfev=max_nfev, ftol=1e-10, xtol=1e-10
    )
    polished = _unpack(sol.x, seed)
    return polished, float(np.linalg.norm(sol.fun))


@dataclass
class SearchTrial:
    trial: int
    residual: float
    period: float
    ok_gate: bool
    path: str | None
    reason: str


def _import_existing_passes(
    store: ChoreographySearchStore,
    out_dir: Path,
    n: int,
    *,
    force: bool = False,
) -> int:
    """
    One-shot migration: pull pass_*.json / best.json into SQLite.

    Skipped when the DB already has trials (SQLite is the source of truth).
    Pass ``force=True`` to scan JSON even then (idempotent via fingerprints).
    """
    if not force and (store.count_passed(n) > 0 or store.count_trials(n) > 0):
        return 0
    imported = 0
    for path in sorted(out_dir.glob(f"pass_{n}_*.json")):
        try:
            seed = load_seed(path)
        except Exception:
            continue
        residual = None
        notes = seed.notes or ""
        if "residual=" in notes:
            try:
                residual = float(notes.split("residual=")[-1].split()[0])
            except ValueError:
                residual = None
        if store.import_seed_pass(seed, residual=residual, reason="imported_pass_json"):
            imported += 1
    best = out_dir / "best.json"
    if best.exists():
        try:
            seed = load_seed(best)
            if store.import_seed_pass(seed, residual=None, reason="imported_best_json"):
                imported += 1
        except Exception:
            pass
    return imported


def archive_pass_json_files(out_dir: Path, n: int) -> int:
    """
    Move legacy ``pass_*.json`` into ``pass_json_archive/`` (SQLite keeps seeds).

    Returns number of files moved. ``best.json`` is left in place.
    """
    out_dir = Path(out_dir)
    archive = out_dir / "pass_json_archive"
    moved = 0
    paths = list(out_dir.glob(f"pass_{n}_*.json"))
    if not paths:
        return 0
    archive.mkdir(parents=True, exist_ok=True)
    for path in paths:
        dest = archive / path.name
        if dest.exists():
            dest = archive / f"{path.stem}_{int(time.time())}{path.suffix}"
        path.rename(dest)
        moved += 1
    return moved


def _write_summary(
    store: ChoreographySearchStore,
    n: int,
    out_dir: Path,
    *,
    wall_hours: float | None,
    status: str,
    skipped_dupes: int,
) -> dict[str, Any]:
    summary = store.summary_dict(n, out_dir=str(out_dir))
    summary["wall_hours"] = wall_hours
    summary["status"] = status
    summary["skipped_dupes"] = skipped_dupes
    best_path = out_dir / "best.json"
    summary["best_path"] = str(best_path) if best_path.exists() else None
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_choreography_search(
    n: int,
    *,
    wall_hours: float | None = None,
    shift: int = 1,
    out_dir: Path | None = None,
    db_path: Path | None = None,
    max_nfev: int = 14,
    fresh: bool = False,
    atol_rel: float = 1e-8,
    max_residual: float = 1e-6,
    write_pass_json: bool = False,
    import_json: bool = False,
    archive_json: bool = True,
    away_family_frac: float | None = None,
    away_min_sep: float = 0.12,
    away_tries: int = 24,
    anneal_window: int = 40,
    anneal_warmup: int = 24,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Long-running multi-start §3.2 polish for free equal-mass N-body.

    Accept only if §3.2 ``atol_rel`` holds, polish residual ``<= max_residual``,
    and the orbit does not maintain a regular n-gon. Resumes from SQLite.

    Start sampling anneals by residual convergence: while the rolling best is
    still improving, prefer baseline ICs (mine families); when residuals are
    already low and improvement stalls, raise the probability of away-family
    starts. Pass ``away_family_frac`` to override with a fixed probability.

    Accepted seeds are stored in SQLite (``seed_json``). Per-pass ``pass_*.json``
    files are off by default; only ``best.json`` is refreshed on disk.
    """
    out_dir = Path(out_dir or f"experiments/output/choreography_search_n{n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path or (out_dir / DEFAULT_SEARCH_DB_NAME))
    log_path = out_dir / "trials.jsonl"
    atol_rel = float(atol_rel)
    max_residual = float(max_residual)
    away_min_sep = float(away_min_sep)
    away_tries = int(away_tries)
    fixed_away = None if away_family_frac is None else float(away_family_frac)

    t_end = (
        None
        if wall_hours is None or wall_hours <= 0
        else time.time() + float(wall_hours) * 3600.0
    )

    with ChoreographySearchStore(db_path) as store:
        if fresh:
            store.clear(n)

        demoted = store.refilter_by_residual(n, max_residual=max_residual)
        imported = _import_existing_passes(store, out_dir, n, force=import_json)
        demoted += store.refilter_by_residual(n, max_residual=max_residual)
        archived = archive_pass_json_files(out_dir, n) if archive_json else 0
        trial_no = store.next_trial_no(n)
        best_rec = store.best_accepted(n)
        best_res = float("inf") if best_rec is None or best_rec.residual is None else float(
            best_rec.residual
        )
        if best_rec is not None and best_rec.seed_json is not None:
            save_seed(OrbitSeed.from_dict(best_rec.seed_json), out_dir / "best.json")

        family_feats = accepted_shape_features(
            store, n, max_residual=max_residual
        )
        annealer = ResidualAnnealer(
            window=int(anneal_window),
            warmup=int(anneal_warmup),
        )
        annealer.seed_from_residuals(
            store.list_recent_residuals(n, limit=int(anneal_window))
        )
        skipped_dupes = 0
        n_away = 0
        n_baseline = 0
        print(
            f"[search n={n}] db={db_path} resume_trial={trial_no} "
            f"stored={store.count_trials(n)} passed={store.count_passed(n)} "
            f"imported={imported} demoted={demoted} archived_json={archived} "
            f"atol_rel={atol_rel:g} max_residual={max_residual:g} "
            f"write_pass_json={write_pass_json} "
            f"away_mode={'fixed:'+str(fixed_away) if fixed_away is not None else 'anneal'} "
            f"away_prob={fixed_away if fixed_away is not None else annealer.away_prob():.3f} "
            f"away_min_sep={away_min_sep:g} family_feats={len(family_feats)}",
            flush=True,
        )

        with log_path.open("a", encoding="utf-8") as logf:
            while t_end is None or time.time() < t_end:
                rng = trial_rng(n, trial_no)
                away_prob = (
                    float(np.clip(fixed_away, 0.0, 1.0))
                    if fixed_away is not None
                    else annealer.away_prob()
                )
                start, start_mode, start_family_dist = sample_search_start(
                    n,
                    rng,
                    family_feats,
                    away_prob=away_prob,
                    away_min_sep=away_min_sep,
                    away_tries=away_tries,
                )
                if start_mode.startswith("away"):
                    n_away += 1
                else:
                    n_baseline += 1
                start_fp = seed_fingerprint(start)

                if store.has_start_fp(n, start_fp):
                    skipped_dupes += 1
                    trial_no += 1
                    continue

                try:
                    polished, res_n = polish_seed(start, shift=shift, max_nfev=max_nfev)
                    acc = accept_free_choreography(
                        polished.to_system(),
                        polished.period,
                        shift=shift,
                        atol_rel=atol_rel,
                        n_outputs=16,
                        ngon_samples=8,
                    )
                    ok = bool(acc.ok)
                    reason = acc.reason
                    if ok and res_n > max_residual:
                        ok = False
                        reason = "failed_residual_too_large"
                    result_fp = seed_fingerprint(polished)
                    path = None
                    duplicate_result = ok and store.has_accepted_result_fp(n, result_fp)
                    save_seed_obj: OrbitSeed | None = None

                    if ok and duplicate_result:
                        skipped_dupes += 1
                        reason = "duplicate_accepted_result"
                        gate_pass = False
                    elif ok:
                        gate_pass = True
                        save_seed_obj = OrbitSeed(
                            id=f"search_n{n}_{trial_no:05d}",
                            family=polished.family,
                            n_bodies=polished.n_bodies,
                            G=polished.G,
                            masses=polished.masses,
                            period=polished.period,
                            positions=polished.positions,
                            velocities=polished.velocities,
                            names=polished.names,
                            symmetry="accepted_non_maintained_ngon",
                            source="choreography_search",
                            notes=f"trial={trial_no} residual={res_n:.3e}",
                            central_index=None,
                            verification=acc.to_dict(),
                        )
                        if write_pass_json:
                            path = str(out_dir / f"pass_{n}_{trial_no:05d}.json")
                            save_seed(save_seed_obj, Path(path))
                        if res_n < best_res:
                            best_res = res_n
                            save_seed(save_seed_obj, out_dir / "best.json")
                        try:
                            family_feats.append(shape_feature_vector(save_seed_obj))
                        except Exception:
                            pass
                    else:
                        gate_pass = False

                    annealer.observe(res_n)

                    row_id = store.insert_trial(
                        n_bodies=n,
                        trial_no=trial_no,
                        start_fp=start_fp,
                        result_fp=result_fp,
                        residual=res_n,
                        period=float(polished.period),
                        ok_gate=gate_pass,
                        reason=reason,
                        maintains_regular_ngon=bool(acc.maintains_regular_ngon),
                        seed=save_seed_obj,
                    )
                    if row_id is None:
                        skipped_dupes += 1
                        trial_no += 1
                        continue

                    row = {
                        "trial": trial_no,
                        "residual": res_n,
                        "period": polished.period,
                        "ok_gate": gate_pass,
                        "reason": reason,
                        "maintains_regular_ngon": acc.maintains_regular_ngon,
                        "path": path,
                        "start_fp": start_fp,
                        "result_fp": result_fp,
                        "duplicate_result": duplicate_result,
                        "atol_rel": atol_rel,
                        "max_residual": max_residual,
                        "start_mode": start_mode,
                        "away_prob": away_prob,
                        "start_family_dist": (
                            None
                            if not np.isfinite(start_family_dist)
                            else float(start_family_dist)
                        ),
                        "t_left_s": None if t_end is None else max(0.0, t_end - time.time()),
                    }
                    logf.write(json.dumps(row) + "\n")
                    logf.flush()
                    if on_progress:
                        on_progress(row)
                    summary = _write_summary(
                        store,
                        n,
                        out_dir,
                        wall_hours=wall_hours,
                        status="running",
                        skipped_dupes=skipped_dupes,
                    )
                    summary["atol_rel"] = atol_rel
                    summary["max_residual"] = max_residual
                    summary["away_family_frac"] = fixed_away
                    summary["away_prob"] = away_prob
                    summary["anneal"] = annealer.status()
                    summary["starts_away"] = n_away
                    summary["starts_baseline"] = n_baseline
                    summary["family_feats"] = len(family_feats)
                    (out_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
                except Exception as exc:  # pragma: no cover
                    store.insert_trial(
                        n_bodies=n,
                        trial_no=trial_no,
                        start_fp=start_fp,
                        result_fp=None,
                        residual=None,
                        period=None,
                        ok_gate=False,
                        reason="error",
                        maintains_regular_ngon=False,
                        error=str(exc),
                    )
                    row = {
                        "trial": trial_no,
                        "error": str(exc),
                        "start_fp": start_fp,
                        "start_mode": start_mode,
                        "start_family_dist": (
                            None
                            if not np.isfinite(start_family_dist)
                            else float(start_family_dist)
                        ),
                        "t_left_s": None if t_end is None else max(0.0, t_end - time.time()),
                    }
                    logf.write(json.dumps(row) + "\n")
                    logf.flush()

                trial_no += 1

        summary = _write_summary(
            store,
            n,
            out_dir,
            wall_hours=wall_hours,
            status="done",
            skipped_dupes=skipped_dupes,
        )
        summary["atol_rel"] = atol_rel
        summary["max_residual"] = max_residual
        summary["away_family_frac"] = fixed_away
        summary["away_min_sep"] = away_min_sep
        summary["away_prob"] = (
            float(np.clip(fixed_away, 0.0, 1.0))
            if fixed_away is not None
            else annealer.away_prob()
        )
        summary["anneal"] = annealer.status()
        summary["starts_away"] = n_away
        summary["starts_baseline"] = n_baseline
        summary["family_feats"] = len(family_feats)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
