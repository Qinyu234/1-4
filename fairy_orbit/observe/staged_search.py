"""Staged search for ABCD→BCDA reachability (then detailed σ-score).

Pipeline:
  Stage-A sigmas (required) →
  一阶: linear-poly coarse find points (soft choreography residual) →
  二阶: local grid slowly expands around seeds →
  三阶: stain flood densify + optional a2→e2→M2→v1 unlock (if needed) →
  细致 score: raw base E_* then σ-weighted score on gate survivors

Stages 1–2 optimize soft residual / gate pass — not full PEO score.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fairy_orbit.observe.bayes import UNLOCK_STAGES, soft_choreography_residual
from fairy_orbit.observe.peo import evaluate_peo
from fairy_orbit.observe.rep_error import (
    CHANNELS,
    RepSigmas,
    is_calibrated_sigmas,
    load_required_sigmas,
)
from fairy_orbit.observe.search import (
    FREE_NAMES,
    HARD_BOUNDS,
    LOSS_FAIL,
    QUAD_ORDER,
    V1_NAMES,
    FreeParams,
    SearchBounds,
    SeedAnchors,
    ic_valid,
    to_manifold,
)

SOFT_ESCAPE = 1.0e3


@dataclass(frozen=True)
class StagedConfig:
    log_m: tuple[float, float] = (-6.0, -2.0)
    e_range: tuple[float, float] = (0.0, 0.20)
    n_m: int = 5
    n_e: int = 4
    # Stage 1 free grid (linear only); keep small — product over listed axes.
    stage1_axes: tuple[str, ...] = ("a1", "e1", "M1")
    stage1_points: int = 3
    stage1_top_k: int = 8
    # Stage 2
    stage2_axes: tuple[str, ...] = FREE_NAMES
    stage2_points: int = 3
    stage2_expand_rounds: int = 3
    stage2_expand_grow: float = 0.35
    stage2_edge_frac: float = 0.08
    stage2_max_product: int = 200
    # Stage 3 stain
    stain_frac: float = 0.25
    stain_max_seeds: int = 12
    flood_chebyshev: int = 1
    stage3_points: int = 3
    unlock_high_order: bool = True
    # Integration
    n_periods: float = 2.0
    n_outputs_coarse: int = 80
    n_outputs_fine: int = 160
    perm_mode: str = "fixed_radial"
    free_bounds: SearchBounds = field(default_factory=SearchBounds)


@dataclass
class ReachSample:
    m: float
    e: float
    free: FreeParams
    status: str
    soft_choreo: float
    score: float | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    stage: str = ""
    stained: bool = False
    unlocked: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "e": self.e,
            "free": self.free.as_dict(),
            "status": self.status,
            "soft_choreo": self.soft_choreo,
            "score": self.score,
            "stage": self.stage,
            "stained": self.stained,
            "unlocked": list(self.unlocked),
            "summary": {
                k: self.summary.get(k)
                for k in (
                    "reason",
                    "choreography_shift_k",
                    "E_r_final",
                    "E_v_final",
                    "score",
                    "sigmas_source",
                )
                if k in self.summary
            },
        }


@dataclass
class StagedSearchResult:
    sigmas: RepSigmas
    config: StagedConfig
    stage1: list[ReachSample]
    stage2: list[ReachSample]
    stained: list[ReachSample]
    stage3: list[ReachSample]
    survivors: list[ReachSample]
    n_evals: int
    wall_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigmas_source": self.sigmas.source,
            "sigmas_n": self.sigmas.n_samples,
            "n_evals": self.n_evals,
            "wall_s": self.wall_s,
            "n_stage1": len(self.stage1),
            "n_stage2": len(self.stage2),
            "n_stained": len(self.stained),
            "n_stage3": len(self.stage3),
            "n_survivors": len(self.survivors),
            "survivors": [s.as_dict() for s in self.survivors],
            "best_soft": min((s.soft_choreo for s in self.stage1 + self.stage2), default=None),
            "best_score": min(
                (s.score for s in self.survivors if s.score is not None),
                default=None,
            ),
        }


EvalReachFn = Callable[[SeedAnchors, FreeParams], ReachSample]


def _grid_1d(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.5 * (lo + hi)], dtype=float)
    return np.linspace(lo, hi, int(n))


def _product_on_axes(
    bounds: SearchBounds,
    axes: tuple[str, ...],
    n: int,
    *,
    base: FreeParams | None = None,
) -> list[FreeParams]:
    """Cartesian product on selected axes; others from base or bounds center."""
    center = base or bounds.center()
    axis_vals = {k: _grid_1d(*getattr(bounds, k), n) for k in axes}
    out: list[FreeParams] = []

    def rec(i: int, cur: dict[str, float]) -> None:
        if i == len(axes):
            d = center.as_dict()
            d.update(cur)
            # Force locked high-order to 0 unless present in cur
            for k in QUAD_ORDER + V1_NAMES:
                if k not in cur:
                    d[k] = float(getattr(center, k, 0.0)) if base is not None else 0.0
            out.append(FreeParams(**{k: float(d[k]) for k in FreeParams.__dataclass_fields__}))
            return
        name = axes[i]
        for x in axis_vals[name]:
            cur[name] = float(x)
            rec(i + 1, cur)

    rec(0, {})
    return out


def evaluate_reachability(
    seed: SeedAnchors,
    free: FreeParams,
    *,
    t_end: float,
    n_outputs: int,
    sigmas: RepSigmas,
    perm_mode: str = "fixed_radial",
    detailed_score: bool = False,
    stage: str = "",
    unlocked: tuple[str, ...] = (),
) -> ReachSample:
    """
    Soft residual for hunt; detailed σ-score only when detailed_score=True
    and choreography gate passes (require calibrated σ).
    """
    if not ic_valid(seed, free):
        return ReachSample(
            m=seed.m,
            e=seed.e,
            free=free,
            status="bad_ic",
            soft_choreo=SOFT_ESCAPE,
            stage=stage,
            unlocked=unlocked,
        )
    params = to_manifold(seed, free)
    res = evaluate_peo(
        params,
        t_end=t_end,
        n_outputs=n_outputs,
        sigmas=sigmas,
        perm_mode=perm_mode,
        require_calibrated_sigmas=detailed_score,
    )
    status = res.status
    summary = dict(res.summary)
    if status in ("escape", "escape_ic", "bad_ic"):
        return ReachSample(
            m=seed.m,
            e=seed.e,
            free=free,
            status=status,
            soft_choreo=SOFT_ESCAPE,
            summary=summary,
            stage=stage,
            unlocked=unlocked,
        )
    if status == "choreography":
        soft = {"soft_choreo": 1.0}
        if res.traj is not None:
            soft = soft_choreography_residual(res.traj)
        return ReachSample(
            m=seed.m,
            e=seed.e,
            free=free,
            status=status,
            soft_choreo=float(soft["soft_choreo"]),
            summary={**summary, **soft},
            stage=stage,
            unlocked=unlocked,
        )
    if status != "success":
        return ReachSample(
            m=seed.m,
            e=seed.e,
            free=free,
            status=status,
            soft_choreo=SOFT_ESCAPE,
            summary=summary,
            stage=stage,
            unlocked=unlocked,
        )
    # Gate passed: soft=0; optionally attach detailed score (base E_* already in summary).
    score = None
    if detailed_score:
        score = float(summary.get("score", LOSS_FAIL))
        if not math.isfinite(score):
            score = LOSS_FAIL
    return ReachSample(
        m=seed.m,
        e=seed.e,
        free=free,
        status="success",
        soft_choreo=0.0,
        score=score,
        summary=summary,
        stage=stage,
        unlocked=unlocked,
    )


def stain_flood(
    samples: list[ReachSample],
    *,
    frac: float = 0.25,
    max_seeds: int = 12,
    flood_chebyshev: int = 1,
    m_axis: np.ndarray | None = None,
    e_axis: np.ndarray | None = None,
) -> list[ReachSample]:
    """
    Mark low-soft cells as stained; flood Chebyshev neighbors on (m,e) grid indices.

    If m/e axes are omitted, stain by soft rank only (no geometric flood).
    """
    if not samples:
        return []
    ranked = sorted(samples, key=lambda s: s.soft_choreo)
    n_seed = max(1, min(max_seeds, int(math.ceil(frac * len(ranked)))))
    seeds = ranked[:n_seed]

    if m_axis is None or e_axis is None or flood_chebyshev <= 0:
        out = []
        for s in seeds:
            s.stained = True
            out.append(s)
        return out

    def idx_m(m: float) -> int:
        return int(np.argmin(np.abs(m_axis - m)))

    def idx_e(e: float) -> int:
        return int(np.argmin(np.abs(e_axis - e)))

    stain_idx: set[tuple[int, int]] = set()
    for s in seeds:
        stain_idx.add((idx_m(s.m), idx_e(s.e)))
    # Flood
    grown = set(stain_idx)
    for im, ie in stain_idx:
        for dm in range(-flood_chebyshev, flood_chebyshev + 1):
            for de in range(-flood_chebyshev, flood_chebyshev + 1):
                if max(abs(dm), abs(de)) > flood_chebyshev:
                    continue
                jm, je = im + dm, ie + de
                if 0 <= jm < len(m_axis) and 0 <= je < len(e_axis):
                    grown.add((jm, je))

    out: list[ReachSample] = []
    seen: set[tuple] = set()
    for s in samples:
        key = (round(s.m, 12), round(s.e, 12), s.free.key())
        if (idx_m(s.m), idx_e(s.e)) in grown:
            if key in seen:
                continue
            seen.add(key)
            s.stained = True
            out.append(s)
    # Always include soft-ranked seeds even if free keys differ
    for s in seeds:
        key = (round(s.m, 12), round(s.e, 12), s.free.key())
        if key not in seen:
            s.stained = True
            out.append(s)
            seen.add(key)
    return out[: max(max_seeds * 4, len(out))]


def _local_bounds_around(
    free: FreeParams,
    bounds: SearchBounds,
    *,
    shrink: float = 0.5,
) -> SearchBounds:
    kwargs = {k: getattr(bounds, k) for k in FreeParams.__dataclass_fields__ if hasattr(bounds, k)}
    for name in FREE_NAMES:
        lo, hi = getattr(bounds, name)
        half = 0.5 * (hi - lo) * shrink
        c = getattr(free, name)
        kwargs[name] = (max(lo, c - half), min(hi, c + half))
    return SearchBounds(**{k: kwargs[k] for k in SearchBounds.__dataclass_fields__ if k in kwargs})


def _expand_if_edged(
    bounds: SearchBounds,
    best: FreeParams,
    *,
    grow: float,
    frac: float,
) -> tuple[SearchBounds, dict[str, tuple[float, float]]]:
    edges = bounds.near_edges(best, frac=frac)
    if not edges:
        return bounds, {}
    return bounds.expand_edges(edges, grow=grow, hard=HARD_BOUNDS)


def run_stage1_coarse(
    *,
    config: StagedConfig,
    eval_fn: EvalReachFn,
) -> tuple[list[ReachSample], np.ndarray, np.ndarray]:
    m_axis = 10.0 ** _grid_1d(config.log_m[0], config.log_m[1], config.n_m)
    e_axis = _grid_1d(config.e_range[0], config.e_range[1], config.n_e)
    free_list = _product_on_axes(
        config.free_bounds, config.stage1_axes, config.stage1_points
    )
    samples: list[ReachSample] = []
    for m in m_axis:
        for e in e_axis:
            seed = SeedAnchors(m=float(m), e=float(e))
            for free in free_list:
                s = eval_fn(seed, free)
                s.stage = "stage1"
                samples.append(s)
    samples.sort(key=lambda s: (s.soft_choreo, 0 if s.status == "success" else 1))
    return samples, m_axis, e_axis


def run_stage2_expand(
    seeds: list[ReachSample],
    *,
    config: StagedConfig,
    eval_fn: EvalReachFn,
) -> list[ReachSample]:
    """Local grids around stage-1 seeds; expand when best hugs edges."""
    out: list[ReachSample] = []
    seen: set[tuple] = set()
    for seed_s in seeds:
        bounds = _local_bounds_around(seed_s.free, config.free_bounds, shrink=0.55)
        best_free = seed_s.free
        for _round in range(config.stage2_expand_rounds):
            frees = _product_on_axes(
                bounds, config.stage2_axes, config.stage2_points, base=best_free
            )
            # Cap product explosion: scan axes 1D + center
            if len(frees) > config.stage2_max_product:
                frees = [best_free]
                for name in config.stage2_axes:
                    for x in _grid_1d(*getattr(bounds, name), config.stage2_points):
                        frees.append(best_free.with_update(**{name: float(x)}))
            round_best: ReachSample | None = None
            for free in frees:
                # linear only
                free = free.with_update(
                    **{k: 0.0 for k in QUAD_ORDER + V1_NAMES}
                )
                key = (round(seed_s.m, 12), round(seed_s.e, 12), free.key())
                if key in seen:
                    continue
                seen.add(key)
                s = eval_fn(SeedAnchors(seed_s.m, seed_s.e), free)
                s.stage = "stage2"
                out.append(s)
                if round_best is None or s.soft_choreo < round_best.soft_choreo:
                    round_best = s
            if round_best is None:
                break
            best_free = round_best.free
            bounds, changed = _expand_if_edged(
                bounds,
                best_free,
                grow=config.stage2_expand_grow,
                frac=config.stage2_edge_frac,
            )
            if not changed and _round > 0:
                break
    out.sort(key=lambda s: s.soft_choreo)
    return out


def run_stage3_stain(
    stained: list[ReachSample],
    *,
    config: StagedConfig,
    eval_fn: EvalReachFn,
    score_fn: EvalReachFn,
) -> tuple[list[ReachSample], list[ReachSample]]:
    """Densify stained islands; unlock high-order one-by-one; score gate survivors."""
    samples: list[ReachSample] = []
    survivors: list[ReachSample] = []
    seen: set[tuple] = set()

    unlock_seq = UNLOCK_STAGES if config.unlock_high_order else ((),)

    for seed_s in stained:
        base_bounds = _local_bounds_around(seed_s.free, config.free_bounds, shrink=0.35)
        for unlocked in unlock_seq:
            axes = FREE_NAMES + unlocked
            # densify: product on a reduced set to control cost
            densify_axes = ("a1", "M1") + unlocked if unlocked else ("a1", "e1", "M1", "vx")
            densify_axes = tuple(a for a in densify_axes if a in axes)
            frees = _product_on_axes(
                base_bounds,
                densify_axes,
                config.stage3_points,
                base=seed_s.free,
            )
            for free in frees:
                # zero knobs not in unlocked
                z = {k: 0.0 for k in QUAD_ORDER + V1_NAMES if k not in unlocked}
                free = free.with_update(**z)
                key = (round(seed_s.m, 12), round(seed_s.e, 12), free.key(), unlocked)
                if key in seen:
                    continue
                seen.add(key)
                # hunt with soft
                s = eval_fn(SeedAnchors(seed_s.m, seed_s.e), free)
                s.stage = "stage3"
                s.unlocked = unlocked
                s.stained = True
                samples.append(s)
                if s.status == "success":
                    scored = score_fn(SeedAnchors(seed_s.m, seed_s.e), free)
                    scored.stage = "score"
                    scored.unlocked = unlocked
                    scored.stained = True
                    survivors.append(scored)
                    samples.append(scored)

    samples.sort(key=lambda s: (0 if s.status == "success" else 1, s.soft_choreo))
    survivors.sort(key=lambda s: s.score if s.score is not None else LOSS_FAIL)
    return samples, survivors


def run_staged_search(
    *,
    sigmas: RepSigmas | Path | str,
    config: StagedConfig | None = None,
    skip_stage3_if_survivors: bool = True,
) -> StagedSearchResult:
    """
    Full ABCD→BCDA reachability pipeline.

    ``sigmas`` must be Stage-A calibrated (path or RepSigmas).
    """
    config = config or StagedConfig()
    if isinstance(sigmas, (str, Path)):
        sigmas = load_required_sigmas(sigmas)
    elif not is_calibrated_sigmas(sigmas):
        raise ValueError(
            f"staged search requires Stage-A calibrated sigmas "
            f"(source={sigmas.source!r}, n={sigmas.n_samples})"
        )

    t_end = float(config.n_periods * 2.0 * math.pi)
    n_evals = 0
    t0 = time.perf_counter()

    def eval_soft(seed: SeedAnchors, free: FreeParams) -> ReachSample:
        nonlocal n_evals
        n_evals += 1
        return evaluate_reachability(
            seed,
            free,
            t_end=t_end,
            n_outputs=config.n_outputs_coarse,
            sigmas=sigmas,
            perm_mode=config.perm_mode,
            detailed_score=False,
        )

    def eval_score(seed: SeedAnchors, free: FreeParams) -> ReachSample:
        nonlocal n_evals
        n_evals += 1
        return evaluate_reachability(
            seed,
            free,
            t_end=t_end,
            n_outputs=config.n_outputs_fine,
            sigmas=sigmas,
            perm_mode=config.perm_mode,
            detailed_score=True,
        )

    stage1, m_axis, e_axis = run_stage1_coarse(config=config, eval_fn=eval_soft)
    top1 = stage1[: config.stage1_top_k]
    # Promote any stage-1 successes to scored survivors early
    early_survivors: list[ReachSample] = []
    for s in top1:
        if s.status == "success":
            scored = eval_score(SeedAnchors(s.m, s.e), s.free)
            scored.stage = "score"
            early_survivors.append(scored)

    stage2 = run_stage2_expand(top1, config=config, eval_fn=eval_soft)
    for s in stage2:
        if s.status == "success":
            scored = eval_score(SeedAnchors(s.m, s.e), s.free)
            scored.stage = "score"
            early_survivors.append(scored)

    need_stage3 = not (skip_stage3_if_survivors and early_survivors)
    stained: list[ReachSample] = []
    stage3: list[ReachSample] = []
    survivors = list(early_survivors)

    if need_stage3:
        pool = stage2 if stage2 else stage1
        stained = stain_flood(
            pool,
            frac=config.stain_frac,
            max_seeds=config.stain_max_seeds,
            flood_chebyshev=config.flood_chebyshev,
            m_axis=m_axis,
            e_axis=e_axis,
        )
        stage3, surv3 = run_stage3_stain(
            stained, config=config, eval_fn=eval_soft, score_fn=eval_score
        )
        survivors.extend(surv3)

    # Dedup survivors by free key + m,e
    uniq: dict[tuple, ReachSample] = {}
    for s in survivors:
        key = (round(s.m, 12), round(s.e, 12), s.free.key())
        prev = uniq.get(key)
        if prev is None or (s.score is not None and (prev.score is None or s.score < prev.score)):
            uniq[key] = s
    survivors = sorted(
        uniq.values(),
        key=lambda s: s.score if s.score is not None else LOSS_FAIL,
    )

    return StagedSearchResult(
        sigmas=sigmas,
        config=config,
        stage1=stage1,
        stage2=stage2,
        stained=stained,
        stage3=stage3,
        survivors=survivors,
        n_evals=n_evals,
        wall_s=time.perf_counter() - t0,
    )
