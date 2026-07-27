"""Grid + beam search over free increments (q̂ fixed).

Per Stage-A seed (m, e), anchors:

    a0 = 1,  e0 = e,  M0 = 0,  μ = m

Linear polys: q_i = q0 + i q1  (no quadratic a2/e2/M2).

Td-symmetric velocity kick: δv_i = R_i · (vx, vy, vz).

Free knobs (6D): a1, e1, M1, vx, vy, vz

IC is built in the inertial COM frame.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from fairy_orbit.design.manifold import ManifoldParams, elements_for_index
from fairy_orbit.observe.peo import PEOFilterResult, evaluate_peo
from fairy_orbit.observe.rep_error import RepSigmas

LOSS_FAIL = 1e12

FREE_NAMES: tuple[str, ...] = ("a1", "e1", "M1", "vx", "vy", "vz")
A0_FIXED: float = 1.0
M0_FIXED: float = 0.0


@dataclass(frozen=True)
class FreeParams:
    a1: float
    e1: float
    M1: float
    vx: float
    vy: float
    vz: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in FREE_NAMES}

    def key(self, ndigits: int = 10) -> tuple[float, ...]:
        return tuple(round(float(getattr(self, k)), ndigits) for k in FREE_NAMES)

    def with_update(self, **kwargs: float) -> FreeParams:
        d = self.as_dict()
        d.update({k: float(v) for k, v in kwargs.items() if k in FREE_NAMES})
        return FreeParams(**d)


@dataclass(frozen=True)
class SeedAnchors:
    m: float
    e: float

    @property
    def e0(self) -> float:
        return self.e


@dataclass(frozen=True)
class SearchBounds:
    """a1>0 for radial ladder; (vx,vy,vz) small vs circular speed ~1."""

    a1: tuple[float, float] = (0.10, 0.30)
    e1: tuple[float, float] = (-0.02, 0.04)
    M1: tuple[float, float] = (0.5, 6.0)
    vx: tuple[float, float] = (-0.05, 0.05)
    vy: tuple[float, float] = (-0.05, 0.05)
    vz: tuple[float, float] = (-0.05, 0.05)

    def span(self, name: str) -> float:
        lo, hi = getattr(self, name)
        return float(hi - lo)

    def clip(self, name: str, value: float) -> float:
        lo, hi = getattr(self, name)
        return float(np.clip(value, lo, hi))

    def center(self) -> FreeParams:
        return FreeParams(**{k: 0.5 * (getattr(self, k)[0] + getattr(self, k)[1]) for k in FREE_NAMES})


@dataclass(frozen=True)
class BeamConfig:
    beam_width: int = 4
    coarse_points: int = 2
    refine_points: tuple[int, ...] = (5, 7)
    window_shrink: float = 0.45
    max_evals: int = 3000
    n_periods: float = 2.0
    t_end: float | None = None
    n_outputs: int = 160
    epsilon: float = 1e-9
    min_dt: float = 1e-6
    bisect_iters: int = 6
    grad_steps: int = 8
    grad_eps_frac: float = 0.02
    grad_lr: float = 0.2
    refine_full_product: bool = False

    def resolve_t_end(self) -> float:
        if self.t_end is not None:
            return float(self.t_end)
        return float(self.n_periods * 2.0 * math.pi)


@dataclass(frozen=True)
class Candidate:
    params: FreeParams
    loss: float
    status: str
    summary: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def key(self) -> tuple[float, ...]:
        return self.params.key()


@dataclass
class BeamSearchResult:
    seed: SeedAnchors
    bounds: SearchBounds
    config: BeamConfig
    history: list[Candidate]
    beams: list[Candidate]
    refined: list[Candidate]
    best: Candidate | None
    n_evals: int
    wall_s: float


EvalFn = Callable[[ManifoldParams], tuple[float, str, dict, float]]


def to_manifold(seed: SeedAnchors, free: FreeParams) -> ManifoldParams:
    return ManifoldParams(
        a0=A0_FIXED,
        a1=free.a1,
        e0=seed.e,
        e1=free.e1,
        M0=M0_FIXED,
        M1=free.M1,
        vx=free.vx,
        vy=free.vy,
        vz=free.vz,
        mu_mass=seed.m,
    )


def ic_valid(seed: SeedAnchors, free: FreeParams) -> bool:
    if seed.m <= 0.0:
        return False
    try:
        params = to_manifold(seed, free)
        for i in range(4):
            elements_for_index(params, i)
        return True
    except ValueError:
        return False


def make_eval_fn(
    *,
    t_end: float,
    n_outputs: int,
    sigmas: RepSigmas | Path | str | None,
    epsilon: float = 1e-9,
    min_dt: float = 1e-6,
    perm_mode: str = "fixed_radial",
) -> EvalFn:
    def _eval(params: ManifoldParams) -> tuple[float, str, dict, float]:
        t0 = time.perf_counter()
        res: PEOFilterResult = evaluate_peo(
            params,
            t_end=t_end,
            n_outputs=n_outputs,
            epsilon=epsilon,
            min_dt=min_dt,
            sigmas=sigmas,
            perm_mode=perm_mode,
        )
        elapsed = time.perf_counter() - t0
        if res.status != "success":
            return LOSS_FAIL, res.status, dict(res.summary), elapsed
        loss = float(res.summary.get("score", LOSS_FAIL))
        if not math.isfinite(loss):
            loss = LOSS_FAIL
        return loss, res.status, dict(res.summary), elapsed

    return _eval


def _grid_1d(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.5 * (lo + hi)], dtype=float)
    return np.linspace(lo, hi, int(n))


def _product_grid(bounds: SearchBounds, n: int) -> Iterable[FreeParams]:
    axes = {k: _grid_1d(*getattr(bounds, k), n) for k in FREE_NAMES}

    def rec(i: int, cur: dict[str, float]):
        if i == len(FREE_NAMES):
            yield FreeParams(**cur)
            return
        name = FREE_NAMES[i]
        for x in axes[name]:
            cur[name] = float(x)
            yield from rec(i + 1, cur)

    yield from rec(0, {})


def select_beams(history: list[Candidate], beam_width: int) -> list[Candidate]:
    best_by_key: dict[tuple[float, ...], Candidate] = {}
    for c in history:
        prev = best_by_key.get(c.key)
        if prev is None or c.loss < prev.loss:
            best_by_key[c.key] = c
    return sorted(best_by_key.values(), key=lambda c: c.loss)[:beam_width]


def evaluate_free(
    seed: SeedAnchors,
    free: FreeParams,
    eval_fn: EvalFn,
    cache: dict[tuple[float, ...], Candidate],
) -> Candidate:
    key = free.key()
    if key in cache:
        return cache[key]
    if not ic_valid(seed, free):
        cand = Candidate(params=free, loss=LOSS_FAIL, status="bad_ic", summary={})
    else:
        loss, status, summary, elapsed = eval_fn(to_manifold(seed, free))
        cand = Candidate(
            params=free,
            loss=loss,
            status=status,
            summary=summary,
            elapsed_s=elapsed,
        )
    cache[key] = cand
    return cand


def _local_bounds(center: FreeParams, bounds: SearchBounds, shrink: float) -> SearchBounds:
    kwargs = {}
    for name in FREE_NAMES:
        lo, hi = getattr(bounds, name)
        half = 0.5 * (hi - lo) * shrink
        c = getattr(center, name)
        kwargs[name] = (max(lo, c - half), min(hi, c + half))
    return SearchBounds(**kwargs)


def grid_beam_search(
    m: float,
    e: float,
    *,
    bounds: SearchBounds | None = None,
    config: BeamConfig | None = None,
    eval_fn: EvalFn | None = None,
    sigmas: RepSigmas | Path | str | None = None,
) -> BeamSearchResult:
    seed = SeedAnchors(m=m, e=e)
    bounds = bounds or SearchBounds()
    config = config or BeamConfig()
    if eval_fn is None:
        eval_fn = make_eval_fn(
            t_end=config.resolve_t_end(),
            n_outputs=config.n_outputs,
            sigmas=sigmas,
            epsilon=config.epsilon,
            min_dt=config.min_dt,
        )

    t0 = time.perf_counter()
    cache: dict[tuple[float, ...], Candidate] = {}
    history: list[Candidate] = []

    n0 = int(config.coarse_points)
    n0_prod = n0 ** len(FREE_NAMES)
    if n0_prod > config.max_evals:
        raise ValueError(
            f"coarse grid {n0}^{len(FREE_NAMES)}={n0_prod} exceeds max_evals={config.max_evals}"
        )

    for free in _product_grid(bounds, n0):
        history.append(evaluate_free(seed, free, eval_fn, cache))

    beams = select_beams(history, config.beam_width)

    for level, n_pts in enumerate(config.refine_points):
        if not beams or len(cache) >= config.max_evals:
            break
        shrink = config.window_shrink ** (level + 1)
        for beam in list(beams):
            loc = _local_bounds(beam.params, bounds, shrink)
            if config.refine_full_product:
                for free in _product_grid(loc, n_pts):
                    if len(cache) >= config.max_evals:
                        break
                    free = FreeParams(**{k: bounds.clip(k, getattr(free, k)) for k in FREE_NAMES})
                    history.append(evaluate_free(seed, free, eval_fn, cache))
            else:
                base = beam.params
                for name in FREE_NAMES:
                    if len(cache) >= config.max_evals:
                        break
                    lo, hi = getattr(loc, name)
                    for x in _grid_1d(lo, hi, n_pts):
                        free = base.with_update(**{name: float(x)})
                        free = FreeParams(**{k: bounds.clip(k, getattr(free, k)) for k in FREE_NAMES})
                        history.append(evaluate_free(seed, free, eval_fn, cache))
            beams = select_beams(history, config.beam_width)

    refined = []
    for b in beams:
        if len(cache) >= config.max_evals:
            break
        refined.append(
            refine_candidate(
                seed, b, bounds=bounds, config=config, eval_fn=eval_fn, cache=cache
            )
        )
    history.extend(refined)
    beams_final = select_beams(history, config.beam_width)
    best = beams_final[0] if beams_final else None
    return BeamSearchResult(
        seed=seed,
        bounds=bounds,
        config=config,
        history=history,
        beams=beams_final,
        refined=refined,
        best=best,
        n_evals=len(cache),
        wall_s=time.perf_counter() - t0,
    )


def refine_candidate(
    seed: SeedAnchors,
    cand: Candidate,
    *,
    bounds: SearchBounds,
    config: BeamConfig,
    eval_fn: EvalFn,
    cache: dict[tuple[float, ...], Candidate],
) -> Candidate:
    free = cand.params
    best = cand

    for _ in range(config.bisect_iters):
        improved = False
        for name in FREE_NAMES:
            lo, hi = getattr(bounds, name)
            cur = getattr(free, name)
            half = max(0.03 * (hi - lo), 1e-12)
            left = max(lo, cur - half)
            right = min(hi, cur + half)
            trials = [left, 0.5 * (left + cur), cur, 0.5 * (cur + right), right]
            for x in trials:
                trial = FreeParams(
                    **{
                        k: bounds.clip(k, float(x) if k == name else getattr(free, k))
                        for k in FREE_NAMES
                    }
                )
                c = evaluate_free(seed, trial, eval_fn, cache)
                if c.loss < best.loss:
                    best = c
                    free = c.params
                    improved = True
        if not improved:
            break

    lr = config.grad_lr
    for _ in range(config.grad_steps):
        c0 = evaluate_free(seed, free, eval_fn, cache)
        if c0.loss < best.loss:
            best = c0
            free = c0.params
        grads = {}
        for name in FREE_NAMES:
            span = bounds.span(name)
            eps = max(config.grad_eps_frac * span, 1e-12)
            trial = FreeParams(
                **{
                    k: bounds.clip(k, getattr(free, k) + (eps if k == name else 0.0))
                    for k in FREE_NAMES
                }
            )
            c1 = evaluate_free(seed, trial, eval_fn, cache)
            grads[name] = (c1.loss - c0.loss) / eps
        new = FreeParams(
            **{
                k: bounds.clip(k, getattr(free, k) - lr * grads[k] * bounds.span(k))
                for k in FREE_NAMES
            }
        )
        c_new = evaluate_free(seed, new, eval_fn, cache)
        if c_new.loss < best.loss:
            best = c_new
            free = c_new.params
            lr *= 1.05
        else:
            lr *= 0.5
            if lr < 1e-4:
                break
    return best


def result_to_dict(res: BeamSearchResult) -> dict:
    def _c(c: Candidate | None) -> dict | None:
        if c is None:
            return None
        return {
            "params": c.params.as_dict(),
            "theta": list(to_manifold(res.seed, c.params).as_theta()),
            "loss": c.loss,
            "status": c.status,
            "elapsed_s": c.elapsed_s,
            "summary": c.summary,
        }

    return {
        "seed": {"m": res.seed.m, "e": res.seed.e, "a0": A0_FIXED, "M0": M0_FIXED, "e0": res.seed.e},
        "free_names": list(FREE_NAMES),
        "poly": "q_i = q0 + i q1",
        "v_kick": "delta_v_i = R_i @ (vx,vy,vz)  (Td via Rodrigues from T1)",
        "frame": "inertial COM",
        "n_evals": res.n_evals,
        "wall_s": res.wall_s,
        "bounds": {k: list(getattr(res.bounds, k)) for k in FREE_NAMES},
        "config": {
            "beam_width": res.config.beam_width,
            "coarse_points": res.config.coarse_points,
            "refine_points": list(res.config.refine_points),
            "window_shrink": res.config.window_shrink,
            "t_end": res.config.resolve_t_end(),
            "n_periods": res.config.n_periods,
            "n_outputs": res.config.n_outputs,
            "max_evals": res.config.max_evals,
        },
        "best": _c(res.best),
        "beams": [_c(b) for b in res.beams],
        "refined": [_c(b) for b in res.refined],
        "history_top": [_c(c) for c in select_beams(res.history, min(20, len(res.history)))],
    }
