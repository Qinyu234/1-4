"""Bayesian search for PEO with staged high-order unlock (Optuna TPE).

Objective (minimize):
  escape / bad IC  → large constant
  choreography fail → mid penalty + soft radial residual (guides the surrogate)
  success          → PEO score (σ-weighted E_r/E_v …)

If no success for `stagnate_trials`, escalate:
  1) expand active free-parameter bounds (toward HARD_BOUNDS)
  2) unlock next high-order knob in order a2 → e2 → M2 → (v1x,v1y,v1z)

Higher-order forms:
  q_i = q0 + i q1 + i² q2   (a, e, M)
  δv_i = R_i · (v + i v₁)   (v 三参高阶)

Each escalate restarts the Optuna study (TPE distributions are fixed per study).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from fairy_orbit.observe.closure import (
    cyclic_shift,
    radial_choreography_shift,
    radial_order,
)
from fairy_orbit.observe.peo import evaluate_peo
from fairy_orbit.observe.rep_error import RepSigmas
from fairy_orbit.observe.search import (
    HARD_BOUNDS,
    LOSS_FAIL,
    QUAD_ORDER,
    V1_NAMES,
    FreeParams,
    SeedAnchors,
    to_manifold,
)

# Soft / hard penalties for BO (must stay << LOSS_FAIL used by beam).
PENALTY_ESCAPE = 1.0e4
PENALTY_CHOREO_BASE = 1.0e2

# Unlock order: a → e → M → v's three components as one stage.
UNLOCK_STAGES: tuple[tuple[str, ...], ...] = (
    (),
    ("a2",),
    ("a2", "e2"),
    ("a2", "e2", "M2"),
    ("a2", "e2", "M2") + V1_NAMES,
)

BASE_FREE: tuple[str, ...] = ("a1", "e1", "M1", "vx", "vy", "vz")
EXPAND_AXES: tuple[str, ...] = ("log_m", "e") + BASE_FREE


@dataclass(frozen=True)
class BayesSpace:
    """Search box; high-order axes exist but are only suggested when unlocked."""

    log_m: tuple[float, float] = (-6.0, -2.0)
    e: tuple[float, float] = (0.0, 0.20)
    a1: tuple[float, float] = (0.05, 1.00)
    e1: tuple[float, float] = (-0.12, 0.25)
    M1: tuple[float, float] = (0.2, 8.0)
    vx: tuple[float, float] = (-0.25, 0.25)
    vy: tuple[float, float] = (-0.25, 0.25)
    vz: tuple[float, float] = (-0.25, 0.25)
    a2: tuple[float, float] = (-0.08, 0.08)
    e2: tuple[float, float] = (-0.04, 0.04)
    M2: tuple[float, float] = (-1.5, 1.5)
    v1x: tuple[float, float] = (-0.08, 0.08)
    v1y: tuple[float, float] = (-0.08, 0.08)
    v1z: tuple[float, float] = (-0.08, 0.08)

    def bounds_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "log_m": self.log_m,
            "e": self.e,
            "a1": self.a1,
            "e1": self.e1,
            "M1": self.M1,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "a2": self.a2,
            "e2": self.e2,
            "M2": self.M2,
            "v1x": self.v1x,
            "v1y": self.v1y,
            "v1z": self.v1z,
        }


# Absolute clamps for Bayesian bound expansion (log_m / e included).
BAYES_HARD: dict[str, tuple[float, float]] = {
    "log_m": (-8.0, -1.0),
    "e": (0.0, 0.45),
    **HARD_BOUNDS,
}


@dataclass
class BayesTrialResult:
    loss: float
    status: str
    m: float
    e: float
    free: FreeParams
    summary: dict = field(default_factory=dict)
    soft_choreo: float = float("nan")
    stage: int = 0
    unlocked: tuple[str, ...] = ()


@dataclass
class EscalateEvent:
    trial_index: int
    action: str  # "expand" | "unlock"
    detail: dict = field(default_factory=dict)


def soft_choreography_residual(
    traj,
    *,
    central_index: int = 0,
    prefer_k: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float]:
    """
    Continuous residual in [0, 1+] for BO when hard gate fails.

    - frac_noncyclic: fraction of frames where S(t) is not a cyclic shift of S(0)
    - identity_at_T: 1 if S(T)=S(0), else 0
    - best_k_mismatch: 0 if S(T) is prefer_k cyclic shift, else 1
    """
    order_0 = radial_order(traj.positions[0], central_index=central_index)
    n = len(traj)
    noncyc = 0
    shifts: list[int | None] = []
    for k in range(n):
        ok = radial_choreography_shift(
            order_0, radial_order(traj.positions[k], central_index=central_index)
        )
        shifts.append(ok)
        if ok is None:
            noncyc += 1
    frac_noncyclic = noncyc / max(n, 1)
    shift_T = shifts[-1]
    identity_at_T = 1.0 if shift_T == 0 else 0.0
    if shift_T in prefer_k:
        best_k_mismatch = 0.0
    elif shift_T is None:
        order_T = radial_order(traj.positions[-1], central_index=central_index)
        ham = min(
            sum(int(a != b) for a, b in zip(order_T, cyclic_shift(order_0, k)))
            for k in prefer_k
        )
        best_k_mismatch = ham / max(len(order_0), 1)
    else:
        best_k_mismatch = 1.0
    residual = frac_noncyclic + 0.5 * identity_at_T + 0.5 * best_k_mismatch
    return {
        "soft_choreo": float(residual),
        "frac_noncyclic": float(frac_noncyclic),
        "identity_at_T": float(identity_at_T),
        "best_k_mismatch": float(best_k_mismatch),
        "shift_T": -1.0 if shift_T is None else float(shift_T),
    }


def bayes_objective_from_peo(
    res,
    *,
    traj_for_soft=None,
) -> tuple[float, dict[str, Any]]:
    """Map PEOFilterResult → (loss_for_BO, extras)."""
    status = res.status
    summary = dict(res.summary)
    if status in ("escape", "escape_ic"):
        return PENALTY_ESCAPE, {"status": status, **summary}
    if status == "choreography":
        soft = {"soft_choreo": 1.0, "frac_noncyclic": 1.0}
        if res.traj is not None:
            soft = soft_choreography_residual(res.traj)
        loss = PENALTY_CHOREO_BASE + float(soft["soft_choreo"])
        return loss, {"status": status, **summary, **soft}
    if status != "success":
        return PENALTY_ESCAPE, {"status": status, **summary}
    score = float(summary.get("score", LOSS_FAIL))
    if not math.isfinite(score):
        score = LOSS_FAIL
    return score, {"status": status, **summary}


def expand_bayes_space(
    space: BayesSpace,
    *,
    grow: float = 0.35,
    hard: dict[str, tuple[float, float]] | None = None,
    axes: tuple[str, ...] | None = None,
) -> tuple[BayesSpace, dict[str, tuple[float, float]]]:
    """Widen each axis by grow*span on both sides, clamped to hard bounds."""
    hard = hard or BAYES_HARD
    axes = axes or EXPAND_AXES
    kwargs = space.bounds_dict()
    changed: dict[str, tuple[float, float]] = {}
    for name in axes:
        if name not in kwargs:
            continue
        lo, hi = kwargs[name]
        span = max(hi - lo, 1e-15)
        delta = grow * span
        hlo, hhi = hard[name]
        new_lo = max(hlo, lo - delta)
        new_hi = min(hhi, hi + delta)
        if new_lo >= new_hi:
            continue
        if (new_lo, new_hi) != (lo, hi):
            kwargs[name] = (float(new_lo), float(new_hi))
            changed[name] = (float(new_lo), float(new_hi))
    if not changed:
        return space, {}
    return replace(space, **kwargs), changed


def suggest_free(
    trial,
    space: BayesSpace,
    e: float,
    *,
    unlocked: tuple[str, ...] = (),
) -> FreeParams:
    """Optuna suggestions; locked high-order knobs stay at 0.

    e1 uses fixed ``space.e1`` (Optuna forbids per-trial distribution changes);
    invalid ladders fail later in ``elements_for_index``.
    """
    del e  # kept for call-site compatibility
    a1 = trial.suggest_float("a1", *space.a1)
    e1 = trial.suggest_float("e1", *space.e1)
    M1 = trial.suggest_float("M1", *space.M1)
    vx = trial.suggest_float("vx", *space.vx)
    vy = trial.suggest_float("vy", *space.vy)
    vz = trial.suggest_float("vz", *space.vz)

    extras: dict[str, float] = {k: 0.0 for k in QUAD_ORDER + V1_NAMES}
    for name in unlocked:
        lo, hi = getattr(space, name)
        extras[name] = float(trial.suggest_float(name, lo, hi))
    return FreeParams(a1=a1, e1=e1, M1=M1, vx=vx, vy=vy, vz=vz, **extras)


def _try_escalate(
    *,
    space: BayesSpace,
    stage_idx: int,
    n_expands: int,
    max_expands: int,
    expand_grow: float,
    trial_index: int,
) -> tuple[BayesSpace, int, int, EscalateEvent | None]:
    """On stagnate: widen bounds (if room) and unlock the next high-order stage."""
    new_space = space
    changed: dict[str, tuple[float, float]] = {}
    new_expands = n_expands
    if n_expands < max_expands:
        new_space, changed = expand_bayes_space(space, grow=expand_grow)
        if changed:
            new_expands = n_expands + 1

    new_stage = stage_idx
    if stage_idx + 1 < len(UNLOCK_STAGES):
        new_stage = stage_idx + 1

    if not changed and new_stage == stage_idx:
        return space, stage_idx, n_expands, None

    action = "expand+unlock" if changed and new_stage != stage_idx else (
        "expand" if changed else "unlock"
    )
    ev = EscalateEvent(
        trial_index=trial_index,
        action=action,
        detail={
            "n_expands": new_expands,
            "stage": new_stage,
            "unlocked": list(UNLOCK_STAGES[new_stage]),
            "changed": {k: list(v) for k, v in changed.items()},
        },
    )
    return new_space, new_stage, new_expands, ev


def run_bayes_search(
    *,
    n_trials: int = 200,
    space: BayesSpace | None = None,
    t_end: float | None = None,
    n_periods: float = 2.0,
    n_outputs: int = 120,
    sigmas: RepSigmas | None = None,
    perm_mode: str = "fixed_radial",
    seed: int = 0,
    study_name: str = "peo_bayes",
    storage: str | None = None,
    stagnate_trials: int = 40,
    expand_grow: float = 0.35,
    max_expands: int = 8,
    unlock_all: bool = False,
    on_event: Callable[[EscalateEvent], None] | None = None,
) -> tuple[Any, list[BayesTrialResult], list[EscalateEvent]]:
    """
    Optuna TPE over free params (+ optional high-order).

    Default: stagnation → expand bounds → unlock a2→e2→M2→v1.
    ``unlock_all=True``: suggest every knob from trial 0 (no staged unlock).

    Returns (last_study, trial_results, escalate_events).
    """
    import optuna
    from optuna.samplers import TPESampler

    space = space or BayesSpace()
    sigmas = sigmas or RepSigmas(source="unit")
    if t_end is None:
        t_end = float(n_periods * 2.0 * math.pi)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    history: list[BayesTrialResult] = []
    events: list[EscalateEvent] = []
    stage_idx = (len(UNLOCK_STAGES) - 1) if unlock_all else 0
    n_expands = 0
    trials_done = 0
    segment = 0
    study: Any = None
    rng_seed = seed

    def _emit(ev: EscalateEvent) -> None:
        events.append(ev)
        if on_event is not None:
            on_event(ev)

    while trials_done < n_trials:
        unlocked = UNLOCK_STAGES[stage_idx]
        sampler = TPESampler(seed=rng_seed + segment, multivariate=True, group=True)
        # Fresh study per segment so float distributions stay fixed.
        name = f"{study_name}_s{segment}"
        study = optuna.create_study(
            study_name=name,
            direction="minimize",
            sampler=sampler,
            storage=storage,
            load_if_exists=False,
        )
        trials_since_success = 0
        stop_escalate = {"event": None}

        space_snap = space
        unlocked_snap = unlocked
        stage_snap = stage_idx

        def objective(trial: optuna.Trial) -> float:
            nonlocal trials_since_success
            log_m = trial.suggest_float("log_m", *space_snap.log_m)
            e = trial.suggest_float("e", *space_snap.e)
            m = 10.0 ** log_m
            free = suggest_free(trial, space_snap, e, unlocked=unlocked_snap)
            seed_a = SeedAnchors(m=m, e=e)
            params = to_manifold(seed_a, free)
            res = evaluate_peo(
                params,
                t_end=t_end,
                n_outputs=n_outputs,
                sigmas=sigmas,
                perm_mode=perm_mode,
            )
            loss, extras = bayes_objective_from_peo(res)
            status = str(extras.get("status"))
            trial.set_user_attr("status", status)
            trial.set_user_attr("m", m)
            trial.set_user_attr("soft_choreo", extras.get("soft_choreo"))
            trial.set_user_attr("score", extras.get("score"))
            trial.set_user_attr("stage", stage_snap)
            trial.set_user_attr("unlocked", list(unlocked_snap))

            if status == "success":
                trials_since_success = 0
            else:
                trials_since_success += 1

            history.append(
                BayesTrialResult(
                    loss=loss,
                    status=status,
                    m=m,
                    e=e,
                    free=free,
                    summary=extras,
                    soft_choreo=float(extras.get("soft_choreo", float("nan"))),
                    stage=stage_snap,
                    unlocked=unlocked_snap,
                )
            )
            n_hist = len(history)
            if n_hist == 1 or n_hist % 25 == 0:
                best_so_far = min(h.loss for h in history)
                print(
                    f"  trial {n_hist}/{n_trials} status={status} "
                    f"loss={loss:.4g} best={best_so_far:.4g} "
                    f"soft={extras.get('soft_choreo', float('nan'))}",
                    flush=True,
                )
            return float(loss)

        def _callback(study_cb: Any, trial: Any) -> None:
            nonlocal space, stage_idx, n_expands
            if unlock_all:
                # Full-param mode: only optional bound expand, never unlock further.
                if stop_escalate["event"] is not None:
                    return
                if trials_since_success < stagnate_trials:
                    return
                if n_expands >= max_expands:
                    return
                new_space, changed = expand_bayes_space(space, grow=expand_grow)
                if not changed:
                    return
                space = new_space
                n_expands += 1
                stop_escalate["event"] = EscalateEvent(
                    trial_index=trials_done + len(study_cb.trials),
                    action="expand",
                    detail={
                        "n_expands": n_expands,
                        "changed": {k: list(v) for k, v in changed.items()},
                        "unlocked": list(UNLOCK_STAGES[stage_idx]),
                    },
                )
                study_cb.stop()
                return
            if stop_escalate["event"] is not None:
                return
            if trials_since_success < stagnate_trials:
                return
            new_space, new_stage, new_expands, ev = _try_escalate(
                space=space,
                stage_idx=stage_idx,
                n_expands=n_expands,
                max_expands=max_expands,
                expand_grow=expand_grow,
                trial_index=trials_done + len(study_cb.trials),
            )
            if ev is None:
                return
            space = new_space
            stage_idx = new_stage
            n_expands = new_expands
            stop_escalate["event"] = ev
            study_cb.stop()

        remaining = n_trials - trials_done
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=False,
            callbacks=[_callback],
        )
        trials_done = len(history)
        if stop_escalate["event"] is not None:
            _emit(stop_escalate["event"])
            segment += 1
            continue
        break

    return study, history, events
