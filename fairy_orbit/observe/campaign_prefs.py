"""Campaign defaults from RESPONSE §6–§7 decisions.

Thread-budget metaphor (was equal 4:4:4:4)::

    choreo_n4 : choreo_n5 : path_a(+Floquet) : branch2  =  7 : 2 : 6 : 1

Default *activation*: only slots with positive weight that are marked active —
n5 and Branch-2 stay dark (weight reserved for opt-in). Active default =
choreo_n4 + Path A when a seed is available.

Priority narrative:
  1. N=4 search + Floquet gate (weight 7)
  2. Path A + Floquet path resweep (weight 6)
  3. N=5 search (weight 2, inactive by default)
  4. Branch-2 probe (weight 1, inactive by default)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fairy_orbit.design.seeds import OrbitSeed, save_seed
from fairy_orbit.store.search_db import (
    DEFAULT_SEARCH_DB_NAME,
    ChoreographySearchStore,
)

# --- Thread metaphor weights (sum = 16) ---
WEIGHT_CHOREO_N4 = 7
WEIGHT_CHOREO_N5 = 2
WEIGHT_PATH_A = 6
WEIGHT_BRANCH2 = 1

SLOT_WEIGHTS: dict[str, int] = {
    "choreo_n4": WEIGHT_CHOREO_N4,
    "choreo_n5": WEIGHT_CHOREO_N5,
    "path_a": WEIGHT_PATH_A,
    "branch2": WEIGHT_BRANCH2,
}

# --- Default activation (n5 + branch2 dark) ---
CHOREO_N4_IN_DEFAULT_CAMPAIGN = True
CHOREO_N5_IN_DEFAULT_CAMPAIGN = False
PATH_A_IN_DEFAULT_CAMPAIGN = True  # when seed / --path-a-from-db
BRANCH2_IN_DEFAULT_CAMPAIGN = False

# --- Floquet certify ---
FLOQUET_GATE_DEFAULT = True
FLOQUET_STABLE_ATOL = 0.05

# --- Path A + path resweep ---
PATH_A_AUTO_FLOQUET_SWEEP = True
ALLOW_UNSTABLE_PATH_A_SEED = True

# --- Branch-2 (lowest; opt-in) ---
BRANCH2_PROBE_DEFAULT_DIVERSE = 4
BRANCH2_PROBE_DEFAULT_SAMPLES = 48

# --- Scout / certify funnel ---
SCOUT_ATOL_REL = 1e-5
SCOUT_MAX_RESIDUAL = 1e-3
CERTIFY_ATOL_REL = 1e-8
CERTIFY_MAX_RESIDUAL = 1e-6


@dataclass(frozen=True)
class ContinuationSeedPick:
    seed: OrbitSeed
    trial_no: int | None
    residual: float | None
    floquet_stable: bool | None
    floquet_max_abs: float | None
    source: str
    note: str


def wall_hours_for_slots(
    total_wall_hours: float,
    active_slots: list[str],
) -> dict[str, float]:
    """
    Split a finite wall clock across active slots by SLOT_WEIGHTS.

    ``total_wall_hours <= 0`` → each active slot gets 0.0 (caller = unlimited).
    Inactive / unknown slots are omitted.
    """
    keys = [k for k in active_slots if k in SLOT_WEIGHTS]
    if not keys:
        return {}
    if total_wall_hours is None or float(total_wall_hours) <= 0:
        return {k: 0.0 for k in keys}
    wsum = sum(SLOT_WEIGHTS[k] for k in keys)
    if wsum <= 0:
        return {k: 0.0 for k in keys}
    total = float(total_wall_hours)
    return {k: total * SLOT_WEIGHTS[k] / wsum for k in keys}


def _floquet_meta(seed: OrbitSeed) -> tuple[bool | None, float | None]:
    ver = seed.verification or {}
    fl = ver.get("floquet")
    if not isinstance(fl, dict):
        return None, None
    stable = fl.get("stable")
    max_abs = fl.get("max_abs")
    return (
        bool(stable) if stable is not None else None,
        float(max_abs) if max_abs is not None else None,
    )


def pick_path_a_seed(
    n_bodies: int = 4,
    *,
    db_path: Path | None = None,
    prefer_floquet_stable: bool = True,
    out_seed_path: Path | None = None,
) -> ContinuationSeedPick | None:
    """Prefer Floquet-stable accepted pass; else best residual for Path A."""
    root = Path(__file__).resolve().parents[2]
    db = db_path or (
        root
        / "experiments/output"
        / f"choreography_search_n{n_bodies}"
        / DEFAULT_SEARCH_DB_NAME
    )
    if not db.is_file():
        return None

    with ChoreographySearchStore(db) as store:
        passes = store.list_passes(
            n_bodies, limit=200, max_residual=CERTIFY_MAX_RESIDUAL
        )
        stable_pick = None
        best_pick = None
        for rec in passes:
            if not rec.seed_json:
                continue
            seed = OrbitSeed.from_dict(rec.seed_json)
            st, mx = _floquet_meta(seed)
            if best_pick is None:
                best_pick = (rec, seed, st, mx)
            if prefer_floquet_stable and st is True and stable_pick is None:
                stable_pick = (rec, seed, st, mx)

        chosen = stable_pick or best_pick
        if chosen is None:
            return None
        rec, seed, st, mx = chosen
        if stable_pick is not None:
            note = "floquet-stable accepted pass"
        elif st is False:
            note = (
                "no Floquet-stable pass in archive; using best residual "
                "(Path A for |λ|=1 crossing hunt, not endpoint shooting)"
            )
        else:
            note = (
                "archive pass has no floquet metadata (pre-gate); "
                "using best residual — Path A + resweep still preferred"
            )

        if out_seed_path is not None:
            out_seed_path.parent.mkdir(parents=True, exist_ok=True)
            save_seed(seed, out_seed_path)

        return ContinuationSeedPick(
            seed=seed,
            trial_no=rec.trial_no,
            residual=rec.residual,
            floquet_stable=st,
            floquet_max_abs=mx,
            source=str(db),
            note=note,
        )


def campaign_priority_blurb() -> str:
    return (
        f"threads 7:2:6:1 = n4:n5:pathA:branch2 "
        f"(default active: n4={'on' if CHOREO_N4_IN_DEFAULT_CAMPAIGN else 'off'} "
        f"n5={'on' if CHOREO_N5_IN_DEFAULT_CAMPAIGN else 'off'} "
        f"pathA={'on' if PATH_A_IN_DEFAULT_CAMPAIGN else 'off'} "
        f"branch2={'on' if BRANCH2_IN_DEFAULT_CAMPAIGN else 'off'})"
    )
