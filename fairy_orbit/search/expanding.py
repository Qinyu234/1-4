"""Time-budgeted expanding 2D grid search over (v_rad, v_tan)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.library.store import save_candidate
from fairy_orbit.search.grid import Candidate, evaluate_params


@dataclass
class Bounds:
    """Axis bounds as multiples of escape speed."""

    rad_lo: float = 0.7
    rad_hi: float = 1.3
    tan_lo: float = 0.7
    tan_hi: float = 1.3

    def expand(self, step: float, *, pad: float = 0.05) -> Bounds:
        """Grow all sides by `step` (in vesc units), keeping a small pad past extrema."""
        return Bounds(
            rad_lo=self.rad_lo - step,
            rad_hi=self.rad_hi + step,
            tan_lo=self.tan_lo - step,
            tan_hi=self.tan_hi + step,
        )

    def expand_toward_best(
        self,
        best_rad_frac: float,
        best_tan_frac: float,
        step: float,
        edge_tol: float = 0.15,
    ) -> Bounds:
        """Expand sides where the best point sits near the boundary."""
        rad_lo, rad_hi = self.rad_lo, self.rad_hi
        tan_lo, tan_hi = self.tan_lo, self.tan_hi
        rad_span = max(rad_hi - rad_lo, 1e-12)
        tan_span = max(tan_hi - tan_lo, 1e-12)
        if (best_rad_frac - rad_lo) / rad_span <= edge_tol:
            rad_lo -= step
        if (rad_hi - best_rad_frac) / rad_span <= edge_tol:
            rad_hi += step
        if (best_tan_frac - tan_lo) / tan_span <= edge_tol:
            tan_lo -= step
        if (tan_hi - best_tan_frac) / tan_span <= edge_tol:
            tan_hi += step
        # Always grow at least a little so the window never stagnates
        if (rad_lo, rad_hi, tan_lo, tan_hi) == (
            self.rad_lo,
            self.rad_hi,
            self.tan_lo,
            self.tan_hi,
        ):
            return self.expand(step * 0.5)
        return Bounds(rad_lo=rad_lo, rad_hi=rad_hi, tan_lo=tan_lo, tan_hi=tan_hi)

    def to_dict(self) -> dict[str, float]:
        return {
            "rad_lo": self.rad_lo,
            "rad_hi": self.rad_hi,
            "tan_lo": self.tan_lo,
            "tan_hi": self.tan_hi,
        }


@dataclass
class ExpandingConfig:
    hours: float = 8.0
    n_per_axis: int = 5
    expand_step: float = 0.2
    edge_tol: float = 0.2
    save_every: int = 1
    checkpoint_path: str = "experiments/output/expanding_checkpoint.json"
    library_dir: str = "orbit_library"
    planet_mass: float = 1.0
    fairy_mass: float = 0.01
    radius: float = 20.0
    G: float = 1.0
    n_periods: float = 5.0
    steps_per_period: int = 60
    record_every: int = 3
    initial_bounds: Bounds = field(default_factory=Bounds)


def _key(v_rad: float, v_tan: float, ndigits: int = 8) -> tuple[float, float]:
    return (round(v_rad, ndigits), round(v_tan, ndigits))


def _grid_points(bounds: Bounds, vesc: float, n: int) -> list[tuple[float, float]]:
    rads = np.linspace(bounds.rad_lo * vesc, bounds.rad_hi * vesc, n)
    tans = np.linspace(bounds.tan_lo * vesc, bounds.tan_hi * vesc, n)
    return [(float(vr), float(vt)) for vr in rads for vt in tans]


def _candidate_payload(cand: Candidate) -> dict[str, Any]:
    return {
        "v_rad": cand.v_rad,
        "v_tan": cand.v_tan,
        "mass_ratio": cand.fairy_mass / cand.planet_mass,
        "planet_mass": cand.planet_mass,
        "fairy_mass": cand.fairy_mass,
        "radius": cand.radius,
        "G": cand.G,
        "initial_position": cand.initial_positions,
        "initial_velocity": cand.initial_velocities,
        "period": cand.period,
        "score": cand.score,
        "metrics": cand.metrics,
    }


def _write_checkpoint(
    path: Path,
    *,
    bounds: Bounds,
    round_idx: int,
    evaluated: int,
    best: Candidate | None,
    elapsed_s: float,
    hours: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bounds": bounds.to_dict(),
        "round": round_idx,
        "evaluated": evaluated,
        "elapsed_hours": elapsed_s / 3600.0,
        "budget_hours": hours,
        "best": None
        if best is None
        else {
            "v_rad": best.v_rad,
            "v_tan": best.v_tan,
            "score": best.score,
            "metrics": best.metrics,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_expanding_search(
    config: ExpandingConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[Candidate]:
    """
    Keep evaluating new (v_rad, v_tan) cells until wall-clock budget is exhausted.

    Each round fills the current bounds on an n×n grid (skipping already-seen
    points). After the round, bounds auto-expand toward the best score (and
    always grow at least a little).
    """
    if config is None:
        config = ExpandingConfig()
    if progress is None:
        progress = print

    vesc = escape_speed(config.G, config.planet_mass, config.radius)
    deadline = time.monotonic() + config.hours * 3600.0
    t0 = time.monotonic()
    bounds = config.initial_bounds
    seen: set[tuple[float, float]] = set()
    all_candidates: list[Candidate] = []
    best: Candidate | None = None
    round_idx = 0
    eval_kwargs = {
        "planet_mass": config.planet_mass,
        "fairy_mass": config.fairy_mass,
        "radius": config.radius,
        "G": config.G,
        "n_periods": config.n_periods,
        "steps_per_period": config.steps_per_period,
        "record_every": config.record_every,
    }

    progress(
        f"expanding search: budget={config.hours}h  vesc={vesc:.6f}  "
        f"n_periods={config.n_periods}  grid={config.n_per_axis}x{config.n_per_axis}"
    )

    while time.monotonic() < deadline:
        round_idx += 1
        points = _grid_points(bounds, vesc, config.n_per_axis)
        new_points = [p for p in points if _key(*p) not in seen]
        if not new_points:
            # Grid fully covered at this resolution — expand and densify next round
            bounds = bounds.expand(config.expand_step)
            progress(f"round {round_idx}: no new cells, force-expand -> {bounds.to_dict()}")
            continue

        progress(
            f"round {round_idx}: bounds={bounds.to_dict()}  "
            f"new_cells={len(new_points)}  elapsed={((time.monotonic()-t0)/3600):.3f}h"
        )

        for vr, vt in new_points:
            if time.monotonic() >= deadline:
                break
            seen.add(_key(vr, vt))
            cand, _, _ = evaluate_params(vr, vt, **eval_kwargs)
            all_candidates.append(cand)
            if best is None or cand.score < best.score:
                best = cand
                progress(
                    f"  NEW BEST score={cand.score:.6f}  "
                    f"(v_rad={cand.v_rad:.5f}, v_tan={cand.v_tan:.5f})"
                )
            else:
                progress(
                    f"  eval score={cand.score:.6f}  "
                    f"(v_rad={cand.v_rad:.5f}, v_tan={cand.v_tan:.5f})"
                )
            if config.save_every > 0 and len(all_candidates) % config.save_every == 0:
                # Always persist improving / periodic samples: save if top-ish
                ranked = sorted(all_candidates, key=lambda c: c.score)
                if cand.score <= ranked[min(4, len(ranked) - 1)].score:
                    save_candidate(_candidate_payload(cand), directory=config.library_dir)

            _write_checkpoint(
                Path(config.checkpoint_path),
                bounds=bounds,
                round_idx=round_idx,
                evaluated=len(all_candidates),
                best=best,
                elapsed_s=time.monotonic() - t0,
                hours=config.hours,
            )

        if time.monotonic() >= deadline:
            break

        if best is not None:
            bounds = bounds.expand_toward_best(
                best.v_rad / vesc,
                best.v_tan / vesc,
                step=config.expand_step,
                edge_tol=config.edge_tol,
            )
        else:
            bounds = bounds.expand(config.expand_step)

    all_candidates.sort(key=lambda c: c.score)
    # Final save of top 10
    for cand in all_candidates[:10]:
        save_candidate(_candidate_payload(cand), directory=config.library_dir)

    _write_checkpoint(
        Path(config.checkpoint_path),
        bounds=bounds,
        round_idx=round_idx,
        evaluated=len(all_candidates),
        best=best,
        elapsed_s=time.monotonic() - t0,
        hours=config.hours,
    )
    progress(
        f"done: evaluated={len(all_candidates)}  "
        f"elapsed={((time.monotonic()-t0)/3600):.3f}h  "
        f"best_score={best.score if best else None}"
    )
    return all_candidates
