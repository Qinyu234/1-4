"""For each mass ratio k≈1 (outer = kM), optimize (v_rad, v_tan) separately."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import optimize

from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.library.store import save_candidate
from fairy_orbit.search.grid import Candidate, evaluate_params
from fairy_orbit.analysis.evaluator import EvaluatorConfig


# Default: a few k near unity (planet M, fairies kM).
DEFAULT_K_NEAR_1 = [0.01, 0.1, 1.0, 10, 100]


@dataclass
class PerKOptimizeConfig:
    """Independently optimize velocity for each k near 1."""

    k_values: list[float] = field(default_factory=lambda: list(DEFAULT_K_NEAR_1))
    planet_mass: float = 1.0
    radius: float = 20.0
    G: float = 1.0
    # Coarse seed grid in units of v_esc before local refine
    alpha_grid: list[float] = field(
        default_factory=lambda: list(np.linspace(-0.5, 1.5, 5))
    )
    beta_grid: list[float] = field(
        default_factory=lambda: list(np.linspace(0.0, 1.5, 5))
    )
    n_periods: float = 12.0
    steps_per_period: int = 120
    record_every: int = 3
    refine_method: str = "Nelder-Mead"
    refine_maxiter: int = 40
    library_dir: str = "orbit_library"
    summary_path: str = "experiments/output/k_near1_summary.json"
    hours: float | None = None  # optional wall budget across all k
    # New options for improved search
    use_continuation: bool = False  # Use parameter continuation across k values
    use_distance_matrix: bool = False  # Use continuous distance matrix loss
    t_scan_range: tuple[float, float] | None = None  # Scan T around period, e.g., (0.8, 1.2)
    t_scan_steps: int = 5  # Number of T scan points
    solver_type: str = "own"  # "own" or "rebound"
    evaluation_mode: str = "event"  # "position" or "event"


def _payload(cand: Candidate, vesc: float) -> dict[str, Any]:
    return {
        "v_rad": cand.v_rad,
        "v_tan": cand.v_tan,
        "v_rad_over_vesc": cand.v_rad / vesc,
        "v_tan_over_vesc": cand.v_tan / vesc,
        "mass_ratio": cand.mass_ratio,
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


def _eval_kwargs(config: PerKOptimizeConfig, k: float) -> dict[str, Any]:
    kwargs = {
        "planet_mass": config.planet_mass,
        "mass_ratio": k,
        "radius": config.radius,
        "G": config.G,
        "n_periods": config.n_periods,
        "steps_per_period": config.steps_per_period,
        "record_every": config.record_every,
    }
    if config.t_scan_range is not None:
        kwargs["t_scan_range"] = config.t_scan_range
        kwargs["t_scan_steps"] = config.t_scan_steps
    if config.use_distance_matrix:
        kwargs["eval_config"] = EvaluatorConfig(use_distance_matrix=True)
    kwargs["solver_type"] = config.solver_type
    kwargs["evaluation_mode"] = config.evaluation_mode
    return kwargs


def seed_best_for_k(
    k: float,
    config: PerKOptimizeConfig,
    vesc: float,
    *,
    progress: Callable[[str], None],
    deadline: float | None,
) -> Candidate:
    """Coarse (alpha, beta) grid → best seed for this k."""
    best: Candidate | None = None
    for alpha in config.alpha_grid:
        for beta in config.beta_grid:
            if deadline is not None and time.monotonic() >= deadline:
                break
            cand, _, _ = evaluate_params(
                float(alpha) * vesc,
                float(beta) * vesc,
                **_eval_kwargs(config, k),
            )
            if best is None or cand.score < best.score:
                best = cand
                progress(
                    f"  k={k} seed  score={cand.score:.4f}  "
                    f"alpha={alpha:.3f} beta={beta:.3f}"
                )
        if deadline is not None and time.monotonic() >= deadline:
            break
    assert best is not None
    return best


def refine_velocity_for_k(
    k: float,
    x0: tuple[float, float],
    config: PerKOptimizeConfig,
    *,
    progress: Callable[[str], None],
) -> Candidate:
    """Local optimize (v_rad, v_tan) at fixed k."""
    kwargs = _eval_kwargs(config, k)

    def loss(x: np.ndarray) -> float:
        cand, _, _ = evaluate_params(float(x[0]), float(x[1]), **kwargs)
        return cand.score

    result = optimize.minimize(
        loss,
        x0=np.asarray(x0, dtype=float),
        method=config.refine_method,
        options={"maxiter": config.refine_maxiter, "xatol": 1e-4, "fatol": 1e-4},
    )
    cand, _, _ = evaluate_params(float(result.x[0]), float(result.x[1]), **kwargs)
    progress(
        f"  k={k} refined  score={cand.score:.4f}  "
        f"v_rad={cand.v_rad:.5f} v_tan={cand.v_tan:.5f}  "
        f"success={result.success}"
    )
    return cand


def optimize_each_k(
    config: PerKOptimizeConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[float, Candidate]:
    """
    For each k near 1: seed grid on velocity fractions, then refine.
    If use_continuation is True, use previous k's solution as initial guess.
    Returns mapping k → best Candidate.
    """
    if config is None:
        config = PerKOptimizeConfig()
    if progress is None:
        progress = print

    M = config.planet_mass
    vesc = escape_speed(config.G, M, config.radius)
    deadline = (
        time.monotonic() + config.hours * 3600.0 if config.hours is not None else None
    )
    best_by_k: dict[float, Candidate] = {}

    progress(
        f"per-k optimize: M={M}, outer=kM, k in {config.k_values}, vesc={vesc:.6f}"
    )
    if config.use_continuation:
        progress("using parameter continuation mode (nearest neighbor in ln(k) space)")

    # Sort k values for continuation (start from ln(k)=0, i.e., k=1.0)
    k_values_sorted = sorted(config.k_values)
    if config.use_continuation and len(k_values_sorted) > 1:
        # Start from k=1.0 (ln(k)=0) if available, otherwise closest to it
        if 1.0 in k_values_sorted:
            k_values_sorted.remove(1.0)
            k_values_sorted = [1.0] + k_values_sorted
        else:
            center_idx = min(range(len(k_values_sorted)), key=lambda i: abs(np.log(k_values_sorted[i])))
            k_values_sorted = [k_values_sorted[center_idx]] + \
                             [k for i, k in enumerate(k_values_sorted) if i != center_idx]

    # For continuation: track optimized k values and their solutions
    optimized_k: dict[float, tuple[float, float]] = {}  # k -> (v_rad, v_tan)

    for k in k_values_sorted:
        if deadline is not None and time.monotonic() >= deadline:
            progress("time budget exhausted")
            break
        progress(f"=== optimize k={k} (fairy_mass={k * M}, ln(k)={np.log(k):.4f}) ===")
        
        if config.use_continuation and optimized_k:
            # Find nearest already-optimized k in ln(k) space
            current_ln_k = np.log(k)
            nearest_k = min(optimized_k.keys(), key=lambda k_opt: abs(np.log(k_opt) - current_ln_k))
            nearest_ln_k = np.log(nearest_k)
            progress(f"  using continuation from nearest k={nearest_k:.6f} (ln(k)={nearest_ln_k:.4f})")
            
            # Get solution from nearest k
            seed_v_rad, seed_v_tan = optimized_k[nearest_k]
            # Scale by sqrt(k_ratio) for approximate momentum conservation
            k_ratio = k / nearest_k
            scale_factor = np.sqrt(k_ratio)
            seed_v_rad *= scale_factor
            seed_v_tan *= scale_factor
            
            # Create a dummy candidate for the seed
            seed_cand, _, _ = evaluate_params(
                seed_v_rad, seed_v_tan, **_eval_kwargs(config, k)
            )
            seed = seed_cand
            progress(f"  continuation seed: score={seed.score:.4f} (scale factor={scale_factor:.4f})")
        else:
            # Standard grid search for seed (first k or no continuation)
            seed = seed_best_for_k(k, config, vesc, progress=progress, deadline=deadline)
        
        if deadline is not None and time.monotonic() >= deadline:
            best_by_k[k] = seed
            save_candidate(_payload(seed, vesc), directory=config.library_dir)
            break
        
        refined = refine_velocity_for_k(
            k, (seed.v_rad, seed.v_tan), config, progress=progress
        )
        best = refined if refined.score <= seed.score else seed
        best_by_k[k] = best
        save_candidate(_payload(best, vesc), directory=config.library_dir)
        
        # Store solution for continuation (use best solution)
        optimized_k[k] = (best.v_rad, best.v_tan)

    _write_summary(config, best_by_k, vesc)
    progress("summary written; best per k:")
    for k, c in sorted(best_by_k.items()):
        progress(
            f"  k={k:.3f}  score={c.score:.4f}  "
            f"alpha={c.v_rad / vesc:.4f}  beta={c.v_tan / vesc:.4f}"
        )
    return best_by_k


def _write_summary(
    config: PerKOptimizeConfig,
    best_by_k: dict[float, Candidate],
    vesc: float,
) -> None:
    path = Path(config.summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, c in sorted(best_by_k.items(), key=lambda kv: kv[0]):
        rows.append(
            {
                "k": k,
                "score": c.score,
                "v_rad": c.v_rad,
                "v_tan": c.v_tan,
                "alpha": c.v_rad / vesc,
                "beta": c.v_tan / vesc,
                "fairy_mass": c.fairy_mass,
                "planet_mass": c.planet_mass,
                "permutation_error": c.metrics.get("permutation_error"),
                "energy_drift": c.metrics.get("energy_drift"),
                "min_pair_distance": c.metrics.get("min_pair_distance"),
            }
        )
    payload = {
        "mode": "per_k_optimize_near_1",
        "vesc": vesc,
        "planet_mass": config.planet_mass,
        "k_values": config.k_values,
        "best_by_k": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
