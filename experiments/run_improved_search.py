"""Improved orbit search using distance matrix loss, T-scanning, and continuation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.search.grid import evaluate_params
from fairy_orbit.search.k_velocity import PerKOptimizeConfig, optimize_each_k
from fairy_orbit.visualization.plots import plot_energy_error, plot_trajectories

OUT = ROOT / "experiments" / "output"
LIB = ROOT / "orbit_library"


def generate_k_values(n_points: int = 10, ln_k_min: float = -3.5, ln_k_max: float = 3.5) -> list[float]:
    """Generate k values uniformly spaced in ln(k) space."""
    k_min = np.exp(ln_k_min)
    k_max = np.exp(ln_k_max)
    k_values = np.logspace(np.log10(k_min), np.log10(k_max), n_points).tolist()
    return k_values


def plot_best_vs_k(summary: dict, path: Path) -> None:
    rows = summary["best_by_k"]
    if not rows:
        return
    ks = [r["k"] for r in rows]
    scores = [r["score"] for r in rows]
    alphas = [r["alpha"] for r in rows]
    betas = [r["beta"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].semilogx(ks, scores, "o-")
    axes[0].set_xlabel("k (fairy = kM)")
    axes[0].set_ylabel("best score")
    axes[0].set_title("Optimized score vs k (log scale)")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(ks, alphas, "o-", label=r"$\alpha=v_{rad}/v_{esc}$")
    axes[1].semilogx(ks, betas, "s-", label=r"$\beta=v_{tan}/v_{esc}$")
    axes[1].set_xlabel("k (log scale)")
    axes[1].set_ylabel("velocity / v_esc")
    axes[1].set_title("Best velocity fractions vs k")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    # Plot ln(k) vs score for linear view
    ln_ks = [np.log(k) for k in ks]
    axes[2].plot(ln_ks, scores, "o-")
    axes[2].set_xlabel("ln(k)")
    axes[2].set_ylabel("best score")
    axes[2].set_title("Optimized score vs ln(k)")
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved orbit search with distance matrix loss and continuation")
    parser.add_argument(
        "--n-points",
        type=int,
        default=10,
        help="Number of k values to sample (default: 00)",
    )
    parser.add_argument(
        "--ln-k-min",
        type=float,
        default=-3.5,
        help="Minimum ln(k) value (default: -3.5)",
    )
    parser.add_argument(
        "--ln-k-max",
        type=float,
        default=3.5,
        help="Maximum ln(k) value (default: 3.5)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=20.0,
        help="Orbital radius R (10 to 100, default: 20.0)",
    )
    parser.add_argument("--n-periods", type=float, default=12.0)
    parser.add_argument("--refine-maxiter", type=int, default=40)
    parser.add_argument("--smoke", action="store_true", help="Tiny grids for CI")
    
    # New options for improved search
    parser.add_argument("--use-distance-matrix", action="store_true", 
                       help="Use continuous distance matrix loss instead of permutation")
    parser.add_argument("--use-continuation", action="store_true",
                       help="Use parameter continuation across k values")
    parser.add_argument("--t-scan-range", type=float, nargs=2, default=None,
                       help="Scan T around period, e.g., --t-scan-range 0.8 1.2")
    parser.add_argument("--t-scan-steps", type=int, default=5,
                       help="Number of T scan points (default: 5)")
    parser.add_argument("--solver-type", choices=["own", "rebound"], default="own",
                       help="Choose integration backend (default: own)")
    parser.add_argument("--evaluation-mode", choices=["position", "event"], default="event",
                       help="Choose evaluation strategy (default: event)")
    
    args, _ = parser.parse_known_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LIB.mkdir(parents=True, exist_ok=True)

    k_values = generate_k_values(args.n_points, args.ln_k_min, args.ln_k_max)
    print(f"Generated {len(k_values)} k values: {k_values}")

    t_scan_range = tuple(args.t_scan_range) if args.t_scan_range else None

    if args.smoke:
        cfg = PerKOptimizeConfig(
            k_values=k_values[:3],  # Only first 3 for smoke test
            radius=args.radius,
            alpha_grid=[-0.2, 0.2],
            beta_grid=[0.4, 0.8],
            n_periods=0.5,
            steps_per_period=30,
            record_every=2,
            refine_maxiter=5,
            library_dir=str(LIB),
            summary_path=str(OUT / "improved_search_summary.json"),
            use_distance_matrix=args.use_distance_matrix,
            use_continuation=args.use_continuation,
            t_scan_range=t_scan_range,
            t_scan_steps=args.t_scan_steps,
            solver_type=args.solver_type,
        )
    else:
        cfg = PerKOptimizeConfig(
            k_values=k_values,
            radius=args.radius,
            n_periods=args.n_periods,
            refine_maxiter=args.refine_maxiter,
            library_dir=str(LIB),
            summary_path=str(OUT / "improved_search_summary.json"),
            use_distance_matrix=args.use_distance_matrix,
            use_continuation=args.use_continuation,
            t_scan_range=t_scan_range,
            t_scan_steps=args.t_scan_steps,
            solver_type=args.solver_type,
            evaluation_mode=args.evaluation_mode,
        )

    log_path = OUT / "improved_search_run.log"
    log_path.write_text("", encoding="utf-8")

    def progress(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    progress(f"Starting improved search with:")
    progress(f"  - Distance matrix loss: {args.use_distance_matrix}")
    progress(f"  - Parameter continuation: {args.use_continuation}")
    progress(f"  - T scanning: {t_scan_range}")
    progress(f"  - Solver: {args.solver_type}")
    progress(f"  - Evaluation mode: {args.evaluation_mode}")
    
    best_by_k = optimize_each_k(cfg, progress=progress)
    summary = json.loads(Path(cfg.summary_path).read_text(encoding="utf-8"))
    plot_best_vs_k(summary, OUT / "improved_search_best_vs_k.png")

    if not best_by_k:
        return

    # Plot trajectory of globally best k
    best_k = min(best_by_k, key=lambda k: best_by_k[k].score)
    best = best_by_k[best_k]
    vesc = escape_speed(cfg.G, cfg.planet_mass, cfg.radius)
    progress(
        f"global best: k={best_k:.6f} ln(k)={np.log(best_k):.4f} score={best.score:.4f} "
        f"alpha={best.v_rad / vesc:.4f} beta={best.v_tan / vesc:.4f}"
    )
    
    # Re-run with full trajectory for visualization
    _, traj, _ = evaluate_params(
        best.v_rad,
        best.v_tan,
        planet_mass=cfg.planet_mass,
        mass_ratio=best_k,
        radius=cfg.radius,
        G=cfg.G,
        n_periods=min(cfg.n_periods, 5.0),
        steps_per_period=cfg.steps_per_period,
        record_every=cfg.record_every,
        eval_config=None,  # Use default for visualization
        solver_type=cfg.solver_type,
    )
    plot_trajectories(traj, OUT / "improved_search_best_trajectories.png")
    plot_energy_error(traj, OUT / "improved_search_best_energy.png")
    progress(f"plots under {OUT}")


if __name__ == "__main__":
    main()
