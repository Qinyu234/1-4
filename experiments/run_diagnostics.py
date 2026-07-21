"""Diagnostic experiments to analyze optimization landscape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.analysis.evaluator import (
    EvalResult,
    EvaluatorConfig,
    collision_penalty,
    distance_matrix_error,
    energy_drift,
    evaluate,
    permutation_error,
)
from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.search.grid import evaluate_params
from fairy_orbit.simulation.runner import Trajectory, run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def load_best_solution(summary_path: Path) -> dict:
    """Load the best solution from summary JSON."""
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    
    # Find global best
    best_row = min(summary["best_by_k"], key=lambda r: r["score"])
    return {
        "k": best_row["k"],
        "v_rad": best_row["v_rad"],
        "v_tan": best_row["v_tan"],
        "alpha": best_row["alpha"],
        "beta": best_row["beta"],
        "score": best_row["score"],
        "planet_mass": summary.get("planet_mass", 1.0),
        "vesc": summary.get("vesc", 0.316228),
    }


def experiment1_score_v_rad(best: dict, n_points: int = 200) -> None:
    """Experiment 1: Score(v_rad) sweep ±30% around optimum."""
    print("Running Experiment 1: Score(v_rad) sweep")
    
    v_rad_center = best["v_rad"]
    v_rad_min = v_rad_center * 0.7
    v_rad_max = v_rad_center * 1.3
    v_rad_values = np.linspace(v_rad_min, v_rad_max, n_points)
    
    scores = []
    perm_errors = []
    dist_errors = []
    collision_penalties = []
    energy_drifts = []
    
    for v_rad in v_rad_values:
        cand, traj, result = evaluate_params(
            v_rad,
            best["v_tan"],
            planet_mass=best["planet_mass"],
            mass_ratio=best["k"],
            n_periods=2.0,
            steps_per_period=60,
            record_every=3,
            eval_config=EvaluatorConfig(use_distance_matrix=True),
        )
        scores.append(cand.score)
        perm_errors.append(result.permutation_error)
        dist_errors.append(result.distance_matrix_error)
        collision_penalties.append(result.collision_penalty)
        energy_drifts.append(result.energy_drift)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    ax = axes[0]
    ax.plot(v_rad_values, scores, "b-", linewidth=2, label="Total score")
    ax.axvline(v_rad_center, color="r", linestyle="--", alpha=0.7, label="Optimum")
    ax.set_xlabel("v_rad")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 1: Score(v_rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(v_rad_values, perm_errors, "g-", label="Permutation error")
    ax.plot(v_rad_values, dist_errors, "m-", label="Distance matrix error")
    ax.plot(v_rad_values, collision_penalties, "orange", label="Collision penalty")
    ax.plot(v_rad_values, energy_drifts, "c-", label="Energy drift")
    ax.axvline(v_rad_center, color="r", linestyle="--", alpha=0.7)
    ax.set_xlabel("v_rad")
    ax.set_ylabel("Component value")
    ax.set_title("Score components vs v_rad")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT / "exp1_score_v_rad.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT / 'exp1_score_v_rad.png'}")


def experiment2_score_v_tan(best: dict, n_points: int = 200) -> None:
    """Experiment 2: Score(v_tan) sweep ±30% around optimum."""
    print("Running Experiment 2: Score(v_tan) sweep")
    
    v_tan_center = best["v_tan"]
    v_tan_min = v_tan_center * 0.7
    v_tan_max = v_tan_center * 1.3
    v_tan_values = np.linspace(v_tan_min, v_tan_max, n_points)
    
    scores = []
    perm_errors = []
    dist_errors = []
    collision_penalties = []
    energy_drifts = []
    
    for v_tan in v_tan_values:
        cand, traj, result = evaluate_params(
            best["v_rad"],
            v_tan,
            planet_mass=best["planet_mass"],
            mass_ratio=best["k"],
            n_periods=2.0,
            steps_per_period=60,
            record_every=3,
            eval_config=EvaluatorConfig(use_distance_matrix=True),
        )
        scores.append(cand.score)
        perm_errors.append(result.permutation_error)
        dist_errors.append(result.distance_matrix_error)
        collision_penalties.append(result.collision_penalty)
        energy_drifts.append(result.energy_drift)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    ax = axes[0]
    ax.plot(v_tan_values, scores, "b-", linewidth=2, label="Total score")
    ax.axvline(v_tan_center, color="r", linestyle="--", alpha=0.7, label="Optimum")
    ax.set_xlabel("v_tan")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 2: Score(v_tan)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(v_tan_values, perm_errors, "g-", label="Permutation error")
    ax.plot(v_tan_values, dist_errors, "m-", label="Distance matrix error")
    ax.plot(v_tan_values, collision_penalties, "orange", label="Collision penalty")
    ax.plot(v_tan_values, energy_drifts, "c-", label="Energy drift")
    ax.axvline(v_tan_center, color="r", linestyle="--", alpha=0.7)
    ax.set_xlabel("v_tan")
    ax.set_ylabel("Component value")
    ax.set_title("Score components vs v_tan")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT / "exp2_score_v_tan.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT / 'exp2_score_v_tan.png'}")


def experiment3_2d_landscape(best: dict, n_grid: int = 150) -> None:
    """Experiment 3: 2D landscape heatmap score(v_rad, v_tan)."""
    print("Running Experiment 3: 2D landscape heatmap")
    
    v_rad_center = best["v_rad"]
    v_tan_center = best["v_tan"]
    
    v_rad_min = v_rad_center * 0.7
    v_rad_max = v_rad_center * 1.3
    v_tan_min = v_tan_center * 0.7
    v_tan_max = v_tan_center * 1.3
    
    v_rad_values = np.linspace(v_rad_min, v_rad_max, n_grid)
    v_tan_values = np.linspace(v_tan_min, v_tan_max, n_grid)
    
    scores = np.zeros((n_grid, n_grid))
    
    for i, v_rad in enumerate(v_rad_values):
        for j, v_tan in enumerate(v_tan_values):
            cand, _, _ = evaluate_params(
                v_rad,
                v_tan,
                planet_mass=best["planet_mass"],
                mass_ratio=best["k"],
                n_periods=2.0,
                steps_per_period=60,
                record_every=3,
                eval_config=EvaluatorConfig(use_distance_matrix=True),
            )
            scores[i, j] = cand.score
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_grid}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use log scale for better visualization if range is large
    if scores.max() / scores.min() > 100:
        scores_plot = np.log10(scores + 1e-10)
        cbar_label = "log10(Score)"
    else:
        scores_plot = scores
        cbar_label = "Score"
    
    im = ax.imshow(scores_plot, extent=[v_tan_min, v_tan_max, v_rad_min, v_rad_max],
                   origin="lower", aspect="auto", cmap="viridis")
    ax.scatter([v_tan_center], [v_rad_center], c="red", s=100, marker="*", 
               label="Optimum", zorder=5)
    ax.set_xlabel("v_tan")
    ax.set_ylabel("v_rad")
    ax.set_title("Experiment 3: 2D Landscape score(v_rad, v_tan)")
    ax.legend()
    plt.colorbar(im, ax=ax, label=cbar_label)
    
    fig.tight_layout()
    fig.savefig(OUT / "exp3_2d_landscape.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT / 'exp3_2d_landscape.png'}")
    
    # Save data
    np.savez(OUT / "exp3_2d_landscape_data.npz",
             v_rad_values=v_rad_values,
             v_tan_values=v_tan_values,
             scores=scores)


def experiment4_time_scan(best: dict, n_points: int = 300) -> None:
    """Experiment 4: Time scan - sweep simulation end time T (MOST IMPORTANT)."""
    print("Running Experiment 4: Time scan (MOST IMPORTANT)")
    
    # Get base simulation parameters
    system = generate_system(
        best["v_rad"],
        best["v_tan"],
        planet_mass=best["planet_mass"],
        fairy_mass=best["k"] * best["planet_mass"],
        radius=20.0,
        G=1.0,
    )
    
    period = orbital_period(1.0, best["planet_mass"], 20.0)
    T0 = 2.0 * period  # Base simulation time (2 periods)
    
    t_min = 0.5 * T0
    t_max = 1.5 * T0
    t_values = np.linspace(t_min, t_max, n_points)
    
    scores = []
    dist_errors = []
    
    # Run simulation once with max time
    dt = period / 60
    traj = run(system, dt=dt, t_end=t_max, record_every=1)
    
    for t in t_values:
        # Find closest time index
        t_idx = int(np.argmin(np.abs(traj.times - t)))
        result = evaluate(traj, config=EvaluatorConfig(use_distance_matrix=True), period_index=t_idx)
        scores.append(result.score)
        dist_errors.append(result.distance_matrix_error)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    ax = axes[0]
    ax.plot(t_values, scores, "b-", linewidth=2, label="Total score")
    ax.axvline(T0, color="r", linestyle="--", alpha=0.7, label="Base T0")
    ax.set_xlabel("Simulation time T")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 4: Score(T) - Time scan")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t_values, dist_errors, "m-", linewidth=2, label="Distance matrix error")
    ax.axvline(T0, color="r", linestyle="--", alpha=0.7)
    ax.set_xlabel("Simulation time T")
    ax.set_ylabel("Distance matrix error")
    ax.set_title("Distance matrix error vs T")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT / "exp4_time_scan.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT / 'exp4_time_scan.png'}")
    
    # Save data
    np.savez(OUT / "exp4_time_scan_data.npz",
             t_values=t_values,
             scores=scores,
             dist_errors=dist_errors)


def experiment5_trajectory_diagnostics(best: dict) -> None:
    """Experiment 5: Trajectory diagnostics - paths, distances, energy, angular momentum."""
    print("Running Experiment 5: Trajectory diagnostics")
    
    cand, traj, _ = evaluate_params(
        best["v_rad"],
        best["v_tan"],
        planet_mass=best["planet_mass"],
        mass_ratio=best["k"],
        n_periods=2.0,
        steps_per_period=60,
        record_every=1,
        eval_config=EvaluatorConfig(use_distance_matrix=True),
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Orbital paths (3D projection)
    ax = axes[0, 0]
    for i in range(5):  # Planet + 4 fairies
        ax.plot(traj.positions[:, i, 0], traj.positions[:, i, 1], 
                label=f"Body {i}", linewidth=1.5)
        ax.scatter(traj.positions[0, i, 0], traj.positions[0, i, 1], 
                   marker="o", s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Orbital paths (xy projection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pairwise distances over time
    ax = axes[0, 1]
    n_bodies = traj.positions.shape[1]
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            distances = np.linalg.norm(traj.positions[:, i] - traj.positions[:, j], axis=1)
            ax.plot(traj.times, distances, label=f"D_{i}{j}", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pairwise distance")
    ax.set_title("Pairwise distances vs time")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total energy over time
    ax = axes[1, 0]
    ax.plot(traj.times, traj.energies, "b-", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total energy")
    ax.set_title("Total energy vs time")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Angular momentum (z-component) over time
    ax = axes[1, 1]
    L_z = []
    for t in range(len(traj.times)):
        pos = traj.positions[t]
        vel = traj.velocities[t]
        # L = r × p = r × mv
        L = np.cross(pos, vel)
        L_z.append(L[:, 2].sum())  # Sum of z-components
    ax.plot(traj.times, L_z, "g-", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total L_z")
    ax.set_title("Total angular momentum (z) vs time")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT / "exp5_trajectory_diagnostics.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT / 'exp5_trajectory_diagnostics.png'}")


def experiment7_poincare_section(best: dict) -> None:
    """Experiment 7: Poincaré section - record (x, vx) when z crosses zero with positive vz."""
    print("Running Experiment 7: Poincaré section")
    
    # Run longer simulation for better statistics
    cand, traj, _ = evaluate_params(
        best["v_rad"],
        best["v_tan"],
        planet_mass=best["planet_mass"],
        mass_ratio=best["k"],
        n_periods=10.0,  # Longer simulation
        steps_per_period=60,
        record_every=1,
        eval_config=EvaluatorConfig(use_distance_matrix=True),
    )
    
    # Find z crossings with positive vz for fairy 1 (index 1)
    fairy_idx = 1
    x_crossings = []
    vx_crossings = []
    
    for i in range(1, len(traj.times)):
        z_prev = traj.positions[i-1, fairy_idx, 2]
        z_curr = traj.positions[i, fairy_idx, 2]
        vz_prev = traj.velocities[i-1, fairy_idx, 2]
        vz_curr = traj.velocities[i, fairy_idx, 2]
        
        # Check for zero crossing with positive velocity
        if z_prev <= 0 and z_curr > 0 and vz_curr > 0:
            # Linear interpolation for better accuracy
            frac = -z_prev / (z_curr - z_prev)
            x_interp = traj.positions[i-1, fairy_idx, 0] + frac * (traj.positions[i, fairy_idx, 0] - traj.positions[i-1, fairy_idx, 0])
            vx_interp = traj.velocities[i-1, fairy_idx, 0] + frac * (traj.velocities[i, fairy_idx, 0] - traj.velocities[i-1, fairy_idx, 0])
            x_crossings.append(x_interp)
            vx_crossings.append(vx_interp)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_crossings, vx_crossings, s=20, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("vx")
    ax.set_title(f"Experiment 7: Poincaré section (fairy {fairy_idx}, z=0, vz>0)")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT / "exp7_poincare_section.png", dpi=140)
    plt.close(fig)
    print(f"Saved: {OUT / 'exp7_poincare_section.png'}")
    print(f"  Found {len(x_crossings)} crossings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diagnostic experiments")
    parser.add_argument(
        "--summary",
        type=str,
        default="experiments/output/improved_search_summary.json",
        help="Path to summary JSON with best solution",
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3, 4, 5, 7],
        help="Run specific experiment (default: all)",
    )
    args, _ = parser.parse_known_args()
    
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        return
    
    best = load_best_solution(summary_path)
    print(f"Best solution: k={best['k']:.6f}, score={best['score']:.4f}")
    print(f"  v_rad={best['v_rad']:.6f}, v_tan={best['v_tan']:.6f}")
    print(f"  alpha={best['alpha']:.4f}, beta={best['beta']:.4f}")
    
    experiments = {
        1: lambda: experiment1_score_v_rad(best),
        2: lambda: experiment2_score_v_tan(best),
        3: lambda: experiment3_2d_landscape(best),
        4: lambda: experiment4_time_scan(best),
        5: lambda: experiment5_trajectory_diagnostics(best),
        7: lambda: experiment7_poincare_section(best),
    }
    
    if args.exp:
        print(f"\nRunning Experiment {args.exp} only")
        experiments[args.exp]()
    else:
        print("\nRunning all experiments")
        for exp_num in sorted(experiments.keys()):
            print(f"\n{'='*50}")
            experiments[exp_num]()
    
    print(f"\nAll diagnostic plots saved to: {OUT}")


if __name__ == "__main__":
    main()
