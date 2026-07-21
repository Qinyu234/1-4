"""Generate score(v_rad, v_tan) heatmap for diagnostic visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.search.grid import evaluate_params

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output"


def generate_heatmap(
    k: float,
    vesc: float,
    alpha_range: tuple[float, float],
    beta_range: tuple[float, float],
    n_grid: int,
    planet_mass: float = 1.0,
    radius: float = 20.0,
    G: float = 1.0,
    n_periods: float = 5.0,
    steps_per_period: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate score heatmap over (alpha, beta) grid."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_grid)
    betas = np.linspace(beta_range[0], beta_range[1], n_grid)
    
    scores = np.zeros((n_grid, n_grid))
    
    print(f"Generating {n_grid}×{n_grid} heatmap for k={k:.6f}")
    print(f"Alpha range: [{alpha_range[0]:.3f}, {alpha_range[1]:.3f}]")
    print(f"Beta range: [{beta_range[0]:.3f}, {beta_range[1]:.3f}]")
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            v_rad = alpha * vesc
            v_tan = beta * vesc
            cand, _, _ = evaluate_params(
                v_rad,
                v_tan,
                planet_mass=planet_mass,
                mass_ratio=k,
                radius=radius,
                G=G,
                n_periods=n_periods,
                steps_per_period=steps_per_period,
                record_every=3,
            )
            scores[i, j] = cand.score
            if (i + 1) * (j + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_grid} × {j+1}/{n_grid}")
    
    return alphas, betas, scores


def plot_heatmap(
    alphas: np.ndarray,
    betas: np.ndarray,
    scores: np.ndarray,
    k: float,
    output_path: Path,
) -> None:
    """Plot and save the heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use log scale for scores if they span multiple orders of magnitude
    score_min = scores.min()
    score_max = scores.max()
    use_log = score_max / score_min > 100
    
    if use_log:
        im = ax.contourf(alphas, betas, scores, levels=50, cmap='viridis')
        im = ax.contourf(alphas, betas, np.log10(scores + 1e-10), levels=50, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('log10(score)')
        title_suffix = " (log scale)"
    else:
        im = ax.contourf(alphas, betas, scores, levels=50, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('score')
        title_suffix = ""
    
    # Mark minimum
    min_idx = np.unravel_index(scores.argmin(), scores.shape)
    min_alpha = alphas[min_idx[0]]
    min_beta = betas[min_idx[1]]
    min_score = scores[min_idx]
    ax.plot(min_alpha, min_beta, 'r*', markersize=15, label=f'Min: {min_score:.2f}')
    
    ax.set_xlabel(r'$\alpha = v_{rad}/v_{esc}$')
    ax.set_ylabel(r'$\beta = v_{tan}/v_{esc}$')
    ax.set_title(f'Score Heatmap for k={k:.6f} (ln(k)={np.log(k):.4f}){title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate score heatmap for diagnostic")
    parser.add_argument(
        "--k",
        type=float,
        default=0.135335,
        help="Mass ratio k (default: 0.135335, best from logrange search)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.0,
        help="Minimum alpha (v_rad/v_esc)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=1.0,
        help="Maximum alpha (v_rad/v_esc)",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.0,
        help="Minimum beta (v_tan/v_esc)",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=1.0,
        help="Maximum beta (v_tan/v_esc)",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=30,
        help="Grid resolution (n×n)",
    )
    parser.add_argument(
        "--n-periods",
        type=float,
        default=5.0,
        help="Number of orbital periods to simulate",
    )
    parser.add_argument(
        "--steps-per-period",
        type=int,
        default=60,
        help="Integration steps per period",
    )
    args, _ = parser.parse_known_args()

    # Load config from previous run if available
    config_path = OUT / "k_logrange_summary.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("best_by_k"):
            # Find the k closest to requested k
            best_k = min(
                summary["best_by_k"],
                key=lambda r: abs(r["k"] - args.k),
            )
            print(f"Using parameters from summary for k={best_k['k']:.6f}")
            args.k = best_k["k"]
            # Adjust ranges based on best alpha, beta
            best_alpha = best_k["alpha"]
            best_beta = best_k["beta"]
            args.alpha_min = max(0.0, best_alpha - 0.3)
            args.alpha_max = min(1.5, best_alpha + 0.3)
            args.beta_min = max(0.0, best_beta - 0.3)
            args.beta_max = min(1.5, best_beta + 0.3)
            print(f"Adjusted ranges: alpha [{args.alpha_min:.3f}, {args.alpha_max:.3f}], beta [{args.beta_min:.3f}, {args.beta_max:.3f}]")

    OUT.mkdir(parents=True, exist_ok=True)

    vesc = escape_speed(1.0, 1.0, 20.0)  # G=1, M=1, R=20
    
    alphas, betas, scores = generate_heatmap(
        k=args.k,
        vesc=vesc,
        alpha_range=(args.alpha_min, args.alpha_max),
        beta_range=(args.beta_min, args.beta_max),
        n_grid=args.n_grid,
        n_periods=args.n_periods,
        steps_per_period=args.steps_per_period,
    )
    
    output_path = OUT / f"heatmap_k_{args.k:.6f}.png"
    plot_heatmap(alphas, betas, scores, args.k, output_path)
    
    # Save data
    data_path = OUT / f"heatmap_k_{args.k:.6f}.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({
            "k": args.k,
            "ln_k": float(np.log(args.k)),
            "alphas": alphas.tolist(),
            "betas": betas.tolist(),
            "scores": scores.tolist(),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "min_alpha": float(alphas[np.unravel_index(scores.argmin(), scores.shape)[0]]),
            "min_beta": float(betas[np.unravel_index(scores.argmin(), scores.shape)[1]]),
        }, f, indent=2)
    print(f"Data saved to {data_path}")


if __name__ == "__main__":
    main()
