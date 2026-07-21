"""Quick simulation entry script to run orbits and generate interactive visualisations."""

from __future__ import annotations
import argparse
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.simulation.simulator import simulate
from fairy_orbit.visualization.viewer import create_interactive_viewer

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Fairy Orbit Simulator")
    parser.add_argument("--lnk", type=float, default=0.0, help="ln(k) where k = exp(lnk)")
    parser.add_argument("--radius", type=float, default=20.0, help="Initial orbital radius")
    parser.add_argument("--alpha", type=float, default=0.5, help="v_rad = alpha * escape_velocity")
    parser.add_argument("--beta", type=float, default=0.6, help="v_tan = beta * escape_velocity")
    parser.add_argument("--solver", type=str, default="own", choices=["own", "rebound"], help="Solver/Integrator to use")
    parser.add_argument("--n-periods", type=float, default=5.0, help="Number of periods to simulate")
    parser.add_argument("--steps-per-period", type=int, default=200, help="Number of integration steps per orbital period")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    
    args, _ = parser.parse_known_args()
    
    k = np.exp(args.lnk)
    planet_mass = 1.0
    fairy_mass = k * planet_mass
    G = 1.0
    
    vesc = escape_speed(G, planet_mass, args.radius)
    v_rad = args.alpha * vesc
    v_tan = args.beta * vesc
    
    print(f"Generating system with k={k:.6f} (lnk={args.lnk:.4f}), radius={args.radius:.2f}")
    print(f"Velocities: alpha={args.alpha:.4f} (v_rad={v_rad:.6f}), beta={args.beta:.4f} (v_tan={v_tan:.6f})")
    
    system = generate_system(
        v_rad=v_rad,
        v_tan=v_tan,
        planet_mass=planet_mass,
        fairy_mass=fairy_mass,
        radius=args.radius,
        G=G,
    )
    
    period = orbital_period(G, planet_mass, args.radius)
    dt = period / args.steps_per_period
    t_end = args.n_periods * period
    
    print(f"Running simulation with solver '{args.solver}' for {args.n_periods} periods (t_end={t_end:.2f}, dt={dt:.6f})...")
    
    trajectory = simulate(
        system=system,
        dt=dt,
        t_end=t_end,
        solver_type=args.solver,
        record_every=1,
    )
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    npy_path = out_dir / "trajectory.npy"
    np.save(npy_path, trajectory.positions)
    print(f"Saved trajectory data to {npy_path}")
    
    html_path = out_dir / "orbit.html"
    create_interactive_viewer(trajectory.positions, trajectory.labels, str(html_path))
    print(f"Saved interactive 3D viewer to {html_path}")

if __name__ == "__main__":
    main()
