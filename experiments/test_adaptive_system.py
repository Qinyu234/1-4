"""Test script for the new event-driven adaptive system."""

from pathlib import Path

from fairy_orbit.analysis.periodicity_evaluator import PeriodicityConfig
from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.search.adaptive_evaluator import evaluate_adaptive
from fairy_orbit.simulation.adaptive_simulator import AdaptiveConfig
from fairy_orbit.visualization.event_plots import plot_all_diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output" / "adaptive_test"
OUT.mkdir(parents=True, exist_ok=True)


def test_adaptive_system():
    """Test the new adaptive event-driven system."""
    print("Testing adaptive event-driven system...")
    
    # Use the best solution from previous search
    k = 1.649  # Best from extended search
    planet_mass = 1.0
    radius = 20.0
    vesc = escape_speed(1.0, planet_mass, radius)
    
    # Best velocities from previous search
    alpha = 1.0881
    beta = 0.3582
    v_rad = alpha * vesc
    v_tan = beta * vesc
    
    print(f"Testing with k={k}, v_rad={v_rad:.6f}, v_tan={v_tan:.6f}")
    
    # Configure adaptive simulator
    adaptive_config = AdaptiveConfig(
        influence_threshold=0.1,
        event_tolerance=1e-4,
        max_iterations=50,
    )
    
    # Configure periodicity evaluator
    periodicity_config = PeriodicityConfig(
        w_time=1.0,
        w_event=1.0,
        w_count=1.0,
        w_center=1.0,
        w_energy=1.0,
        min_cycles=2,
    )
    
    # Run adaptive evaluation
    cand, adaptive_result = evaluate_adaptive(
        v_rad,
        v_tan,
        planet_mass=planet_mass,
        mass_ratio=k,
        radius=radius,
        G=1.0,
        n_periods=5.0,
        steps_per_period=60,
        adaptive_config=adaptive_config,
        periodicity_config=periodicity_config,
    )
    
    print(f"\nResults:")
    print(f"  Total score: {adaptive_result.score:.4f}")
    print(f"  Number of events: {adaptive_result.n_events}")
    print(f"  Total time: {adaptive_result.total_time:.4f}")
    print(f"  Center displacement: {adaptive_result.center_displacement:.6f}")
    print(f"  Energy drift: {adaptive_result.energy_drift:.6f}")
    
    print(f"\nScore breakdown:")
    print(f"  Time variance: {adaptive_result.periodicity_score.time_variance:.4f}")
    print(f"  Event sequence error: {adaptive_result.periodicity_score.event_sequence_error:.4f}")
    print(f"  Event count difference: {adaptive_result.periodicity_score.event_count_difference:.4f}")
    print(f"  Center motion error: {adaptive_result.periodicity_score.center_motion_error:.4f}")
    print(f"  Energy drift penalty: {adaptive_result.periodicity_score.energy_drift:.4f}")
    
    # Generate diagnostic plots
    print(f"\nGenerating diagnostic plots...")
    
    # Need to re-run simulation to get events for visualization
    from fairy_orbit.icgen.generator import generate_system
    from fairy_orbit.simulation.adaptive_simulator import AdaptiveSimulator
    
    system = generate_system(
        v_rad,
        v_tan,
        planet_mass=planet_mass,
        fairy_mass=k * planet_mass,
        radius=radius,
        G=1.0,
    )
    
    from fairy_orbit.icgen.generator import orbital_period
    period = orbital_period(1.0, planet_mass, radius)
    dt = period / 60
    t_end = 5.0 * period
    
    simulator = AdaptiveSimulator(adaptive_config)
    events = simulator.run(system, dt, t_end)
    
    print(f"  Generated {len(events)} events")
    
    # Plot diagnostics
    plot_all_diagnostics(events, adaptive_result.periodicity_score, OUT)
    
    print(f"\nTest complete. Results saved to {OUT}")


if __name__ == "__main__":
    test_adaptive_system()
