import numpy as np

from fairy_orbit.physics.body import Body, System
from fairy_orbit.simulation.trajectory import Trajectory

from experiments import compare_solvers


def test_score_breakdown_contains_position_and_event_components():
    system = System(
        bodies=[
            Body(mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], name="planet"),
            Body(mass=0.01, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.1, 0.0], name="fairy_1"),
            Body(mass=0.01, position=[0.0, 1.0, 0.0], velocity=[-0.1, 0.0, 0.0], name="fairy_2"),
            Body(mass=0.01, position=[-1.0, 0.0, 0.0], velocity=[0.0, -0.1, 0.0], name="fairy_3"),
            Body(mass=0.01, position=[0.0, -1.0, 0.0], velocity=[0.1, 0.0, 0.0], name="fairy_4"),
        ],
        G=1.0,
    )

    traj = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        positions=np.array([
            system.positions(),
            system.positions(),
            system.positions(),
        ]),
        velocities=np.array([
            system.velocities(),
            system.velocities(),
            system.velocities(),
        ]),
        energies=np.array([1.0, 1.0, 1.0]),
        angular_momenta=np.zeros((3, 3)),
        labels=system.labels,
        G=system.G,
        masses=system.masses(),
    )

    entry = compare_solvers.score_breakdown("demo", traj, system, dt=1.0, t_end=2.0)

    assert "position" in entry
    assert "event" in entry
    assert set(entry["position"].keys()) >= {"score", "permutation_error", "collision_penalty", "energy_drift"}
    assert set(entry["event"].keys()) >= {"total_score", "time_variance", "event_sequence_error", "event_count_difference"}


def test_default_simulation_duration_uses_10000_hours():
    assert compare_solvers.default_simulation_duration() == 10000.0


def test_score_breakdown_returns_component_dicts():
    system = System(
        bodies=[
            Body(mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], name="planet"),
            Body(mass=0.01, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.1, 0.0], name="fairy_1"),
            Body(mass=0.01, position=[0.0, 1.0, 0.0], velocity=[-0.1, 0.0, 0.0], name="fairy_2"),
            Body(mass=0.01, position=[-1.0, 0.0, 0.0], velocity=[0.0, -0.1, 0.0], name="fairy_3"),
            Body(mass=0.01, position=[0.0, -1.0, 0.0], velocity=[0.1, 0.0, 0.0], name="fairy_4"),
        ],
        G=1.0,
    )

    traj = Trajectory(
        times=np.array([0.0, 1.0, 2.0]),
        positions=np.array([system.positions(), system.positions(), system.positions()]),
        velocities=np.array([system.velocities(), system.velocities(), system.velocities()]),
        energies=np.array([1.0, 1.0, 1.0]),
        angular_momenta=np.zeros((3, 3)),
        labels=system.labels,
        G=system.G,
        masses=system.masses(),
    )

    entry = compare_solvers.score_breakdown("demo", traj, system, dt=1.0, t_end=2.0)

    assert entry["position"]["score"] >= 0.0
    assert entry["event"]["total_score"] >= 0.0
