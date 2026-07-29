"""Tests for equal-mass continuation seed catalogue."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fairy_orbit.design.seeds import (
    OrbitSeed,
    assert_com_frame,
    build_free_polygon_seed,
    build_hier_1plus4_manifold_seed,
    load_catalogue,
    load_seed,
    polygon_force_factor,
    regenerate_canonical_seeds,
    role_shifted_positions,
    shape_congruence_residual,
    verify_free_shape_congruence,
    verify_seed_model,
    SEEDS_DIR,
)
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate


def test_polygon_force_factor_n4():
    # S_4 = 1/(4*sin(π/4)) + 1/(4*sin(π/2)) + 1/(4*sin(3π/4))
    #     = 1/(2√2) + 1/4 + 1/(2√2) = 1/√2 + 1/4
    assert polygon_force_factor(4) == pytest.approx(1.0 / math.sqrt(2.0) + 0.25)


def test_free_4_com_and_period():
    seed = build_free_polygon_seed(4, seed_id="free_4_square_re", family="free_4")
    assert_com_frame(seed)
    assert seed.n_bodies == 4
    assert seed.period == pytest.approx(
        2.0 * math.pi / math.sqrt(1.0 / math.sqrt(2.0) + 0.25)
    )


def test_free_5_com():
    seed = build_free_polygon_seed(5, seed_id="free_5_pentagon_re", family="free_5")
    assert_com_frame(seed)
    assert seed.n_bodies == 5
    assert seed.period > 0


def test_hier_seed_has_central():
    seed = build_hier_1plus4_manifold_seed()
    assert seed.central_index == 0
    assert seed.n_bodies == 5
    assert seed.masses[0] == pytest.approx(1.0)
    assert_com_frame(seed, atol=1e-9)


def test_roundtrip_dict():
    seed = build_free_polygon_seed(4, seed_id="t", family="free_4")
    s2 = OrbitSeed.from_dict(seed.to_dict())
    assert s2.id == seed.id
    assert np.allclose(s2.positions, seed.positions)


def test_instant_cyclic_shape_congruence():
    seed = build_free_polygon_seed(4, seed_id="free_4_square_re", family="free_4")
    r = shape_congruence_residual(
        seed.positions, role_shifted_positions(seed.positions, 1)
    )
    v = shape_congruence_residual(
        seed.velocities, role_shifted_positions(seed.velocities, 1)
    )
    assert r < 1e-12
    assert v < 1e-12


def test_free_shape_congruence_with_traj():
    seed = build_free_polygon_seed(4, seed_id="free_4_square_re", family="free_4")
    traj = integrate(
        seed.to_system(),
        t_end=float(seed.period),
        n_outputs=120,
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=1e-9,
            min_dt=1e-8,
        ),
    )
    out = verify_free_shape_congruence(seed, traj=traj, atol=1e-5)
    assert out["ok"]
    assert out["shape_residual_max_r"] < 1e-5
    assert out["shape_residual_max_v"] < 1e-5
    assert out["kabsch_rel_max_v"] < 1e-4


def test_verify_seed_model_hier_ok():
    seed = build_hier_1plus4_manifold_seed()
    assert verify_seed_model(seed)["ok"]


def test_regenerate_writes_catalogue():
    seeds = regenerate_canonical_seeds()
    assert len(seeds) == 3
    cat = load_catalogue()
    ids = {e["id"] for e in cat["seeds"]}
    assert ids == {"free_4_square_re", "free_5_pentagon_re", "hier_1plus4_manifold"}
    by_id = {e["id"]: e for e in cat["seeds"]}
    assert by_id["free_4_square_re"]["orbit_class"] == "free_relative_equilibrium"
    assert by_id["free_5_pentagon_re"]["orbit_class"] == "free_relative_equilibrium"
    assert by_id["hier_1plus4_manifold"]["orbit_class"] == "hier_baseline_ic"
    for e in cat["seeds"]:
        s = load_seed(SEEDS_DIR / e["path"])
        assert s.id == e["id"]
