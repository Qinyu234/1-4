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
from fairy_orbit.observe.choreography_verify import (
    accept_seed_choreography,
    is_regular_equal_ngon,
    maintains_regular_equal_ngon,
)


def test_polygon_force_factor_n4():
    assert polygon_force_factor(4) == pytest.approx(1.0 / math.sqrt(2.0) + 0.25)


def test_maintained_regular_ngon_rejected():
    seed = build_free_polygon_seed(4, seed_id="tmp4", family="free_4")
    assert is_regular_equal_ngon(seed.positions)
    assert maintains_regular_equal_ngon(seed.to_system(), seed.period)
    acc = accept_seed_choreography(seed, atol_rel=1e-5)
    assert acc.choreography.ok  # §3.2 can pass
    assert acc.maintains_regular_ngon
    assert not acc.ok
    assert acc.reason == "rejected_maintained_regular_ngon"


def test_regular_pentagon_maintained_rejected():
    seed = build_free_polygon_seed(5, seed_id="tmp5", family="free_5")
    assert maintains_regular_equal_ngon(seed.to_system(), seed.period)
    acc = accept_seed_choreography(seed, atol_rel=1e-5)
    assert not acc.ok
    assert acc.reason == "rejected_maintained_regular_ngon"


def test_instantaneous_polygon_ic_does_not_force_reject():
    """Scrambled velocities: looks regular at t=0 but does not stay RE."""
    seed = build_free_polygon_seed(4, seed_id="tmp4b", family="free_4")
    v = np.asarray(seed.velocities, dtype=float).copy()
    v[0] *= 1.7
    v[1] *= 0.4
    v[2] += np.array([0.3, -0.2, 0.0])
    broken = OrbitSeed(
        id="tmp4_broken_v",
        family=seed.family,
        n_bodies=seed.n_bodies,
        G=seed.G,
        masses=seed.masses,
        period=seed.period,
        positions=seed.positions,
        velocities=v,
        names=seed.names,
        symmetry=seed.symmetry,
        source="test",
        notes="scrambled v",
        central_index=None,
    )
    assert is_regular_equal_ngon(broken.positions)
    assert not maintains_regular_equal_ngon(broken.to_system(), broken.period)
    acc = accept_seed_choreography(broken, atol_rel=1e-5)
    assert not acc.maintains_regular_ngon
    assert acc.reason != "rejected_maintained_regular_ngon"


def test_hier_seed_has_central():
    seed = build_hier_1plus4_manifold_seed()
    assert seed.central_index == 0
    assert seed.n_bodies == 5
    assert_com_frame(seed, atol=1e-9)


def test_regenerate_writes_catalogue_without_polygon_re():
    seeds = regenerate_canonical_seeds()
    assert len(seeds) == 1
    cat = load_catalogue()
    ids = {e["id"] for e in cat["seeds"]}
    assert ids == {"hier_1plus4_manifold"}
    assert not (SEEDS_DIR / "free_4_square_re.json").exists()
    assert not (SEEDS_DIR / "free_5_pentagon_re.json").exists()
