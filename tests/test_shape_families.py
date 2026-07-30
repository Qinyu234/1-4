"""Tests for shape-family diversity selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed
from fairy_orbit.observe.shape_families import (
    select_diverse_families,
    shape_distance,
    shape_feature_vector,
)
from fairy_orbit.store.search_db import ChoreographySearchStore


def _seed_circle(n: int, scale: float = 1.0, z_amp: float = 0.0) -> OrbitSeed:
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for i in range(n):
        ang = 2 * np.pi * i / n
        pos[i] = (scale * np.cos(ang), scale * np.sin(ang), z_amp * ((-1) ** i))
        vel[i] = (-np.sin(ang), np.cos(ang), 0.1 * z_amp)
    return OrbitSeed(
        id=f"c{n}_{scale}_{z_amp}",
        family=f"free_{n}",
        n_bodies=n,
        G=1.0,
        masses=tuple(1.0 for _ in range(n)),
        period=6.0 * scale,
        positions=pos,
        velocities=vel,
        names=tuple(f"B{i+1}" for i in range(n)),
        symmetry="test",
        source="test",
    )


def _seed_line(n: int) -> OrbitSeed:
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for i in range(n):
        pos[i] = (float(i) - (n - 1) / 2.0, 0.05 * i, 0.0)
        vel[i] = (0.0, 0.4 + 0.1 * i, 0.0)
    return OrbitSeed(
        id=f"line{n}",
        family=f"free_{n}",
        n_bodies=n,
        G=1.0,
        masses=tuple(1.0 for _ in range(n)),
        period=5.0,
        positions=pos,
        velocities=vel,
        names=tuple(f"B{i+1}" for i in range(n)),
        symmetry="test",
        source="test",
    )


def test_shape_feature_differs_for_distinct_geometries():
    a = shape_feature_vector(_seed_circle(4, 1.0, 0.0))
    b = shape_feature_vector(_seed_line(4))
    c = shape_feature_vector(_seed_circle(4, 1.0, 0.8))
    assert shape_distance(a, b) > shape_distance(a, a)
    assert shape_distance(a, c) > 0.05


def test_refilter_by_residual(tmp_path: Path):
    db = tmp_path / "refilter.sqlite"
    with ChoreographySearchStore(db) as store:
        for i, res in enumerate([1e-9, 1e-4, 1e-8], start=1):
            seed = _seed_circle(4, 1.0 + 0.01 * i)
            store.insert_trial(
                n_bodies=4,
                trial_no=i,
                start_fp=f"s{i}",
                result_fp=f"r{i}",
                residual=res,
                period=seed.period,
                ok_gate=True,
                reason="ok",
                maintains_regular_ngon=False,
                seed=seed,
            )
        n = store.refilter_by_residual(4, max_residual=1e-6)
        assert n == 1
        assert store.count_passed(4) == 2


def test_select_diverse_families(tmp_path: Path):
    db = tmp_path / "div.sqlite"
    seeds = [
        (_seed_circle(4, 1.0, 0.0), 1e-8),
        (_seed_circle(4, 1.0, 0.05), 2e-8),
        (_seed_line(4), 3e-8),
        (_seed_circle(4, 1.0, 1.2), 4e-8),
    ]
    with ChoreographySearchStore(db) as store:
        for i, (seed, res) in enumerate(seeds, start=1):
            store.insert_trial(
                n_bodies=4,
                trial_no=i,
                start_fp=f"s{i}",
                result_fp=f"r{i}",
                residual=res,
                period=seed.period,
                ok_gate=True,
                reason="ok",
                maintains_regular_ngon=False,
                seed=seed,
            )
        picks = select_diverse_families(store, 4, n_families=3, min_sep=0.05)
        assert len(picks) >= 2
        assert picks[0].record.trial_no == 1
        ids = {p.seed.id for p in picks}
        assert "line4" in ids or any("line" in i for i in ids)
