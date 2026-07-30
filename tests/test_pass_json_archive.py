"""Tests for SQLite-first pass storage helpers."""

from __future__ import annotations

from pathlib import Path

from fairy_orbit.design.seeds import OrbitSeed, save_seed
from fairy_orbit.observe.choreography_search import (
    archive_pass_json_files,
    _import_existing_passes,
)
from fairy_orbit.store.search_db import ChoreographySearchStore
import numpy as np


def _toy(n: int = 4) -> OrbitSeed:
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for i in range(n):
        ang = 2 * np.pi * i / n
        pos[i] = (np.cos(ang), np.sin(ang), 0.0)
        vel[i] = (-np.sin(ang), np.cos(ang), 0.0)
    return OrbitSeed(
        id=f"t{n}",
        family=f"free_{n}",
        n_bodies=n,
        G=1.0,
        masses=tuple(1.0 for _ in range(n)),
        period=6.0,
        positions=pos,
        velocities=vel,
        names=tuple(f"B{i+1}" for i in range(n)),
        symmetry="test",
        source="test",
        notes="residual=1e-9",
    )


def test_archive_pass_json_files(tmp_path: Path):
    n = 4
    p = tmp_path / f"pass_{n}_00001.json"
    save_seed(_toy(n), p)
    assert p.exists()
    moved = archive_pass_json_files(tmp_path, n)
    assert moved == 1
    assert not p.exists()
    assert (tmp_path / "pass_json_archive" / p.name).exists()


def test_import_skipped_when_db_has_rows(tmp_path: Path):
    from fairy_orbit.store.search_db import seed_fingerprint

    db = tmp_path / "s.sqlite"
    seed = _toy(4)
    save_seed(seed, tmp_path / "pass_4_00001.json")
    fp = seed_fingerprint(seed)
    with ChoreographySearchStore(db) as store:
        store.insert_trial(
            n_bodies=4,
            trial_no=1,
            start_fp="a",
            result_fp=fp,
            residual=1e-9,
            period=6.0,
            ok_gate=True,
            reason="ok",
            maintains_regular_ngon=False,
            seed=seed,
        )
        assert _import_existing_passes(store, tmp_path, 4) == 0
        assert _import_existing_passes(store, tmp_path, 4, force=True) == 0
