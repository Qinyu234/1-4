"""Tests for choreography search SQLite store (resume + dedupe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed
from fairy_orbit.store.search_db import (
    ChoreographySearchStore,
    seed_fingerprint,
    state_fingerprint,
    trial_rng,
)


def _toy_seed(n: int = 4, scale: float = 1.0) -> OrbitSeed:
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for i in range(n):
        ang = 2 * np.pi * i / n
        pos[i] = (scale * np.cos(ang), scale * np.sin(ang), 0.0)
        vel[i] = (-np.sin(ang), np.cos(ang), 0.0)
    return OrbitSeed(
        id=f"toy_{n}",
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


def test_fingerprint_stable_and_sensitive():
    a = _toy_seed(4, 1.0)
    b = _toy_seed(4, 1.0)
    assert seed_fingerprint(a) == seed_fingerprint(b)
    c = _toy_seed(4, 1.01)
    assert seed_fingerprint(a) != seed_fingerprint(c)


def test_trial_rng_deterministic():
    r1 = trial_rng(4, 3).random(5)
    r2 = trial_rng(4, 3).random(5)
    r3 = trial_rng(4, 4).random(5)
    assert np.allclose(r1, r2)
    assert not np.allclose(r1, r3)


def test_store_resume_and_dedupe(tmp_path: Path):
    db = tmp_path / "search.sqlite"
    seed = _toy_seed(4)
    fp = seed_fingerprint(seed)
    with ChoreographySearchStore(db) as store:
        assert store.next_trial_no(4) == 1
        rid = store.insert_trial(
            n_bodies=4,
            trial_no=1,
            start_fp="start_a",
            result_fp=fp,
            residual=1e-3,
            period=seed.period,
            ok_gate=True,
            reason="ok",
            maintains_regular_ngon=False,
            seed=seed,
        )
        assert rid is not None
        assert store.next_trial_no(4) == 2
        assert store.has_start_fp(4, "start_a")
        assert store.has_accepted_result_fp(4, fp)
        # duplicate start_fp → None
        assert (
            store.insert_trial(
                n_bodies=4,
                trial_no=2,
                start_fp="start_a",
                result_fp="other",
                residual=1.0,
                period=1.0,
                ok_gate=False,
                reason="failed",
                maintains_regular_ngon=False,
            )
            is None
        )
        best = store.best_accepted(4)
        assert best is not None
        assert best.residual == 1e-3
        summary = store.summary_dict(4)
        assert summary["trials"] == 1
        assert summary["passed_gate"] == 1

    # reopen resumes
    with ChoreographySearchStore(db) as store:
        assert store.next_trial_no(4) == 2
        assert store.count_passed(4) == 1


def test_import_seed_pass_dedupes(tmp_path: Path):
    db = tmp_path / "search.sqlite"
    seed = _toy_seed(5, 1.2)
    with ChoreographySearchStore(db) as store:
        a = store.import_seed_pass(seed, residual=0.01)
        b = store.import_seed_pass(seed, residual=0.01)
        assert a is not None
        assert b is None
        assert store.count_passed(5) == 1


def test_state_fingerprint_round():
    pos = np.array([[1.0000004, 0.0, 0.0]])
    vel = np.zeros((1, 3))
    assert state_fingerprint(pos, vel, 1.0) == state_fingerprint(
        np.array([[1.0, 0.0, 0.0]]), vel, 1.0
    )
