"""Tests for grid search + library (T7)."""

from pathlib import Path

import numpy as np

from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.library.store import list_candidates, load_candidate, save_candidate
from fairy_orbit.search.grid import evaluate_params, scan


def test_save_and_load_candidate(tmp_path: Path):
    path = save_candidate(
        {"v_rad": 0.1, "v_tan": 0.2, "score": 1.5, "period": 10.0},
        directory=tmp_path,
    )
    assert path.name == "orbit_001.json"
    data = load_candidate(path)
    assert data["score"] == 1.5
    assert len(list_candidates(tmp_path)) == 1


def test_evaluate_params_returns_candidate():
    vesc = escape_speed(1.0, 1.0, 20.0)
    cand, traj, result = evaluate_params(
        0.8 * vesc,
        0.9 * vesc,
        n_periods=0.25,
        steps_per_period=40,
        record_every=2,
    )
    assert cand.score == result.score
    assert traj.positions.shape[1] == 5


def test_tiny_grid_scan(tmp_path: Path):
    vesc = escape_speed(1.0, 1.0, 20.0)
    rads = [0.85 * vesc, 0.95 * vesc]
    tans = [0.85 * vesc]
    cands = scan(
        rads,
        tans,
        save_top=1,
        library_dir=str(tmp_path),
        n_periods=0.2,
        steps_per_period=30,
        record_every=2,
    )
    assert len(cands) == 2
    assert cands[0].score <= cands[1].score
    assert len(list_candidates(tmp_path)) == 1
