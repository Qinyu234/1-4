"""Branch-2 probe and Floquet stability helpers."""

from __future__ import annotations

import numpy as np

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.observe.branch2_probe import branch2_existence_probe
from fairy_orbit.observe.choreography_search import (
    random_asymmetric_seed,
    scout_then_certify,
)
from fairy_orbit.observe.stability import floquet_multipliers_fd
from fairy_orbit.store.search_db import trial_rng


def test_branch2_probe_runs_on_polygon() -> None:
    seed4 = build_free_polygon_seed(4, seed_id="poly4", family="free_4")
    res = branch2_existence_probe(seed4, n_samples=8, seed=1)
    assert res.n_samples == 8
    assert np.isfinite(res.best_residual)
    assert "hopeful" in res.to_dict()


def test_floquet_fd_on_polygon_returns_eigs() -> None:
    seed4 = build_free_polygon_seed(4, seed_id="poly4", family="free_4")
    fl = floquet_multipliers_fd(seed4, shift=1, eps=1e-6, stable_atol=0.2)
    assert fl.multipliers.size == 24
    assert np.isfinite(fl.max_abs)
    d = fl.to_dict()
    assert "max_abs" in d and isinstance(d["multipliers"][0], dict)


def test_scout_certify_can_skip_floquet() -> None:
    seed = random_asymmetric_seed(4, trial_rng(4, 3))
    out = scout_then_certify(
        seed,
        residual=10.0,
        shift=1,
        scout_atol_rel=1e-5,
        scout_max_residual=1e-3,
        certify_atol_rel=1e-8,
        certify_max_residual=1e-6,
        require_floquet_stable=True,
    )
    assert out["reason"] == "failed_scout_residual"
