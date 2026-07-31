"""Scout / certify funnel."""

from __future__ import annotations

import numpy as np

from fairy_orbit.observe.choreography_search import (
    random_asymmetric_seed,
    scout_then_certify,
)
from fairy_orbit.store.search_db import trial_rng


def test_scout_fail_on_huge_residual() -> None:
    seed = random_asymmetric_seed(4, trial_rng(4, 1))
    out = scout_then_certify(
        seed,
        residual=10.0,
        shift=1,
        scout_atol_rel=1e-5,
        scout_max_residual=1e-3,
        certify_atol_rel=1e-8,
        certify_max_residual=1e-6,
        require_floquet_stable=False,
    )
    assert out["scout_ok"] is False
    assert out["certified"] is False
    assert out["reason"] == "failed_scout_residual"


def test_funnel_dict_shape_on_random_ic() -> None:
    seed = random_asymmetric_seed(4, trial_rng(4, 2))
    out = scout_then_certify(
        seed,
        residual=1e-4,
        shift=1,
        scout_atol_rel=1e-5,
        scout_max_residual=1e-3,
        certify_atol_rel=1e-8,
        certify_max_residual=1e-6,
        require_floquet_stable=False,
    )
    assert "scout_ok" in out and "certified" in out and "reason" in out
    assert out["certified"] in (True, False)
    if out["scout_ok"]:
        assert out["family_key"] and "n4|" in out["family_key"]
        assert out["action_proxy"] is None or np.isfinite(out["action_proxy"])
