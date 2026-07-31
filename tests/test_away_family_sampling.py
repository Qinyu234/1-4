"""Family-hit anneal and §3.3 classification keys."""

from __future__ import annotations

import numpy as np

from fairy_orbit.observe.choreography_search import (
    random_asymmetric_seed,
    sample_search_start,
)
from fairy_orbit.observe.family_class import (
    FamilyHitAnnealer,
    action_proxy,
    family_classification_key,
)
from fairy_orbit.observe.shape_families import shape_feature_vector


def test_family_key_uses_perm_and_action_bucket() -> None:
    seed = random_asymmetric_seed(4, np.random.default_rng(0))
    k1 = family_classification_key(seed, perm_label="(1 2 3 0)", action=1.0e-2)
    k2 = family_classification_key(seed, perm_label="(1 2 3 0)", action=1.0e-2)
    k3 = family_classification_key(seed, perm_label="(1 2 3 0)", action=1.0e2)
    k4 = family_classification_key(seed, perm_label="(0 1 2 3)", action=1.0e-2)
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4
    assert "n4|" in k1 and "|S" in k1
    assert np.isfinite(action_proxy(seed))


def test_family_hit_annealer_raises_away_on_rediscovery() -> None:
    ann = FamilyHitAnnealer(window=20, warmup=8, away_min=0.05, away_max=0.9)
    for i in range(10):
        ann.observe_scout(f"fam_{i}")
    p_novel = ann.away_prob()
    for _ in range(30):
        ann.observe_scout("fam_0")
    p_rediscover = ann.away_prob()
    assert ann.family_hit_rate() is not None
    assert p_rediscover > p_novel
    assert p_rediscover > 0.5


def test_sample_uses_away_prob() -> None:
    seed, mode, _ = sample_search_start(
        4,
        np.random.default_rng(7),
        [np.zeros(8)],
        away_prob=0.0,
    )
    assert mode == "baseline"
    assert seed.source == "random_asymmetric_ic"
    _ = shape_feature_vector(seed)
