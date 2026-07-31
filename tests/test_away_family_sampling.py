"""Residual-annealed away-from-family start sampling."""

from __future__ import annotations

import numpy as np

from fairy_orbit.observe.choreography_search import (
    ResidualAnnealer,
    min_distance_to_features,
    random_asymmetric_seed,
    sample_search_start,
)
from fairy_orbit.observe.shape_families import shape_feature_vector


def test_away_mode_uses_distinct_source() -> None:
    rng = np.random.default_rng(0)
    base = random_asymmetric_seed(4, rng, mode="baseline")
    away = random_asymmetric_seed(4, rng, mode="away")
    assert base.source == "random_asymmetric_ic"
    assert away.source == "random_asymmetric_ic_away"


def test_annealer_stays_low_while_improving() -> None:
    ann = ResidualAnnealer(window=20, warmup=8, away_min=0.05, away_max=0.9)
    # Rapid improvement from O(1) toward low residual
    for r in np.geomspace(2.0, 1e-2, 20):
        ann.observe(float(r))
    p_improving = ann.away_prob()
    assert p_improving < 0.45


def test_annealer_raises_when_stalled_low() -> None:
    ann = ResidualAnnealer(window=20, warmup=8, away_min=0.05, away_max=0.9)
    for _ in range(30):
        ann.observe(5e-7)
    p = ann.away_prob()
    assert p > 0.55


def test_sample_uses_away_prob() -> None:
    seed, mode, _ = sample_search_start(
        4,
        np.random.default_rng(7),
        [np.zeros(8)],
        away_prob=0.0,
        away_min_sep=0.12,
    )
    assert mode == "baseline"
    assert seed.source == "random_asymmetric_ic"
    assert min_distance_to_features(shape_feature_vector(seed), [np.zeros(8)]) >= 0.0


def test_away_without_families() -> None:
    seed, mode, d = sample_search_start(
        5,
        np.random.default_rng(3),
        [],
        away_prob=1.0,
    )
    assert mode == "away_no_families"
    assert seed.source == "random_asymmetric_ic_away"
    assert d == float("inf")
