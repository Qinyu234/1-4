"""Post-continuation visual classify (not free-N)."""

from __future__ import annotations

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.observe.visual_classify import classify_continued_orbit


def test_classify_continued_requires_positive_Mc() -> None:
    seed = build_free_polygon_seed(4, seed_id="poly4", family="free_4")
    rep0 = classify_continued_orbit(seed, 0.0, n_outputs=8)
    assert rep0.klass == "quiet"
    assert "M_c<=0" in rep0.note


def test_classify_continued_runs_with_central() -> None:
    seed = build_free_polygon_seed(4, seed_id="poly4", family="free_4")
    rep = classify_continued_orbit(seed, 1e-3, n_outputs=16, log_rho=0.0)
    assert rep.klass in {"light_swap", "gravity_only", "angular_distant", "quiet", "error"}
    assert rep.M_c == 1e-3
