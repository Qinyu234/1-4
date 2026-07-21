"""Tests for per-k near-1 velocity optimization."""

from fairy_orbit.search.k_velocity import PerKOptimizeConfig, optimize_each_k


def test_optimize_each_k_smoke(tmp_path):
    cfg = PerKOptimizeConfig(
        k_values=[0.9, 1.0],
        alpha_grid=[0.4, 0.7],
        beta_grid=[0.7, 1.0],
        n_periods=0.25,
        steps_per_period=25,
        record_every=2,
        refine_maxiter=3,
        library_dir=str(tmp_path / "lib"),
        summary_path=str(tmp_path / "summary.json"),
    )
    best = optimize_each_k(cfg, progress=lambda _m: None)
    assert set(best.keys()) == {0.9, 1.0}
    assert (tmp_path / "summary.json").exists()
    for cand in best.values():
        assert cand.score >= 0.0
        assert abs(cand.fairy_mass / cand.planet_mass - cand.mass_ratio) < 1e-12
