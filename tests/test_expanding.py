"""Tests for expanding-bounds search."""

from fairy_orbit.search.expanding import Bounds, ExpandingConfig, run_expanding_search


def test_expand_toward_best_grows_near_edge():
    b = Bounds(0.7, 1.3, 0.7, 1.3)
    # Best near upper-right corner
    grown = b.expand_toward_best(1.28, 1.25, step=0.2, edge_tol=0.2)
    assert grown.rad_hi > b.rad_hi
    assert grown.tan_hi > b.tan_hi


def test_expand_always_grows_when_interior():
    b = Bounds(0.7, 1.3, 0.7, 1.3)
    grown = b.expand_toward_best(1.0, 1.0, step=0.2, edge_tol=0.15)
    # Interior best still forces a small expand
    assert grown.rad_lo < b.rad_lo or grown.rad_hi > b.rad_hi


def test_short_expanding_run(tmp_path):
    cfg = ExpandingConfig(
        hours=45.0 / 3600.0,  # 45 seconds
        n_per_axis=2,
        expand_step=0.15,
        n_periods=0.2,
        steps_per_period=25,
        record_every=2,
        save_every=1,
        checkpoint_path=str(tmp_path / "ckpt.json"),
        library_dir=str(tmp_path / "lib"),
        initial_bounds=Bounds(0.9, 1.1, 0.9, 1.1),
    )
    cands = run_expanding_search(cfg, progress=lambda _m: None)
    assert len(cands) >= 1
    assert (tmp_path / "ckpt.json").exists()
    assert cands[0].score <= cands[-1].score
