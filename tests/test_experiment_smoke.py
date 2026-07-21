"""Smoke test for first_grid_scan experiment (T8)."""

from pathlib import Path

from fairy_orbit.icgen.tetrahedron import escape_speed
from fairy_orbit.search.grid import scan
from fairy_orbit.visualization.plots import plot_score_heatmap


def test_smoke_grid_and_plot(tmp_path: Path):
    vesc = escape_speed(1.0, 1.0, 20.0)
    rads = [0.9 * vesc]
    tans = [0.95 * vesc]
    cands = scan(
        rads,
        tans,
        save_top=1,
        library_dir=str(tmp_path / "lib"),
        n_periods=0.2,
        steps_per_period=30,
        record_every=2,
    )
    out = tmp_path / "heatmap.png"
    plot_score_heatmap(cands, out)
    assert out.exists()
    assert cands[0].score >= 0.0
