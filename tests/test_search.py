"""Tests for grid+beam search (mock eval) and COM helpers."""

from __future__ import annotations

import pytest

from fairy_orbit.design.manifold import ManifoldParams, poly_linear
from fairy_orbit.observe.search import (
    FREE_NAMES,
    BeamConfig,
    Candidate,
    FreeParams,
    SearchBounds,
    grid_beam_search,
    select_beams,
)


def test_free_names():
    assert FREE_NAMES == ("a1", "e1", "M1", "vx", "vy", "vz")


def test_select_beams_keeps_historical_best():
    hist = [
        Candidate(FreeParams(0.12, 0.0, 1.0, 0.0, 0.0, 0.0), 5.0, "ok"),
        Candidate(FreeParams(0.15, 0.0, 2.0, 0.0, 0.0, 0.0), 1.0, "ok"),
        Candidate(FreeParams(0.12, 0.0, 1.0, 0.0, 0.0, 0.0), 4.0, "ok"),
        Candidate(FreeParams(0.20, 0.0, 3.0, 0.0, 0.0, 0.0), 2.0, "ok"),
    ]
    beams = select_beams(hist, 2)
    assert beams[0].loss == pytest.approx(1.0)
    assert beams[1].loss == pytest.approx(2.0)


def test_default_bounds_require_a_ladder():
    b = SearchBounds()
    assert b.a1[0] > 0.0


def test_grid_beam_finds_bowl_minimum():
    def eval_fn(params: ManifoldParams):
        loss = (params.a1 - 0.18) ** 2 + (params.M1 - 3.0) ** 2 + (params.vx - 0.01) ** 2
        return float(loss), "success", {"score": float(loss)}, 0.0

    res = grid_beam_search(
        m=1e-3,
        e=0.05,
        bounds=SearchBounds(
            a1=(0.10, 0.30),
            e1=(0.0, 0.0),
            M1=(0.5, 6.0),
            vx=(-0.05, 0.05),
            vy=(0.0, 0.0),
            vz=(0.0, 0.0),
        ),
        config=BeamConfig(
            beam_width=3,
            coarse_points=3,
            refine_points=(5,),
            max_evals=2000,
            n_periods=1.0,
        ),
        eval_fn=eval_fn,
    )
    assert res.best is not None
    assert res.best.params.a1 == pytest.approx(0.18, abs=0.03)
    assert res.best.params.M1 == pytest.approx(3.0, abs=0.4)
    assert res.best.params.vx == pytest.approx(0.01, abs=0.015)


def test_free_params_optional_quad_defaults():
    f = FreeParams(0.12, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert f.a2 == 0.0 and f.v1z == 0.0
    assert "a2" in f.as_dict()


def test_near_edges_and_expand():
    b = SearchBounds(a1=(0.10, 0.30), e1=(-0.02, 0.04), M1=(0.5, 6.0), vx=(-0.05, 0.05), vy=(-0.05, 0.05), vz=(-0.05, 0.05))
    edges = b.near_edges(FreeParams(0.30, 0.04, 3.0, 0.0, -0.05, 0.05), frac=0.05)
    assert edges["a1"] == "hi"
    assert edges["e1"] == "hi"
    assert edges["vy"] == "lo"
    assert edges["vz"] == "hi"
    assert "M1" not in edges
    b2, changed = b.expand_edges(edges, grow=0.5)
    assert b2.a1[1] > b.a1[1]
    assert b2.vy[0] < b.vy[0]
    assert "a1" in changed
