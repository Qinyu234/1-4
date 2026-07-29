"""Tests for staged ABCD→BCDA search and Stage-A σ requirement."""

from __future__ import annotations

from pathlib import Path

import pytest

from fairy_orbit.design.manifold import ManifoldParams
from fairy_orbit.observe.peo import evaluate_peo
from fairy_orbit.observe.rep_error import (
    RepSigmas,
    is_calibrated_sigmas,
    load_required_sigmas,
)
from fairy_orbit.observe.search import FreeParams, SeedAnchors
from fairy_orbit.observe.staged_search import (
    ReachSample,
    StagedConfig,
    run_stage1_coarse,
    run_stage2_expand,
    run_stage3_stain,
    stain_flood,
)


def test_is_calibrated_sigmas():
    assert not is_calibrated_sigmas(RepSigmas(source="unit"))
    assert not is_calibrated_sigmas(RepSigmas(source="unit_default", n_samples=10))
    assert is_calibrated_sigmas(RepSigmas(source="rep_error_scan_finals", n_samples=5))


def test_load_required_sigmas_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Stage-A"):
        load_required_sigmas(tmp_path / "missing.json")


def test_load_required_sigmas_ok(tmp_path: Path):
    path = tmp_path / "sigmas.json"
    RepSigmas(source="rep_error_scan_finals", n_samples=12, E_r=0.1, E_v=0.2).to_json(path)
    sig = load_required_sigmas(path)
    assert sig.n_samples == 12
    assert is_calibrated_sigmas(sig)


def test_evaluate_peo_require_calibrated_rejects_unit():
    with pytest.raises(ValueError, match="calibrated"):
        evaluate_peo(
            ManifoldParams(),
            t_end=0.1,
            n_outputs=4,
            sigmas=RepSigmas(source="unit"),
            require_calibrated_sigmas=True,
            skip_choreography_gate=True,
        )


def test_stain_flood_ranks_and_neighbors():
    m_axis = __import__("numpy").array([1e-4, 1e-3, 1e-2])
    e_axis = __import__("numpy").array([0.0, 0.1, 0.2])
    samples = []
    for i, m in enumerate(m_axis):
        for j, e in enumerate(e_axis):
            soft = 1.0 if (i, j) != (1, 1) else 0.05
            samples.append(
                ReachSample(
                    m=float(m),
                    e=float(e),
                    free=FreeParams(0.15, 0.0, 1.0, 0.0, 0.0, 0.0),
                    status="choreography",
                    soft_choreo=soft,
                )
            )
    stained = stain_flood(
        samples,
        frac=0.15,
        max_seeds=1,
        flood_chebyshev=1,
        m_axis=m_axis,
        e_axis=e_axis,
    )
    assert any(s.stained for s in stained)
    # center + neighbors → more than 1
    assert len(stained) >= 5


def test_stage_pipeline_with_mock_eval():
    calls = {"n": 0}

    def eval_fn(seed: SeedAnchors, free: FreeParams) -> ReachSample:
        calls["n"] += 1
        # Prefer larger a1 as "better" soft residual
        soft = abs(free.a1 - 0.22) + abs(free.M1 - 2.0) * 0.01
        status = "success" if soft < 0.02 else "choreography"
        return ReachSample(
            m=seed.m,
            e=seed.e,
            free=free,
            status=status,
            soft_choreo=0.0 if status == "success" else soft,
            score=0.5 if status == "success" else None,
        )

    cfg = StagedConfig(
        n_m=2,
        n_e=2,
        stage1_axes=("a1", "M1"),
        stage1_points=2,
        stage1_top_k=2,
        stage2_points=2,
        stage2_expand_rounds=1,
        stage3_points=2,
        stain_frac=0.5,
        stain_max_seeds=2,
        unlock_high_order=True,
    )
    stage1, m_axis, e_axis = run_stage1_coarse(config=cfg, eval_fn=eval_fn)
    assert len(stage1) == 2 * 2 * 2 * 2  # m×e×a1×M1
    top = stage1[: cfg.stage1_top_k]
    stage2 = run_stage2_expand(top, config=cfg, eval_fn=eval_fn)
    assert calls["n"] > len(stage1)

    stained = stain_flood(
        stage2 or stage1,
        frac=0.5,
        max_seeds=2,
        flood_chebyshev=1,
        m_axis=m_axis,
        e_axis=e_axis,
    )
    assert stained

    def score_fn(seed: SeedAnchors, free: FreeParams) -> ReachSample:
        s = eval_fn(seed, free)
        s.status = "success"
        s.soft_choreo = 0.0
        s.score = 1.23
        s.summary = {"E_r_final": 0.01, "E_v_final": 0.02, "score": 1.23}
        return s

    stage3, survivors = run_stage3_stain(
        stained[:1], config=cfg, eval_fn=eval_fn, score_fn=score_fn
    )
    # unlock stages include (), (a2,), ... so some samples should have unlocked a2
    assert any(s.unlocked[:1] == ("a2",) or s.unlocked == ("a2",) for s in stage3) or any(
        "a2" in s.unlocked for s in stage3
    )


def test_unlock_order_in_stage3_config():
    from fairy_orbit.observe.bayes import UNLOCK_STAGES

    assert UNLOCK_STAGES[1] == ("a2",)
    assert UNLOCK_STAGES[-1][-3:] == ("v1x", "v1y", "v1z")
