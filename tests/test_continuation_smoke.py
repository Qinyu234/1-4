"""Smoke test for Path A mass-continuation stub."""

from __future__ import annotations

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.observe.continuation import mass_continuation_smoke


def test_mass_continuation_smoke_mc0_gate():
    seed = build_free_polygon_seed(4, seed_id="free_4_square_re", family="free_4")
    # Gate + residual at tiny M_c only (skip LM — each eval is an IAS15 integrate).
    res = mass_continuation_smoke(seed, M_c=1e-6, shift=1, correct=False)
    assert res.gate0.ok
    assert res.residual0_norm < 1e-8
    assert res.success
