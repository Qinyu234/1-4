"""SQLite orbit store tests."""

from __future__ import annotations

from pathlib import Path

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder
from fairy_orbit.observe import diagnose
from fairy_orbit.store import OrbitStore, make_param_class


def test_save_query_load(tmp_path: Path):
    db = tmp_path / "orbits.sqlite"
    cfg = SystemConfig(mass_ratio=1e-3)
    params = LadderParams(eccentricity=0.12, tetrahedral=True)
    system = build_orbital_ladder(cfg, params)
    d = diagnose(
        system, cfg, t_end=30.0, n_outputs=40, ladder=params, run_megno=False
    )
    with OrbitStore(db) as store:
        run_id = store.save(
            config=cfg,
            params=params,
            diagnosis=d,
            source="test",
            t_end=30.0,
            n_outputs=40,
            store_trajectory=True,
        )
        assert run_id >= 1
        cls = make_param_class(
            eccentricity=0.12,
            mass_ratio=1e-3,
            tetrahedral=True,
            period_ratios=params.period_ratios,
        )
        hits = store.query(param_class=cls, limit=5)
        assert any(r.id == run_id for r in hits)
        by_mu = store.query(mu_min=5e-4, mu_max=2e-3, e_min=0.1, e_max=0.15)
        assert any(r.id == run_id for r in by_mu)
        traj = store.load_trajectory(run_id)
        assert traj is not None
        assert traj.positions.shape[0] == 40
        classes = store.list_classes()
        assert any(c["param_class"] == cls for c in classes)


def test_param_class_stable():
    a = make_param_class(
        eccentricity=0.15,
        mass_ratio=1e-3,
        tetrahedral=True,
        period_ratios=(1.5, 5 / 3, 1.4),
    )
    b = make_param_class(
        eccentricity=0.15,
        mass_ratio=1e-3,
        tetrahedral=True,
        period_ratios=(1.5, 5 / 3, 1.4),
    )
    assert a == b
    assert "tet1" in a
    assert "e0.15" in a
