"""Path A accepts steps on residual even when LM success=False."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed, build_free_polygon_seed
from fairy_orbit.observe.continuation import run_path_a_continuation


def test_path_a_accepts_on_residual_despite_ls_failure(tmp_path: Path) -> None:
    seed = build_free_polygon_seed(4, seed_id="poly4_patha", family="free_4")

    def fake_correct(
        current: OrbitSeed,
        target: float,
        **kwargs: object,
    ) -> tuple[OrbitSeed, float, bool]:
        # Tiny residual but LM "failed" — previously blocked Mc advance.
        out = OrbitSeed(
            id=current.id,
            family=current.family,
            n_bodies=current.n_bodies,
            G=current.G,
            masses=current.masses,
            period=current.period,
            positions=np.asarray(current.positions, dtype=float).copy(),
            velocities=np.asarray(current.velocities, dtype=float).copy(),
            names=current.names,
            symmetry=current.symmetry,
            source=current.source,
            notes=f"fake M_c={target}",
            central_index=None,
        )
        return out, 1e-6, False

    with patch(
        "fairy_orbit.observe.continuation.correct_at_mass", side_effect=fake_correct
    ):
        with patch(
            "fairy_orbit.observe.continuation.verify_choreography_Tn"
        ) as gate:
            gate.return_value.ok = True
            gate.return_value.to_dict.return_value = {"ok": True}
            summary = run_path_a_continuation(
                seed,
                wall_hours=None,
                M_c_max=2e-3,
                dM0=1e-3,
                max_nfev=80,
                res_tol=1e-4,
                out_dir=tmp_path,
                optics_soft=False,
            )

    assert summary["M_c_final"] >= 2e-3 - 1e-15
    assert list(tmp_path.glob("state_Mc_*.json"))
    assert summary["res_tol"] == 1e-4
