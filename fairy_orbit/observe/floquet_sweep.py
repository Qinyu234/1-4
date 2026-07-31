"""Floquet batch sweeps: continuation path and equal-mass family archive."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fairy_orbit.design.seeds import load_seed
from fairy_orbit.observe.campaign_prefs import FLOQUET_STABLE_ATOL
from fairy_orbit.observe.stability import floquet_multipliers_fd

_MC_RE = re.compile(r"state_Mc_([0-9.eE+-]+)\.json$")


def mc_from_checkpoint_name(path: Path) -> float | None:
    m = _MC_RE.search(path.name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def floquet_path_sweep(
    cont_dir: Path,
    *,
    stable_atol: float = FLOQUET_STABLE_ATOL,
    out: Path | None = None,
    on_row: Any = None,
) -> dict[str, Any]:
    """
    Floquet certify every ``state_Mc_*.json`` under ``cont_dir``.

    Returns payload with ``rows`` and ``unit_circle_crossings`` (max|λ| vs M_c).
    Raises ``FileNotFoundError`` / ``ValueError`` if no checkpoints.
    """
    cont = Path(cont_dir)
    if not cont.is_dir():
        raise FileNotFoundError(f"no continuation dir {cont}")

    paths = sorted(
        (p for p in cont.glob("state_Mc_*.json") if mc_from_checkpoint_name(p) is not None),
        key=lambda p: mc_from_checkpoint_name(p) or 0.0,
    )
    if not paths:
        raise ValueError(
            f"no state_Mc_*.json under {cont} — run Path A first; "
            "equal-mass proxy: experiments/run_floquet_family_sweep.py"
        )

    rows: list[dict[str, Any]] = []
    for path in paths:
        seed = load_seed(path)
        Mc = mc_from_checkpoint_name(path)
        assert Mc is not None
        if on_row is not None:
            on_row(path, Mc, seed)
        fl = floquet_multipliers_fd(seed, shift=1, stable_atol=stable_atol)
        abs_sorted = sorted(
            (m["abs"] for m in fl.to_dict()["multipliers"]), reverse=True
        )
        clearly = [a for a in abs_sorted if a > 1.0 + float(stable_atol)]
        near_unit = [a for a in abs_sorted if abs(a - 1.0) < 0.02]
        rows.append(
            {
                "path": str(path),
                "M_c": Mc,
                "period": float(seed.period),
                "map_residual": fl.map_residual,
                "max_abs": fl.max_abs,
                "stable": fl.stable,
                "n_unstable": fl.n_unstable,
                "clearly_unstable_abs": clearly,
                "n_near_unit_0.02": len(near_unit),
                "top_abs": abs_sorted[:8],
            }
        )

    ordered = sorted(rows, key=lambda r: r["M_c"])
    crossings: list[dict[str, float]] = []
    for a, b in zip(ordered, ordered[1:]):
        if (a["max_abs"] - 1.0) * (b["max_abs"] - 1.0) < 0:
            crossings.append(
                {
                    "M_c_lo": a["M_c"],
                    "M_c_hi": b["M_c"],
                    "max_abs_lo": a["max_abs"],
                    "max_abs_hi": b["max_abs"],
                }
            )

    payload = {
        "cont_dir": str(cont),
        "stable_atol": float(stable_atol),
        "n": len(rows),
        "n_stable": sum(1 for r in rows if r["stable"]),
        "unit_circle_crossings": crossings,
        "rows": rows,
    }
    out_path = Path(out) if out is not None else (cont / "floquet_path_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["out"] = str(out_path)
    return payload
