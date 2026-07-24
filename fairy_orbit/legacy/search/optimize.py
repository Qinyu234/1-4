"""Stage2 local optimization stub (not required for v1 green line)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairy_orbit.search.grid import Candidate, evaluate_params


def refine(
    x0: tuple[float, float],
    *,
    method: str = "Nelder-Mead",
    **eval_kwargs: Any,
) -> Candidate:
    """
    Locally refine (v_rad, v_tan) with scipy.optimize.

    Thin stub for Stage2 — available but not used by first_grid_scan.
    """
    from scipy import optimize

    def loss(x: np.ndarray) -> float:
        cand, _, _ = evaluate_params(float(x[0]), float(x[1]), **eval_kwargs)
        return cand.score

    result = optimize.minimize(loss, x0=np.asarray(x0, dtype=float), method=method)
    cand, _, _ = evaluate_params(float(result.x[0]), float(result.x[1]), **eval_kwargs)
    return cand
