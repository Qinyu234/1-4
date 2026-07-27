"""Score form with numerical floors from tetrahedron dual-ε calibration (PROMPT §2.4.1 / §7.5)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ScoreFloors:
    """
    Noise floors from the regular-tetrahedron dual-ε experiment.

    A real-chain metric is "significant" only if it exceeds k × floor.
    """

    energy_drift: float = 1e-12
    amd_total_ptp: float = 1e-10
    a_ptp_mean: float = 1e-6
    shape_error: float = 1e-6  # only meaningful if baseline stays regular
    k_sigma: float = 10.0
    source: str = "defaults"


@dataclass
class ScoreBreakdown:
    total: float
    floors: ScoreFloors
    terms: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)


def _sig(value: float, floor: float, k: float) -> bool:
    return float(value) > k * float(floor)


def score_summary(summary: dict, floors: ScoreFloors | None = None) -> ScoreBreakdown:
    """
    Reasonable score for resonant-chain verification.

    Design rules (from tetra dual-ε calibration + PROMPT taste):
    1. Fail hard on collision/escape.
    2. Reject static rings (no significant a/AMD/encounter/swap above floors).
    3. Reward secular change: a migration, AMD exchange, encounters, a-order swap.
    4. Penalize energy drift only when it exceeds the numerical floor (integrator fault).
    5. MEGNO: mild OK; deep chaos sea penalized.
    """
    floors = floors or ScoreFloors()
    k = floors.k_sigma
    terms: dict[str, float] = {}
    flags: dict[str, bool] = {}

    status = summary.get("status")
    if status != "success":
        return ScoreBreakdown(total=-1e9, floors=floors, terms={"status": -1e9}, flags={"success": False})

    flags["success"] = True
    a_rms = float(summary.get("a_delta_rms") or 0.0)
    a_ptp = float(summary.get("a_ptp_mean") or 0.0)
    n_enc = int(summary.get("n_encounters") or 0)
    e_max = max(summary.get("e_max") or [0.0])
    amd_ptp = float(summary.get("amd_total_ptp") or 0.0)
    energy_drift = float(summary.get("energy_drift") or 0.0)
    order_swap = bool(summary.get("a_order_changed", False))
    megno = summary.get("megno")

    flags["sig_a"] = _sig(a_ptp, floors.a_ptp_mean, k) or _sig(a_rms, floors.a_ptp_mean, k)
    flags["sig_amd"] = _sig(amd_ptp, floors.amd_total_ptp, k)
    flags["sig_enc"] = n_enc > 0
    flags["sig_swap"] = order_swap
    # Long-chain runs accumulate more roundoff than one-orbit calibration;
    # treat E-drift as integrator fault only above a practical absolute cap.
    e_lim = max(k * floors.energy_drift, 1e-8)
    flags["integrator_ok"] = energy_drift <= e_lim

    # Static = nothing significant above numerical floor
    static = not (flags["sig_a"] or flags["sig_amd"] or flags["sig_enc"] or flags["sig_swap"])
    flags["static"] = static
    if static:
        terms["static"] = -100.0
        return ScoreBreakdown(total=-100.0, floors=floors, terms=terms, flags=flags)

    migrate = min(a_rms, 0.5) * 20.0
    amd_term = min(amd_ptp / max(floors.amd_total_ptp, 1e-30), 50.0) * 0.15
    enc = min(n_enc, 30) * 0.35
    swap = 8.0 if order_swap else 0.0
    e_term = min(max(float(e_max) - 0.05, 0.0), 0.6) * 4.0

    if not flags["integrator_ok"]:
        excess = energy_drift / max(e_lim, 1e-30)
        terms["energy_penalty"] = -min(50.0, 5.0 * math.log10(max(excess, 1.0)))
    else:
        terms["energy_penalty"] = 0.0

    megno_term = 0.0
    if megno is None or not math.isfinite(float(megno)):
        megno_term = -1.0
    else:
        m = float(megno)
        if m > 8.0:
            megno_term = -20.0 * (m - 8.0)
        elif m > 4.0:
            megno_term = -2.0 * (m - 4.0)
        else:
            megno_term = 0.5

    terms.update(
        {
            "migrate": migrate,
            "amd": amd_term,
            "encounters": enc,
            "swap": swap,
            "e_pump": e_term,
            "megno": megno_term,
        }
    )
    total = sum(terms.values())
    return ScoreBreakdown(total=float(total), floors=floors, terms=terms, flags=flags)


def interestingness(summary: dict, floors: ScoreFloors | None = None) -> float:
    """Backward-compatible scalar score."""
    return score_summary(summary, floors).total
