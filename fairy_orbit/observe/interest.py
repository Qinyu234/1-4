"""Interestingness score: prefer secular change without chaos/escape (PROMPT §3)."""

from __future__ import annotations

import math

import numpy as np


def a_order_changed(a_initial: list[float], a_final: list[float]) -> bool:
    """True if fairy semi-major-axis ranking changed (role migration proxy)."""
    if len(a_initial) < 2 or len(a_final) < 2:
        return False
    return tuple(np.argsort(a_initial).tolist()) != tuple(np.argsort(a_final).tolist())


def interestingness(summary: dict) -> float:
    """
    Higher is better.

    Rewards: a migration, encounters, a-order swaps, mild e pumping.
    Rejects/penalizes: collision/escape, near-static a, MEGNO deep in chaos sea.
    """
    if summary.get("status") != "success":
        return -1e9

    a_rms = float(summary.get("a_delta_rms") or 0.0)
    n_enc = int(summary.get("n_encounters") or 0)
    e_max = max(summary.get("e_max") or [0.0])
    megno = summary.get("megno")
    order_swap = bool(summary.get("a_order_changed", False))

    # Static orbits are the failure mode the user rejected
    if a_rms < 1e-3 and n_enc == 0 and not order_swap:
        return -100.0

    # Prefer visible migration, but not unbound blow-ups
    migrate = min(a_rms, 0.5) * 20.0
    enc = min(n_enc, 30) * 0.35
    swap = 8.0 if order_swap else 0.0
    e_term = min(max(float(e_max) - 0.05, 0.0), 0.6) * 4.0

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
            # mild chaos / secular OK; MEGNO~2 with no motion already handled above
            megno_term = 0.5

    return migrate + enc + swap + e_term + megno_term
