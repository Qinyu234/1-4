"""8-channel representation errors for PEO / error calibration.

Channels (always scored together):
  E_r, E_v  — relative position/velocity closure with shared R*
  E_a, E_e, E_i, E_Omega, E_omega, E_M — element residuals after (R*,P)

E_energy is recorded for numerical validation only (not one of the 8).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fairy_orbit.design.elements import OrbitalElements
from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.closure import (
    closure_for_perm,
    fairy_states,
    radial_order,
    radial_permutation,
)

CHANNELS: tuple[str, ...] = (
    "E_r",
    "E_v",
    "E_a",
    "E_e",
    "E_i",
    "E_Omega",
    "E_omega",
    "E_M",
)


def frobenius(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    return float(np.sqrt(np.sum(X * X)))


def wrap_angle(delta: float) -> float:
    """Map angle difference into (−π, π]."""
    return float((delta + np.pi) % (2.0 * np.pi) - np.pi)


def relative_Er(
    r: np.ndarray,
    r0: np.ndarray,
    R: np.ndarray,
    perm: tuple[int, ...],
) -> float:
    """||Q − R P Q0||_F / ||Q0||_F."""
    r = np.asarray(r, dtype=float).reshape(-1, 3)
    r0 = np.asarray(r0, dtype=float).reshape(-1, 3)
    src = np.stack([r0[j] for j in perm], axis=0)
    pred = (R @ src.T).T
    denom = frobenius(src)
    if denom < 1e-300:
        return float("nan")
    return frobenius(r - pred) / denom


def relative_Ev(
    v: np.ndarray,
    v0: np.ndarray,
    R: np.ndarray,
    perm: tuple[int, ...],
) -> float:
    """||V − R P V0||_F / ||V0||_F (same R*)."""
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    v0 = np.asarray(v0, dtype=float).reshape(-1, 3)
    src = np.stack([v0[j] for j in perm], axis=0)
    pred = (R @ src.T).T
    denom = frobenius(src)
    if denom < 1e-300:
        return float("nan")
    return frobenius(v - pred) / denom


def element_channel_errors(
    r: np.ndarray,
    v: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    R: np.ndarray,
    perm: tuple[int, ...],
    mu: float,
) -> dict[str, float]:
    """
    Compare osculating elements of (r_i,v_i) vs (R r0_{P(i)}, R v0_{P(i)}).

    Aggregates as mean over fairies of relative |Δa|/a0, |Δe|, and wrapped
    angle abs for i, Ω, ω, M.
    """
    r = np.asarray(r, dtype=float).reshape(-1, 3)
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    r0 = np.asarray(r0, dtype=float).reshape(-1, 3)
    v0 = np.asarray(v0, dtype=float).reshape(-1, 3)
    n = r.shape[0]
    ea = ee = ei = eOm = eom = eM = 0.0
    for i in range(n):
        j = perm[i]
        el_t = OrbitalElements.from_state(r[i], v[i], mu)
        r_ref = R @ r0[j]
        v_ref = R @ v0[j]
        el_0 = OrbitalElements.from_state(r_ref, v_ref, mu)
        a0 = abs(el_0.a)
        ea += abs(el_t.a - el_0.a) / max(a0, 1e-300)
        ee += abs(el_t.e - el_0.e)
        ei += abs(wrap_angle(el_t.i - el_0.i))
        eOm += abs(wrap_angle(el_t.Omega - el_0.Omega))
        eom += abs(wrap_angle(el_t.omega - el_0.omega))
        eM += abs(wrap_angle(el_t.M - el_0.M))
    inv = 1.0 / float(n)
    return {
        "E_a": ea * inv,
        "E_e": ee * inv,
        "E_i": ei * inv,
        "E_Omega": eOm * inv,
        "E_omega": eom * inv,
        "E_M": eM * inv,
    }


@dataclass(frozen=True)
class RepErrorSnapshot:
    E_r: float
    E_v: float
    E_a: float
    E_e: float
    E_i: float
    E_Omega: float
    E_omega: float
    E_M: float
    E_energy: float
    R: np.ndarray
    perm: tuple[int, ...]

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in CHANNELS}

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in CHANNELS], dtype=float)


def rep_error_for_perm(
    r: np.ndarray,
    v: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    perm: tuple[int, ...],
    *,
    mu: float,
    E_energy: float = float("nan"),
) -> RepErrorSnapshot:
    """Fixed P → R* from positions → relative E_r/E_v + element channels."""
    closed = closure_for_perm(r, v, r0, v0, perm)
    R = closed.R
    els = element_channel_errors(r, v, r0, v0, R, perm, mu)
    return RepErrorSnapshot(
        E_r=relative_Er(r, r0, R, perm),
        E_v=relative_Ev(v, v0, R, perm),
        E_a=els["E_a"],
        E_e=els["E_e"],
        E_i=els["E_i"],
        E_Omega=els["E_Omega"],
        E_omega=els["E_omega"],
        E_M=els["E_M"],
        E_energy=float(E_energy),
        R=R,
        perm=tuple(perm),
    )


@dataclass
class RepErrorSeries:
    times: np.ndarray
    channels: dict[str, np.ndarray]
    E_energy: np.ndarray
    perm: tuple[int, ...]
    R_final: np.ndarray
    order_0: tuple[int, ...]
    order_final: tuple[int, ...]

    def final_snapshot(self) -> dict[str, float]:
        return {k: float(self.channels[k][-1]) for k in CHANNELS}


def energy_relative_drift(traj: Trajectory) -> np.ndarray:
    E = np.asarray(traj.energies, dtype=float)
    if E.size == 0:
        return np.full(len(traj), np.nan)
    e0 = float(E[0])
    denom = max(abs(e0), 1e-300)
    return np.abs(E - e0) / denom


def rep_error_series(
    traj: Trajectory,
    *,
    mu: float | None = None,
    central_index: int = 0,
    perm: tuple[int, ...] | None = None,
    mode: str = "identity",
) -> RepErrorSeries:
    """
    All 8 channels vs time. P fixed first (identity | fixed_radial | fixed).
    """
    if mu is None:
        if traj.masses is not None and traj.masses.size > central_index:
            mu = float(traj.G * traj.masses[central_index])
        else:
            mu = float(traj.G)

    # COM-frame absolute states for E_r/E_v (translation already removed).
    r_ref, v_ref = fairy_states(traj, 0, central_index=central_index, frame="com")
    order_0 = radial_order(traj.positions[0], central_index=central_index)
    order_f = radial_order(traj.positions[-1], central_index=central_index)
    radial_perm = radial_permutation(order_f, order_0)

    if mode == "fixed":
        if perm is None:
            raise ValueError("mode='fixed' requires perm")
    elif mode == "identity":
        perm = tuple(range(len(r_ref)))
    elif mode == "fixed_radial":
        perm = radial_perm
    else:
        raise ValueError(f"unknown mode {mode!r}")

    T = len(traj)
    ch = {k: np.empty(T) for k in CHANNELS}
    E_en = energy_relative_drift(traj)
    R_last = np.eye(3)
    for k in range(T):
        r, v = fairy_states(traj, k, central_index=central_index, frame="com")
        snap = rep_error_for_perm(r, v, r_ref, v_ref, perm, mu=mu, E_energy=float(E_en[k]))
        for name in CHANNELS:
            ch[name][k] = getattr(snap, name)
        R_last = snap.R

    return RepErrorSeries(
        times=traj.times.copy(),
        channels=ch,
        E_energy=E_en,
        perm=perm,
        R_final=R_last,
        order_0=order_0,
        order_final=order_f,
    )


@dataclass(frozen=True)
class RepSigmas:
    E_r: float = 1.0
    E_v: float = 1.0
    E_a: float = 1.0
    E_e: float = 1.0
    E_i: float = 1.0
    E_Omega: float = 1.0
    E_omega: float = 1.0
    E_M: float = 1.0
    n_samples: int = 0
    source: str = "default"

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in CHANNELS}

    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**asdict(self)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> RepSigmas:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            **{k: float(data[k]) for k in CHANNELS},
            n_samples=int(data.get("n_samples", 0)),
            source=str(data.get("source", "loaded")),
        )


def compute_sigmas(
    samples: list[dict[str, float]] | list[RepErrorSnapshot],
    *,
    floor: float = 1e-16,
    source: str = "rep_error_scan",
) -> RepSigmas:
    """σ_i = std of channel i over samples (population); floored."""
    if not samples:
        return RepSigmas(source="empty")
    mats = []
    for s in samples:
        if isinstance(s, RepErrorSnapshot):
            mats.append(s.vector())
        else:
            mats.append(np.array([float(s[k]) for k in CHANNELS], dtype=float))
    X = np.vstack(mats)
    # Prefer std; if nearly constant use mean abs as scale
    std = np.nanstd(X, axis=0, ddof=0)
    mean_abs = np.nanmean(np.abs(X), axis=0)
    scale = np.where(std > floor, std, np.maximum(mean_abs, floor))
    scale = np.maximum(scale, floor)
    return RepSigmas(
        **{CHANNELS[i]: float(scale[i]) for i in range(8)},
        n_samples=len(samples),
        source=source,
    )


def apply_sigmas(
    snap: RepErrorSnapshot | dict[str, float],
    sigmas: RepSigmas,
) -> dict[str, float]:
    """Ẽ_i = E_i / σ_i."""
    raw = snap.as_dict() if isinstance(snap, RepErrorSnapshot) else {k: float(snap[k]) for k in CHANNELS}
    out = {}
    for k in CHANNELS:
        s = max(float(getattr(sigmas, k)), 1e-300)
        out[k] = float(raw[k]) / s
    return out


def weighted_score(
    tilde: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Σ w_i Ẽ_i. Default equal weights; pass zeros to drop channels."""
    w = weights or {k: 1.0 for k in CHANNELS}
    return float(sum(float(w.get(k, 0.0)) * float(tilde[k]) for k in CHANNELS))


# PEO search default: close on shape/velocity; light element; ignore noisy angles.
SEARCH_SCORE_WEIGHTS: dict[str, float] = {
    "E_r": 1.0,
    "E_v": 1.0,
    "E_a": 0.25,
    "E_e": 0.25,
    "E_i": 0.0,
    "E_Omega": 0.0,
    "E_omega": 0.0,
    "E_M": 0.0,
}
