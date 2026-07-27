"""Osculating orbital-element time series from a Trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.design.elements import OrbitalElements
from fairy_orbit.engine.trajectory import Trajectory


@dataclass
class ElementSeries:
    times: np.ndarray
    labels: list[str]
    a: np.ndarray  # (T, N_fairy)
    e: np.ndarray
    i: np.ndarray
    omega: np.ndarray
    Omega: np.ndarray
    M: np.ndarray


def extract_element_series(
    trajectory: Trajectory,
    mu: float,
    *,
    central_index: int = 0,
) -> ElementSeries:
    """Extract osculating elements for all non-central bodies."""
    T, N, _ = trajectory.positions.shape
    fairy_idx = [i for i in range(N) if i != central_index]
    labels = [trajectory.labels[i] for i in fairy_idx]
    n_f = len(fairy_idx)

    a = np.zeros((T, n_f))
    e = np.zeros((T, n_f))
    inc = np.zeros((T, n_f))
    omega = np.zeros((T, n_f))
    Omega = np.zeros((T, n_f))
    M = np.zeros((T, n_f))

    for t in range(T):
        r0 = trajectory.positions[t, central_index]
        v0 = trajectory.velocities[t, central_index]
        for j, i in enumerate(fairy_idx):
            r = trajectory.positions[t, i] - r0
            v = trajectory.velocities[t, i] - v0
            el = OrbitalElements.from_state(r, v, mu)
            a[t, j] = el.a
            e[t, j] = el.e
            inc[t, j] = el.i
            omega[t, j] = el.omega
            Omega[t, j] = el.Omega
            M[t, j] = el.M

    return ElementSeries(
        times=trajectory.times.copy(),
        labels=labels,
        a=a,
        e=e,
        i=inc,
        omega=omega,
        Omega=Omega,
        M=M,
    )


def extract_aei_series(
    trajectory: Trajectory,
    mu: float,
    *,
    central_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized (a, e, i) arrays of shape (T, N_fairy) for all bodies at once.

    Only computes the elements the dynamics score needs (a, e, i), skipping the
    per-sample Python object construction of `extract_element_series`.
    """
    pos = trajectory.positions
    vel = trajectory.velocities
    T, N, _ = pos.shape
    fairy_idx = [k for k in range(N) if k != central_index]

    r0 = pos[:, central_index : central_index + 1, :]
    v0 = vel[:, central_index : central_index + 1, :]
    R = pos[:, fairy_idx, :] - r0  # (T, n, 3)
    V = vel[:, fairy_idx, :] - v0

    r = np.sqrt(np.sum(R * R, axis=2))  # (T, n)
    v2 = np.sum(V * V, axis=2)
    energy = 0.5 * v2 - mu / np.maximum(r, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(np.abs(energy) > 1e-14, -mu / (2.0 * energy), np.inf)

    # h = R × V
    hx = R[..., 1] * V[..., 2] - R[..., 2] * V[..., 1]
    hy = R[..., 2] * V[..., 0] - R[..., 0] * V[..., 2]
    hz = R[..., 0] * V[..., 1] - R[..., 1] * V[..., 0]
    h_mag = np.sqrt(hx * hx + hy * hy + hz * hz)

    # e_vec = (V × h)/mu − R/r
    ex = (V[..., 1] * hz - V[..., 2] * hy) / mu - R[..., 0] / np.maximum(r, 1e-300)
    ey = (V[..., 2] * hx - V[..., 0] * hz) / mu - R[..., 1] / np.maximum(r, 1e-300)
    ez = (V[..., 0] * hy - V[..., 1] * hx) / mu - R[..., 2] / np.maximum(r, 1e-300)
    e = np.sqrt(ex * ex + ey * ey + ez * ez)

    with np.errstate(invalid="ignore"):
        inc = np.where(
            h_mag > 1e-14,
            np.arccos(np.clip(hz / np.maximum(h_mag, 1e-300), -1.0, 1.0)),
            0.0,
        )
    return a, e, inc
