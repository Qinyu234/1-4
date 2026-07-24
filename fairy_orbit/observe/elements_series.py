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
