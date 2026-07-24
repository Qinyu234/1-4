"""Encounter Event index derived from Trajectory (PROMPT §6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.engine.trajectory import Trajectory


@dataclass
class EncounterEvent:
    time: float
    i: int
    j: int
    label_i: str
    label_j: str
    distance: float
    position_mid: np.ndarray


def find_encounters(
    trajectory: Trajectory,
    *,
    threshold: float,
    central_index: int = 0,
    min_separation_steps: int = 1,
) -> list[EncounterEvent]:
    """
    Local minima of pairwise fairy–fairy distance below `threshold`.
    """
    T, N, _ = trajectory.positions.shape
    events: list[EncounterEvent] = []
    fairy_idx = [k for k in range(N) if k != central_index]

    for a, i in enumerate(fairy_idx):
        for j in fairy_idx[a + 1 :]:
            dists = np.linalg.norm(
                trajectory.positions[:, i, :] - trajectory.positions[:, j, :],
                axis=1,
            )
            for t in range(1, T - 1):
                if dists[t] > threshold:
                    continue
                if dists[t] <= dists[t - 1] and dists[t] <= dists[t + 1]:
                    if events and events[-1].i == i and events[-1].j == j:
                        if t - int(np.searchsorted(trajectory.times, events[-1].time)) < min_separation_steps:
                            # replace if closer
                            if dists[t] < events[-1].distance:
                                events[-1] = EncounterEvent(
                                    time=float(trajectory.times[t]),
                                    i=i,
                                    j=j,
                                    label_i=trajectory.labels[i],
                                    label_j=trajectory.labels[j],
                                    distance=float(dists[t]),
                                    position_mid=0.5
                                    * (
                                        trajectory.positions[t, i]
                                        + trajectory.positions[t, j]
                                    ).copy(),
                                )
                            continue
                    events.append(
                        EncounterEvent(
                            time=float(trajectory.times[t]),
                            i=i,
                            j=j,
                            label_i=trajectory.labels[i],
                            label_j=trajectory.labels[j],
                            distance=float(dists[t]),
                            position_mid=0.5
                            * (
                                trajectory.positions[t, i] + trajectory.positions[t, j]
                            ).copy(),
                        )
                    )
    events.sort(key=lambda e: e.time)
    return events
