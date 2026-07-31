"""PROMPT §3.3 classification helpers: P cycle structure + action magnitude."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from fairy_orbit.core.body import kinetic_energy, potential_energy
from fairy_orbit.design.seeds import OrbitSeed


def action_proxy(seed: OrbitSeed) -> float:
    """
    Cheap Lagrangian proxy S ≈ T · (K − U) at the IC (COM frame).

    Enough for §3.3 magnitude bucketing; not a variational action integral.
    """
    sys = seed.to_system()
    return float(seed.period) * (kinetic_energy(sys) - potential_energy(sys))


def action_magnitude_bucket(action: float, *, bins_per_decade: float = 2.0) -> int:
    mag = abs(float(action)) + 1e-300
    return int(round(math.log10(mag) * float(bins_per_decade)))


def family_classification_key(
    seed: OrbitSeed,
    *,
    perm_label: str,
    action: float | None = None,
) -> str:
    """
    Compact §3.3 family id: N + P cycle label + action log-magnitude bucket.
    """
    s = float(action) if action is not None else action_proxy(seed)
    bucket = action_magnitude_bucket(s)
    return f"n{int(seed.n_bodies)}|{perm_label}|S{bucket}"


@dataclass
class FamilyHitAnnealer:
    """
    Drive away-start probability from rolling *new-family hit rate*.

    Each scout-level accept reports a §3.3 family key. Hit rate = fraction of
    recent scout accepts whose key was new. Low hit rate (rediscovering) →
    raise away_prob; high hit rate → keep mining near baseline.
    """

    window: int = 40
    warmup: int = 24
    away_min: float = 0.05
    away_max: float = 0.9
    seen: set[str] = field(default_factory=set)
    _new_flags: deque[bool] = field(default_factory=deque, repr=False)
    n_obs: int = 0
    n_new: int = 0

    def __post_init__(self) -> None:
        self._new_flags = deque(maxlen=int(self.window))
        self.seen = set(self.seen)

    def seed_seen(self, keys: Iterable[str]) -> None:
        for k in keys:
            if k:
                self.seen.add(str(k))

    def observe_scout(self, family_key: str) -> bool:
        """Record one scout accept; return True if family key is new."""
        key = str(family_key)
        is_new = key not in self.seen
        if is_new:
            self.seen.add(key)
            self.n_new += 1
        self._new_flags.append(is_new)
        self.n_obs += 1
        return is_new

    def family_hit_rate(self) -> float | None:
        if not self._new_flags:
            return None
        return float(sum(self._new_flags) / len(self._new_flags))

    def away_prob(self) -> float:
        if self.n_obs < int(self.warmup) or len(self._new_flags) < 4:
            return float(self.away_min)
        rate = self.family_hit_rate()
        assert rate is not None
        # rediscovery → away; novelty → stay baseline
        return float(
            np.clip(
                self.away_min + (1.0 - rate) * (self.away_max - self.away_min),
                self.away_min,
                self.away_max,
            )
        )

    def status(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "n_new": self.n_new,
            "n_seen_families": len(self.seen),
            "family_hit_rate": self.family_hit_rate(),
            "away_prob": self.away_prob(),
            "window": int(self.window),
            "warmup": int(self.warmup),
        }
