"""Orbital ladder initial-condition builder (PROMPT §3 / §5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairy_orbit.core.body import Body, System
from fairy_orbit.core.config import SystemConfig
from fairy_orbit.design.elements import OrbitalElements, orbital_period
from fairy_orbit.design.tetrahedron import (
    FAIRY_ORDER,
    tetrahedral_ladder_states,
    tetrahedral_phase_offsets,
)


# Adjacent period ratios: T2/T1 ≈ 3/2, T3/T2 ≈ 5/3, T4/T3 ≈ 7/5
DEFAULT_PERIOD_RATIOS: tuple[float, float, float] = (3.0 / 2.0, 5.0 / 3.0, 7.0 / 5.0)


@dataclass
class LadderParams:
    """Parameters for a four-rung orbital ladder."""

    eccentricity: float = 0.15
    a_inner: float = 1.0
    period_ratios: tuple[float, float, float] = DEFAULT_PERIOD_RATIOS
    inclination: float = 0.0  # unused when tetrahedral=True
    base_omega: float = 0.0
    base_Omega: float = 0.0
    phase_offsets: dict[str, float] | None = None
    fairy_names: tuple[str, ...] = FAIRY_ORDER
    # PROMPT §5: non-coplanar tetrahedral periapsis directions (default on)
    tetrahedral: bool = True

    def semi_major_axes(self, mu: float) -> list[float]:
        """Nested a from a_inner and adjacent period ratios (Kepler: T ∝ a^{3/2})."""
        a = [float(self.a_inner)]
        for ratio in self.period_ratios:
            a.append(a[-1] * float(ratio) ** (2.0 / 3.0))
        return a

    def period_targets(self, mu: float) -> list[float]:
        return [orbital_period(ai, mu) for ai in self.semi_major_axes(mu)]


def build_orbital_ladder(
    config: SystemConfig,
    params: LadderParams | None = None,
) -> System:
    """
    Build 1 central + 4 fairy System on an orbital ladder.

    Default: each fairy's periapsis lies along a regular-tetrahedron vertex
    (non-coplanar, mutually asymmetric). Not the abandoned same-radius
    Rodrigues copy — nested a's still follow the period ladder.
    """
    params = params or LadderParams()
    mu = config.mu
    axes = params.semi_major_axes(mu)

    if len(axes) != len(params.fairy_names):
        raise ValueError("Need one semi-major axis per fairy")

    central = Body(
        mass=config.central_mass,
        position=np.zeros(3),
        velocity=np.zeros(3),
        name="central",
        radius=config.central_radius,
    )

    fairies: list[Body] = []
    if params.tetrahedral:
        states = tetrahedral_ladder_states(
            axes, params.eccentricity, mu, names=params.fairy_names
        )
        for name in params.fairy_names:
            r, v = states[name]
            fairies.append(
                Body(
                    mass=config.fairy_mass,
                    position=r,
                    velocity=v,
                    name=name,
                    radius=config.fairy_radius,
                )
            )
    else:
        # Legacy coplanar ladder: only mean-anomaly offsets from tetra azimuth
        phases = params.phase_offsets or tetrahedral_phase_offsets()
        for name, a in zip(params.fairy_names, axes, strict=True):
            elems = OrbitalElements(
                a=a,
                e=params.eccentricity,
                i=params.inclination,
                omega=params.base_omega,
                Omega=params.base_Omega,
                M=float(phases.get(name, 0.0)),
            )
            r, v = elems.to_state(mu)
            fairies.append(
                Body(
                    mass=config.fairy_mass,
                    position=r,
                    velocity=v,
                    name=name,
                    radius=config.fairy_radius,
                )
            )

    bodies = [central, *fairies]
    return System(bodies=bodies, G=config.G, labels=[b.name for b in bodies])


def ladder_period_ratios(system: System, config: SystemConfig) -> list[float]:
    """Measured adjacent period ratios from fairy osculating a at t=0."""
    mu = config.mu
    central = system.bodies[0]
    periods: list[float] = []
    for body in system.bodies[1:]:
        r = body.position - central.position
        v = body.velocity - central.velocity
        elems = OrbitalElements.from_state(r, v, mu)
        periods.append(elems.period(mu))
    return [periods[i + 1] / periods[i] for i in range(len(periods) - 1)]
