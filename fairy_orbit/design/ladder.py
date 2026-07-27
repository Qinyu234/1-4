"""Hierarchical resonant orbit chain IC builder (PROMPT §2.4 / §3 / §5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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

# PROMPT §2.4 stage 1: T1,T3 prograde; T2,T4 retrograde (adjacent pairs counter-rotating)
ALTERNATING_SENSE: dict[str, int] = {"T1": +1, "T2": -1, "T3": +1, "T4": -1}

Geometry = Literal["planar_alternating", "tetrahedral_3d", "calibration"]


@dataclass
class LadderParams:
    """Parameters for a four-rung hierarchical resonant chain."""

    eccentricity: float = 0.15
    a_inner: float = 1.0
    period_ratios: tuple[float, float, float] = DEFAULT_PERIOD_RATIOS
    inclination: float = 0.0  # unused for planar_alternating / tetrahedral_3d / calibration
    base_omega: float = 0.0
    base_Omega: float = 0.0
    phase_offsets: dict[str, float] | None = None
    fairy_names: tuple[str, ...] = FAIRY_ORDER
    # PROMPT §2.4 / §5 default: planar + alternating pro/retro + tetrahedral M offsets
    geometry: Geometry = "planar_alternating"
    # Backward-compat flag; ignored when geometry is set explicitly at call sites
    tetrahedral: bool | None = None

    def resolved_geometry(self) -> Geometry:
        if self.tetrahedral is True:
            return "tetrahedral_3d"
        if self.tetrahedral is False and self.geometry == "planar_alternating":
            # Explicit tetrahedral=False with default geometry → planar alternating
            return "planar_alternating"
        if self.tetrahedral is False:
            return self.geometry
        return self.geometry

    def semi_major_axes(self, mu: float) -> list[float]:
        """Nested a from a_inner and adjacent period ratios (Kepler: T ∝ a^{3/2})."""
        if self.resolved_geometry() == "calibration":
            return [float(self.a_inner)] * len(self.fairy_names)
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
    Build 1 central + 4 fairy System.

    Geometries (PROMPT §2.4):
    - planar_alternating: coplanar, nested a, T1/T3 prograde + T2/T4 retrograde,
      tetrahedral azimuth as mean-anomaly offsets (stage-1 / §5 default).
    - tetrahedral_3d: nested a, periapsis along tetrahedron vertices (non-coplanar).
    - calibration: same a,e; Rodrigues (v_rad,v_tan) copy — regular tetrahedron
      at every later time (PROMPT §2.4.1 noise baseline only).
    """
    params = params or LadderParams()
    mu = config.mu
    geom = params.resolved_geometry()
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

    if geom == "calibration":
        fairies = _fairies_calibration(config, params, axes)
    elif geom == "tetrahedral_3d":
        fairies = _fairies_tetrahedral_3d(config, params, axes)
    else:
        fairies = _fairies_planar_alternating(config, params, axes)

    # Central stays fixed at the origin (CanonicalUnits / heavy-primary limit).
    # Do not dump fairy momentum onto it — that would spoil the Kepler (a,e)
    # we just built. Legacy search did cancel because the planet was free.

    bodies = [central, *fairies]
    return System(bodies=bodies, G=config.G, labels=[b.name for b in bodies])


def _body(config: SystemConfig, name: str, r: np.ndarray, v: np.ndarray) -> Body:
    return Body(
        mass=config.fairy_mass,
        position=np.asarray(r, dtype=float),
        velocity=np.asarray(v, dtype=float),
        name=name,
        radius=config.fairy_radius,
    )


def _fairies_tetrahedral_3d(
    config: SystemConfig, params: LadderParams, axes: list[float]
) -> list[Body]:
    states = tetrahedral_ladder_states(
        axes, params.eccentricity, config.mu, names=params.fairy_names
    )
    return [_body(config, name, *states[name]) for name in params.fairy_names]


def _fairies_calibration(
    config: SystemConfig, params: LadderParams, axes: list[float]
) -> list[Body]:
    """Equal a,e; Rodrigues Newton (r,v) — orbit form is the same state rewritten."""
    from fairy_orbit.design.tetrahedron import calibration_tetrahedron_states

    states = calibration_tetrahedron_states(
        axes[0], params.eccentricity, config.mu, names=params.fairy_names
    )
    return [_body(config, name, *states[name]) for name in params.fairy_names]


def _fairies_planar_alternating(
    config: SystemConfig, params: LadderParams, axes: list[float]
) -> list[Body]:
    """
    Coplanar resonant chain with alternating sense (PROMPT §2.4 stage 1).

    Adjacent pairs are counter-rotating ⇒ larger v∞, smaller deflection per
    encounter — matches the "many weak kicks" target.
    """
    phases = params.phase_offsets or tetrahedral_phase_offsets()
    out: list[Body] = []
    for name, a in zip(params.fairy_names, axes, strict=True):
        sense = ALTERNATING_SENSE.get(name, +1)
        i = 0.0 if sense > 0 else float(np.pi)
        elems = OrbitalElements(
            a=a,
            e=params.eccentricity,
            i=i,
            omega=params.base_omega,
            Omega=params.base_Omega,
            M=float(phases.get(name, 0.0)),
        )
        r, v = elems.to_state(config.mu)
        # Force exact planarity (numerical noise from i=π path)
        r = np.array([r[0], r[1], 0.0])
        v = np.array([v[0], v[1], 0.0])
        out.append(_body(config, name, r, v))
    return out


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
