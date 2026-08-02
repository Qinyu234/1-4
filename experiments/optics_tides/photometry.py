"""Central-observer photometry via ``starry`` reflected-light ``Map.flux``.

Engine
------
Uniform Lambertian spheres: ``starry.Map(ydeg=1, reflected=True)``, ``amp = A``
(spherical albedo). Illumination source at ``(xs, ys, zs)`` in units of the
body radius; observer along ``+z`` after a local rotation.

Starry normalization (full phase, unit radius, unit source distance)::

    flux → (2/3) A     # geometric albedo of a Lambert sphere

So disk-integrated brightness we report is::

    far:   L = flux(xs,ys,zs) * r_s² * R²     # cancel 1/r_s² → (2/3) A Φ(α) R²
    near:  L = flux(xs,ys,zs) * R²            # keep 1/r_s² from finite Sun

1 Sun + 3 Moons: tagged Sun is self-luminous (``L_sun``); other fairies reflect.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from fairy_orbit.engine.trajectory import Trajectory

# Theano (starry backend) + NumPy 2: force Python ops (no C compile / elsize).
os.environ.setdefault("THEANO_FLAGS", "device=cpu,floatX=float64,cxx=")

DEFAULT_ALBEDO = 0.25
DEFAULT_ALBEDO_EARTH = 0.30
DEFAULT_L_SUN = 1.0
DEFAULT_PHASE_POWER = 1.0  # CLI compat; unused with starry
FAR_SOURCE_RADII = 1.0e6

Illumination = Literal["far", "near"]

_MAP = None
_STARRY_READY = False


def fairy_indices(n_bodies: int, *, central_index: int = 0) -> list[int]:
    return [k for k in range(int(n_bodies)) if k != int(central_index)]


def _ensure_starry():
    global _MAP, _STARRY_READY
    if _STARRY_READY:
        return _MAP
    # starry 1.2 + numpy≥2: ``np.ones(None)`` in Map.reset
    _orig = np.ones

    def _ones(shape, *args, **kwargs):  # type: ignore[no-untyped-def]
        if shape is None:
            return np.asarray(1.0)
        return _orig(shape, *args, **kwargs)

    np.ones = _ones  # type: ignore[assignment]
    import starry

    starry.config.lazy = False
    starry.config.quiet = True
    _MAP = starry.Map(ydeg=1, reflected=True)
    _STARRY_READY = True
    return _MAP


def _rotation_obs_to_z(obs_hat: np.ndarray) -> np.ndarray:
    """3×3 R with ``R @ obs_hat = (0,0,1)`` (rows = new basis in world)."""
    z = np.asarray(obs_hat, dtype=float).ravel()
    zn = float(np.linalg.norm(z))
    if zn < 1e-300:
        return np.eye(3)
    z = z / zn
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    xn = float(np.linalg.norm(x))
    if xn < 1e-300:
        tmp = np.array([0.0, 0.0, 1.0])
        x = np.cross(tmp, z)
        xn = float(np.linalg.norm(x))
    x = x / xn
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def starry_reflected_flux(
    R: float,
    A: float,
    r_body: np.ndarray,
    *,
    s_hat: np.ndarray | None = None,
    r_source: np.ndarray | None = None,
    illumination: Illumination = "far",
    far_source_radii: float = FAR_SOURCE_RADII,
) -> tuple[float, float]:
    """
    Return ``(L, alpha_deg)`` for one reflecting sphere.

    ``r_body`` / ``r_source`` are Planet-relative. Observer at Planet.
    """
    R = float(R)
    A = float(A)
    if R <= 0.0 or A <= 0.0:
        return 0.0, 180.0

    r = np.asarray(r_body, dtype=float).ravel()
    d_obs = float(np.linalg.norm(r))
    if d_obs < 1e-300:
        return 0.0, 180.0

    # From body toward observer (Planet)
    obs_hat = -r / d_obs
    Rot = _rotation_obs_to_z(obs_hat)

    if illumination == "far":
        if s_hat is None:
            raise ValueError("far illumination needs s_hat")
        s = np.asarray(s_hat, dtype=float).ravel()
        sn = float(np.linalg.norm(s))
        if sn < 1e-300:
            return 0.0, 180.0
        s = s / sn
        src_radii = Rot @ (s * float(far_source_radii))
        rs = float(far_source_radii)
        # cos α = −r̂·ŝ  (Planet at origin, far sun)
        cα = float(np.clip(-np.dot(r / d_obs, s), -1.0, 1.0))
    else:
        if r_source is None:
            raise ValueError("near illumination needs r_source")
        src_world = np.asarray(r_source, dtype=float).ravel() - r
        dist = float(np.linalg.norm(src_world))
        if dist < 1e-300 or R < 1e-300:
            return 0.0, 180.0
        src_radii = Rot @ (src_world / R)
        rs = dist / R
        # cos α from (−r)·(r_s−r)
        cα = float(
            np.clip(np.dot(-r, src_world) / (d_obs * dist), -1.0, 1.0)
        )

    m = _ensure_starry()
    m.amp = A
    flux = float(
        np.asarray(
            m.flux(
                xs=float(src_radii[0]),
                ys=float(src_radii[1]),
                zs=float(src_radii[2]),
            )
        ).ravel()[0]
    )
    if not np.isfinite(flux) or flux < 0.0:
        flux = 0.0

    if illumination == "far":
        L = flux * (rs * rs) * (R * R)
    else:
        L = flux * (R * R)

    alpha_deg = math.degrees(math.acos(cα))
    return float(L), float(alpha_deg)


def rotate_sun_at_swap(sun: int, body_i: int, body_j: int) -> int:
    if sun == body_i:
        return int(body_j)
    if sun == body_j:
        return int(body_i)
    return int(sun)


def sun_role_timeline(
    n_frames: int,
    swaps: Sequence[Any],
    *,
    initial_sun: int,
    role_rotate: bool = True,
) -> np.ndarray:
    sun = int(initial_sun)
    ordered = sorted(swaps, key=lambda s: int(s.swap_frame)) if role_rotate else []
    out = np.empty(int(n_frames), dtype=int)
    k = 0
    for t in range(int(n_frames)):
        out[t] = sun
        while k < len(ordered) and int(ordered[k].swap_frame) == t:
            sun = rotate_sun_at_swap(
                sun, int(ordered[k].body_i), int(ordered[k].body_j)
            )
            k += 1
    return out


def _ray_sphere_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    t_max: float,
) -> bool:
    o = np.asarray(origin, dtype=float).ravel()
    d = np.asarray(direction, dtype=float).ravel()
    c = np.asarray(center, dtype=float).ravel()
    dn = float(np.linalg.norm(d))
    if dn < 1e-300:
        return False
    d = d / dn
    oc = o - c
    b = float(np.dot(oc, d))
    disc = b * b - float(np.dot(oc, oc)) + float(radius) * float(radius)
    if disc < 0.0:
        return False
    sd = math.sqrt(disc)
    for t in (-b - sd, -b + sd):
        if 1e-9 < t < t_max - 1e-9:
            return True
    return False


def is_occulted(
    traj: Trajectory,
    frame: int,
    target: int,
    *,
    central_index: int,
    radii: np.ndarray,
    check_planet: bool = True,
    check_fairies: bool = True,
) -> bool:
    pos = traj.positions[int(frame)]
    r_c = pos[int(central_index)]
    r_t = pos[int(target)]
    R_c = float(radii[int(central_index)])
    to_t = r_t - r_c
    dist = float(np.linalg.norm(to_t))
    if dist <= R_c + 1e-12:
        return True
    rhat = to_t / dist
    obs = r_c + R_c * rhat
    los = r_t - obs
    los_n = float(np.linalg.norm(los))
    if los_n < 1e-300:
        return True
    if check_planet and _ray_sphere_hit(obs, los, r_c, R_c, t_max=los_n):
        return True
    if check_fairies:
        for k in range(traj.n_bodies):
            if k in (int(central_index), int(target)):
                continue
            if _ray_sphere_hit(obs, los, pos[k], float(radii[k]), t_max=los_n):
                return True
    return False


@dataclass(frozen=True)
class PhotoConfig:
    central_index: int
    L_sun: float
    A: float
    A_earth: float
    initial_sun: int
    illumination: str
    role_rotate: bool
    occultation: bool
    engine: str = "starry"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FluxSample:
    time: float
    frame: int
    sun_index: int
    s_hat: tuple[float, float, float]
    fairy_indices: tuple[int, ...]
    L_fairies: tuple[float, ...]
    occulted: tuple[bool, ...]
    alpha_deg: tuple[float, ...]
    F_sun: float
    F_moons: float
    F_total: float
    r_SO: float
    engine: str = "starry"


def _s_hat_far(pos: np.ndarray, sun_index: int, central_index: int) -> np.ndarray:
    v = pos[int(sun_index)] - pos[int(central_index)]
    n = float(np.linalg.norm(v))
    if n < 1e-300:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def flux_at_frame(
    traj: Trajectory,
    frame: int,
    radii: np.ndarray,
    *,
    sun_index: int,
    central_index: int,
    L_sun: float,
    A: float,
    illumination: Illumination = "far",
    occultation: bool = True,
    s_hat_fixed: np.ndarray | None = None,
    phase_power: float | None = None,  # unused; CLI/API compat
) -> FluxSample:
    """Planet-centric flux via starry reflected maps + self-luminous Sun."""
    del phase_power
    t = int(frame)
    pos = traj.positions[t]
    r_c = pos[int(central_index)]
    r_s = pos[int(sun_index)]
    fairies = tuple(fairy_indices(traj.n_bodies, central_index=central_index))

    s_hat = (
        np.asarray(s_hat_fixed, dtype=float).ravel()
        if s_hat_fixed is not None
        else _s_hat_far(pos, sun_index, central_index)
    )
    sn = float(np.linalg.norm(s_hat))
    s_hat = s_hat / sn if sn > 1e-300 else np.array([1.0, 0.0, 0.0])

    r_SO = float(np.linalg.norm(r_s - r_c))
    if illumination == "far":
        F_sun = float(L_sun)
    else:
        F_sun = (
            float("inf")
            if r_SO < 1e-300
            else float(L_sun) / (4.0 * math.pi * r_SO * r_SO)
        )

    L_list: list[float] = []
    occ_list: list[bool] = []
    alpha_list: list[float] = []

    for i in fairies:
        occ = (
            is_occulted(
                traj, t, i, central_index=central_index, radii=radii
            )
            if occultation
            else False
        )
        occ_list.append(occ)

        if i == int(sun_index):
            # self-luminous Sun: no reflected self-term
            L_list.append(0.0)
            alpha_list.append(0.0)
            continue

        if occ:
            L_list.append(0.0)
            r_i = pos[i] - r_c
            if illumination == "far":
                cα = float(
                    np.clip(
                        -np.dot(
                            r_i / (float(np.linalg.norm(r_i)) + 1e-300), s_hat
                        ),
                        -1.0,
                        1.0,
                    )
                )
            else:
                v = (r_s - r_c) - r_i
                cα = float(
                    np.clip(
                        np.dot(-r_i, v)
                        / (
                            (float(np.linalg.norm(r_i)) + 1e-300)
                            * (float(np.linalg.norm(v)) + 1e-300)
                        ),
                        -1.0,
                        1.0,
                    )
                )
            alpha_list.append(math.degrees(math.acos(cα)))
            continue

        L, a_deg = starry_reflected_flux(
            float(radii[i]),
            float(A),
            pos[i] - r_c,
            s_hat=s_hat,
            r_source=(r_s - r_c),
            illumination=illumination,
        )
        L_list.append(L)
        alpha_list.append(a_deg)

    F_moons = float(sum(L_list))
    return FluxSample(
        time=float(traj.times[t]),
        frame=t,
        sun_index=int(sun_index),
        s_hat=(float(s_hat[0]), float(s_hat[1]), float(s_hat[2])),
        fairy_indices=fairies,
        L_fairies=tuple(L_list),
        occulted=tuple(occ_list),
        alpha_deg=tuple(alpha_list),
        F_sun=float(F_sun),
        F_moons=F_moons,
        F_total=float(F_sun + F_moons),
        r_SO=r_SO,
        engine="starry",
    )


def flux_series_with_roles(
    traj: Trajectory,
    radii: np.ndarray,
    sun_per_frame: np.ndarray,
    *,
    central_index: int,
    L_sun: float,
    A: float,
    illumination: Illumination = "far",
    occultation: bool = True,
    s_hat_fixed: np.ndarray | None = None,
    phase_power: float | None = None,
) -> list[FluxSample]:
    return [
        flux_at_frame(
            traj,
            t,
            radii,
            sun_index=int(sun_per_frame[t]),
            central_index=central_index,
            L_sun=L_sun,
            A=A,
            illumination=illumination,
            occultation=occultation,
            s_hat_fixed=s_hat_fixed,
            phase_power=phase_power,
        )
        for t in range(len(traj))
    ]


def period_variance_stats(F: np.ndarray) -> dict[str, float]:
    F = np.asarray(F, dtype=float)
    mean = float(np.mean(F))
    var = float(np.var(F))
    std = float(np.std(F))
    return {
        "F_mean": mean,
        "F_var": var,
        "F_std": std,
        "F_cv": float(std / (mean + 1e-300)),
        "F_min": float(np.min(F)),
        "F_max": float(np.max(F)),
        "F_contrast": float((np.max(F) - np.min(F)) / (mean + 1e-300)),
    }
