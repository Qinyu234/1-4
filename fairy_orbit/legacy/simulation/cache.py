"""SQLite-backed trajectory cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from fairy_orbit.physics.body import System
from fairy_orbit.simulation.trajectory import Trajectory


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _system_signature(system: System) -> dict[str, Any]:
    positions = system.positions()
    velocities = system.velocities()
    masses = system.masses()
    return {
        "n": int(system.n),
        "g": float(system.G),
        "masses": [float(x) for x in masses],
        "positions": positions.reshape(-1).tolist(),
        "velocities": velocities.reshape(-1).tolist(),
        "labels": list(system.labels),
    }


def _simulation_signature(dt: float, t_end: float, record_every: int, solver_type: str) -> dict[str, Any]:
    return {
        "dt": float(dt),
        "t_end": float(t_end),
        "record_every": int(record_every),
        "solver_type": solver_type,
    }


def build_cache_key(system: System, dt: float, t_end: float, record_every: int, solver_type: str) -> str:
    payload = {
        "system": _system_signature(system),
        "simulation": _simulation_signature(dt, t_end, record_every, solver_type),
        "kind": "trajectory(seed)",
    }
    return _hash_payload(payload)


def _resolve_cache_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.suffix:
        return candidate
    return candidate / "trajectory_cache.sqlite"


def ensure_cache_db(path: str | Path | None) -> sqlite3.Connection | None:
    if path is None:
        return None
    db_path = _resolve_cache_path(path)
    if db_path is None:
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trajectories (
            cache_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_cached_trajectory(conn: sqlite3.Connection | None, cache_key: str) -> Trajectory | None:
    if conn is None:
        return None
    row = conn.execute(
        "SELECT payload FROM trajectories WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row[0])
    positions = np.asarray(payload["positions"], dtype=float)
    velocities = np.asarray(payload.get("velocities", np.zeros_like(positions)), dtype=float)
    energies = np.asarray(payload.get("energies", np.zeros(positions.shape[0])), dtype=float)
    angular_momenta = np.asarray(payload.get("angular_momenta", np.zeros((positions.shape[0], 3))), dtype=float)
    times = np.asarray(payload.get("times", np.arange(positions.shape[0], dtype=float)), dtype=float)
    return Trajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        energies=energies,
        angular_momenta=angular_momenta,
        labels=payload.get("labels", []),
        G=payload.get("G", 1.0),
        masses=np.asarray(payload.get("masses", []), dtype=float),
    )


def save_cached_trajectory(conn: sqlite3.Connection | None, cache_key: str, traj: Trajectory) -> None:
    if conn is None:
        return
    payload = {
        "times": traj.times.tolist(),
        "positions": traj.positions.tolist(),
        "velocities": traj.velocities.tolist(),
        "energies": traj.energies.tolist(),
        "angular_momenta": traj.angular_momenta.tolist(),
        "labels": traj.labels,
        "G": float(traj.G),
        "masses": [float(x) for x in traj.masses] if traj.masses is not None else [],
    }
    conn.execute(
        "INSERT OR REPLACE INTO trajectories(cache_key, kind, payload, created_at) VALUES (?, ?, ?, datetime('now'))",
        (cache_key, "trajectory(seed)", json.dumps(payload),),
    )
    conn.commit()
