"""SQLite orbit-result store, keyed / classed by initial parameters."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fairy_orbit.core.config import SystemConfig
from fairy_orbit.design.ladder import DEFAULT_PERIOD_RATIOS, LadderParams
from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.diagnose import Diagnosis

DEFAULT_DB = Path("experiments/output/orbit_db/orbits.sqlite")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def period_ratio_scale(ratios: tuple[float, ...] | list[float]) -> float:
    """Geometric-mean scale vs default 3:2 / 5:3 / 7:5."""
    base = DEFAULT_PERIOD_RATIOS
    logs = [math.log(float(ratios[i]) / float(base[i])) for i in range(3)]
    return float(math.exp(sum(logs) / 3.0))


def make_param_class(
    *,
    eccentricity: float,
    mass_ratio: float,
    tetrahedral: bool,
    period_ratios: tuple[float, ...] | list[float],
    a_inner: float = 1.0,
) -> str:
    """
    Classification key for browsing / querying by IC family.

    Example: e0.15_mu1e-03_tet1_s1.00_a1.00
    """
    scale = period_ratio_scale(period_ratios)
    return (
        f"e{eccentricity:.2f}_mu{mass_ratio:.0e}_"
        f"tet{int(tetrahedral)}_s{scale:.2f}_a{a_inner:.2f}"
    )


@dataclass
class RunRecord:
    id: int
    created_at: str
    source: str
    param_class: str
    eccentricity: float
    mass_ratio: float
    a_inner: float
    period_ratios: list[float]
    tetrahedral: bool
    t_end: float
    n_outputs: int
    status: str
    interest: float | None
    a_delta_rms: float | None
    a_ptp_mean: float | None
    n_encounters: int | None
    a_order_changed: bool | None
    megno: float | None
    energy_drift: float | None
    summary: dict
    traj_path: str | None = None


class OrbitStore:
    """Persist diagnoses in SQLite; trajectories as sidecar .npz files."""

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.traj_dir = self.db_path.parent / "trajectories"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> OrbitStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                param_class TEXT NOT NULL,
                eccentricity REAL NOT NULL,
                mass_ratio REAL NOT NULL,
                a_inner REAL NOT NULL,
                period_r01 REAL NOT NULL,
                period_r12 REAL NOT NULL,
                period_r23 REAL NOT NULL,
                tetrahedral INTEGER NOT NULL,
                t_end REAL NOT NULL,
                n_outputs INTEGER NOT NULL,
                status TEXT NOT NULL,
                interest REAL,
                a_delta_rms REAL,
                a_ptp_mean REAL,
                n_encounters INTEGER,
                a_order_changed INTEGER,
                megno REAL,
                energy_drift REAL,
                summary_json TEXT NOT NULL,
                traj_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_runs_param_class ON runs(param_class);
            CREATE INDEX IF NOT EXISTS idx_runs_e_mu ON runs(eccentricity, mass_ratio);
            CREATE INDEX IF NOT EXISTS idx_runs_interest ON runs(interest DESC);
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            """
        )
        self._conn.commit()

    def save(
        self,
        *,
        config: SystemConfig,
        params: LadderParams,
        diagnosis: Diagnosis,
        source: str = "ladder",
        t_end: float,
        n_outputs: int,
        store_trajectory: bool = True,
        elapsed_s: float | None = None,
    ) -> int:
        ratios = list(params.period_ratios)
        while len(ratios) < 3:
            ratios.append(1.0)
        summary = dict(diagnosis.summary)
        if elapsed_s is not None:
            summary["elapsed_s"] = elapsed_s
        param_class = make_param_class(
            eccentricity=params.eccentricity,
            mass_ratio=config.mass_ratio,
            tetrahedral=params.resolved_geometry() == "tetrahedral_3d",
            period_ratios=params.period_ratios,
            a_inner=params.a_inner,
        )

        cur = self._conn.execute(
            """
            INSERT INTO runs (
                created_at, source, param_class,
                eccentricity, mass_ratio, a_inner,
                period_r01, period_r12, period_r23, tetrahedral,
                t_end, n_outputs, status,
                interest, a_delta_rms, a_ptp_mean, n_encounters,
                a_order_changed, megno, energy_drift,
                summary_json, traj_path
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                _now(),
                source,
                param_class,
                float(params.eccentricity),
                float(config.mass_ratio),
                float(params.a_inner),
                float(ratios[0]),
                float(ratios[1]),
                float(ratios[2]),
                int(params.resolved_geometry() == "tetrahedral_3d"),
                float(t_end),
                int(n_outputs),
                str(summary.get("status", "unknown")),
                summary.get("interest"),
                summary.get("a_delta_rms"),
                summary.get("a_ptp_mean"),
                summary.get("n_encounters"),
                int(bool(summary.get("a_order_changed")))
                if summary.get("a_order_changed") is not None
                else None,
                summary.get("megno"),
                summary.get("energy_drift"),
                json.dumps(_jsonable(summary)),
                None,
            ),
        )
        run_id = int(cur.lastrowid)
        traj_path = None
        if store_trajectory:
            traj_path = self._write_trajectory(run_id, diagnosis)
            self._conn.execute(
                "UPDATE runs SET traj_path=? WHERE id=?",
                (str(traj_path), run_id),
            )
        self._conn.commit()
        return run_id

    def _write_trajectory(self, run_id: int, diagnosis: Diagnosis) -> Path:
        path = self.traj_dir / f"run_{run_id:06d}.npz"
        traj = diagnosis.trajectory
        el = diagnosis.elements
        buf_enc = [
            {
                "time": e.time,
                "i": e.i,
                "j": e.j,
                "label_i": e.label_i,
                "label_j": e.label_j,
                "distance": e.distance,
                "position_mid": e.position_mid.tolist(),
            }
            for e in diagnosis.encounters
        ]
        np.savez_compressed(
            path,
            times=traj.times,
            positions=traj.positions,
            velocities=traj.velocities,
            energies=traj.energies,
            angular_momenta=traj.angular_momenta,
            labels=np.array(traj.labels),
            masses=traj.masses if traj.masses is not None else np.array([]),
            G=np.array([traj.G]),
            status=np.array([traj.status]),
            el_times=el.times,
            el_a=el.a,
            el_e=el.e,
            el_i=el.i,
            el_labels=np.array(el.labels),
            resonance_angles=diagnosis.resonance.angles,
            resonance_times=diagnosis.resonance.times,
            encounters_json=np.array([json.dumps(buf_enc)]),
        )
        return path

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=int(row["id"]),
            created_at=row["created_at"],
            source=row["source"],
            param_class=row["param_class"],
            eccentricity=float(row["eccentricity"]),
            mass_ratio=float(row["mass_ratio"]),
            a_inner=float(row["a_inner"]),
            period_ratios=[
                float(row["period_r01"]),
                float(row["period_r12"]),
                float(row["period_r23"]),
            ],
            tetrahedral=bool(row["tetrahedral"]),
            t_end=float(row["t_end"]),
            n_outputs=int(row["n_outputs"]),
            status=row["status"],
            interest=row["interest"],
            a_delta_rms=row["a_delta_rms"],
            a_ptp_mean=row["a_ptp_mean"],
            n_encounters=row["n_encounters"],
            a_order_changed=None
            if row["a_order_changed"] is None
            else bool(row["a_order_changed"]),
            megno=row["megno"],
            energy_drift=row["energy_drift"],
            summary=json.loads(row["summary_json"]),
            traj_path=row["traj_path"],
        )

    def get(self, run_id: int) -> RunRecord | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,))
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def load_trajectory(self, run_id: int) -> Trajectory | None:
        rec = self.get(run_id)
        if rec is None or not rec.traj_path:
            return None
        path = Path(rec.traj_path)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        masses = data["masses"]
        return Trajectory(
            times=data["times"],
            positions=data["positions"],
            velocities=data["velocities"],
            energies=data["energies"],
            angular_momenta=data["angular_momenta"],
            labels=[str(x) for x in data["labels"].tolist()],
            G=float(data["G"][0]),
            masses=masses if masses.size else None,
            status=str(data["status"][0]),
        )

    def query(
        self,
        *,
        param_class: str | None = None,
        e_min: float | None = None,
        e_max: float | None = None,
        mu_min: float | None = None,
        mu_max: float | None = None,
        tetrahedral: bool | None = None,
        status: str | None = "success",
        min_interest: float | None = None,
        a_order_changed: bool | None = None,
        source: str | None = None,
        order_by: str = "interest",
        limit: int = 50,
    ) -> list[RunRecord]:
        clauses: list[str] = []
        args: list[Any] = []
        if param_class is not None:
            clauses.append("param_class = ?")
            args.append(param_class)
        if e_min is not None:
            clauses.append("eccentricity >= ?")
            args.append(e_min)
        if e_max is not None:
            clauses.append("eccentricity <= ?")
            args.append(e_max)
        if mu_min is not None:
            clauses.append("mass_ratio >= ?")
            args.append(mu_min)
        if mu_max is not None:
            clauses.append("mass_ratio <= ?")
            args.append(mu_max)
        if tetrahedral is not None:
            clauses.append("tetrahedral = ?")
            args.append(int(tetrahedral))
        if status is not None:
            clauses.append("status = ?")
            args.append(status)
        if min_interest is not None:
            clauses.append("interest >= ?")
            args.append(min_interest)
        if a_order_changed is not None:
            clauses.append("a_order_changed = ?")
            args.append(int(a_order_changed))
        if source is not None:
            clauses.append("source = ?")
            args.append(source)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order_sql = {
            "interest": "interest IS NULL, interest DESC",
            "a_delta_rms": "a_delta_rms IS NULL, a_delta_rms DESC",
            "id": "id DESC",
            "created_at": "created_at DESC",
        }.get(order_by, "interest IS NULL, interest DESC")

        sql = f"SELECT * FROM runs{where} ORDER BY {order_sql} LIMIT ?"
        args.append(int(limit))
        cur = self._conn.execute(sql, args)
        return [self._row_to_record(r) for r in cur.fetchall()]

    def list_classes(self) -> list[dict[str, Any]]:
        """Aggregate runs by param_class for browsing."""
        cur = self._conn.execute(
            """
            SELECT param_class,
                   COUNT(*) AS n,
                   AVG(interest) AS mean_interest,
                   MAX(interest) AS max_interest,
                   AVG(a_delta_rms) AS mean_a_rms,
                   SUM(CASE WHEN a_order_changed=1 THEN 1 ELSE 0 END) AS n_swap
            FROM runs
            GROUP BY param_class
            ORDER BY max_interest IS NULL, max_interest DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
