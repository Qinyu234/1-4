"""SQLite store for choreography multi-start search (resume + dedupe)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed


DEFAULT_SEARCH_DB_NAME = "search.sqlite"
MASTER_SEED = 10007


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
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def state_fingerprint(
    positions: np.ndarray,
    velocities: np.ndarray,
    period: float,
    *,
    decimals: int = 6,
) -> str:
    """Stable hash of quantized (r, v, T) for exact-duplicate detection."""
    r = np.round(np.asarray(positions, dtype=float), decimals)
    v = np.round(np.asarray(velocities, dtype=float), decimals)
    T = round(float(period), decimals)
    payload = json.dumps(
        {"r": r.tolist(), "v": v.tolist(), "T": T},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def seed_fingerprint(seed: OrbitSeed, *, decimals: int = 6) -> str:
    return state_fingerprint(
        seed.positions, seed.velocities, seed.period, decimals=decimals
    )


def trial_rng(n_bodies: int, trial_no: int, *, master: int = MASTER_SEED) -> np.random.Generator:
    """Deterministic per-trial RNG so resumed runs never re-draw past ICs."""
    return np.random.default_rng([int(n_bodies), int(master), int(trial_no)])


@dataclass(frozen=True)
class TrialRecord:
    id: int
    n_bodies: int
    trial_no: int
    created_at: str
    start_fp: str
    result_fp: str | None
    residual: float | None
    period: float | None
    ok_gate: bool
    reason: str
    maintains_regular_ngon: bool
    seed_json: dict[str, Any] | None
    error: str | None


class ChoreographySearchStore:
    """
    Persist search trials in SQLite.

    - Resume from ``max(trial_no)+1``
    - Skip if ``start_fp`` already evaluated
    - Skip saving pass if ``result_fp`` already accepted
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=60.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ChoreographySearchStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                n_bodies INTEGER NOT NULL,
                trial_no INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                start_fp TEXT NOT NULL,
                result_fp TEXT,
                residual REAL,
                period REAL,
                ok_gate INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                maintains_regular_ngon INTEGER NOT NULL DEFAULT 0,
                seed_json TEXT,
                error TEXT,
                UNIQUE(n_bodies, trial_no),
                UNIQUE(n_bodies, start_fp)
            );
            CREATE INDEX IF NOT EXISTS idx_trials_n_ok
                ON trials(n_bodies, ok_gate);
            CREATE INDEX IF NOT EXISTS idx_trials_result_fp
                ON trials(n_bodies, result_fp);
            """
        )
        self._conn.commit()

    def clear(self, n_bodies: int | None = None) -> None:
        if n_bodies is None:
            self._conn.execute("DELETE FROM trials")
        else:
            self._conn.execute("DELETE FROM trials WHERE n_bodies=?", (int(n_bodies),))
        self._conn.commit()

    def next_trial_no(self, n_bodies: int) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(MAX(trial_no), 0) FROM trials WHERE n_bodies=?",
            (int(n_bodies),),
        )
        return int(cur.fetchone()[0]) + 1

    def has_start_fp(self, n_bodies: int, start_fp: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM trials WHERE n_bodies=? AND start_fp=? LIMIT 1",
            (int(n_bodies), start_fp),
        )
        return cur.fetchone() is not None

    def has_accepted_result_fp(self, n_bodies: int, result_fp: str) -> bool:
        cur = self._conn.execute(
            """
            SELECT 1 FROM trials
            WHERE n_bodies=? AND result_fp=? AND ok_gate=1
            LIMIT 1
            """,
            (int(n_bodies), result_fp),
        )
        return cur.fetchone() is not None

    def count_trials(self, n_bodies: int) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM trials WHERE n_bodies=?",
            (int(n_bodies),),
        )
        return int(cur.fetchone()[0])

    def count_passed(self, n_bodies: int) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM trials WHERE n_bodies=? AND ok_gate=1",
            (int(n_bodies),),
        )
        return int(cur.fetchone()[0])

    def count_maintained_rejects(self, n_bodies: int) -> int:
        cur = self._conn.execute(
            """
            SELECT COUNT(*) FROM trials
            WHERE n_bodies=? AND maintains_regular_ngon=1
            """,
            (int(n_bodies),),
        )
        return int(cur.fetchone()[0])

    def best_accepted(self, n_bodies: int) -> TrialRecord | None:
        cur = self._conn.execute(
            """
            SELECT * FROM trials
            WHERE n_bodies=? AND ok_gate=1 AND residual IS NOT NULL
            ORDER BY residual ASC
            LIMIT 1
            """,
            (int(n_bodies),),
        )
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def insert_trial(
        self,
        *,
        n_bodies: int,
        trial_no: int,
        start_fp: str,
        result_fp: str | None,
        residual: float | None,
        period: float | None,
        ok_gate: bool,
        reason: str,
        maintains_regular_ngon: bool,
        seed: OrbitSeed | None = None,
        error: str | None = None,
    ) -> int | None:
        """
        Insert one trial. Returns row id, or None if start_fp already present
        (unique conflict → treat as already done).
        """
        seed_json = None if seed is None else json.dumps(_jsonable(seed.to_dict()))
        try:
            cur = self._conn.execute(
                """
                INSERT INTO trials (
                    n_bodies, trial_no, created_at, start_fp, result_fp,
                    residual, period, ok_gate, reason, maintains_regular_ngon,
                    seed_json, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(n_bodies),
                    int(trial_no),
                    _now(),
                    start_fp,
                    result_fp,
                    residual,
                    period,
                    int(bool(ok_gate)),
                    str(reason),
                    int(bool(maintains_regular_ngon)),
                    seed_json,
                    error,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            self._conn.rollback()
            return None

    def import_seed_pass(
        self,
        seed: OrbitSeed,
        *,
        trial_no: int | None = None,
        residual: float | None = None,
        reason: str = "imported",
    ) -> int | None:
        """Import an existing accepted seed JSON into the store (deduped)."""
        n = int(seed.n_bodies)
        fp = seed_fingerprint(seed)
        if self.has_accepted_result_fp(n, fp):
            return None
        tno = int(trial_no) if trial_no is not None else self.next_trial_no(n)
        # synthetic start_fp distinct from result
        start_fp = f"import:{fp}"
        if self.has_start_fp(n, start_fp):
            return None
        return self.insert_trial(
            n_bodies=n,
            trial_no=tno,
            start_fp=start_fp,
            result_fp=fp,
            residual=residual,
            period=float(seed.period),
            ok_gate=True,
            reason=reason,
            maintains_regular_ngon=False,
            seed=seed,
        )

    def list_passes(self, n_bodies: int, *, limit: int = 50) -> list[TrialRecord]:
        cur = self._conn.execute(
            """
            SELECT * FROM trials
            WHERE n_bodies=? AND ok_gate=1
            ORDER BY residual ASC
            LIMIT ?
            """,
            (int(n_bodies), int(limit)),
        )
        return [self._row_to_record(r) for r in cur.fetchall()]

    def summary_dict(self, n_bodies: int, *, out_dir: str | None = None) -> dict[str, Any]:
        best = self.best_accepted(n_bodies)
        return {
            "n": int(n_bodies),
            "trials": self.count_trials(n_bodies),
            "passed_gate": self.count_passed(n_bodies),
            "rejected_maintained_regular_ngon": self.count_maintained_rejects(n_bodies),
            "best_residual": None if best is None else best.residual,
            "best_trial_no": None if best is None else best.trial_no,
            "db_path": str(self.db_path),
            "out_dir": out_dir,
        }

    def _row_to_record(self, row: sqlite3.Row) -> TrialRecord:
        seed_raw = row["seed_json"]
        return TrialRecord(
            id=int(row["id"]),
            n_bodies=int(row["n_bodies"]),
            trial_no=int(row["trial_no"]),
            created_at=str(row["created_at"]),
            start_fp=str(row["start_fp"]),
            result_fp=None if row["result_fp"] is None else str(row["result_fp"]),
            residual=None if row["residual"] is None else float(row["residual"]),
            period=None if row["period"] is None else float(row["period"]),
            ok_gate=bool(row["ok_gate"]),
            reason=str(row["reason"] or ""),
            maintains_regular_ngon=bool(row["maintains_regular_ngon"]),
            seed_json=None if seed_raw is None else json.loads(seed_raw),
            error=None if row["error"] is None else str(row["error"]),
        )
