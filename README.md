# Fairy Orbit — Design → REBOUND Verify

Design orbital ladders, then verify with REBOUND. Goal (see `PROMPT.md` §3):  
**slow energy exchange / a-migration / role swap** — not rigid, unchanging orbits.

Blind `(v_rad, v_tan)` search and the old “soak for MEGNO≈2 static score” campaign
are abandoned.

Source of truth for intent: [`PROMPT.md`](PROMPT.md) (raw dialogue).  
Distilled judgments: [`docs/DIRECTION.md`](docs/DIRECTION.md).  
Module map: [`design/design.html`](design/design.html).

## Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e .
```

## Single ladder observation

```powershell
.\.venv\Scripts\python experiments\run_orbital_ladder.py --smoke
.\.venv\Scripts\python experiments\run_orbital_ladder.py --t-end 800
```

## Dynamics scan (prefer changing orbits)

Scans `e × μ ×` mild period-ratio detunes; ranks by **interest** (a migration,
encounters, a-order swap; rejects near-static success). Results go to SQLite.

```powershell
.\.venv\Scripts\python scripts\run_campaign.py smoke
.\.venv\Scripts\python scripts\run_campaign.py dynamics --t-end 800
```

Outputs: `experiments/output/dynamics/` + DB `experiments/output/orbit_db/orbits.sqlite`.

## Query saved orbits (by initial params)

```powershell
.\.venv\Scripts\python scripts\run_campaign.py classes
.\.venv\Scripts\python scripts\run_campaign.py query --mu-min 1e-3 --min-interest 1 --swap
.\.venv\Scripts\python experiments\query_orbits.py get 3 --traj
```

`param_class` example: `e0.15_mu1e-03_tet1_s1.00_a1.00`.

## Perf

```powershell
.\.venv\Scripts\python scripts\run_campaign.py perf
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Package layout

| Package | Role |
|---------|------|
| `fairy_orbit.core` | Units, config, Body/System, criteria |
| `fairy_orbit.design` | Ladder IC + Kepler elements + tetrahedral phases |
| `fairy_orbit.engine` | REBOUND → Trajectory |
| `fairy_orbit.observe` | Elements, resonance, MEGNO, encounters, **interest** |
| `fairy_orbit.viz` | Reports + orbit plots |
| `fairy_orbit.store` | SQLite results by `param_class` + trajectory sidecars |
| `fairy_orbit.legacy` | Quarantined search / hierarchical code |
