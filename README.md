# Fairy Orbit Periodic Search Platform

Experiment platform for finding tetrahedron-symmetric weaving orbits
(Planet + 4 fairies) under Newtonian gravity — not a game physics engine.

## Environment

Python **3.11** via project `.venv`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e .
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## First experiment

Smoke (fast):

```powershell
.\.venv\Scripts\python experiments\first_grid_scan.py --smoke
```

Full near-escape 2D grid (~100 orbital periods; slower):

```powershell
.\.venv\Scripts\python experiments\first_grid_scan.py
```

**Per-k optimize near k=1** (planet `M`, fairies `kM`; optimize velocity for each k):

```powershell
.\.venv\Scripts\python experiments\run_k_near1.py
.\.venv\Scripts\python experiments\run_k_near1.py --smoke
```

Summary: `experiments/output/k_near1_summary.json`, `k_near1_best_vs_k.png`.

**8-hour expanding search** (auto-widens `(v_rad, v_tan)` bounds; checkpoint + log):

```powershell
.\.venv\Scripts\python experiments\run_8h.py --hours 8
```

Smoke (~30s):

```powershell
.\.venv\Scripts\python experiments\run_8h.py --smoke
```

Outputs: `orbit_library/orbit_*.json`, `experiments/output/*.png`,
`experiments/output/expanding_checkpoint.json`, `experiments/output/expanding_run.log`.

## Design

Source of truth: `design/design.html` (run `design/parser.py` to regenerate README/MANUAL).
Workflow docs: `docs/v1/`.

## New workflow

- Event-based search:
  `./.venv/Scripts/python.exe experiments/run_improved_search.py --evaluation-mode event --solver-type rebound --smoke`
- Interactive viewer:
  `./.venv/Scripts/python.exe fairy_orbit/visualization/interactive_viewer.py --out experiments/output/orbit_viewer.html --solver-type rebound`
