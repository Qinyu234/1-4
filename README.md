# Fairy Orbit — Design → REBOUND Verify

Layered search for **relative periodic encounter orbits (PEO)**.  
Architecture (see [`PROMPT.md`](PROMPT.md)):

1. **Initial manifold generator** — \(\theta\in\mathbb{R}^7 \to X_0=f(\theta)\)
2. **Simulator** — REBOUND \(\Phi_T\) only (no PEO judgment)
3. **PEO filter pipeline** — escape/collision → encounter choreography → SO(3) closure → velocity match → archive → refine

Distilled status: [`docs/DIRECTION.md`](docs/DIRECTION.md).  
Module map: [`design/design.html`](design/design.html).

## Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e .
```

## Error experiments (kept)

Integrator floors + Td group-orbit error base (exponential normalize).

```powershell
.\.venv\Scripts\python scripts\run_campaign.py smoke
.\.venv\Scripts\python scripts\run_campaign.py calib
.\.venv\Scripts\python scripts\run_campaign.py td_group --smoke
.\.venv\Scripts\python scripts\run_campaign.py td_beta_e --smoke
.\.venv\Scripts\python scripts\run_campaign.py td_growth
```

Outputs under `experiments/output/`:

| Dir | Content |
|-----|---------|
| `calibration/` | \(\varepsilon_{\mathrm{numerical}}(N)\) |
| `td_group_orbit_*` | Td ABC error growth |
| `td_error_growth/` | Multi-integrator tetra error |
| `td_beta_e_scan/` | \((\beta,e)\) breaking map |
| `td_growth_law/` | lin vs exp growth law |

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Package layout

| Package | Role |
|---------|------|
| `fairy_orbit.core` | Units, config, Body/System, criteria |
| `fairy_orbit.design` | Ladder / graded / Td group-orbit IC |
| `fairy_orbit.engine` | REBOUND → Trajectory |
| `fairy_orbit.observe` | Elements, encounters, AMD, **error_base** |
| `fairy_orbit.viz` | Reports + orbit plots |
| `fairy_orbit.store` | SQLite results (PEO archive, later) |
| `fairy_orbit.legacy` | Quarantined old search / hierarchical code |
