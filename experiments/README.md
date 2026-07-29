# Experiments

## Active (PEO mainline)

**Invariant:** Stage-A base error (`run_rep_error_scan` → `sigmas.json`) **before** any detailed σ-weighted score. Campaign / beam / heatmap / Bayes / staged CLIs hard-fail if `sigmas.json` is missing or uncalibrated.

| Script | Role |
|--------|------|
| `run_staged_peo.py` | **ABCD→BCDA reachability:** 一阶 coarse → 二阶 expand grid → 三阶 stain+unlock → 细致 score |
| `run_bayes_peo.py` | Optional TPE helper (also requires σ; stagnate → expand+unlock) |
| `run_long_campaign.py` | Beam search until solve / plateau / wall (edge expand) |
| `run_me_heatmap.py` | Progressive `(m,e)` heatmap cells |
| `run_10h_campaign.py` | Orchestrator: σ → heatmap → beam → plots |
| `run_beam_search.py` | One-shot multi-seed beam |
| `run_rep_error_scan.py` | Stage-A σ calibration (base error) |
| `fit_rep_sigmas.py` | Fit `sigmas.json` from scan |
| `plot_campaign_orbits.py` | Re-integrate + plot campaign bests |
| `run_peo_smoke.py` | Short PEO smoke |

Typical run:

```powershell
# 1) base error first
.\.venv\Scripts\python.exe experiments\run_rep_error_scan.py
# 2) staged ABCD→BCDA hunt
.\.venv\Scripts\python.exe experiments\run_staged_peo.py --out experiments\output\staged_peo
# smoke:
.\.venv\Scripts\python.exe experiments\run_staged_peo.py --smoke
```

Outputs: `experiments/output/staged_peo/`, `experiments/output/campaign_10h/`, `experiments/output/rep_error/`.

## Legacy (`experiments/legacy/`)

Td μ_eff / calibration / old stereo & perf probes. Kept for reference; not on the PEO search path.

| Script | Notes |
|--------|-------|
| `run_td_*`, `fit_td_growth_law.py` | Td symmetry / β–e scans |
| `run_tetra_error_growth.py` | Integrator error growth |
| `run_calibration.py` | Old calibration IC |
| `run_stereo_from_seed.py` | Seed stereo probe |
| `run_search_perf.py` | Eval timing probe |

Launcher for legacy modes: `scripts/run_campaign.py` (Td modes only).
