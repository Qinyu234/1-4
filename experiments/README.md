# Experiments

**Design log:** repo-root `PROMPT.md` (overrides other docs).

## Active (PROMPT mainline)

| Script | Role |
|--------|------|
| `verify_orbit_seeds.py` | §3.2 gate: \(x_i(T/n)=R\,x_{P(i)}(0)\) for **r and v**; update catalogue |
| `run_mass_continuation_smoke.py` | Path A stub: \(M_c=0\) gate → tiny \(M_c\) + LS corrector |
| `run_choreography_search.py` | Free-N multi-start §3.2 polish (`--n 4|5 --wall-hours`) |
| `run_mass_continuation_campaign.py` | Path A \(M_c\uparrow\) (n=4) / μ↓ (n=5) |
| `run_prompt_8h.py` | Launch all four campaigns |

```powershell
# four parallel 8h campaigns (or start each manually):
.\.venv\Scripts\python.exe experiments\run_choreography_search.py --n 4 --wall-hours 8
.\.venv\Scripts\python.exe experiments\run_choreography_search.py --n 5 --wall-hours 8
.\.venv\Scripts\python.exe experiments\run_mass_continuation_campaign.py --n 4 --wall-hours 8
.\.venv\Scripts\python.exe experiments\run_mass_continuation_campaign.py --n 5 --wall-hours 8
```

Docs: `docs/continuation/`. Outputs:
`experiments/output/choreography_search_n{4,5}/`,
`experiments/output/continuation_n{4,5}/`,
`experiments/output/prompt_8h_logs/`.

## Legacy (`experiments/legacy/`)

Demoted Bayes / staged / campaign / σ-scan / Td probes. Not on the mainline.

| Group | Scripts |
|-------|---------|
| Bayes / staged / campaign | `run_bayes_peo.py`, `run_staged_peo.py`, `run_long_campaign.py`, `run_10h_campaign.py`, `run_beam_search.py`, `run_me_heatmap.py`, `run_peo_smoke.py`, `plot_campaign_orbits.py` |
| σ Stage-A (old score hunt) | `run_rep_error_scan.py`, `fit_rep_sigmas.py` |
| Td / calibration | `run_td_*`, `fit_td_growth_law.py`, `run_tetra_error_growth.py`, `run_calibration.py`, … |

Launcher for old Td modes: `scripts/run_campaign.py`.
