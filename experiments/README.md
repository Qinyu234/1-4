# Experiments

**Design log:** repo-root `PROMPT.md` (overrides other docs).

## Active (PROMPT mainline)

| Script | Role |
|--------|------|
| `verify_orbit_seeds.py` | §3.2 gate (r+v at \(T/n\)); update catalogue |
| `run_choreography_search.py` | Free-N multi-start §3.2 polish; SQLite resume (`search.sqlite`) |
| `query_search_db.py` | Query search DB summary / passes |
| `run_mass_continuation_campaign.py` | Path A \(M_c\uparrow\) (n=4) / μ↓ (n=5); requires `--seed` |
| `run_prompt_campaign.py` | Self-expanding parallel launcher (default unlimited; optional `--wall-hours`) |
| `run_auto_pipeline.py` | Auto: ensure searches + periodic shape-family replot |
| `plot_best_orbits.py` | Plot `best.json` / `final.json` orbits |

```powershell
# Unlimited self-expanding choreography search (N=4 and N=5); resumes SQLite
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py

# Optional wall clock stop after 12h
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --wall-hours 12

# Wipe SQLite + output dirs then relaunch
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --fresh

# Auto pipeline: search (resume) + replot shape families every 30 min
.\.venv\Scripts\python.exe experiments\run_auto_pipeline.py --plot-every-min 30

.\.venv\Scripts\python.exe experiments\query_search_db.py --n 4 --passes
.\.venv\Scripts\python.exe experiments\plot_best_orbits.py
# HTML 时间滑条动画：experiments/output/best_orbit_plots/choreo_n*_best/orbit_anim.html
# 形状差异族（非仅 residual 最优）：
.\.venv\Scripts\python.exe experiments\plot_shape_families.py --n-families 6
# → best_orbit_plots/choreo_n{4,5}_families/family_*/orbit_anim.html
```

`--wall-hours <=0` means unlimited. Each run **continues** from `search.sqlite`
(no re-do of stored `start_fp` / accepted `result_fp`). Accepted seeds live in
SQLite (`seed_json`); per-pass `pass_*.json` are **not** written by default
(legacy files are moved to `pass_json_archive/` on startup). Only `best.json`
plus `trials.jsonl` / `summary.json` stay as light sidecars.

Kept outputs: `choreography_search_n{4,5}/`, `continuation_n{4,5}/`,
`prompt_campaign_logs/`, `best_orbit_plots/`.

**Policy:** reject orbits that *maintain* a regular equal n-gon over \(T\)
(rigid RE). A polygonal snapshot alone is not enough to reject.

## Legacy (`experiments/legacy/`)

Td / calibration probes only. Bayes/campaign CLIs removed.
