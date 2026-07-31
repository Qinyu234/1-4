# Experiments

**Design log:** repo-root `PROMPT.md` (overrides other docs).

## Active (PROMPT mainline)

**Priority (RESPONSE §6–§7):** thread budget **7:2:6:1** =
`choreo_n4 : choreo_n5 : path_a(+Floquet) : branch2`.
Default **active**: n4 + Path A; **n5 and Branch-2 dark** (`--with-n5` / `--with-branch2-probe` to opt in).

| Script | Role |
|--------|------|
| `verify_orbit_seeds.py` | §3.2 gate (r+v at \(T/n\)); update catalogue |
| `run_choreography_search.py` | Free-N multi-start; Floquet certify ON by default |
| `query_search_db.py` | Query search DB summary / passes |
| `run_mass_continuation_campaign.py` | Path A \(M_c\uparrow\) / Path B μ↓; auto Floquet path sweep after Path A; default optics soft extras (`--log-rho`, `--no-optics-soft`) |
| `run_floquet_path_sweep.py` | Resweep `state_Mc_*.json` → `max|λ|` vs `M_c` |
| `run_floquet_family_sweep.py` | Equal-mass archive Floquet proxy (no continuation yet) |
| `run_branch2_probe.py` | Lowest-priority multi-family Branch-2 probe (default `--diverse 4`) |
| `run_prompt_campaign.py` | Launcher; default n4 only; `--path-a-from-db`; `--with-n5` / `--with-branch2-probe` |
| `run_auto_pipeline.py` | Auto: ensure N=4 search + periodic shape-family replot |
| `plot_best_orbits.py` | Plot `best.json` / `final.json` orbits |

```powershell
# Default: N=4 search only (n5/branch2 off)
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py

# N=4 search + Path A + Floquet resweep; 13h total → 7h search / 6h Path A
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --path-a-from-db --wall-hours 13

# Opt-in N=5 (weight 2) or Branch-2 (weight 1)
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --with-n5
.\.venv\Scripts\python.exe experiments\run_branch2_probe.py

# Wipe then relaunch
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --fresh

# Auto pipeline: N=4 search (resume) + replot every 30 min
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
(rigid RE). A polygonal snapshot alone is not enough to reject. Floquet-unstable
equal-mass seeds may still start Path A (crossing hunt); Branch-2 and choreo_n5
are deprioritized / dark by default, not deleted.

**Visual overlap (PROMPT §4.1):** equal density via `logρ∈[-1,1]`.
Reclassify **post-continuation only** (not free-N):

```powershell
.\.venv\Scripts\python.exe experiments\run_visual_reclassify.py --n 4 --cont-dir experiments\output\continuation_n4_visual
```

Path A soft extras use encounter-conditioned `|Δr_perp|`; any-time verify via
`optical_overlap_angular` / `scan_visual_overlaps`.

## Legacy (`experiments/legacy/`)

Td / calibration probes only. Bayes/campaign CLIs removed.
