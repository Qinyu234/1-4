# Experiments

**Design log:** repo-root `PROMPT.md` (overrides other docs).

## Active (PROMPT mainline)

**Priority:** thread budget **7:2:6:1** =
`choreo_n4 : choreo_n5 : path_a(+Floquet) : branch2`.
Default **active**: n4 + Path A; **n5 and Branch-2 dark**.

### Scripts

| Script | Role |
|--------|------|
| `run_choreography_search.py` | Free-N multi-start; Floquet certify ON |
| `run_mass_continuation_campaign.py` | Path A / Path B; `--res-tol`, `--m-c-max`, `--horizon-periods`, `--correct-only --m-c` |
| `run_path_a_cycle.py` | Multi-seed Path A; `--horizon-periods`; `--scan-top-k` multi-period+Floquet scan |
| `run_prompt_campaign.py` | Campaign launcher (7:2:6:1; n5/branch2 opt-in) |
| `run_live_monitor.py` | Throughput + funnel sample → `profile/live_monitor.jsonl` |
| `optics_tides/` | Earth-scale central observer; `starry` flux; brightness swaps |
| `plot_best_orbits.py` | Best free-N + best Path-A HTML/PNG |
| `query_search_db.py` | SQLite summary / passes |
| `verify_orbit_seeds.py` | §3.2 catalogue gate |

```powershell
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --path-a-from-db --wall-hours 0
.\.venv\Scripts\python.exe experiments\run_path_a_cycle.py --res-tol 1e-3 --m-c-max 1 --horizon-periods 4
.\.venv\Scripts\python.exe experiments\run_path_a_cycle.py --scan-top-k 8 --horizon-periods 3 --res-tol 1e-2 --m-c-max 0.1
.\.venv\Scripts\python.exe experiments\run_mass_continuation_campaign.py --n 4 --seed path\to\state.json --correct-only --m-c 0.06876 --horizon-periods 3 --no-optics-soft --out experiments\output\best_orbit_plots\path_a_best_Mc
.\.venv\Scripts\python.exe experiments\run_live_monitor.py --interval-s 600
.\.venv\Scripts\python.exe experiments\plot_best_orbits.py --horizon-periods 3,4
# → experiments/output/best_orbit_plots/choreo_n4_best/orbit_anim.html
# optics (needs: pip install -r requirements-optics.txt)
.\.venv\Scripts\python.exe experiments\optics_tides\run_central_optics.py --seed ... --m-c 0.06876 --A 0.25 --evenness uneven
```

### Output layout (kept)

| Path | Contents |
|------|----------|
| `choreography_search_n{4,5}/` | `search.sqlite`, `best.json`, `summary.json` |
| `continuation_n4/` | Latest single-seed Path A |
| `continuation_n4_cycle/` | Multi-seed Path A archive (`trial_*`) |
| `best_orbit_plots/` | Regenerable PNG + `orbit_anim.html` |
| `profile/` | Monitor + multiperiod scan JSON |

SQLite is the search source of truth.

**Policy:** reject maintained regular n-gon REs. Optical verify is **post-continuation
only**. Path A accepts steps when `||F|| < res_tol` even if LM `success=False`.
