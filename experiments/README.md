# Experiments

**Design log:** repo-root `PROMPT.md` (overrides other docs).

## Active (PROMPT mainline)

| Script | Role |
|--------|------|
| `verify_orbit_seeds.py` | §3.2 gate (r+v at \(T/n\)); update catalogue |
| `run_choreography_search.py` | Free-N multi-start §3.2 polish (`--n 4|5`) |
| `run_mass_continuation_campaign.py` | Path A \(M_c\uparrow\) (n=4) / μ↓ (n=5); requires `--seed` |
| `run_prompt_campaign.py` | Self-expanding parallel launcher (default unlimited; optional `--wall-hours`) |
| `plot_best_orbits.py` | Plot `best.json` / `final.json` orbits |

```powershell
# Unlimited self-expanding choreography search (N=4 and N=5)
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py

# Optional wall clock stop after 12h
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --wall-hours 12

# Wipe prior search dirs then relaunch
.\.venv\Scripts\python.exe experiments\run_prompt_campaign.py --fresh

.\.venv\Scripts\python.exe experiments\plot_best_orbits.py
```

`--wall-hours <=0` means unlimited. Outputs append: `trials.jsonl` / `steps.jsonl`
grow; `summary.json` updates live.

Kept outputs: `choreography_search_n{4,5}/`, `continuation_n{4,5}/`,
`prompt_campaign_logs/`, `best_orbit_plots/`.

**Policy:** reject orbits that *maintain* a regular equal n-gon over \(T\)
(rigid RE). A polygonal snapshot alone is not enough to reject.

## Legacy (`experiments/legacy/`)

Td / calibration probes only. Bayes/campaign CLIs removed.
