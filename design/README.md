<!-- generated: 2026-07-24T17:32 from design.html@untracked -->

# design — module map

## Modules

- **core** — CanonicalUnits(a_ref, GM), SystemConfig (mass_ratio, R_central),
Body/System state carriers, and §6 criteria: collision (r_ij < R_i+R_j)
and energy-based escape (E = v²/2 − GM/r > 0, equiv. a ≤ 0).
- **design** — Hierarchical Resonant Orbit Chain IC: nested a from period ratios
≈ 3:2 / 5:3 / 7:5, shared e, non-coplanar tetrahedral periapses
(PROMPT / docs/DIRECTION.md). T1<T2<T3<T4 is initial ordering only.
- **engine** — Single REBOUND (IAS15) integration path: System → Trajectory
(positions, velocities, energies, angular momenta). Simulator is the
physical truth source for verification, not discovery.
- **observe** — Diagnostics on Trajectory: a(t)/e(t), resonance angles, MEGNO, encounter
Event index, interest score (prefer secular change), and AMD (angular
momentum deficit) as exchange / stability language.
- **viz** — Ladder experiment report plots: element evolution, encounter locations,
MEGNO / stability summary. No search heatmaps as main UX.
- **store** — SQLite persistence of ladder runs, classified by initial-parameter key
(param_class: e, μ, tetrahedral, period-ratio scale). Trajectories stored
as sidecar .npz for later reload / query.

## Behaviours (cross-module paths)

- **build_ladder**: design → core
  Construct a 1-central + 4-fairy System from ladder parameters
(shared e, period ratios, tetrahedral phases) and CanonicalUnits.
- **integrate**: core → engine
  REBOUND-integrate a System and emit Trajectory plus conserved-quantity
time series; apply collision/escape criteria during or after the run.
- **diagnose**: engine → observe
  From Trajectory (and optional MEGNO pass), extract a(t), e(t), resonance
angles, encounter events, and stability flags.
- **run_ladder_experiment**: design → core → engine → observe → viz → store
  End-to-end §5 experiment: build ladder → integrate → diagnose → write
plots and persist the run (params + metrics + trajectory) in SQLite.
- **query_orbits**: store
  Filter saved runs by param_class / e / μ / interest / a-order swap and
reload Trajectory for follow-up analysis.

## Budget

- modules: 6 (soft 7 / hard 9)
- behaviours: 5 (soft 12 / hard 15)
