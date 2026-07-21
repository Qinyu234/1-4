<!-- generated: 2026-07-19T04:29 from design.html@untracked -->

# design — module map

## Modules

- **physics** — Newtonian N-body: Body, System, pairwise gravity, Leapfrog integrator,
energy and angular-momentum diagnostics.
- **icgen** — Tetrahedron geometry, Rodrigues rotations mapping A→B/C/D, and 2D
near-escape (v_rad, v_tan) initial-condition generation with momentum
cancellation.
- **simulation** — Trajectory runner: integrates a System over time and records positions,
velocities, energy, and angular momentum at each step.
- **analysis** — OrbitEvaluator: permutation-periodic residual over S4, collision
penalty, energy-drift check, and combined loss score.
- **search** — 2D grid scan over (v_rad, v_tan); optional Stage2 optimize stub for
later local refinement.
- **library** — Persist and load ranked orbit candidates as JSON under orbit_library/.
- **visualization** — Plots for trajectories, energy error, and candidate ranking from a scan.

## Behaviours (cross-module paths)

- **generate_ic**: icgen → physics
  Build a 5-body System from (v_rad, v_tan) using tetrahedron positions
and Rodrigues-propagated velocities.
- **integrate**: physics → simulation
  Leapfrog-integrate a System and emit trajectory plus conserved-quantity
time series.
- **evaluate**: analysis → simulation
  Score a trajectory for weaving periodicity, collisions, and energy drift.
- **grid_search**: search → icgen → simulation → analysis → library
  Enumerate (v_rad, v_tan), run integrate→evaluate, rank and save candidates.
- **first_experiment**: search → icgen → simulation → analysis → library → visualization
  Default Planet/Fairy experiment: near-escape 2D grid, ~100 orbital
periods, plots and orbit_library output.

## Budget

- modules: 7 (soft 7 / hard 9)
- behaviours: 5 (soft 12 / hard 15)
