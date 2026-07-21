<!-- generated: 2026-07-19T04:29 from design.html@untracked -->

# design — behaviour manual

## generate_ic
Spans: icgen, physics

Build a 5-body System from (v_rad, v_tan) using tetrahedron positions
and Rodrigues-propagated velocities.

## integrate
Spans: physics, simulation

Leapfrog-integrate a System and emit trajectory plus conserved-quantity
time series.

## evaluate
Spans: analysis, simulation

Score a trajectory for weaving periodicity, collisions, and energy drift.

## grid_search
Spans: search, icgen, simulation, analysis, library

Enumerate (v_rad, v_tan), run integrate→evaluate, rank and save candidates.

## first_experiment
Spans: search, icgen, simulation, analysis, library, visualization

Default Planet/Fairy experiment: near-escape 2D grid, ~100 orbital
periods, plots and orbit_library output.
