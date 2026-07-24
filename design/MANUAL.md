<!-- generated: 2026-07-24T17:32 from design.html@untracked -->

# design — behaviour manual

## build_ladder
Spans: design, core

Construct a 1-central + 4-fairy System from ladder parameters
(shared e, period ratios, tetrahedral phases) and CanonicalUnits.

## integrate
Spans: core, engine

REBOUND-integrate a System and emit Trajectory plus conserved-quantity
time series; apply collision/escape criteria during or after the run.

## diagnose
Spans: engine, observe

From Trajectory (and optional MEGNO pass), extract a(t), e(t), resonance
angles, encounter events, and stability flags.

## run_ladder_experiment
Spans: design, core, engine, observe, viz, store

End-to-end §5 experiment: build ladder → integrate → diagnose → write
plots and persist the run (params + metrics + trajectory) in SQLite.

## query_orbits
Spans: store

Filter saved runs by param_class / e / μ / interest / a-order swap and
reload Trajectory for follow-up analysis.
