# Path B — 5-body / hierarchical mass scan

Subordinate to [`PROMPT.md`](../../PROMPT.md) §2.5 / §6. On conflict, PROMPT wins.

## Goal (PROMPT)

Verified equal-mass **5-body choreography** → fix one body mass \(=1\), take the
other four masses **down in logspace** from 1. Newton each step from the last
converged state; fold → pseudo-arclength.

Also try the two **E-branch** ICs from PROMPT §2.4 (axis oscillation vs
counter-rotation); keep what converges.

## Hierarchical project seed

`hier_1plus4_manifold` remains a Path-B-style **baseline IC**
(`orbit_class=hier_baseline_ic`); it is **not** a verified choreography and
must not start continuation until a 5-body (or 1+4 PEO) gate passes.

## Gate

Same PROMPT §3.2 residual on r and v (and E-branch relation when applicable).

## Score / diagnostics

Same as Path A: last Newton as seed; zero-order + first-order as diagnostic
baseline (PROMPT §2.5).

## Out of scope

Bayes / “nonlinear param search” language on the mainline.
