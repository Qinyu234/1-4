# Path A — 4-body: raise \(M_{\mathrm{central}}\) from 0

Subordinate to [`PROMPT.md`](../../PROMPT.md) §2.5 / §6. On conflict, PROMPT wins.

## Goal

Verified equal-mass **4-body choreography** → introduce central mass
\(M_c\) from **0** upward; each step **Newton** on the full coupled system,
seeded by the previous converged solution. Fold → pseudo-arclength.

## Gate before any continuation

\[
x_i(T/n)=R\,x_{P(i)}(0)
\quad\text{(positions and velocities)}
\]

Binary residual. See PROMPT §3.2. Free-\(N\) catalogue checks may use shape
congruence as a COM-frame equivalent when \(R\) is fitted diagnostically —
promotion still requires the PROMPT residual.

## Score / diagnostics

- **Continuation seed:** last Newton solution (never stick to the \(M_c=0\) IC alone).
- **Diagnostic baseline:** zero-order choreography + first-order central perturbation
  (PROMPT §2.5), not bare zero-order.

## Script checklist (to implement)

1. Load verified `free_4_*` / literature choreography (re-converged).
2. \(M_c=0\): confirm gate.
3. Predictor–corrector in \(M_c\); Newton on \((X,T)\) (and symmetry residuals).
4. On stall: arclength, not only step refinement.

## Out of scope

Bayes / black-box “nonlinear param search.” Hip-hop only if PROMPT construct
list is used and gate passes (not a scheduled Path A start).
