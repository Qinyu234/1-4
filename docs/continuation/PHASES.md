# Layered pipeline

**Source of truth:** repo-root [`PROMPT.md`](../../PROMPT.md) (design log).  
This file only mirrors its layering; if anything conflicts, **PROMPT wins**.

```mermaid
flowchart TB
  L1[1. N=4 construct + store + classify]
  L2[2. N=5 construct + store + classify]
  G4{N=4: x_i(T/n)=R x_P(i)(0)  r and v}
  G5{N=5: same gate}
  L3[3. N=4 mass continuation + Newton]
  L4[4. N=5 mass continuation + Newton]
  L1 --> G4
  L2 --> G5
  G4 -->|pass| L3
  G5 -->|pass| L4
```

Do **not** call Layers 3–4 “nonlinear parameter search” or Bayes.  
PROMPT name: **质量延拓 + Newton 修正** (fold → pseudo-arclength).

---

## Layers 1–2 — Construct + store + classify

Per PROMPT §3:

| Step | Rule |
|------|------|
| Construct | Literature IC (re-converge) and/or Fourier + action; analytic RE only as weak baseline |
| Verify | Explicit \(x_i(T/n)=R\,x_{P(i)}(0)\) for **positions and velocities** |
| Classify | Store \(N\), cycle type of \(P\), \(R\) (axis/angle), action, Floquet if available, source |

Gate is binary algebraic residual → 0, not fuzzy periodicity scores.

## Layer 3 — Four-body mass continuation (PROMPT §2.5 / §6)

After a verified equal-mass 4-body choreography:

- Introduce a central body; **\(M_{\mathrm{central}}\) from 0 upward**.
- At \(M_c=0\) the free choreography is the exact start (no solve).
- Each step: previous converged solution = Newton guess for the **full coupled** system.
- On Newton failure: suspect fold → **pseudo-arclength**, not tinier steps.

## Layer 4 — Five-body mass continuation (PROMPT §2.5 / §6)

After a verified equal-mass 5-body choreography:

- Fix one body mass \(=1\) (“center” role); other four masses down in **logspace** from 1.
- Fifth-body symmetry \(x_E(t)=R\,x_E(t+\tau)\) → try **both** branches (axis oscillation / counter-rotation); keep what converges.

## Score / diagnostic baselines (PROMPT §2.5)

Two different roles — do not mix:

| Role | What |
|------|------|
| Continuation seed | Always the **last converged** Newton solution |
| Diagnostic baseline | Prefer **zero-order choreography + first-order perturbation** (central as perturber), not bare zero-order |

Raw \(m_c\in\{0,1\}\) anchors may still be logged; the informative residual is vs zero+first order.

## Hard rules

1. No Layer 3/4 until that \(N\) passes §3.2 (r **and** v).
2. No Bayes / GP / black-box “param search” on the mainline.
3. Literature digits → re-converge before catalogue promotion.
4. Detail and abandoned routes: only in PROMPT (esp. §7).
