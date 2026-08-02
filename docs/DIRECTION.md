# Direction Log — Hierarchical Resonant Orbit Chain

Distilled from the dialogue now in [`PROMPT.md`](../PROMPT.md).  
Raw chat is exploratory; this file records **what we keep**, **what we reject**, and **what to run now**.

Generated: 2026-07-24

---

## 1. Core research question (current)

Not “how do I invent a custom N-body simulator / free-form interaction kernel?”

> **Which initial orbit families (hierarchical resonant chains), under Newtonian gravity, naturally produce quasi-periodic encounter / secular exchange events?**

Naming: prefer **Hierarchical Resonant Orbit Chain** over vague “orbital ladder.”  
`T1 < T2 < T3 < T4` is an **initial** semi-major-axis ordering, not a permanent physical constraint (crossing / high-e can break “adjacency”).

---

## 2. Judgments: keep vs reject / demote

### Keep (architecture)

| Judgment | Why |
|----------|-----|
| Do **not** start derivation from Lagrange Planetary Equations | LPE assumes a known disturbing function \(R\); mechanical once \(R\) exists |
| Innovation lives in **interaction structure** / orbit family, not rewriting LPE | Matches “design → verify” |
| **Three layers**: Orbit family → Secular (slow) → Encounter/Event (fast) | Separates timescales cleanly |
| **REBOUND = truth** for verification | Half-analytic models only screen / explain |
| Adjacent-pair emphasis as a **working hypothesis** | Hierarchical \(\alpha = a_i/a_j\) intuition — must be checked numerically |
| Analogues: Laplace resonance / Galilean moons; **secular angular-momentum exchange**, not flyby | Aligns with “relay / slow swap” goal |
| First implementation: generate chain → long REBOUND → \(a(t), e(t), d_{ij}\), events | Prove phenomenon before analytic toys |
| **AMD (Angular Momentum Deficit)** as a secular diagnostic | Standard tool (Laskar) for exchange + stability language |

### Reject or demote (appeared in dialogue, then corrected)

| Proposal | Status | Reason |
|----------|--------|--------|
| Freely designed \(R_{ij} = W_{ij}\,K_{ij}\) with hand-tuned \(W_{ij}=f(\Delta a,\Delta e,\Delta\omega)\) | **Reject as Newtonian claim** | Real Laplace coefficients already encode strength; free \(W\) is a phenomenological toy, not gravity |
| Instantaneous distance kernel \(K_{ij}(t)=\exp(-d^2/\sigma^2)\) multiplied into secular \(R\) | **Reject as mixed timescale** | Secular theory averages fast angles; \(d_{ij}(t)\) belongs to encounter / direct N-body, not averaged LPE |
| Merge η / Subsystem / Encke “close-encounter line” into the same \(R_{ij}\) as the ladder secular model | **Demote / keep separate** | Same idea as archived §8; useful for strong flybys (toolkit §7), not the main secular-chain path |
| Blind \((v_{\mathrm{rad}},v_{\mathrm{tan}})\) search / soak for MEGNO≈2 static orbits | **Abandoned earlier** | Candidate generation bottleneck; static rings are not the desired phenomenon |
| Rodrigues `(v_rad,v_tan)` equal copy as Td ∀t | **Reject** | Not a group orbit; \(D_{Td}\to O(1)\) in ≪1 orbit |
| μ_eff **Kepler ellipses** as Td reference | **Reject** | Orbit is \(\rho(t)R(t)q_i\), not planar ellipses |
| **Td group orbit** \(r_i=\rho R q_i\) from reduced Lagrangian | **Keep / SOT** | \(A\ddot\rho=-C/\rho^2+J^2/(B\rho^3)\); shape \(\theta(\rho)\) = elliptic integral |
| Drop all non-adjacent \(R_{ij}\) without checking | **Not yet accepted** | For 3:2×5:3×7:5, \(a_4/a_1\approx2.30\) ⇒ \(\alpha_{14}\approx0.43\) — Laplace terms not obviously negligible; verify before assuming |

### Optimized middle ground

- **Secular Resonant Chain Approximation** (future): standard averaged \(R_{ij}\) (Laplace), optionally adjacent-only **after** coefficient check; role = intuition / pre-screen / explain REBOUND — **not** a new gravity model and **not** a substitute for REBOUND.
- **Encounter line**: keep REBOUND + event index; optional later kernels / §7 patched-conic for strong one-shot exchanges.

---

## 3. Pipeline to execute (now)

Aligned with [`PROMPT.md`](../PROMPT.md). Collision ignored on PEO path.

```text
[done] Generator + Simulator + closure scores
  θ∈ℝ⁷ → ManifoldParams → REBOUND Φ_T
  Orbit error := E_r(t), E_v(t)  (shape / velocity under best (R,P))
        │
        ▼
[next] Filter thresholds + archive + refinement F(θ)=0
  Level 0 escape → Level 1 radial S(T)=P(S(0)) → Level 2–3 cut on E_r, E_v
```

**Orbit error definition:** not D_Td / energy proxies. For each sample t,
Kabsch \(R\in SO(3)\) and \(P\in S_4\) minimize

\[
E_r=\sum_i\|r_i(t)-R r_{P(i)}(0)\|^2,\quad
E_v=\sum_i\|v_i(t)-R v_{P(i)}(0)\|^2.
\]

Drift of \((E_r(t),E_v(t))\) is the error time series (`observe.closure`).

---

## 4. Engineering map (repo)

| Piece | Role |
|-------|------|
| `fairy_orbit.design.manifold` | 7D θ → X₀ on Td rays |
| `fairy_orbit.design.tetra_eff` | Td group-orbit analytic (legacy calib) |
| `fairy_orbit.engine` | REBOUND Φ_T |
| `fairy_orbit.observe.closure` | Kabsch, E_r, E_v, radial S(t) |
| `fairy_orbit.observe.peo` | Filter: escape → choreography → closure |
| `fairy_orbit.observe.error_base` | Exp normalize of E_r/E_v |
| `experiments/verify_orbit_seeds.py` | PROMPT §3.2 seed gate |
| `experiments/run_mass_continuation_campaign.py` | Path A / Path B continuation |
| `experiments/run_path_a_cycle.py` | Multi-seed Path A + multiperiod scan |
| `experiments/optics_tides/` | Central-observer starry photometry |
| `fairy_orbit.design.seeds` | Equal-mass / hierarchical continuation seed catalogue |
| `docs/continuation/` | Path A (\(m_c\) from 0) / Path B (\(m_c=1\)) specs |

---

## 5. Success criteria

**Closure (now):** E_r(0)=E_v(0)=0; growth of E_r, E_v is the orbit error.

**PEO (PROMPT):** \(\Phi_T(X_0)\approx(R,P)X_0\) after escape + radial
choreography filters — archive then refine.

---

## 6. Negative result: Bayesian / GP search (2026-07-28)

Wide full-parameter Optuna TPE (`experiments/output/bayes_full_wide/`, 2000 trials)
reached **success=0** (best ≈ choreography soft floor). This is treated as a
**method mismatch**, not undersampling:

- Resonance / radial-choreography residuals live on **fractal / separatrix-sensitive**
  landscapes; nearby parameters can jump between escape, identity-at-T, and near-miss.
- Bayesian optimization assumes a **smooth** surrogate (GP / TPE clustering). That
  assumption does not hold on such landscapes.

**Demote:** free-parameter Bayes / staged soft-residual BO as mainline PEO hunt.  
**Promote:** PROMPT mainline — equal-mass choreography → **mass continuation + Newton**
(pseudo-arclength on folds). Layers mirrored in
[`docs/continuation/PHASES.md`](continuation/PHASES.md).  
**On conflict, [`PROMPT.md`](../PROMPT.md) covers all other docs.**

Construct sources / Fourier+action: PROMPT §3.1 and
[`docs/continuation/LITERATURE_SEEDS.md`](continuation/LITERATURE_SEEDS.md).
