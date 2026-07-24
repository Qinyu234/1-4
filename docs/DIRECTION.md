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
| Same-radius tetrahedron + Rodrigues velocity copy as main IC | **Trap** | Exact symmetry → rigid evolution; use **non-coplanar tetrahedral periapses on nested \(a\)** instead |
| Drop all non-adjacent \(R_{ij}\) without checking | **Not yet accepted** | For 3:2×5:3×7:5, \(a_4/a_1\approx2.30\) ⇒ \(\alpha_{14}\approx0.43\) — Laplace terms not obviously negligible; verify before assuming |

### Optimized middle ground

- **Secular Resonant Chain Approximation** (future): standard averaged \(R_{ij}\) (Laplace), optionally adjacent-only **after** coefficient check; role = intuition / pre-screen / explain REBOUND — **not** a new gravity model and **not** a substitute for REBOUND.
- **Encounter line**: keep REBOUND + event index; optional later kernels / §7 patched-conic for strong one-shot exchanges.

---

## 3. Pipeline to execute (now)

```text
Orbit-family generator (nested a + shared e + tetrahedral 3D periapses)
        │
        ▼
REBOUND IAS15  (parameter scan → SQLite by param_class)
        │
        ▼
Observe: a(t), e(t), encounters, interest, AMD
        │
        ▼
Rank / query / plot  (prefer change: migration, swap — not static)
```

**Deferred:** full LPE integrator, free \(W_{ij}\), distance-kernel secular hybrid.

---

## 4. Engineering map (repo)

| Piece | Role |
|-------|------|
| `fairy_orbit.design` | Hierarchical resonant chain IC |
| `fairy_orbit.engine` | REBOUND truth |
| `fairy_orbit.observe` | Elements, encounters, interest, AMD |
| `fairy_orbit.store` | SQLite + traj sidecars, classed by IC |
| `experiments/run_dynamics_scan.py` | Wide IC scan |
| `experiments/query_orbits.py` | Call saved runs later |
| `experiments/report_scan.py` | Post-scan: re-integrate top hits + AMD plots |
| `scripts/run_campaign.py report` | Launcher for the report |
| `fairy_orbit.legacy` | Old search / hierarchical sim (quarantine) |

---

## 5. Success criteria (from §1 + dialogue)

A chain is interesting if REBOUND shows, without requiring a custom gravity model:

1. Bound / long-lived enough to study (not instant escape/collision),
2. Visible **semi-major-axis / AMD exchange** (not frozen rings),
3. Quasi-periodic **encounters** (event index),
4. Optionally later: secular approximation that **explains** the same trends.

Hard reject for product taste: high MEGNO≈2 **and** near-zero \(a\) drift (static).
