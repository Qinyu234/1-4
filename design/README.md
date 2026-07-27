<!-- generated: 2026-07-27T19:56 from design.html@untracked -->

# design — module map

## Modules

- **core** — CanonicalUnits, SystemConfig, Body/System carriers.
Inertial ↔ COM frame (translation only: r'=r−R_cm, v'=v−V_cm).
Escape criterion. Collision ignored on the PEO path.
- **design** — Manifold on fixed Td rays q̂_i. Linear polys a_i=a0+i a1 (same for e,M).
Td-symmetric velocity kick δv_i=R_i·(vx,vy,vz). IC shifted to inertial COM
frame. Seed anchors a0=1, e0=e, M0=0, μ=m; free (a1,e1,M1,vx,vy,vz).
- **engine** — REBOUND IAS15 simulator Φ_T only: Trajectory (r,v,E,L). No PEO judgment.
- **observe** — PEO in inertial COM frame: radial order S(t); Kabsch R*∈SO(3);
E_r/E_v with r'_i(T)≈R r'_{P(i)}(0) (P∈S₄). Angles optional. Escape only.
- **viz** — Rep-error and stereo reports: 8-channel curves, σ tables, seed rankings.
- **store** — Candidate archive for filter survivors (θ, T, P, R*, 8-channel E, σ-tilde).

## Behaviours (cross-module paths)

- **generate_manifold**: design → core
  Build System from θ or seed: a0=1, M0=0, e0=e, μ=m; a_i=a0+i a1;
δv via Rodrigues Td map; shift to inertial COM (Σmr=Σmv=0).
- **simulate**: core → engine
  Integrate System with REBOUND; stop on escape only (collision off).
- **radial_choreography**: observe → engine
  Compute radial order S(t); fix P from S(T) vs S(0) (Level 1).
- **closure_errors**: observe → engine
  Fixed P → R* from positions; emit relative E_r/E_v and element residuals
(E_a,E_e,E_i,E_Ω,E_ω,E_M) with the same R*.
- **rep_error_scan**: design → engine → observe → viz
  Td Stage A scan: all 8 channels together vs (m,e,…); write series + REPORT.
Re-run whenever scan knobs change.
- **normalize_sigmas**: observe → store → viz
  From scan sample cloud compute σ_i and write sigmas.json for consumers.
- **stereo_from_seed**: design → engine → observe → store
  For each (m,e): grid+beam over (a1,e1,M1,vx,vy,vz), keep low historical
loss beams, refine (bisect/FD), score 8 channels with σ.
- **peo_filter**: design → core → engine → observe → store
  Pipeline: generate → simulate → escape → choreography → 8-channel errors
→ σ-normalize → archive. Refinement F(θ)=0 is later.

## Budget

- modules: 6 (soft 7 / hard 9)
- behaviours: 8 (soft 12 / hard 15)
