<!-- generated: 2026-07-27T19:56 from design.html@untracked -->

# design — behaviour manual

## generate_manifold
Spans: design, core

Build System from θ or seed: a0=1, M0=0, e0=e, μ=m; a_i=a0+i a1;
δv via Rodrigues Td map; shift to inertial COM (Σmr=Σmv=0).

## simulate
Spans: core, engine

Integrate System with REBOUND; stop on escape only (collision off).

## radial_choreography
Spans: observe, engine

Compute radial order S(t); fix P from S(T) vs S(0) (Level 1).

## closure_errors
Spans: observe, engine

Fixed P → R* from positions; emit relative E_r/E_v and element residuals
(E_a,E_e,E_i,E_Ω,E_ω,E_M) with the same R*.

## rep_error_scan
Spans: design, engine, observe, viz

Td Stage A scan: all 8 channels together vs (m,e,…); write series + REPORT.
Re-run whenever scan knobs change.

## normalize_sigmas
Spans: observe, store, viz

From scan sample cloud compute σ_i and write sigmas.json for consumers.

## stereo_from_seed
Spans: design, engine, observe, store

For each (m,e): grid+beam over (a1,e1,M1,vx,vy,vz), keep low historical
loss beams, refine (bisect/FD), score 8 channels with σ.

## peo_filter
Spans: design, core, engine, observe, store

Pipeline: generate → simulate → escape → choreography → 8-channel errors
→ σ-normalize → archive. Refinement F(θ)=0 is later.
