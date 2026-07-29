# Seed classification

Subordinate to [`PROMPT.md`](../../PROMPT.md) §3.3. On conflict, PROMPT wins.

Gate: \(x_i(T/n)=R\,x_{P(i)}(0)\) for **r and v** (binary).

## Classes

| `orbit_class` | Meaning | Next layer |
|---------------|---------|------------|
| `free_relative_equilibrium` | Polygon RE; weak baseline, not full choreography BVP | Polish / replace with true choreography before Layer 3/4 if needed |
| `literature_choreography` | Imported IC, re-converged, §3.2 OK | Path A/B continuation |
| `action_fourier_new` | Fourier+action under our \((R,P)\) | Continuation |
| `hier_baseline_ic` | Hierarchical IC; **not** claimed periodic | Do not continue until gate |
| `unverified` / fail | Gate failed | Reconstruct |

## In-repo status

| id | \(N\) | class | note |
|----|------|-------|------|
| `free_4_square_re` | 4 | `free_relative_equilibrium` | Analytic RE; Layer-1 placeholder |
| `free_5_pentagon_re` | 5 | `free_relative_equilibrium` | Analytic RE; Layer-2 placeholder |
| `hier_1plus4_manifold` | 1+4 | `hier_baseline_ic` | Not a choreography seed |

True choreography seeds (PROMPT §3.1 / §6) still to import + re-converge.
