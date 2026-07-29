# Seed classification

Subordinate to [`PROMPT.md`](../../PROMPT.md) §3.3. On conflict, PROMPT wins.

Gate: \(x_i(T/n)=R\,x_{P(i)}(0)\) for **r and v** (binary).

## Classes

| `orbit_class` | Meaning | Next layer |
|---------------|---------|------------|
| `literature_choreography` / `action_fourier_new` | §3.2 OK **and** does not maintain regular n-gon | Path A/B continuation |
| `rejected_maintained_regular_ngon` | Orbit **keeps** square/pentagon A=B=… RE over \(T\) — **policy reject** | Do not continue |
| `hier_baseline_ic` | Hierarchical IC; **not** claimed periodic | Do not continue until gate |
| `unverified` / fail | Gate failed | Reconstruct |

**Policy:** reject solutions whose dynamics **maintain** a regular equal-mass n-gon
(正四边形 / 正五边形 with roles A=B=C=… for the whole period). A snapshot that
looks polygonal at one time is **not** enough to reject.
