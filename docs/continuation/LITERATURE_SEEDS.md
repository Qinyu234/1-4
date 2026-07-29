# Literature seeds & standard construction

Subordinate to [`PROMPT.md`](../../PROMPT.md) §3.1. On conflict, PROMPT wins.

Two layers (PROMPT):

1. **Downloadable / table IC** — limited coverage & digits; **re-converge** before use.
2. **Standard construction** — Fourier truncation + action minimization; run it for
   custom \((R,P)\).

## Sources (PROMPT §3.1)

| Source | Role |
|--------|------|
| Torus-knot choreographies (arXiv 1901.03738), e.g. `pts_four_bodies.mat` via ref [70] | Best machine-readable **4-body** start found |
| Simó 5-body ECM tables (pentagon, 4-chain, …) | Main **5-body** table source (hand-transcribe) |
| Vanderbei gallery | Historical; **digits often insufficient** — re-solve always |
| Hip-hop (Chenciner–Venturelli 2000) | Listed in PROMPT as known 3D solution; use only if §3.2 gate passes |
| In-repo polygon RE | Analytic placeholders only — not full choreography BVP |

## Construction algorithm (PROMPT)

Parameterize path by truncated Fourier series → minimize Newtonian action
(collision barrier) → recover IC → verify §3.2 (r and v).

## Policy

`orbit_library/` until §3.2 passes → then slim JSON in `fairy_orbit/design/seeds/`
with citation + `orbit_class`.
