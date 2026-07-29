# Orbit seed store

Subordinate to [`PROMPT.md`](../../PROMPT.md) §3. On conflict, PROMPT wins.

## Decisions

| Choice | Decision |
|--------|----------|
| Design log | **PROMPT.md** |
| Canonical seeds | `fairy_orbit/design/seeds/` |
| Raw imports | `orbit_library/` (gitignored) |
| Promote to catalogue | Only after PROMPT §3.2 gate (r **and** v) |

## Acceptance (PROMPT §3.2)

Must check, after integrating \(T/n\):

\[
x_i\!\left(\tfrac{T}{n}\right)=R\,x_{P(i)}(0)
\quad\text{(position and velocity)}
\]

Record concrete \(R\) (axis/angle) and whether \(P\) is the target \(n\)-cycle.
Shape-congruence helpers in code are COM-frame tools when no absolute frame is
fixed a priori; they do not replace this algebraic gate for promotion.

```powershell
.\.venv\Scripts\python.exe experiments\verify_orbit_seeds.py
```

## Catalogue fields (PROMPT §3.3)

Prefer: \(N\), cycle type of \(P\), \(R\) axis/angle, action, Floquet multipliers
if available, source (self-built / downloaded+rechecked).

Current slots: see [`SEED_CLASSIFICATION.md`](SEED_CLASSIFICATION.md).
