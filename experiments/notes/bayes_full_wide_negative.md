# Bayes PEO full-wide: negative result

Run: `experiments/run_bayes_peo.py --full --wide --n-trials 2000`  
Output: `experiments/output/bayes_full_wide/` (local; gitignored)

| Metric | Value |
|--------|-------|
| trials | 2000 |
| wall | ~39 min |
| success (gate pass) | **0** |
| choreography rejects | ~1034 |
| best BO loss | ~100.04 (`PENALTY_CHOREO_BASE` + soft residual) |
| escalations (bound expand) | 1 |

## Interpretation

Not “need more trials.” Target (radial ABCD→BCDA + closure) is sparse and
**non-smooth** in free θ; GP/TPE interpolators are the wrong tool class.

Direction update: [`docs/DIRECTION.md`](../../docs/DIRECTION.md) §6 →
[`docs/continuation/`](../../docs/continuation/).
