#!/usr/bin/env python3
"""Path A smoke: free-4 §3.2 gate at M_c=0, tiny M_c step + LS corrector stub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.design.seeds import SEEDS_DIR, load_seed
from fairy_orbit.observe.continuation import mass_continuation_smoke

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Mass continuation smoke (PROMPT Path A)")
    p.add_argument(
        "--seed",
        type=Path,
        default=SEEDS_DIR / "free_4_square_re.json",
    )
    p.add_argument("--M-c", type=float, default=1e-4, dest="M_c")
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=8)
    p.add_argument("--no-correct", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "continuation_smoke" / "report.json",
    )
    args = p.parse_args()

    seed = load_seed(args.seed)
    res = mass_continuation_smoke(
        seed,
        M_c=args.M_c,
        shift=args.shift,
        correct=not args.no_correct,
        max_nfev=args.max_nfev,
    )
    print(
        f"gate0_ok={res.gate0.ok}  ||F||(0)={res.residual0_norm:.3e}  "
        f"||F||(Mc)={res.residual_mc_norm:.3e}  "
        f"||F||(corr)={res.residual_corrected_norm}  success={res.success}",
        flush=True,
    )
    print(res.message, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    print(f"report → {args.out}", flush=True)
    raise SystemExit(0 if res.success else 1)


if __name__ == "__main__":
    main()
