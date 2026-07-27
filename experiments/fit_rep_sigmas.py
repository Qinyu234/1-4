#!/usr/bin/env python3
"""Recompute sigmas.json from an existing rep_error runs.jsonl (no re-integrate)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.observe.rep_error import CHANNELS, compute_sigmas

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Fit σ from rep_error runs.jsonl")
    p.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "experiments" / "output" / "rep_error" / "runs.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "rep_error" / "sigmas.json",
    )
    args = p.parse_args()
    samples = []
    for line in args.runs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append({k: float(row[f"{k}_final"]) for k in CHANNELS})
    sig = compute_sigmas(samples, source=f"from:{args.runs.name}")
    sig.to_json(args.out)
    print(json.dumps({"n": sig.n_samples, "sigmas": sig.as_dict(), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
