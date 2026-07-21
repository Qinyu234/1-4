"""Run the experiment scripts with profiling and heatmap generation support."""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
PROFILES_DIR = ROOT / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def _iter_scripts() -> list[Path]:
    return sorted([p for p in EXPERIMENTS_DIR.glob("*.py") if p.name != "__init__.py"])


def _run_script(script: Path, *, use_scalene: bool, use_cprofile: bool) -> tuple[int, float]:
    python_exe = sys.executable
    script_path = str(script)
    command = [python_exe, script_path]
    if use_scalene and shutil.which("scalene"):
        command = ["scalene", "--html", "--outfile", str(PROFILES_DIR / f"{script.stem}.html"), python_exe, script_path]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - start
    (PROFILES_DIR / f"{script.stem}.stdout.log").write_text(stdout, encoding="utf-8", errors="ignore")
    (PROFILES_DIR / f"{script.stem}.stderr.log").write_text(stderr, encoding="utf-8", errors="ignore")
    if use_cprofile and process.returncode == 0:
        profiler = cProfile.Profile()
        profiler.enable()
        prof_process = subprocess.Popen([python_exe, str(script)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        prof_stdout, prof_stderr = prof_process.communicate()
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats(str(PROFILES_DIR / f"{script.stem}.prof"))
    return process.returncode, elapsed


def _run_heatmap(script: Path) -> None:
    if script.stem not in {"generate_heatmap", "run_diagnostics"}:
        return
    if shutil.which("py-spy"):
        subprocess.run(
            ["py-spy", "record", "-o", str(PROFILES_DIR / f"{script.stem}.svg"), "--", sys.executable, str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment scripts with profiling")
    parser.add_argument("--no-scalene", action="store_true")
    parser.add_argument("--no-cprofile", action="store_true")
    parser.add_argument("--scripts", nargs="*", default=None)
    args = parser.parse_args()

    scripts = [Path(p) for p in args.scripts] if args.scripts else _iter_scripts()
    print(f"Found {len(scripts)} experiment(s).")
    for script in scripts:
        print("=" * 80)
        print(f"Running {script.name}")
        try:
            code, elapsed = _run_script(script, use_scalene=not args.no_scalene, use_cprofile=not args.no_cprofile)
            _run_heatmap(script)
            print(f"Finished in {elapsed:.2f}s")
            print(f"Exit code: {code}")
        except Exception as exc:  # pragma: no cover - defensive wrapper
            print(f"Failed with error: {exc}")


if __name__ == "__main__":
    main()
