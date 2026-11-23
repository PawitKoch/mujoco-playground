"""Run every example headless and report pass/fail + wall time.

    uv run python scripts/run_all.py              # all except the Gemini demo
    uv run python scripts/run_all.py 01 03        # only examples whose folder starts with these
    uv run python scripts/run_all.py --with-gemini

Each example's run.py asserts its own "the feature did what it should" check,
so a non-zero exit means the demo no longer shows what its README claims.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Examples that need extra flags to run as a short, non-interactive smoke test.
SMOKE_ARGS = {"08": ["--quick"], "10": ["--smoke"]}
GEMINI = "09"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("only", nargs="*", help="folder prefixes to run, e.g. 01 03")
    p.add_argument("--with-gemini", action="store_true", help="also run a scripted vibe-sim turn (uses the API)")
    args = p.parse_args()

    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    results = []
    for folder in sorted((REPO / "examples").iterdir()):
        prefix = folder.name[:2]
        if args.only and prefix not in args.only:
            continue
        if prefix == GEMINI:
            if not args.with_gemini:
                results.append((folder.name, "skipped", 0.0))
                continue
            cmd = [sys.executable, "vibe_sim.py", "--thinking", "low",
                   "--script", "make the cube blue and twice as big, then simulate 1 second"]
        else:
            if not (folder / "run.py").exists():
                continue
            cmd = [sys.executable, "run.py", *SMOKE_ARGS.get(prefix, [])]
        print(f"\n=== {folder.name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, cwd=folder, env=env).returncode
        results.append((folder.name, "ok" if rc == 0 else f"FAIL ({rc})", time.perf_counter() - t0))

    print("\n" + "\n".join(f"{status:>10}  {secs:6.1f}s  {name}" for name, status, secs in results))
    return int(any(s.startswith("FAIL") for _, s, _ in results))


if __name__ == "__main__":
    sys.exit(main())
