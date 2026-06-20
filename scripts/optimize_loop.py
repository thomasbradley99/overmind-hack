#!/usr/bin/env python3
"""
The full loop: score all clips → Gemini edits prompt.txt → score again → repeat.

This is what `overmind optimize` was supposed to do; we use Gemini coding-agent instead.

Usage (from repo root, with .venv + .env):

  source .venv/bin/activate
  export $(grep -v '^#' .env | xargs)
  python scripts/optimize_loop.py

  python scripts/optimize_loop.py --iterations 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _optimizer_python() -> str:
    uv_tool = Path.home() / ".local/share/uv/tools/overmind/bin/python"
    if uv_tool.is_file():
        return str(uv_tool)
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def _classify_python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def main() -> int:
    p = argparse.ArgumentParser(description="Score → edit prompt → score loop")
    p.add_argument("--dataset", default="9-8GT-right")
    p.add_argument("--iterations", type=int, default=int(os.environ.get("OPT_ITERATIONS", "3")))
    args = p.parse_args()

    print("=" * 60)
    print("OPTIMIZE LOOP  (classify → edit prompt.txt → classify …)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}   iterations: {args.iterations}")
    print(f"Logs: .overmind/agents/goal-classifier/experiments/gemini_prompt_loop/")
    print()

    # Step 0: fresh baseline score
    print(">>> Baseline classify (temperature=0, deterministic)")
    subprocess.run(
        [
            _classify_python(),
            str(ROOT / "classify.py"),
            "--dataset",
            args.dataset,
            "--temperature",
            "0",
        ],
        cwd=ROOT,
        check=True,
    )

    # Steps 1..N: diagnose failures → coding agent edits prompt → re-score → keep best
    print("\n>>> Prompt optimization (Gemini coding agent)")
    opt_py = _optimizer_python()
    cmd = [
        opt_py,
        str(ROOT / "scripts" / "gemini_prompt_optimize.py"),
        "--apply",
        "--eval",
        "--keep-best",
        "--focus",
        "both",
        "--dataset",
        args.dataset,
        "--iterations",
        str(args.iterations),
    ]
    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
