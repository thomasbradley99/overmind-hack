#!/usr/bin/env python3
"""
Export clip labels from data/<dataset>/ into Overmind dataset.json format.

Each row: { "input": { "clip_path", "dataset" }, "expected_output": { "goal", "team" } }

Usage:
  python3 scripts/export_overmind_dataset.py --dataset 9-8GT-right
  python3 scripts/export_overmind_dataset.py --dataset 9-8GT-right --out data/seed.json
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="9-8GT-right")
    p.add_argument("--out", default=None, help="Output path (default: data/seed.json)")
    args = p.parse_args()

    clip_dir = DATA_DIR / args.dataset
    rows = []
    for clip_json in sorted(clip_dir.glob("*.json")):
        data = json.loads(clip_json.read_text())
        clip_path = f"data/{args.dataset}/{clip_json.stem}.mp4"
        if data.get("label") == "not_goal" or data.get("action") is None:
            expected = {"goal": False, "team": None}
        else:
            expected = {"goal": True, "team": data.get("team")}
        rows.append({
            "input": {"clip_path": clip_path, "dataset": args.dataset},
            "expected_output": expected,
        })

    out = Path(args.out) if args.out else DATA_DIR / "seed.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} cases to {out}")


if __name__ == "__main__":
    main()
