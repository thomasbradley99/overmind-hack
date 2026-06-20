#!/usr/bin/env python3
"""
Cut random NON-GOAL (negative) clips from a game, avoiding all ground-truth events.

These are the "easy negatives" for a goal-vs-not control set: random windows of
play that contain no labeled event (goal, near miss, or big hit).

Writes into data/<game>/:
  nongoal_NN_<time>s.mp4
  nongoal_NN_<time>s.json   {action: null, label: "not_goal", ...}

Usage:
  python3 scripts/make_negatives.py --game 9-8GT-right --count 17
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
GAMES_DIR = ROOT / "games"
CLIPS_DIR = ROOT / "data"


def video_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def main():
    p = argparse.ArgumentParser(description="Cut random non-goal clips as control negatives")
    p.add_argument("--game", required=True)
    p.add_argument("--count", type=int, default=17, help="How many negative clips")
    p.add_argument("--before", type=float, default=8.0)
    p.add_argument("--after", type=float, default=4.0)
    p.add_argument("--gap", type=float, default=25.0,
                   help="Min seconds a clip must stay away from any ground-truth event")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    game_dir = GAMES_DIR / args.game
    video = game_dir / "video.mp4"
    gt_path = game_dir / "ground-truth.json"
    if not video.exists() or not gt_path.exists():
        sys.exit(f"ERROR: need {video} and {gt_path}")

    duration = video_duration(video)
    clip_len = args.before + args.after
    event_times = [float(e.get("time", 0)) for e in json.loads(gt_path.read_text()).get("events", [])]

    random.seed(args.seed)
    out_dir = CLIPS_DIR / args.game
    out_dir.mkdir(parents=True, exist_ok=True)

    def far_from_events(center: float) -> bool:
        lo, hi = center - args.before - args.gap, center + args.after + args.gap
        return all(not (lo <= t <= hi) for t in event_times)

    chosen: list[float] = []
    attempts = 0
    while len(chosen) < args.count and attempts < args.count * 200:
        attempts += 1
        center = random.uniform(args.before + 5, duration - args.after - 5)
        if far_from_events(center) and all(abs(center - c) > clip_len for c in chosen):
            chosen.append(round(center, 1))
    chosen.sort()

    print(f"Game: {args.game} ({duration/60:.1f} min) — cutting {len(chosen)} non-goal clips")
    for i, center in enumerate(chosen, 1):
        start = max(0.0, center - args.before)
        dur = round(min(duration, center + args.after) - start, 2)
        stem = f"nongoal_{i:02d}_{int(center)}s"
        clip_path = out_dir / f"{stem}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-i", str(video), "-t", str(dur),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(clip_path)],
            capture_output=True,
        )
        (out_dir / f"{stem}.json").write_text(json.dumps({
            "game": args.game,
            "action": None,
            "label": "not_goal",
            "center_in_game": center,
            "clip_start": round(start, 2),
            "clip_duration": dur,
        }, indent=2))
        ok = "OK" if clip_path.exists() else "FAIL"
        print(f"  [{ok}] {stem}.mp4 ({dur:.0f}s @ {int(center)}s)")

    if len(chosen) < args.count:
        print(f"WARNING: only placed {len(chosen)}/{args.count} (try smaller --gap)")
    print("Done.")


if __name__ == "__main__":
    main()
