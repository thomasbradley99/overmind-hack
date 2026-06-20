#!/usr/bin/env python3
"""
Cut labeled event clips from a full game video using its ground-truth.json.

For each matching event it writes:
  clips/<game>/<action>_NN_<time>s_<team>.mp4    the video clip
  clips/<game>/<action>_NN_<time>s_<team>.json   the ground-truth answer (no description)

Usage:
  python3 scripts/make_clips.py --game 9-8GT-right
  python3 scripts/make_clips.py --game 9-8GT-right --actions Goal "Near Miss" --before 8 --after 4
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
GAMES_DIR = ROOT / "games"
CLIPS_DIR = ROOT / "clips"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def video_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def main():
    p = argparse.ArgumentParser(description="Cut labeled event clips from a game video")
    p.add_argument("--game", required=True, help="Game folder name under games/")
    p.add_argument("--actions", nargs="+", default=["Goal"],
                   help='Event actions to cut (default: Goal). e.g. --actions Goal "Near Miss"')
    p.add_argument("--before", type=float, default=8.0, help="Seconds of lead-up before the event")
    p.add_argument("--after", type=float, default=4.0, help="Seconds after the event")
    args = p.parse_args()

    game_dir = GAMES_DIR / args.game
    video = game_dir / "video.mp4"
    gt_path = game_dir / "ground-truth.json"
    if not video.exists():
        sys.exit(f"ERROR: no video at {video}")
    if not gt_path.exists():
        sys.exit(f"ERROR: no ground-truth.json at {gt_path}")

    duration = video_duration(video)
    events = json.loads(gt_path.read_text()).get("events", [])
    targets = [e for e in events if e.get("action") in args.actions]
    targets.sort(key=lambda e: e.get("time", 0))

    out_dir = CLIPS_DIR / args.game
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Game: {args.game}  ({duration/60:.1f} min)")
    print(f"Cutting {len(targets)} clip(s) for actions {args.actions} "
          f"[-{args.before}s / +{args.after}s] into {out_dir}")

    for i, ev in enumerate(targets, 1):
        t = float(ev.get("time", 0))
        start = max(0.0, t - args.before)
        end = min(duration, t + args.after)
        dur = round(end - start, 2)
        action = slug(ev.get("action", "Event"))
        team = slug(ev.get("team", "unknown"))
        stem = f"{action.lower()}_{i:02d}_{int(t)}s_{team}"

        clip_path = out_dir / f"{stem}.mp4"
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(video),
               "-t", str(dur), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "28", "-c:a", "aac", "-b:a", "96k",
               "-movflags", "+faststart", str(clip_path)]
        subprocess.run(cmd, capture_output=True)

        # Ground-truth answer for this clip (no description by design)
        (out_dir / f"{stem}.json").write_text(json.dumps({
            "game": args.game,
            "action": ev.get("action"),
            "team": ev.get("team"),
            "time_in_game": t,
            "time_in_clip": round(t - start, 2),
            "clip_start": round(start, 2),
            "clip_duration": dur,
        }, indent=2))

        ok = "OK" if clip_path.exists() else "FAIL"
        print(f"  [{ok}] {stem}.mp4  ({dur:.0f}s @ {int(t)}s)")

    print("Done.")


if __name__ == "__main__":
    main()
