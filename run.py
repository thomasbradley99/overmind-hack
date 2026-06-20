#!/usr/bin/env python3
"""
Run the VLM football analysis pipeline locally on one game.

Pipeline:
  1. Spot      - chunk video, Gemini describes each chunk (plain text)
  2. Synthesize- Gemini turns descriptions into a narrative + identified events
  3. Extract   - Gemini converts the narrative into structured JSON events

Outputs land in: games/<game>/runs/<timestamp>/
The final detected events are in stage3_events.json (schema: {events, metadata}).

Usage:
  GEMINI_API_KEY=xxx python3 run.py --game 9-8GT-right
  GEMINI_API_KEY=xxx python3 run.py --game 9-8GT-right --minutes 5   # quick/cheap test
  GEMINI_API_KEY=xxx python3 run.py --game 9-8GT-right --eval        # also score vs ground-truth
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GAMES_DIR = SCRIPT_DIR / "games"

# Load .env (KEY=VALUE lines) if present, without overriding real env vars.
env_file = SCRIPT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(SCRIPT_DIR))
from pipeline import spot, synthesize, extract


def get_video_duration(video_path: Path) -> float:
    import subprocess
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def load_teams(game_dir: Path) -> list[str]:
    for name in ("info.json", "ground-truth.json"):
        p = game_dir / name
        if p.exists():
            data = json.loads(p.read_text())
            if data.get("teams"):
                return data["teams"]
    return ["Team A", "Team B"]


def run_pipeline(video_path: Path, teams: list[str], out_root: Path,
                 api_key: str, max_minutes: float | None) -> Path:
    if not video_path.exists():
        sys.exit(f"ERROR: video not found at {video_path}")

    duration = get_video_duration(video_path)
    max_duration = max_minutes * 60 if max_minutes else None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="vlm-run-"))

    (out_dir / "config.json").write_text(json.dumps({
        "video": str(video_path),
        "teams": teams,
        "video_duration_seconds": duration,
        "analyzed_minutes": max_minutes or round(duration / 60, 1),
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))

    print("=" * 60)
    print(f"VIDEO:   {video_path.name}")
    print(f"TEAMS:   {teams[0]} vs {teams[1]}")
    print(f"LENGTH:  {duration/60:.1f} min" + (f" (analyzing first {max_minutes} min)" if max_minutes else ""))
    print(f"OUTPUT:  {out_dir}")
    print("=" * 60)

    timing = {}

    # ---- Stage 1: Spot ----
    print("\n[1/3] Spotting (describe video chunks)...")
    t0 = time.time()
    spot_result = spot.run(str(video_path), work_dir, teams, api_key, max_duration=max_duration)
    observations = spot_result.get("combined_observations", "")
    timing["spot_seconds"] = round(time.time() - t0, 1)
    (out_dir / "stage1_observations.json").write_text(json.dumps(spot_result, indent=2))
    (out_dir / "stage1_observations.txt").write_text(observations)
    print(f"      done in {timing['spot_seconds']}s")

    # ---- Stage 2: Synthesize ----
    print("\n[2/3] Synthesizing (narrative + events)...")
    t0 = time.time()
    narrative = synthesize.run(observations, teams, duration, api_key)
    timing["synthesize_seconds"] = round(time.time() - t0, 1)
    (out_dir / "stage2_narrative.txt").write_text(narrative)
    print(f"      done in {timing['synthesize_seconds']}s")

    # ---- Stage 3: Extract ----
    print("\n[3/3] Extracting structured JSON...")
    t0 = time.time()
    events = extract.run(narrative, teams, duration, api_key)
    timing["extract_seconds"] = round(time.time() - t0, 1)
    (out_dir / "stage3_events.json").write_text(json.dumps(events, indent=2))
    print(f"      done in {timing['extract_seconds']}s")

    timing["total_seconds"] = round(sum(timing.values()), 1)
    (out_dir / "timing.json").write_text(json.dumps(timing, indent=2))

    print("\n" + "=" * 60)
    print(f"COMPLETE in {timing['total_seconds']}s")
    print(f"Final events: {out_dir / 'stage3_events.json'}")
    print("=" * 60)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Run the VLM football analysis pipeline locally")
    parser.add_argument("--game", default=None,
                        help="Game folder name under games/ (has video.mp4 + ground-truth.json)")
    parser.add_argument("--video", default=None,
                        help="Path to a single video file to analyze (e.g. a clip in clips/)")
    parser.add_argument("--minutes", type=float, default=None,
                        help="Only analyze the first N minutes (cheaper/faster for testing)")
    parser.add_argument("--eval", action="store_true",
                        help="After running, score detected goals against ground-truth.json (--game only)")
    parser.add_argument("--tolerance", type=float, default=30.0,
                        help="Eval match tolerance in seconds (default 30)")
    args = parser.parse_args()

    if not args.game and not args.video:
        args.game = "9-8GT-right"  # default sample game
    if args.game and args.video:
        sys.exit("ERROR: pass either --game or --video, not both")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: set GEMINI_API_KEY (export it or put it in .env)")

    # --- Single video / clip mode ---
    if args.video:
        video_path = Path(args.video)
        # Look for a sibling info.json for team names, else default
        info = video_path.parent / "info.json"
        teams = json.loads(info.read_text()).get("teams", ["Team A", "Team B"]) if info.exists() else ["Team A", "Team B"]
        out_root = SCRIPT_DIR / "outputs" / video_path.stem
        out_dir = run_pipeline(video_path, teams, out_root, api_key, args.minutes)
        print(f"\nDetected events: {out_dir / 'stage3_events.json'}")
        return

    # --- Full game mode ---
    game_dir = GAMES_DIR / args.game
    video_path = game_dir / "video.mp4"
    teams = load_teams(game_dir)
    out_dir = run_pipeline(video_path, teams, game_dir, api_key, args.minutes)

    if args.eval:
        gt = game_dir / "ground-truth.json"
        events = out_dir / "stage3_events.json"
        if not gt.exists():
            print(f"\n(skip eval: no ground-truth.json in games/{args.game}/)")
            return
        print("\nRunning evaluation...\n")
        import subprocess
        subprocess.run([sys.executable, str(SCRIPT_DIR / "eval.py"),
                        "--ai-output", str(events), "--gt", str(gt),
                        "--tolerance", str(args.tolerance)])
    else:
        print("\nTo evaluate this run:")
        print(f"  python3 eval.py --ai-output {out_dir / 'stage3_events.json'} "
              f"--gt games/{args.game}/ground-truth.json")


if __name__ == "__main__":
    main()
