#!/usr/bin/env python3
"""
Goal classifier — the core experiment loop.

Runs every clip in a dataset through a local VLM (Ollama) and asks:
  1) was there a goal?
  2) if yes, which team scored? (sportswear vs suits)

Scores goal detection (F1) and team assignment on goal clips.

Knobs for Overmind:
  - PROMPT (--prompt, default prompt.txt; use {team1} / {team2} placeholders)
  - MODEL  (--model or OLLAMA_MODEL env var)
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import requests



ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
GAMES_DIR = ROOT / "games"
RESULTS_DIR = ROOT / "results"

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "richardyoung/smolvlm2-2.2b-instruct:latest")


def load_teams(dataset: str) -> list[str]:
    info = GAMES_DIR / dataset / "info.json"
    if info.exists():
        teams = json.loads(info.read_text()).get("teams")
        if teams and len(teams) >= 2:
            return teams[:2]
    return ["Team A", "Team B"]


def build_prompt(template: str, teams: list[str]) -> str:
    """Fill {team1} / {team2} in prompt.txt from game info."""
    t1, t2 = teams[0], teams[1]
    return (template.replace("{team1}", t1).replace("{team2}", t2))


def labels_for(clip_json: Path) -> tuple[str, str | None]:
    data = json.loads(clip_json.read_text())
    if data.get("label") == "not_goal":
        return "not_goal", None
    if data.get("label") == "goal" or data.get("action") == "Goal":
        return "goal", data.get("team")
    return "not_goal", None


def normalize_team(text: str | None, teams: list[str]) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    for team in teams:
        if team.lower() == t or team.lower() in t or t in team.lower():
            return team
    return text.strip()


def extract_clip_frames(clip_path: str, num_frames: int = 3, max_size: int = 336) -> list[str]:
    """Extract evenly spaced frames from a short video clip, optionally resized."""
    work_dir = Path(tempfile.mkdtemp(prefix="classify-"))
    # Get clip duration
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", clip_path
    ]
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        duration = float(duration_result.stdout.strip())
    except ValueError:
        duration = 0.0
    frames = []
    for i in range(num_frames):
        ts = duration * i / max(num_frames - 1, 1)
        frame = work_dir / f"frame_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", str(ts),
            "-i", clip_path, "-vframes", "1",
        ]
        if max_size and max_size > 0:
            cmd += ["-vf", f"scale='min({max_size},iw)':-1"]
        cmd += ["-q:v", "2", str(frame)]
        subprocess.run(cmd, capture_output=True, timeout=10)
        if frame.exists():
            frames.append(str(frame))
    return frames


def _ollama_chat(payload: dict) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload.setdefault("stream", False)
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            f"Is ollama running? Error: {e}"
        )


def _extract_prediction(raw: str, teams: list[str]) -> tuple[bool, str | None]:
    """
    Robustly parse SmolVLM2 output into (goal, team).
    Handles multiple template examples, markdown, and nested JSON.
    """
    import re

    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1].lstrip("json").strip()

    # Strategy 1: Find all valid JSON objects in the text and pick the one
    # that matches the expected schema (has 'goal' and 'team' keys)
    candidates = []
    # Match JSON objects
    for obj_match in re.finditer(r'\{[^{}]*\}', txt):
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict) and "goal" in obj:
                candidates.append(obj)
        except json.JSONDecodeError:
            continue

    # Also try nested JSON with "goals" wrapper
    if not candidates:
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, dict):
                # Flatten nested structures
                for key, val in parsed.items():
                    if isinstance(val, dict) and "goal" in val:
                        candidates.append(val)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict) and "goal" in item:
                                candidates.append(item)
        except json.JSONDecodeError:
            pass

    if not candidates:
        return False, None

    # Pick the candidate that says goal=true (most likely the actual prediction)
    # If multiple goal=true, pick the one with a valid team name
    goal_candidates = [c for c in candidates if c.get("goal") is True]
    if goal_candidates:
        for c in goal_candidates:
            team = c.get("team")
            if team in teams:
                return True, team
        # Fallback: first goal candidate
        return True, str(goal_candidates[0].get("team", "")) if goal_candidates[0].get("team") else (False, None)

    # If no goal=true, use first goal=false
    no_goal = [c for c in candidates if c.get("goal") is False or c.get("goal") is None]
    if no_goal:
        return False, None

    # Fallback: last resort
    return False, None


def classify_one(clip: Path, prompt: str, teams: list[str],
                 model_name: str, api_key: str | None = None) -> dict:
    """Classify a single clip using the local Ollama vision model."""
    truth, truth_team = labels_for(clip.with_suffix(".json"))
    try:
        # Extract frames from the clip
        frames = extract_clip_frames(str(clip), num_frames=3)
        if not frames:
            raise RuntimeError("Could not extract frames from clip")

        # Encode frames to base64
        images_b64 = []
        for f in frames:
            with open(f, "rb") as img:
                images_b64.append(base64.b64encode(img.read()).decode("utf-8"))

        # Build Ollama payload
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        raw = _ollama_chat(payload).strip()
        goal, pred_team = _extract_prediction(raw, teams)
        pred = "goal" if goal else "not_goal"
        pred_team = normalize_team(pred_team, teams) if pred == "goal" else None
        return {
            "clip": clip.name,
            "truth": truth,
            "truth_team": truth_team,
            "pred": pred,
            "pred_team": pred_team,
            "raw": raw,
        }
    except Exception as e:
        return {
            "clip": clip.name,
            "truth": truth,
            "truth_team": truth_team,
            "pred": "error",
            "pred_team": None,
            "raw": str(e)[:160],
        }


def score_goals(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
    fn = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "not_goal")
    tn = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "not_goal")
    fp = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "goal")
    errors = sum(1 for r in results if r["pred"] == "error")
    n = tp + fn + tn + fp
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "errors": errors,
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
    }


def score_teams(results: list[dict], teams: list[str]) -> dict:
    """Team accuracy on clips that are actually goals (and we predicted goal)."""
    goal_clips = [r for r in results if r["truth"] == "goal"]
    detected = [r for r in goal_clips if r["pred"] == "goal"]
    team_correct = sum(
        1 for r in detected
        if r["truth_team"] and r["pred_team"]
        and normalize_team(r["truth_team"], teams) == normalize_team(r["pred_team"], teams)
    )
    team_wrong = len(detected) - team_correct
    spurious_team = sum(
        1 for r in results
        if r["truth"] == "not_goal" and r["pred"] == "goal" and r.get("pred_team")
    )
    n_goals = len(goal_clips)
    acc_on_goals = team_correct / len(detected) if detected else 0.0
    return {
        "goal_clips": n_goals,
        "goals_detected": len(detected),
        "team_correct": team_correct,
        "team_wrong": team_wrong,
        "team_accuracy_on_detected_goals": round(acc_on_goals, 4),
        "spurious_team_on_non_goals": spurious_team,
        "teams": teams,
    }


def classify_dataset(prompt_template: str, dataset: str = "9-8GT-right",
                     model: str | None = None, workers: int = 8,
                     limit: int | None = None, verbose: bool = False) -> dict:
    model = model or os.environ.get("CLASSIFIER_MODEL", OLLAMA_MODEL)
    teams = load_teams(dataset)
    prompt = build_prompt(prompt_template, teams)

    clips = sorted((DATA_DIR / dataset).glob("*.mp4"))
    if limit:
        clips = clips[:limit]
    if not clips:
        raise RuntimeError(f"No clips in {DATA_DIR / dataset}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(classify_one, c, prompt, teams, model, None) for c in clips]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if verbose:
                goal_ok = r["pred"] == r["truth"]
                team_ok = (
                    r["truth"] != "goal"
                    or r["pred"] != "goal"
                    or normalize_team(r.get("truth_team"), teams) == normalize_team(r.get("pred_team"), teams)
                )
                mark = "OK " if goal_ok and team_ok else ("ERR" if r["pred"] == "error" else "X  ")
                tt = r.get("truth_team") or "-"
                pt = r.get("pred_team") or "-"
                print(f"  {mark} {r['clip']:<36} goal {r['truth']}/{r['pred']}  team {tt}/{pt}")

    results.sort(key=lambda r: r["clip"])
    goal_metrics = score_goals(results)
    team_metrics = score_teams(results, teams)
    metrics = {**goal_metrics, **team_metrics}
    metrics["model"] = model
    metrics["dataset"] = dataset
    metrics["n_clips"] = len(clips)
    metrics["results"] = results
    return metrics


def main():
    p = argparse.ArgumentParser(description="VLM goal + team classifier (local Ollama model)")
    p.add_argument("--dataset", default="9-8GT-right")
    p.add_argument("--prompt", default=str(ROOT / "prompt.txt"))
    p.add_argument("--model", default=None)
    p.add_argument("--workers", type=int, default=2,
                     help="Concurrent workers (default 2; keep low for local GPU)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    template = Path(args.prompt).read_text().strip()
    teams = load_teams(args.dataset)
    print(f"Dataset: {args.dataset}   Teams: {teams[0]} vs {teams[1]}")
    print(f"Prompt: {args.prompt}")
    print(f"Model:  {args.model or os.environ.get('CLASSIFIER_MODEL', OLLAMA_MODEL)}")
    m = classify_dataset(template, dataset=args.dataset, model=args.model,
                         workers=args.workers, limit=args.limit, verbose=True)

    print("\n" + "=" * 50)
    print(f"GOAL DETECTION  model={m['model']}")
    print("=" * 50)
    print("                 pred goal   pred not_goal")
    print(f"  truth goal        {m['tp']:^9}   {m['fn']:^11}")
    print(f"  truth not_goal    {m['fp']:^9}   {m['tn']:^11}")
    if m["errors"]:
        print(f"  (errors: {m['errors']})")
    print("-" * 50)
    print(f"  F1       : {m['f1']*100:5.1f}%")
    print(f"  Accuracy : {m['accuracy']*100:5.1f}%")

    print("\n" + "=" * 50)
    print("TEAM ASSIGNMENT (on detected goals)")
    print("=" * 50)
    print(f"  Goal clips in dataset     : {m['goal_clips']}")
    print(f"  Goals detected by VLM   : {m['goals_detected']}")
    print(f"  Correct team            : {m['team_correct']}")
    print(f"  Wrong team              : {m['team_wrong']}")
    print(f"  Team accuracy (detected): {m['team_accuracy_on_detected_goals']*100:.1f}%")
    print("=" * 50)

    save_per_clip_results(m["results"], args.dataset, m["model"], teams)
    print(f"\nPer-clip results saved under: {RESULTS_DIR / args.dataset}")
    print(f"SCORE f1={m['f1']:.4f} team_acc={m['team_accuracy_on_detected_goals']:.4f}")

def save_per_clip_results(results: list[dict], dataset: str, model: str, teams: list[str]):
    """Write/update one JSON per clip, keyed by model name for comparison."""
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        clip_name = r["clip"]
        # Strip .mp4 extension for the filename base
        base = clip_name.rsplit(".", 1)[0]
        path = out_dir / f"{base}.json"

        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = {
                "clip": clip_name,
                "truth": r["truth"],
                "truth_team": r["truth_team"],
                "models": {},
            }

        goal_ok = r["pred"] == r["truth"]
        team_ok = (
            r["truth"] != "goal"
            or r["pred"] != "goal"
            or normalize_team(r.get("truth_team"), teams)
            == normalize_team(r.get("pred_team"), teams)
        )

        data["models"][model] = {
            "pred": r["pred"],
            "pred_team": r.get("pred_team"),
            "raw": r.get("raw", ""),
            "goal_correct": goal_ok,
            "team_correct": team_ok,
        }
        path.write_text(json.dumps(data, indent=2))



if __name__ == "__main__":
    main()
