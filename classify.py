#!/usr/bin/env python3
"""
Goal classifier — the core experiment loop.

Runs every clip in a dataset through a VLM and asks:
  1) was there a goal?
  2) if yes, which team scored? (sportswear vs suits)

Scores goal detection (F1) and team assignment on goal clips.

Knobs for Overmind:
  - PROMPT (--prompt, default prompt.txt; use {team1} / {team2} placeholders)
  - MODEL  (--model or CLASSIFIER_MODEL)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.generativeai as genai

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


def classify_one(clip: Path, prompt: str, teams: list[str],
                 model_name: str, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name, generation_config=genai.GenerationConfig(temperature=0.0)
    )
    truth, truth_team = labels_for(clip.with_suffix(".json"))
    try:
        resp = model.generate_content(
            [{"mime_type": "video/mp4", "data": clip.read_bytes()}, prompt]
        )
        raw = resp.text.strip()
        txt = raw
        if txt.startswith("```"):
            txt = txt.split("```")[1].lstrip("json").strip()
        parsed = json.loads(txt)
        pred = "goal" if parsed.get("goal") else "not_goal"
        pred_team = normalize_team(parsed.get("team"), teams) if pred == "goal" else None
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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (export it or put it in .env)")
    model = model or os.environ.get("CLASSIFIER_MODEL", "gemini-3.5-flash")
    teams = load_teams(dataset)
    prompt = build_prompt(prompt_template, teams)

    clips = sorted((DATA_DIR / dataset).glob("*.mp4"))
    if limit:
        clips = clips[:limit]
    if not clips:
        raise RuntimeError(f"No clips in {DATA_DIR / dataset}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(classify_one, c, prompt, teams, model, api_key) for c in clips]
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
    p = argparse.ArgumentParser(description="VLM goal + team classifier")
    p.add_argument("--dataset", default="9-8GT-right")
    p.add_argument("--prompt", default=str(ROOT / "prompt.txt"))
    p.add_argument("--model", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    template = Path(args.prompt).read_text().strip()
    teams = load_teams(args.dataset)
    print(f"Dataset: {args.dataset}   Teams: {teams[0]} vs {teams[1]}")
    print(f"Prompt: {args.prompt}")
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

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.dataset}_{m['model']}.json"
    out.write_text(json.dumps(m, indent=2))
    print(f"\nSaved: {out}")
    print(f"SCORE f1={m['f1']:.4f} team_acc={m['team_accuracy_on_detected_goals']:.4f}")


if __name__ == "__main__":
    main()
