#!/usr/bin/env python3
"""
Goal classifier — the core experiment loop.

Runs every clip in a dataset through a VLM and asks:
  1) was there a goal?
  2) if yes, which team scored? (sportswear vs suits)

Scores goal detection (F1) and team assignment on goal clips.

Default: one clip → one Gemini call → JSON {goal, team} using prompt.txt.
Optional: --mode twostage for split goal/team prompts (experiments only).
"""

import argparse
import json
import os
import re
import sys
import time
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


def infer_from_prose(raw: str, teams: list[str]) -> dict:
    """Fallback when local VLMs answer in prose instead of JSON."""
    lower = raw.lower()
    team = None
    for t in teams:
        if t.lower() in lower:
            team = t

    scoring_line = re.search(
        r"scoring team[^.\n]{0,60}(dark suits|dark sportswear)",
        lower,
    )
    if scoring_line:
        label = scoring_line.group(1)
        team = "Dark suits" if "suits" in label else "Dark sportswear"
        return {"goal": True, "team": team}

    goal_hit = any(
        re.search(p, lower)
        for p in (
            r"goal (?:has |was )?scored",
            r"a goal has been scored",
            r"indicating that a goal",
            r'"goal"\s*:\s*true',
        )
    )
    if goal_hit:
        return {"goal": True, "team": normalize_team(team, teams)}

    return {"goal": False, "team": None}


def parse_model_json(raw: str, teams: list[str] | None = None) -> dict | list:
    txt = raw.strip()
    if not txt:
        raise ValueError("empty model response")

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL | re.IGNORECASE)
    if fence:
        txt = fence.group(1).strip()

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
            match = re.search(pattern, txt)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
        if teams:
            return infer_from_prose(raw, teams)
        raise


def aggregate_predictions(parsed: dict | list, teams: list[str]) -> dict:
    """Collapse per-frame / per-grid-cell VLM output into one {goal, team}."""
    if isinstance(parsed, dict):
        return parsed

    if not isinstance(parsed, list) or not parsed:
        return {"goal": False, "team": None}

    if all(isinstance(x, bool) for x in parsed):
        return {"goal": any(parsed), "team": None}

    if all(isinstance(x, dict) for x in parsed):
        goals = [bool(x.get("goal")) for x in parsed]
        n = len(goals)
        goal_votes = sum(goals)
        not_goal_votes = n - goal_votes
        if goal_votes == 0:
            is_goal = False
        elif not_goal_votes == 0:
            is_goal = True
        elif goal_votes == 1 and not_goal_votes >= 2:
            is_goal = True
        elif goal_votes > not_goal_votes:
            is_goal = True
        else:
            is_goal = False
        if not is_goal:
            return {"goal": False, "team": None}
        team_votes = [
            normalize_team(x.get("team"), teams) for x in parsed if x.get("goal")
        ]
        team_votes = [t for t in team_votes if t]
        team = majority_team(team_votes, teams) if team_votes else None
        return {"goal": True, "team": team}

    return {"goal": False, "team": None}


def vlm_generate(
    clip: Path,
    prompt: str,
    model_name: str,
    api_key: str,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    backend = os.environ.get("CLASSIFIER_BACKEND", "gemini").lower()
    if backend in ("qwen", "qwen3", "ollama", "local"):
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        use_qwen = backend in ("qwen", "qwen3") or "qwen" in model_name.lower()
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                if use_qwen:
                    from vlm_local import qwen_generate

                    return qwen_generate(
                        clip,
                        prompt,
                        model_name,
                        host=host,
                        temperature=temperature,
                    )
                from vlm_local import ollama_generate

                n_frames = int(os.environ.get("LOCAL_VLM_FRAMES", "4"))
                return ollama_generate(
                    clip,
                    prompt,
                    model_name,
                    host=host,
                    temperature=temperature,
                    n_frames=n_frames,
                )
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise last_err or RuntimeError("local VLM failed")

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name,
                generation_config=genai.GenerationConfig(temperature=temperature),
            )
            resp = model.generate_content(
                [{"mime_type": "video/mp4", "data": clip.read_bytes()}, prompt]
            )
            raw = (resp.text or "").strip()
            if not raw:
                raise ValueError("empty model response")
            return raw
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_err or RuntimeError("vlm_generate failed")


def majority_team(votes: list[str | None], teams: list[str]) -> str | None:
    from collections import Counter

    valid = [v for v in votes if v]
    if not valid:
        return None
    counts = Counter(valid)
    top = counts.most_common()
    if len(top) == 1:
        return top[0][0]
    if top[0][1] > top[1][1]:
        return top[0][0]
    # tie — prefer vote from first pass
    for v in votes:
        if v:
            return v
    return None


def classify_one(
    clip: Path,
    prompt: str,
    teams: list[str],
    model_name: str,
    api_key: str,
    temperature: float = 0.0,
) -> dict:
    truth, truth_team = labels_for(clip.with_suffix(".json"))
    try:
        raw = vlm_generate(clip, prompt, model_name, api_key, temperature=temperature)
        parsed = aggregate_predictions(parse_model_json(raw, teams), teams)
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


def classify_one_twostage(
    clip: Path,
    goal_template: str,
    team_templates: list[str],
    teams: list[str],
    model_name: str,
    api_key: str,
    team_votes: int = 1,
    team_temperature: float = 0.0,
) -> dict:
    truth, truth_team = labels_for(clip.with_suffix(".json"))
    try:
        goal_prompt = build_prompt(goal_template, teams)
        raw_goal = vlm_generate(clip, goal_prompt, model_name, api_key, temperature=0.0)
        goal_parsed = aggregate_predictions(parse_model_json(raw_goal, teams), teams)
        is_goal = bool(goal_parsed.get("goal"))

        if not is_goal:
            return {
                "clip": clip.name,
                "truth": truth,
                "truth_team": truth_team,
                "pred": "not_goal",
                "pred_team": None,
                "raw": raw_goal,
                "raw_team": None,
                "mode": "twostage",
            }

        team_votes_list: list[str | None] = []
        team_raws: list[str] = []
        n_calls = max(1, team_votes) * len(team_templates)
        call_idx = 0
        for tmpl in team_templates:
            team_prompt = build_prompt(tmpl, teams)
            repeats = max(1, team_votes) if len(team_templates) == 1 else 1
            for r in range(repeats):
                temp = team_temperature if (n_calls > 1 and call_idx > 0) else 0.0
                raw_team = vlm_generate(clip, team_prompt, model_name, api_key, temperature=temp)
                team_parsed = aggregate_predictions(parse_model_json(raw_team, teams), teams)
                team_votes_list.append(normalize_team(team_parsed.get("team"), teams))
                team_raws.append(raw_team)
                call_idx += 1

        pred_team = majority_team(team_votes_list, teams)
        combined_raw = f"GOAL: {raw_goal}\nTEAM: " + " | ".join(team_raws)
        return {
            "clip": clip.name,
            "truth": truth,
            "truth_team": truth_team,
            "pred": "goal",
            "pred_team": pred_team,
            "raw": combined_raw,
            "raw_team": team_raws,
            "team_votes": team_votes_list,
            "mode": "twostage",
        }
    except Exception as e:
        return {
            "clip": clip.name,
            "truth": truth,
            "truth_team": truth_team,
            "pred": "error",
            "pred_team": None,
            "raw": str(e)[:200],
            "mode": "twostage",
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


def classify_dataset(
    prompt_template: str,
    dataset: str = "9-8GT-right",
    model: str | None = None,
    workers: int = 8,
    limit: int | None = None,
    verbose: bool = False,
    verbose_raw: bool = False,
    mode: str = "onestage",
    goal_template: str | None = None,
    team_templates: list[str] | None = None,
    team_votes: int = 1,
    team_temperature: float = 0.0,
    temperature: float = 0.0,
) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    backend = os.environ.get("CLASSIFIER_BACKEND", "gemini").lower()
    if backend == "gemini" and not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (export it or put it in .env)")
    model = model or os.environ.get("CLASSIFIER_MODEL", "gemini-3.5-flash")
    if backend in ("qwen", "qwen3"):
        model = os.environ.get("CLASSIFIER_MODEL", "qwen3-vl:8b")
    elif backend in ("ollama", "local"):
        model = os.environ.get("CLASSIFIER_MODEL", "llava-llama3")
    teams = load_teams(dataset)
    prompt = build_prompt(prompt_template, teams)

    clips = sorted((DATA_DIR / dataset).glob("*.mp4"))
    if limit:
        clips = clips[:limit]
    if not clips:
        raise RuntimeError(f"No clips in {DATA_DIR / dataset}")

    twostage = mode == "twostage"
    if twostage and not goal_template:
        goal_template = (ROOT / "prompt_goal.txt").read_text().strip()
    if twostage and not team_templates:
        team_templates = [
            (ROOT / "prompt_team.txt").read_text().strip(),
            (ROOT / "prompt_team_alt.txt").read_text().strip(),
        ]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        if twostage:
            futs = [
                ex.submit(
                    classify_one_twostage,
                    c,
                    goal_template,
                    team_templates,
                    teams,
                    model,
                    api_key,
                    team_votes,
                    team_temperature,
                )
                for c in clips
            ]
        else:
            futs = [ex.submit(classify_one, c, prompt, teams, model, api_key, temperature) for c in clips]
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
                if verbose_raw:
                    print(f"       raw API response:")
                    for line in r.get("raw", "").splitlines():
                        print(f"         {line}")
                    print()

    results.sort(key=lambda r: r["clip"])
    goal_metrics = score_goals(results)
    team_metrics = score_teams(results, teams)
    metrics = {**goal_metrics, **team_metrics}
    metrics["model"] = model
    metrics["dataset"] = dataset
    metrics["n_clips"] = len(clips)
    metrics["mode"] = mode
    metrics["temperature"] = temperature
    metrics["backend"] = backend
    if twostage:
        metrics["team_votes"] = team_votes
        metrics["team_temperature"] = team_temperature
        metrics["team_prompts"] = len(team_templates or [])
    metrics["results"] = results
    return metrics


def _run_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": mean, "std": 0.0, "min": mean, "max": mean}
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = var ** 0.5
    return {"mean": mean, "std": std, "min": min(values), "max": max(values)}


def main():
    p = argparse.ArgumentParser(description="VLM goal + team classifier")
    p.add_argument("--dataset", default="9-8GT-right")
    p.add_argument("--prompt", default=str(ROOT / "prompt.txt"))
    p.add_argument("--mode", choices=("onestage", "twostage"), default="onestage",
                   help="onestage=one prompt, one call (default); twostage=experimental split")
    p.add_argument("--prompt-goal", default=str(ROOT / "prompt_goal.txt"))
    p.add_argument("--prompt-team", default=str(ROOT / "prompt_team.txt"))
    p.add_argument("--prompt-team-alt", default=str(ROOT / "prompt_team_alt.txt"),
                   help="Second team prompt for dual-prompt voting in twostage mode")
    p.add_argument("--team-votes", type=int, default=2,
                   help="Extra team-stage calls per prompt when only one team prompt (majority vote)")
    p.add_argument("--team-temperature", type=float, default=0.15,
                   help="Temperature for team vote calls after the first (0 = all deterministic)")
    p.add_argument("--model", default=None)
    p.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("CLASSIFIER_TEMPERATURE", "0")),
        help="VLM sampling temperature (0=deterministic). Use 0.1–0.3 to see run variance.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat full dataset eval N times (reports mean±std when N>1)",
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose-raw", action="store_true",
                   help="Print each clip's raw Gemini API response as it completes")
    args = p.parse_args()

    mode = args.mode
    workers = args.workers
    backend = os.environ.get("CLASSIFIER_BACKEND", "gemini").lower()
    if workers is None:
        workers = 1 if backend in ("ollama", "local", "qwen", "qwen3") else (4 if mode == "twostage" else 8)
    template = Path(args.prompt).read_text().strip()
    teams = load_teams(args.dataset)
    print(f"Dataset: {args.dataset}   Teams: {teams[0]} vs {teams[1]}")
    print(f"Mode: {mode}   temperature: {args.temperature}   runs: {args.runs}")
    print(f"Backend: {backend}")
    if mode == "twostage":
        print(f"Goal prompt: {args.prompt_goal}")
        print(f"Team prompts: {args.prompt_team}, {args.prompt_team_alt}")
        print(f"Team votes: {args.team_votes}  team_temperature: {args.team_temperature}")
    else:
        print(f"Prompt: {args.prompt}")

    team_templates = [
        Path(args.prompt_team).read_text().strip(),
        Path(args.prompt_team_alt).read_text().strip(),
    ]

    run_metrics: list[dict] = []
    for run_i in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {run_i + 1}/{args.runs} ---")
        m = classify_dataset(
            template,
            dataset=args.dataset,
            model=args.model,
            workers=workers,
            limit=args.limit,
            verbose=args.verbose_raw or (args.runs == 1),
            verbose_raw=args.verbose_raw,
            mode=mode,
            goal_template=Path(args.prompt_goal).read_text().strip() if mode == "twostage" else None,
            team_templates=team_templates if mode == "twostage" else None,
            team_votes=args.team_votes,
            team_temperature=args.team_temperature,
            temperature=args.temperature,
        )
        run_metrics.append(m)
        if args.runs > 1:
            print(
                f"  run {run_i + 1}: f1={m['f1']:.4f} "
                f"team_acc={m['team_accuracy_on_detected_goals']:.4f}"
            )

    m = run_metrics[-1]
    if args.runs > 1:
        f1_stats = _run_stats([x["f1"] for x in run_metrics])
        team_stats = _run_stats([x["team_accuracy_on_detected_goals"] for x in run_metrics])
        m["run_summaries"] = [
            {"f1": x["f1"], "team_accuracy_on_detected_goals": x["team_accuracy_on_detected_goals"]}
            for x in run_metrics
        ]
        m["f1_mean"] = round(f1_stats["mean"], 4)
        m["f1_std"] = round(f1_stats["std"], 4)
        m["team_acc_mean"] = round(team_stats["mean"], 4)
        m["team_acc_std"] = round(team_stats["std"], 4)
        print("\n" + "=" * 50)
        print(f"VARIANCE over {args.runs} runs (temperature={args.temperature})")
        print("=" * 50)
        print(
            f"  F1:   {f1_stats['mean']*100:.1f}% ± {f1_stats['std']*100:.1f}% "
            f"(min {f1_stats['min']*100:.1f}% max {f1_stats['max']*100:.1f}%)"
        )
        print(
            f"  Team: {team_stats['mean']*100:.1f}% ± {team_stats['std']*100:.1f}% "
            f"(min {team_stats['min']*100:.1f}% max {team_stats['max']*100:.1f}%)"
        )
        print(
            f"\nPer-run F1: "
            + ", ".join(f"{x['f1']*100:.1f}%" for x in run_metrics)
        )

    if args.runs == 1:
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
    suffix = "_twostage" if m.get("mode") == "twostage" else ""
    if backend in ("ollama", "local"):
        from vlm_local import model_slug

        model_file = model_slug(m["model"])
    else:
        model_file = m["model"]
    out = RESULTS_DIR / f"{args.dataset}_{model_file}{suffix}.json"
    if args.runs > 1:
        save_doc = {
            "dataset": m["dataset"],
            "model": m["model"],
            "mode": m["mode"],
            "temperature": args.temperature,
            "runs": args.runs,
            "f1_mean": m["f1_mean"],
            "f1_std": m["f1_std"],
            "team_acc_mean": m["team_acc_mean"],
            "team_acc_std": m["team_acc_std"],
            "run_summaries": m["run_summaries"],
            "last_run_detail": run_metrics[-1],
        }
    else:
        save_doc = m
    out.write_text(json.dumps(save_doc, indent=2))
    print(f"\nSaved: {out}")
    if args.runs > 1:
        print(
            f"SCORE f1={m['f1_mean']:.4f}±{m['f1_std']:.4f} "
            f"team_acc={m['team_acc_mean']:.4f}±{m['team_acc_std']:.4f}"
        )
    else:
        print(f"SCORE f1={m['f1']:.4f} team_acc={m['team_accuracy_on_detected_goals']:.4f}")


if __name__ == "__main__":
    main()
