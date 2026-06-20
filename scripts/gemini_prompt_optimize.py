#!/usr/bin/env python3
"""
Hybrid Overmind + Gemini prompt loop (Gemini-only keys).

  ~/.local/share/uv/tools/overmind/bin/python scripts/gemini_prompt_optimize.py --apply --eval

Team-focused (goal rules frozen, invent discrimination tactics):

  ~/.local/share/uv/tools/overmind/bin/python scripts/gemini_prompt_optimize.py \\
    --focus team --apply --eval --iterations 3 --keep-best
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROMPT_PATH = ROOT / "prompt.txt"
PROMPT_GOAL_PATH = ROOT / "prompt_goal.txt"
PROMPT_TEAM_PATH = ROOT / "prompt_team.txt"


def _load_dotenv() -> None:
    for path in (ROOT / ".env", ROOT / ".overmind" / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _load_teams(dataset: str) -> list[str]:
    info = ROOT / "games" / dataset / "info.json"
    if info.is_file():
        teams = json.loads(info.read_text()).get("teams")
        if teams and len(teams) >= 2:
            return teams[:2]
    return ["Team A", "Team B"]


def _latest_results_path() -> Path | None:
    candidates = sorted((ROOT / "results").glob("*_gemini*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _row_goal_ok(r: dict) -> bool:
    return (r.get("truth") == "goal") == (r.get("pred") == "goal")


def _row_team_ok(r: dict) -> bool:
    truth_goal = r.get("truth") == "goal"
    pred_goal = r.get("pred") == "goal"
    if not truth_goal:
        return True
    if not pred_goal:
        return False
    tt = (r.get("truth_team") or "").lower()
    pt = (r.get("pred_team") or "").lower()
    return tt == pt


def _failure_digest_from_results(
    results_path: Path,
    teams: list[str],
    focus: str,
) -> str | None:
    data = json.loads(results_path.read_text())
    rows = data.get("results") or []
    lines: list[str] = []
    team_only = 0
    goal_err = 0
    for r in rows:
        goal_ok = _row_goal_ok(r)
        team_ok = _row_team_ok(r)
        if goal_ok and team_ok:
            continue
        if not goal_ok:
            goal_err += 1
        if goal_ok and not team_ok:
            team_only += 1

        if focus == "team" and not (goal_ok and not team_ok):
            continue
        if focus == "goal" and goal_ok:
            continue

        truth_goal = r.get("truth") == "goal"
        tt = r.get("truth_team")
        pt = r.get("pred_team")
        lines.append(
            f"- {r.get('clip')}: truth goal={truth_goal} team={tt!r} | "
            f"pred goal={r.get('pred') == 'goal'} team={pt!r}"
        )

    if focus == "team" and not lines:
        return None

    focus_note = {
        "all": "all failures",
        "both": "all failures (optimize goal + team)",
        "goal": "goal detection failures only",
        "team": "team assignment failures only (goal call was correct)",
    }[focus]
    header = (
        f"Dataset {data.get('dataset')} — {len(lines)} listed ({focus_note}), "
        f"{team_only} team-only / {goal_err} goal errors / {len(rows)} clips. "
        f"Teams: {teams[0]} vs {teams[1]}."
    )
    if focus == "team" and lines:
        header += (
            "\nPattern: model confuses dark suits vs dark sportswear — "
            "both wear dark kits; use clothing *style* not color."
        )
    return header + "\n" + "\n".join(lines[:30])


def _diagnosis_from_overmind() -> str | None:
    state = ROOT / ".overmind/agents/goal-classifier/experiments/skill_state.json"
    if not state.is_file():
        return None
    cmd = [
        "overmind",
        "optimize-step",
        "diagnose",
        "--state",
        str(state.relative_to(ROOT)),
        "--iteration",
        "1",
    ]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = out.stdout or ""
    start = text.find("{")
    if start < 0:
        return None
    try:
        envelope = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    warning = envelope.get("diagnose_warning") or {}
    if envelope.get("status") == "warn" and warning.get("all_failed"):
        return None
    candidates = envelope.get("candidates") or []
    if not candidates:
        return None
    plan_path = Path(candidates[0].get("plan_path") or "")
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        diag = plan.get("diagnosis") or {}
        if diag:
            return json.dumps(diag, indent=2)
    return None


def _build_instruction(focus: str, diagnosis: str, teams: list[str]) -> str:
    t1, t2 = teams[0], teams[1]
    common = f"""Edit ONLY prompt.txt. Do not modify agent.py or classify.py.

## Failure analysis
{diagnosis}

## Shared rules
- Keep {{team1}} and {{team2}} placeholders (filled at runtime from info.json).
- goal=false must imply team=null in the prompt instructions.
- Response format: JSON only, exact team name strings.
- Stay under 500 words.
"""

    if focus == "team":
        return f"""You are improving TEAM ASSIGNMENT for a football clip goal classifier.

Goal detection is already strong (~94% F1). Your job is ONLY to improve which team scored.

{common}
## Team focus — do NOT weaken goal detection
- Keep the existing **Goal Definition** section substantially unchanged (minor wording ok).
- Expand or rewrite the **Team Identification** section with concrete visual tactics.
- Invent and add tactics such as (use your judgment — not an exhaustive list):
  - Identify the **scorer** (or last attacker who shot) at the moment the ball crosses the line.
  - **Ignore defenders and the goalkeeper** for team assignment unless they clearly scored.
  - `{t1}`: formal **suits** — collared dress shirts, ties, blazers/jackets, trousers, office wear.
  - `{t2}`: **sportswear** — jerseys, shorts, tracksuits, athletic training gear.
  - Look at the player who strikes the ball and nearest attacking teammates, not crowd or sidelines.
  - If kit style is ambiguous, describe what you see before choosing.
- Address the symmetric errors in the failure list (sportswear called suits, suits called sportswear).
"""
    if focus in ("all", "both"):
        return f"""You are improving a football clip goal classifier (goal detection + team assignment).

{common}
## Improve BOTH dimensions
- **Goal errors** in the failure list: tighten goal definition (full ball crossing line; not saves/posts/wide/early end).
- **Team errors** (goal was right, team wrong): add scorer/kit tactics — identify scorer at crossing,
  ignore GK/defenders, suits = collars/ties/jackets, sportswear = jerseys/shorts/tracksuits.
- Do not trade away goal F1 for team accuracy or vice versa; aim for composite improvement.
- Both teams: {t1} vs {t2}.
"""
    if focus == "goal":
        return f"""You are improving GOAL DETECTION for a football clip classifier.

{common}
## Goal focus
- Tighten what counts as a goal (full ball crossing line; not saves/posts/wide/early clip end).
- Minimize changes to team identification unless needed for consistency.
- Both teams: {t1} vs {t2}.
"""
    return f"""You are improving a football clip goal classifier.

{common}
## General
- Both teams wear dark kits — discriminate by clothing style (sportswear vs suits).
- Saves, post hits, shots wide are NOT goals.
"""


def _apply_prompt_edit(
    diagnosis: str,
    analyzer_model: str,
    focus: str,
    teams: list[str],
    prompt_path: Path,
) -> bool:
    try:
        import overmind
        from overmind.coding_agent import apply_code_changes
    except ImportError:
        print(
            "Run with Overmind's Python:\n"
            "  ~/.local/share/uv/tools/overmind/bin/python scripts/gemini_prompt_optimize.py ...",
            file=sys.stderr,
        )
        return False

    _load_dotenv()
    overmind.init(overmind_api_key=os.environ.get("OVERMIND_API_KEY"))

    files = {
        "agent.py": (ROOT / "agent.py").read_text(),
        "classify.py": (ROOT / "classify.py").read_text(),
        "prompt.txt": PROMPT_PATH.read_text(),
        "prompt_goal.txt": PROMPT_GOAL_PATH.read_text(),
        "prompt_team.txt": PROMPT_TEAM_PATH.read_text(),
    }
    target_name = prompt_path.name
    instruction = _build_instruction(focus, diagnosis, teams).replace(
        "prompt.txt", target_name
    )
    result = apply_code_changes(
        files,
        instruction,
        analyzer_model,
        entry_file="agent.py",
        max_steps=25,
    )
    if target_name not in result.file_updates:
        print(f"Coding agent made no {target_name} changes.", file=sys.stderr)
        print((result.text or "")[:500], file=sys.stderr)
        return False
    backup = prompt_path.with_suffix(f".{focus}.bak")
    backup.write_text(prompt_path.read_text())
    prompt_path.write_text(result.file_updates[target_name])
    print(f"Updated {target_name} (backup: {backup.relative_to(ROOT)})")
    print(f"Coding agent steps: {result.steps_taken}")
    return True


def _classify_python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def _run_classify(dataset: str, mode: str = "onestage") -> Path:
    cmd = [_classify_python(), str(ROOT / "classify.py"), "--dataset", dataset]
    if mode == "twostage":
        cmd.extend(["--mode", "twostage"])
    subprocess.run(cmd, cwd=ROOT, check=True)
    suffix = "_twostage" if mode == "twostage" else ""
    return ROOT / "results" / f"{dataset}_gemini-3.5-flash{suffix}.json"


def _score_metrics(results_path: Path) -> dict:
    m = json.loads(results_path.read_text())
    return {
        "accuracy": float(m.get("accuracy") or 0),
        "f1": float(m.get("f1") or 0),
        "team_on_goals": float(m.get("team_accuracy_on_detected_goals") or 0),
        "spurious_team": int(m.get("spurious_team_on_non_goals") or 0),
    }


def _score_summary(metrics: dict) -> str:
    return (
        f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
        f"team_on_goals={metrics['team_on_goals']:.4f} spurious_team={metrics['spurious_team']}"
    )


def _is_better(new: dict, best: dict, focus: str) -> bool:
    if focus == "team":
        # Prefer higher team accuracy; require F1 not to drop more than 0.05
        if new["f1"] < best["f1"] - 0.05:
            return False
        if new["team_on_goals"] > best["team_on_goals"]:
            return True
        if new["team_on_goals"] == best["team_on_goals"] and new["f1"] > best["f1"]:
            return True
        return False
    if focus == "goal":
        if new["f1"] > best["f1"]:
            return True
        if new["f1"] == best["f1"] and new["team_on_goals"] > best["team_on_goals"]:
            return True
        return False
    # all / both: eval-spec blend (48 goal + 32 team ≈ 60/40)
    score_new = new["f1"] * 0.6 + new["team_on_goals"] * 0.4
    score_best = best["f1"] * 0.6 + best["team_on_goals"] * 0.4
    return score_new > score_best


def _build_diagnosis(
    teams: list[str],
    focus: str,
    from_overmind: bool,
    results_path: Path | None,
) -> str | None:
    parts: list[str] = []
    if from_overmind and focus != "team":
        od = _diagnosis_from_overmind()
        if od:
            parts.append("## Overmind diagnose JSON\n" + od)
        else:
            print("Overmind diagnose unavailable or failed; using results only.")
    if results_path and results_path.is_file():
        digest_focus = "all" if focus == "both" else focus
        digest = _failure_digest_from_results(results_path, teams, digest_focus)
        if digest:
            parts.append("## Misclassified clips\n" + digest)
        elif focus == "team":
            return None
    if not parts:
        return None
    return "\n\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Gemini coding-agent prompt revision (Overmind hybrid)")
    p.add_argument("--dataset", default="9-8GT-right")
    p.add_argument("--from-overmind", action="store_true")
    p.add_argument("--results", type=Path)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument(
        "--focus",
        choices=("all", "both", "goal", "team"),
        default="both",
        help="both=all failures, optimize goal+team in prompt.txt (default)",
    )
    p.add_argument(
        "--keep-best",
        action="store_true",
        help="After all iterations, restore prompt with best eval metrics for --focus",
    )
    p.add_argument(
        "--classifier-mode",
        choices=("onestage", "twostage"),
        default=os.environ.get("CLASSIFIER_MODE", "onestage"),
        help="How classify.py scores after each iteration",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        help="Prompt file to edit (default: prompt_team.txt for --focus team, else prompt.txt)",
    )
    p.add_argument(
        "--analyzer-model",
        default=os.environ.get("ANALYZER_MODEL", "gemini/gemini-2.5-pro"),
    )
    args = p.parse_args()
    _load_dotenv()

    prompt_path = args.prompt_file
    if prompt_path is None:
        prompt_path = PROMPT_TEAM_PATH if args.focus == "team" else PROMPT_PATH
    prompt_path = prompt_path if prompt_path.is_absolute() else ROOT / prompt_path

    log_dir = ROOT / ".overmind/agents/goal-classifier/experiments/gemini_prompt_loop"
    if args.focus not in ("all", "both"):
        log_dir = log_dir / f"focus_{args.focus}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 1

    teams = _load_teams(args.dataset)
    results_path = args.results or _latest_results_path()

    diagnosis = _build_diagnosis(teams, args.focus, args.from_overmind, results_path)
    if not diagnosis:
        if args.focus == "team":
            print("No team-only failures — team assignment may already be perfect.", file=sys.stderr)
        else:
            print("No results file. Run: python classify.py --dataset", args.dataset, file=sys.stderr)
        return 1

    print(diagnosis[:2500])
    if len(diagnosis) > 2500:
        print(f"... ({len(diagnosis)} chars total)")

    if not args.apply:
        print("\nDry run. Pass --apply to edit prompt.txt, --eval to re-score.")
        return 0

    if args.eval and not (results_path and results_path.is_file()):
        print("No results yet — running baseline classify…")
        results_path = _run_classify(args.dataset, args.classifier_mode)
        print(f"Baseline: {_score_summary(_score_metrics(results_path))}")

    baseline_prompt = prompt_path.read_text()
    (log_dir / f"prompt_baseline_{prompt_path.stem}.txt").write_text(baseline_prompt)

    history: list[dict] = []
    best_metrics: dict | None = None
    best_prompt: str | None = None
    if results_path and results_path.is_file():
        best_metrics = _score_metrics(results_path)
        best_prompt = baseline_prompt

    for i in range(1, args.iterations + 1):
        print(f"\n{'='*60}\nIteration {i}/{args.iterations} (focus={args.focus})\n{'='*60}")
        iter_diagnosis = _build_diagnosis(
            teams, args.focus, args.from_overmind and i == 1, results_path
        )
        if not iter_diagnosis:
            print("No failure signal for this focus; stopping.")
            break

        before_m = _score_metrics(results_path) if results_path and results_path.is_file() else {}
        before = _score_summary(before_m) if before_m else "n/a"

        if not _apply_prompt_edit(iter_diagnosis, args.analyzer_model, args.focus, teams, prompt_path):
            return 1

        iter_prompt = log_dir / f"{prompt_path.stem}_iter_{i:03d}.txt"
        iter_prompt.write_text(prompt_path.read_text())

        after = before
        after_m = before_m
        if args.eval:
            results_path = _run_classify(args.dataset, args.classifier_mode)
            after_m = _score_metrics(results_path)
            after = _score_summary(after_m)
            print(f"Before: {before}")
            print(f"After:  {after}")

            if best_metrics is not None and best_prompt is not None:
                if _is_better(after_m, best_metrics, args.focus):
                    best_metrics = after_m
                    best_prompt = prompt_path.read_text()
                    print("New best for focus metric.")
                else:
                    print("Not better than best so far.")

        history.append(
            {
                "iteration": i,
                "focus": args.focus,
                "before": before,
                "after": after,
                "prompt": str(iter_prompt.relative_to(ROOT)),
            }
        )

    if args.keep_best and best_prompt is not None and best_metrics is not None:
        current_m = _score_metrics(results_path) if results_path else {}
        if best_prompt != prompt_path.read_text() and _is_better(best_metrics, current_m, args.focus):
            prompt_path.write_text(best_prompt)
            (log_dir / f"prompt_best_{prompt_path.stem}.txt").write_text(best_prompt)
            print(
                f"\nRestored best {prompt_path.name} (focus={args.focus}): "
                f"{_score_summary(best_metrics)}"
            )
        else:
            (log_dir / f"prompt_best_{prompt_path.stem}.txt").write_text(prompt_path.read_text())
            print(f"\nCurrent {prompt_path.name} is best: {_score_summary(best_metrics)}")

    log_path = log_dir / "history.json"
    if log_path.is_file():
        try:
            prev = json.loads(log_path.read_text())
        except json.JSONDecodeError:
            prev = []
    else:
        prev = []
    prev.extend(history)
    log_path.write_text(json.dumps(prev, indent=2))
    print(f"\nSaved {len(history)} iteration(s) to {log_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
