"""
Overmind entrypoint for the football goal classifier.

Overmind calls:
    run(input_data) -> output dict

One test case = one clip. Input is a clip path (+ dataset name).
Output matches our eval fields: goal (bool) and team (string or null).

Register with Overmind:
    overmind agent register goal-classifier agent:run
"""

from __future__ import annotations

import os
from pathlib import Path

from classify import (
    build_prompt,
    classify_one,
    load_teams,
    normalize_team,
)

ROOT = Path(__file__).parent
DEFAULT_PROMPT = ROOT / "prompt.txt"


def _load_prompt() -> str:
    template = DEFAULT_PROMPT.read_text().strip()
    teams = load_teams("9-8GT-right")
    return build_prompt(template, teams)


def run(input_data: dict) -> dict:
    """
    Classify a single football clip.

    input_data:
        clip_path: path to .mp4 (relative to repo root or absolute)
        dataset:   optional, default 9-8GT-right (for team names)

    returns:
        goal: bool
        team: str | None  (only when goal is true)
        raw:  str         (verbatim model response, for traces)
    """
    clip_path = input_data.get("clip_path") or input_data.get("video_path")
    if not clip_path:
        return {"goal": False, "team": None, "raw": "missing clip_path", "error": "clip_path required"}

    clip = Path(clip_path)
    if not clip.is_absolute():
        clip = ROOT / clip
    if not clip.exists():
        return {"goal": False, "team": None, "raw": f"file not found: {clip}", "error": "clip not found"}

    dataset = input_data.get("dataset", "9-8GT-right")
    teams = load_teams(dataset)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"goal": False, "team": None, "raw": "GEMINI_API_KEY not set", "error": "no api key"}

    model = os.environ.get("CLASSIFIER_MODEL", "gemini-3.5-flash")
    prompt_template = DEFAULT_PROMPT.read_text().strip()
    prompt = build_prompt(prompt_template, teams)

    result = classify_one(clip, prompt, teams, model, api_key)
    goal = result.get("pred") == "goal"
    team = normalize_team(result.get("pred_team"), teams) if goal else None

    return {
        "goal": goal,
        "team": team,
        "raw": result.get("raw", ""),
    }
