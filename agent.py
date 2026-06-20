"""
Overmind entrypoint — one VLM call per clip via prompt.txt.

Env:
  CLASSIFIER_BACKEND=gemini | qwen | ollama
  CLASSIFIER_MODEL=gemini-3.5-flash | qwen3-vl:8b | llava-llama3
"""

from __future__ import annotations

import os
from pathlib import Path

from classify import build_prompt, classify_one, load_teams, normalize_team

ROOT = Path(__file__).parent
PROMPT_PATH = ROOT / "prompt.txt"


def run(input_data: dict) -> dict:
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
    backend = os.environ.get("CLASSIFIER_BACKEND", "gemini").lower()
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if backend == "gemini" and not api_key:
        return {"goal": False, "team": None, "raw": "GEMINI_API_KEY not set", "error": "no api key"}

    if backend in ("qwen", "qwen3"):
        default_model = "qwen3-vl:8b"
    elif backend in ("ollama", "local"):
        default_model = "llava-llama3"
    else:
        default_model = "gemini-3.5-flash"
    model = os.environ.get("CLASSIFIER_MODEL", default_model)
    prompt = build_prompt(PROMPT_PATH.read_text().strip(), teams)
    result = classify_one(clip, prompt, teams, model, api_key)

    goal = result.get("pred") == "goal"
    team = normalize_team(result.get("pred_team"), teams) if goal else None

    return {
        "goal": goal,
        "team": team,
        "raw": result.get("raw", ""),
    }
