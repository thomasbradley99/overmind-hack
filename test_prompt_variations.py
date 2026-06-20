#!/usr/bin/env python3
"""Test different prompt variations with smolvlm2 on a few clips."""
import json, base64, subprocess, tempfile, os, time
from pathlib import Path
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "richardyoung/smolvlm2-2.2b-instruct:latest"
DATASET = "9-8GT-right-quarter"
DATA_DIR = Path("data") / DATASET

def extract_frame(clip_path: str, max_size: int = 224) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="prompt-"))
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
    dr = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        duration = float(dr.stdout.strip())
    except ValueError:
        duration = 0.0
    ts = duration / 2
    frame = work_dir / "frame.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return str(frame) if frame.exists() else None

def classify(model: str, clip_path: str, prompt: str, timeout: int = 60) -> dict:
    frame = extract_frame(clip_path)
    with open(frame, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 100}
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data["message"]["content"].strip()
    return {"raw": raw, "latency": t1 - t0}

def parse_json(raw: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1].lstrip("json").strip()
    import re
    for m in re.finditer(r'\{[^{}]*\}', txt):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "goal" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {}

# Test clips: 2 goals, 2 non-goals
clips = [
    ("data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4", "goal", "Dark sportswear"),
    ("data/9-8GT-right-quarter/goal_05_606s_Dark-suits.mp4", "goal", "Dark suits"),
    ("data/9-8GT-right-quarter/nongoal_04_189s.mp4", "not_goal", None),
    ("data/9-8GT-right-quarter/nongoal_08_1198s.mp4", "not_goal", None),
]

prompts = {
    "original": Path("prompt.txt").read_text().strip().replace("{team1}", "Dark suits").replace("{team2}", "Dark sportswear"),
    
    "confidence": """Look at this football image carefully.

The two teams are:
- Dark suits: players in dark suits (jackets, dress shirts, office-style)
- Dark sportswear: players in dark athletic sportswear / tracksuits

Was a goal scored in this frame? A goal means the ball is fully in the net.
If you are NOT certain, say goal=false.
Rate your confidence: 1-10.

YOU MUST OUTPUT ONLY THIS JSON:
{"goal": true/false, "team": "Dark suits" or "Dark sportswear" or null, "confidence": 1-10}""",

    "describe_first": """Look at this football image carefully.

The two teams are:
- Dark suits: players in dark suits (jackets, dress shirts, office-style)
- Dark sportswear: players in dark athletic sportswear / tracksuits

Step 1: Describe what you see in 1 sentence.
Step 2: Was a goal scored? (ball fully in net = goal)

YOU MUST OUTPUT ONLY THIS JSON:
{"description": "...", "goal": true/false, "team": "Dark suits" or "Dark sportswear" or null}""",

    "strict": """You are a football referee. Look at this image.

Teams: Dark suits (formal) vs Dark sportswear (athletic)

RULES:
- ONLY say goal=true if you CLEARLY see the ball fully inside the goal net
- If the ball is near the goal but not clearly in, say goal=false
- If you cannot see the ball clearly, say goal=false

OUTPUT ONLY JSON:
{"goal": true/false, "team": "Dark suits" or "Dark sportswear" or null}""",

    "binary": """Is this a football goal? Yes or No.

If Yes, which team scored? Dark suits or Dark sportswear?

Output: {"goal": true/false, "team": "..." or null}""",
}

print(f"Testing {len(prompts)} prompt variations on {len(clips)} clips")
print(f"Model: {MODEL}")
print(f"{'='*70}")

for prompt_name, prompt in prompts.items():
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt_name}")
    print(f"{'='*70}")
    correct = 0
    for clip_path, truth, truth_team in clips:
        result = classify(MODEL, clip_path, prompt)
        parsed = parse_json(result["raw"])
        pred = "goal" if parsed.get("goal") else "not_goal"
        pred_team = parsed.get("team")
        ok = pred == truth
        if truth == "goal" and pred == "goal":
            team_ok = truth_team == pred_team
        else:
            team_ok = True
        correct += (1 if ok and team_ok else 0)
        print(f"  {Path(clip_path).name:<40} truth={truth:<8} pred={pred:<8} team={pred_team or 'None':<20} [{ok and team_ok}] lat={result['latency']:.1f}s")
        print(f"    raw: {result['raw'][:100]}")
    print(f"  Score: {correct}/{len(clips)} correct")
