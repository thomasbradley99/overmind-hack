#!/usr/bin/env python3
import base64, subprocess, tempfile, requests, time
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL = "moondream:latest"

def classify(clip_path, prompt):
    work_dir = Path(tempfile.mkdtemp())
    cmd = ["ffmpeg", "-y", "-ss", "5", "-i", clip_path, "-vframes", "1",
           "-vf", "scale=112:-1", "-q:v", "5", str(work_dir / "frame.jpg")]
    subprocess.run(cmd, capture_output=True, timeout=30)
    with open(work_dir / "frame.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip()
    return raw, t1-t0

# Test on goal and non-goal clips
clips = [
    ("data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4", "goal"),
    ("data/9-8GT-right-quarter/goal_05_606s_Dark-suits.mp4", "goal"),
    ("data/9-8GT-right-quarter/nongoal_04_189s.mp4", "not_goal"),
    ("data/9-8GT-right-quarter/nongoal_08_1198s.mp4", "not_goal"),
]

prompts = [
    "Is this a football goal? The ball must be fully in the net. Answer yes or no only.",
    "Was a goal scored in this image? Answer yes or no.",
    "Look at this football image. Is the ball fully inside the goal net? Say yes or no.",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt[:50]}...")
    print(f"{'='*60}")
    for clip, truth in clips:
        raw, lat = classify(clip, prompt)
        pred = "goal" if "yes" in raw.lower() else "not_goal"
        ok = pred == truth
        print(f"  {Path(clip).name:<40} truth={truth:<8} pred={pred:<8} [{ok}] lat={lat:.1f}s raw={raw[:50]}")
