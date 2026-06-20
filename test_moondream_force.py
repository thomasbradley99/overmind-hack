#!/usr/bin/env python3
import base64, subprocess, tempfile, requests
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL = "moondream:latest"

def classify(clip_path, prompt, max_size=224):
    work_dir = Path(tempfile.mkdtemp())
    cmd = ["ffmpeg", "-y", "-ss", "6", "-i", clip_path, "-vframes", "1",
           "-vf", f"scale={max_size}:-1", "-q:v", "5", str(work_dir / "frame.jpg")]
    subprocess.run(cmd, capture_output=True, timeout=30)
    with open(work_dir / "frame.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()

clips = [
    ("data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4", "goal"),
    ("data/9-8GT-right-quarter/goal_05_606s_Dark-suits.mp4", "goal"),
    ("data/9-8GT-right-quarter/nongoal_04_189s.mp4", "not_goal"),
    ("data/9-8GT-right-quarter/nongoal_08_1198s.mp4", "not_goal"),
]

prompts = [
    "You MUST answer this question about the image. Is this a football goal? Say yes or no.",
    "Analyze this image and tell me: is the ball inside the goal net? Answer yes or no. Do not leave blank.",
    "What is happening in this football image? Is it a goal? Describe and answer.",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt[:50]}...")
    for clip, truth in clips:
        raw = classify(clip, prompt)
        pred = "goal" if "yes" in raw.lower() or "goal" in raw.lower() else "not_goal"
        ok = pred == truth
        print(f"  {Path(clip).name:<40} truth={truth:<8} pred={pred:<8} [{ok}] raw={raw[:60]}")
