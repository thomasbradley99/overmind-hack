#!/usr/bin/env python3
import base64, subprocess, tempfile, requests, time
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL = "moondream:latest"

def classify(clip_path, prompt, max_size=224):
    work_dir = Path(tempfile.mkdtemp())
    cmd = ["ffmpeg", "-y", "-ss", "5", "-i", clip_path, "-vframes", "1",
           "-vf", f"scale={max_size}:-1", "-q:v", "5", str(work_dir / "frame.jpg")]
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

clips = [
    ("data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4", "goal"),
    ("data/9-8GT-right-quarter/goal_05_606s_Dark-suits.mp4", "goal"),
    ("data/9-8GT-right-quarter/nongoal_04_189s.mp4", "not_goal"),
    ("data/9-8GT-right-quarter/nongoal_08_1198s.mp4", "not_goal"),
]

prompt = "Look at this football image carefully. Was a goal scored? The ball must be fully inside the goal net. Answer ONLY yes or no."

print(f"{'='*60}")
print(f"224px test")
print(f"{'='*60}")
for clip, truth in clips:
    raw, lat = classify(clip, prompt, 224)
    pred = "goal" if "yes" in raw.lower() else "not_goal"
    ok = pred == truth
    print(f"  {Path(clip).name:<40} truth={truth:<8} pred={pred:<8} [{ok}] lat={lat:.1f}s raw='{raw[:50]}'")

print(f"\n{'='*60}")
print(f"112px test (mid-frame)")
print(f"{'='*60}")
for clip, truth in clips:
    work_dir = Path(tempfile.mkdtemp())
    cmd = ["ffmpeg", "-y", "-ss", "6", "-i", clip, "-vframes", "1",
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
    raw = resp.json().get("response", "").strip()
    pred = "goal" if "yes" in raw.lower() else "not_goal"
    ok = pred == truth
    print(f"  {Path(clip).name:<40} truth={truth:<8} pred={pred:<8} [{ok}] lat={t1-t0:.1f}s raw='{raw[:50]}'")
