#!/usr/bin/env python3
import base64, subprocess, tempfile, requests, time
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL = "moondream:latest"
clip = "data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4"

work_dir = Path(tempfile.mkdtemp())
cmd = ["ffmpeg", "-y", "-ss", "5", "-i", clip, "-vframes", "1",
       "-vf", "scale=112:-1", "-q:v", "5", str(work_dir / "frame.jpg")]
subprocess.run(cmd, capture_output=True, timeout=30)

with open(work_dir / "frame.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

prompt = "Is this a football goal? Answer yes or no only."

for np in [50, 200, 500, 1000]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": np}
    }
    print(f"\n--- num_predict={np} ---")
    t0 = time.time()
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
        t1 = time.time()
        resp.raise_for_status()
        data = resp.json()
        raw = data["message"]["content"].strip()
        print(f"Latency: {t1-t0:.1f}s, raw: '{raw}'")
    except Exception as e:
        print(f"Error: {e}")
