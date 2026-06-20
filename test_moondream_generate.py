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

payload = {
    "model": MODEL,
    "prompt": prompt,
    "images": [img_b64],
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 200}
}

print(f"Testing {MODEL} with /api/generate...")
t0 = time.time()
try:
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip()
    print(f"Latency: {t1-t0:.1f}s")
    print(f"Raw: '{raw}'")
    print(f"Success!")
except Exception as e:
    print(f"Error: {e}")
