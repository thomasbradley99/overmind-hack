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

# Test 1: Simple description
payload1 = {
    "model": MODEL,
    "prompt": "Describe this image in one sentence.",
    "images": [img_b64],
    "stream": False,
}
print("Test 1: Describe image")
t0 = time.time()
try:
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload1, timeout=60)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip()
    print(f"Latency: {t1-t0:.1f}s, raw: '{raw[:100]}'")
except Exception as e:
    print(f"Error: {e}")

# Test 2: With keep_alive
payload2 = {
    "model": MODEL,
    "prompt": "What sport is being played?",
    "images": [img_b64],
    "stream": False,
    "keep_alive": "30m",
}
print("\nTest 2: What sport")
t0 = time.time()
try:
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload2, timeout=60)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip()
    print(f"Latency: {t1-t0:.1f}s, raw: '{raw[:100]}'")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check if model is actually running
print("\nTest 3: Listing running models")
resp = requests.get(f"{OLLAMA_HOST}/api/tags")
models = resp.json()
for m in models.get("models", []):
    if "moon" in m.get("name", "").lower():
        print(f"  Found: {m.get('name')} size={m.get('size')}")
