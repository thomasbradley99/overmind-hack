#!/usr/bin/env python3
import json, base64, subprocess, tempfile, os, time
from pathlib import Path
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3-vl:8b"
CLIP = "data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4"
PROMPT = Path("prompt.txt").read_text().strip().replace("{team1}", "Dark suits").replace("{team2}", "Dark sportswear")

def extract_frames(clip_path: str, fps: float = 2.0, max_size: int = 56) -> list:
    work_dir = Path(tempfile.mkdtemp(prefix="multi-"))
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
    dr = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        duration = float(dr.stdout.strip())
    except ValueError:
        duration = 0.0
    
    frames = []
    interval = 1.0 / fps
    t = interval
    while t < duration:
        frame = work_dir / f"frame_{t:.2f}.jpg"
        cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", clip_path, "-vframes", "1",
               "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
        subprocess.run(cmd, capture_output=True, timeout=30)
        if frame.exists():
            frames.append(str(frame))
        t += interval
    return frames

frames = extract_frames(CLIP, fps=2.0, max_size=56)
print(f"Extracted {len(frames)} frames")

images_b64 = []
for f in frames:
    with open(f, "rb") as fh:
        images_b64.append(base64.b64encode(fh.read()).decode("utf-8"))

print(f"Total base64 size: {sum(len(x) for x in images_b64) / 1024:.1f} KB")

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT, "images": images_b64}],
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 50}
}

print(f"Sending to {MODEL}... timeout=300s")
t0 = time.time()
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
t1 = time.time()
print(f"Status: {resp.status_code}")
print(f"Latency: {t1 - t0:.1f}s")
if resp.status_code == 200:
    data = resp.json()
    raw = data["message"]["content"].strip()
    print(f"Response: {raw[:300]}")
else:
    print(f"Error: {resp.text[:500]}")
