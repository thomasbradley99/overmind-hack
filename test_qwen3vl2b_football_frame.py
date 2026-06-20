#!/usr/bin/env python3
import base64, subprocess, tempfile, time
from pathlib import Path
import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3-vl:2b"
CLIP = "data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4"
PROMPT = "Describe what you see in this image."

def extract_frame(clip_path: str, ts: float, max_size: int) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="football-"))
    frame = work_dir / f"frame_{max_size}.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return str(frame) if frame.exists() else None

sizes = [56, 112, 224]
ts = 6.0

for size in sizes:
    frame = extract_frame(CLIP, ts, size)
    if not frame:
        print(f"\n[{size}px] FAILED to extract frame")
        continue
    
    r = subprocess.run(["file", frame], capture_output=True, text=True)
    print(f"\n[{size}px] {r.stdout.strip()}")
    
    with open(frame, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 100}
    }
    
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
    t1 = time.time()
    
    print(f"  Status: {resp.status_code}, Latency: {t1-t0:.1f}s")
    if resp.status_code == 200:
        data = resp.json()
        raw = data["message"]["content"].strip()
        print(f"  Raw: '{raw[:200]}'")
    else:
        print(f"  Error: {resp.text[:200]}")
