#!/usr/bin/env python3
"""Test qwen3-vl:2b with different image sizes."""
import base64, subprocess, tempfile, os, time
from pathlib import Path
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3-vl:2b"
CLIP = "data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4"
PROMPT = Path("prompt.txt").read_text().strip().replace("{team1}", "Dark suits").replace("{team2}", "Dark sportswear")

def extract_frame(clip_path: str, ts: float, max_size: int) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="size-"))
    frame = work_dir / f"frame_{max_size}.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    if not frame.exists():
        raise RuntimeError(f"Could not extract frame")
    return str(frame)

def classify(model: str, frame: str, prompt: str, timeout: int = 120) -> dict:
    with open(frame, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 50}
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data["message"]["content"].strip()
    return {"raw": raw, "latency": t1 - t0}

sizes = [56, 112, 160, 192, 224]
ts = 6.0  # middle frame

for size in sizes:
    print(f"\n{'='*50}")
    print(f"Testing {size}px frame")
    print(f"{'='*50}")
    try:
        frame = extract_frame(CLIP, ts, size)
        result = classify(MODEL, frame, PROMPT, timeout=120 if size <= 160 else 300)
        print(f"  Latency: {result['latency']:.1f}s")
        raw = result['raw']
        if raw:
            print(f"  Raw: '{raw[:200]}'")
        else:
            print(f"  Raw: EMPTY")
    except Exception as e:
        print(f"  ERROR: {e}")
