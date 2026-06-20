#!/usr/bin/env python3
"""Test qwen3-vl:2b with 1, 2, and 3 frames at 56px."""
import json, base64, subprocess, tempfile, os, time
from pathlib import Path
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3-vl:2b"
CLIP = "data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4"
PROMPT = Path("prompt.txt").read_text().strip().replace("{team1}", "Dark suits").replace("{team2}", "Dark sportswear")

def extract_frame_at(clip_path: str, ts: float, max_size: int = 56) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="few-"))
    frame = work_dir / f"frame_{ts:.2f}.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    if not frame.exists():
        raise RuntimeError(f"Could not extract frame at {ts}s")
    return str(frame)

def get_duration(clip_path: str) -> float:
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
    dr = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        return float(dr.stdout.strip())
    except ValueError:
        return 0.0

def classify(model: str, frames: list, prompt: str, timeout: int = 120) -> dict:
    images_b64 = []
    for f in frames:
        with open(f, "rb") as fh:
            images_b64.append(base64.b64encode(fh.read()).decode("utf-8"))
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 50}
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data["message"]["content"].strip()
    return {"raw": raw, "latency": t1 - t0, "n_frames": len(frames)}

duration = get_duration(CLIP)
print(f"Clip duration: {duration:.1f}s")
print(f"Model: {MODEL}")
print(f"{'='*60}")

# Test with 1 frame (middle)
frames_1 = [extract_frame_at(CLIP, duration / 2, 56)]
print(f"\n[1 FRAME] Middle frame at {duration/2:.1f}s")
print(f"  Frame: {frames_1[0]}")
result = classify(MODEL, frames_1, PROMPT)
print(f"  Latency: {result['latency']:.1f}s")
print(f"  Raw: '{result['raw'][:200]}'")

# Test with 2 frames (1/3 and 2/3)
frames_2 = [extract_frame_at(CLIP, duration / 3, 56), extract_frame_at(CLIP, 2 * duration / 3, 56)]
print(f"\n[2 FRAMES] At {duration/3:.1f}s and {2*duration/3:.1f}s")
result = classify(MODEL, frames_2, PROMPT)
print(f"  Latency: {result['latency']:.1f}s")
print(f"  Raw: '{result['raw'][:200]}'")

# Test with 3 frames (1/4, 1/2, 3/4)
frames_3 = [extract_frame_at(CLIP, duration / 4, 56), 
            extract_frame_at(CLIP, duration / 2, 56),
            extract_frame_at(CLIP, 3 * duration / 4, 56)]
print(f"\n[3 FRAMES] At {duration/4:.1f}s, {duration/2:.1f}s, {3*duration/4:.1f}s")
result = classify(MODEL, frames_3, PROMPT)
print(f"  Latency: {result['latency']:.1f}s")
print(f"  Raw: '{result['raw'][:200]}'")

print(f"\n{'='*60}")
print("Done.")
