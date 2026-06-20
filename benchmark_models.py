#!/usr/bin/env python3
"""Benchmark inference time for both models on the same frames."""
import json, base64, subprocess, tempfile, os, time
from pathlib import Path
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CLIP = "data/9-8GT-right/goal_01_283s_Dark-sportswear.mp4"
PROMPT = Path("prompt.txt").read_text().strip().replace("{team1}", "Dark suits").replace("{team2}", "Dark sportswear")

def extract_frame(clip_path: str, ts: float, max_size: int = 224) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="bench-"))
    frame = work_dir / f"frame_{ts:.1f}.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=10)
    return str(frame)

def benchmark_model(model_name: str, frame_path: str, n_runs: int = 3) -> dict:
    with open(frame_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    # Warmup (not counted)
    print(f"  {model_name}: Warmup...")
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
    if resp.status_code != 200:
        print(f"  Warmup failed: {resp.status_code}")
        return None
    
    times = []
    for i in range(n_runs):
        t0 = time.time()
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
        t1 = time.time()
        if resp.status_code == 200:
            times.append(t1 - t0)
            print(f"  Run {i+1}: {times[-1]:.1f}s")
        else:
            print(f"  Run {i+1}: FAILED ({resp.status_code})")
    
    if not times:
        return None
    
    return {
        "model": model_name,
        "n_runs": len(times),
        "times": times,
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }

# Extract 3 frames from the clip for testing
print("Extracting frames from test clip...")
frames = []
for ts in [2.0, 5.0, 8.0]:
    f = extract_frame(CLIP, ts)
    frames.append(f)
    print(f"  Frame at {ts}s: {f}")

# Use the middle frame for benchmarking
frame = frames[1]

print(f"\nBenchmarking on single frame (224px, middle of clip)...")
print(f"Frame size: {Path(frame).stat().st_size} bytes")

models = ["qwen3-vl:8b", "richardyoung/smolvlm2-2.2b-instruct:latest"]
results = {}

for model in models:
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"{'='*50}")
    result = benchmark_model(model, frame, n_runs=3)
    if result:
        results[model] = result
        print(f"\n  Mean: {result['mean']:.1f}s")
        print(f"  Min:  {result['min']:.1f}s")
        print(f"  Max:  {result['max']:.1f}s")

# Extrapolate to full dataset
print(f"\n{'='*50}")
print(f"EXTRAPOLATION TO FULL DATASET (34 clips)")
print(f"{'='*50}")

data_dir = Path("data/9-8GT-right")
clips = sorted(data_dir.glob("*.mp4"))
print(f"Total clips: {len(clips)}")

for model, result in results.items():
    mean_time = result['mean']
    total_time = mean_time * len(clips)
    print(f"\n{model}:")
    print(f"  Per-clip inference: {mean_time:.1f}s")
    print(f"  Total for 34 clips: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Per-second of video: {mean_time/10:.2f}s (approx, 10s clips)")

