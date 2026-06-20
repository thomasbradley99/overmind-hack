#!/usr/bin/env python3
import base64, subprocess, tempfile, time, os
from pathlib import Path
import requests
from PIL import Image
import io

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3-vl:2b"
CLIP = "data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4"

def extract_frame(clip_path: str, ts: float, max_size: int) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="diag-"))
    frame = work_dir / f"frame_{max_size}.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return str(frame) if frame.exists() else None

# Test 1: Football frame with simple prompt
print("Test 1: Football frame 224px + 'Describe this image'")
frame = extract_frame(CLIP, 6.0, 224)
with open(frame, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Describe this image", "images": [img_b64]}],
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 100}
}
t0 = time.time()
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
t1 = time.time()
print(f"  Latency: {t1-t0:.1f}s, Raw: '{resp.json()['message']['content'][:100] if resp.status_code==200 else 'ERROR'}'")

# Test 2: Football frame with num_predict=500
print("\nTest 2: Same frame + num_predict=500")
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Describe this image", "images": [img_b64]}],
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 500}
}
t0 = time.time()
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
t1 = time.time()
print(f"  Latency: {t1-t0:.1f}s, Raw: '{resp.json()['message']['content'][:100] if resp.status_code==200 else 'ERROR'}'")

# Test 3: Convert to PNG
print("\nTest 3: Same frame converted to PNG")
img = Image.open(frame)
png_path = Path(frame).with_suffix('.png')
img.save(png_path, 'PNG')
with open(png_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Describe this image", "images": [img_b64]}],
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 100}
}
t0 = time.time()
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
t1 = time.time()
print(f"  Latency: {t1-t0:.1f}s, Raw: '{resp.json()['message']['content'][:100] if resp.status_code==200 else 'ERROR'}'")

# Test 4: Use a different video frame (nongoal)
print("\nTest 4: Different clip (nongoal_04) 224px")
frame2 = extract_frame("data/9-8GT-right-quarter/nongoal_04_189s.mp4", 6.0, 224)
with open(frame2, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Describe this image", "images": [img_b64]}],
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 100}
}
t0 = time.time()
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
t1 = time.time()
print(f"  Latency: {t1-t0:.1f}s, Raw: '{resp.json()['message']['content'][:100] if resp.status_code==200 else 'ERROR'}'")

# Test 5: Check what happens with the actual prompt but no JSON constraint
print("\nTest 5: Football frame + goal prompt (no JSON constraint)")
with open(frame, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Is this a football goal?", "images": [img_b64]}],
    "stream": False,
    "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 100}
}
t0 = time.time()
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
t1 = time.time()
print(f"  Latency: {t1-t0:.1f}s, Raw: '{resp.json()['message']['content'][:100] if resp.status_code==200 else 'ERROR'}'")
