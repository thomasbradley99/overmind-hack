#!/usr/bin/env python3
import base64, json, subprocess, tempfile, requests, time
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL = "moondream:latest"

clip = "data/9-8GT-right-quarter/goal_05_606s_Dark-suits.mp4"

def extract_frames(clip_path, num_frames=3, max_size=336):
    work_dir = Path(tempfile.mkdtemp())
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
    dr = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(dr.stdout.strip())
    except ValueError:
        duration = 0.0
    frames = []
    for i in range(num_frames):
        ts = duration * i / max(num_frames - 1, 1)
        frame = work_dir / f"frame_{i:03d}.jpg"
        cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
               "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "2", str(frame)]
        subprocess.run(cmd, capture_output=True, timeout=10)
        if frame.exists():
            frames.append(str(frame))
    return frames

prompt = """Analyze this football clip. Look at the frame carefully.

The two teams are:
- Dark sportswear: players in dark athletic sportswear / tracksuits
- Dark suits: players in dark suits (jackets, dress shirts, office-style)

Your task: Decide if a goal was scored in this clip, and if so, which team scored it.

A goal means the ball FULLY crosses the goal line into the net. You must see the ball in the net.
A save, block, post hit, shot wide, or the ball near but not in the goal is NOT a goal.

ONLY say goal=true if you are CERTAIN a goal was scored. If you are unsure or the ball is not clearly in the net, say goal=false.

YOU MUST OUTPUT ONLY THIS JSON - nothing else, no explanation, no markdown:

{"goal": true, "team": "Dark sportswear"}   if goal by sportswear team
{"goal": true, "team": "Dark suits"}   if goal by suits team
{"goal": false, "team": null}       if no goal or uncertain"""

frames = extract_frames(clip, num_frames=3, max_size=336)
images_b64 = []
for f in frames:
    with open(f, "rb") as img:
        images_b64.append(base64.b64encode(img.read()).decode("utf-8"))

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": prompt, "images": images_b64}
    ],
    "stream": False,
    "options": {"temperature": 0.0},
}

resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
resp.raise_for_status()
data = resp.json()
raw = data.get("message", {}).get("content", "").strip()
print(f"FULL RAW ({len(raw)} chars):")
print(raw)
print("\n---\n")

# Try to parse JSON
import re
for obj_match in re.finditer(r'\{[^{}]*\}', raw):
    try:
        obj = json.loads(obj_match.group(0))
        if isinstance(obj, dict) and "goal" in obj:
            print(f"Found JSON with 'goal': {json.dumps(obj, indent=2)}")
    except:
        pass
