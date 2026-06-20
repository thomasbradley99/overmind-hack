#!/usr/bin/env python3
import subprocess, tempfile
from pathlib import Path

clip = "data/9-8GT-right-quarter/goal_01_283s_Dark-sportswear.mp4"
work_dir = Path(tempfile.mkdtemp())
cmd = ["ffmpeg", "-y", "-ss", "5", "-i", clip, "-vframes", "1",
       "-vf", "scale=112:-1", "-q:v", "5", str(work_dir / "frame.jpg")]
result = subprocess.run(cmd, capture_output=True, timeout=30)
print(f"ffmpeg exit code: {result.returncode}")
print(f"stderr: {result.stderr.decode()[:200]}")
frame = work_dir / "frame.jpg"
if frame.exists():
    import os
    size = os.path.getsize(frame)
    print(f"Frame exists: {frame}, size: {size} bytes")
else:
    print("Frame not found!")
