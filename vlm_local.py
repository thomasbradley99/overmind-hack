"""Local VLM backends via Ollama (Qwen3-VL single frame, LLaVA frame grid)."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def _clip_duration(clip: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 12.0


def _ffmpeg_tile(clip: Path, n_frames: int = 4, scale: int = 768) -> bytes:
    """
    Sample frames across the clip and stitch into one grid image.
    Ollama LLaVA accepts one image per request (not a list).
    """
    duration = max(_clip_duration(clip), 1.0)
    # ~n_frames samples spread over the clip
    fps = max(n_frames / duration, 0.25)
    cols = 2
    rows = max(2, (n_frames + 1) // 2)
    if n_frames <= 4:
        rows, cols = 2, 2
    vf = f"fps={fps:.4f},scale={scale}:-1,tile={cols}x{rows}"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg tile failed for {clip}: {(proc.stderr or b'').decode()[:200]}"
        )
    return proc.stdout


def _ffmpeg_single_frame(clip: Path, max_size: int = 384) -> bytes:
    """One frame from the middle of the clip (used by Qwen3-VL)."""
    duration = max(_clip_duration(clip), 1.0)
    ts = duration / 2
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(ts),
            "-i",
            str(clip),
            "-vframes",
            "1",
            "-vf",
            f"scale='min({max_size},iw)':-1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg frame failed for {clip}: {(proc.stderr or b'').decode()[:200]}"
        )
    return proc.stdout


def _ollama_chat(
    host: str,
    model: str,
    prompt: str,
    images: list[str],
    temperature: float,
    timeout: int,
) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature},
    }
    url = host.rstrip("/") + "/api/chat"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed ({url}): {e}") from e
    msg = data.get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise RuntimeError("empty response from Ollama")
    return text


def qwen_generate(
    clip: Path,
    prompt: str,
    model: str,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
    timeout: int = 600,
) -> str:
    max_size = int(os.environ.get("LOCAL_VLM_SCALE", "384"))
    frame = _ffmpeg_single_frame(clip, max_size=max_size)
    images = [base64.b64encode(frame).decode("ascii")]
    return _ollama_chat(host, model, prompt, images, temperature, timeout)


def ollama_generate(
    clip: Path,
    prompt: str,
    model: str,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
    n_frames: int = 4,
    timeout: int = 600,
) -> str:
    scale = int(os.environ.get("LOCAL_VLM_SCALE", "768"))
    grid = _ffmpeg_tile(clip, n_frames=n_frames, scale=scale)
    grid_prompt = (
        "The image is a 2x2 grid of frames from one short football clip "
        "(reading order: top-left → top-right → bottom-left → bottom-right; "
        "time advances across the grid). "
        "Some local models answer with one JSON object for the whole clip; "
        "others return a JSON array with one object per grid cell — both are fine. "
        + prompt
    )
    images = [base64.b64encode(grid).decode("ascii")]
    return _ollama_chat(host, model, grid_prompt, images, temperature, timeout)


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")
