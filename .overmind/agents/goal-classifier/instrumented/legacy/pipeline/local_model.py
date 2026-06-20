import os
import base64
from pathlib import Path
from typing import Optional
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "richardyoung/smolvlm2-2.2b-instruct:latest")
MAX_TOKENS = int(os.environ.get("LOCAL_MODEL_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.environ.get("LOCAL_MODEL_TEMPERATURE", "0.0"))


def _ollama_chat(payload: dict) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload.setdefault("stream", False)
    payload.setdefault("options", {})
    payload["options"].setdefault("temperature", TEMPERATURE)
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            f"Is ollama running? Error: {e}"
        )


def generate_text(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Generate text using the Ollama chat API."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    options = {}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if options:
        payload["options"] = options

    return _ollama_chat(payload).strip()


def generate_with_images(
    prompt: str,
    image_paths: list[str],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Generate text with images using the Ollama vision API."""
    images_b64 = []
    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        with open(img_path, "rb") as f:
            images_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
    }
    options = {}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if options:
        payload["options"] = options

    return _ollama_chat(payload).strip()
