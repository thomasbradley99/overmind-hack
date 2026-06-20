#!/usr/bin/env python3
import requests, json

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3-vl:2b"

# Test 1: Simple text prompt
print("Test 1: Simple text prompt")
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": False,
}
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Response: '{data['message']['content'][:100]}'")
else:
    print(f"Error: {resp.text[:200]}")

# Test 2: With a tiny white image
print("\nTest 2: Text + tiny white image")
import base64
from PIL import Image
import io, tempfile

img = Image.new('RGB', (56, 32), color='white')
buf = io.BytesIO()
img.save(buf, format='JPEG')
img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What do you see?", "images": [img_b64]}],
    "stream": False,
}
resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Response: '{data['message']['content'][:100]}'")
else:
    print(f"Error: {resp.text[:200]}")
