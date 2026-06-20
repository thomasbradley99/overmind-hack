#!/usr/bin/env python3
"""Run qwen3-vl:8b on all 34 clips, one at a time."""
import json, base64, subprocess, tempfile, os
from pathlib import Path
import requests
import time

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3-vl:8b"

def extract_1_frame(clip_path: str, max_size: int = 224) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="qwen-"))
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
    dr = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        duration = float(dr.stdout.strip())
    except ValueError:
        duration = 0.0
    ts = duration / 2
    frame = work_dir / "frame.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=10)
    if not frame.exists():
        raise RuntimeError(f"Could not extract frame from {clip_path}")
    return str(frame)

def run_qwen3vl(clip_path: str, prompt: str) -> dict:
    frame = extract_1_frame(clip_path)
    with open(frame, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data["message"]["content"].strip()
    return {"raw": raw, "latency": t1 - t0}

def parse_json(raw: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1].lstrip("json").strip()
    import re
    for m in re.finditer(r'\{[^{}]*\}', txt):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "goal" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {}

if __name__ == "__main__":
    dataset = "9-8GT-right"
    data_dir = Path("data") / dataset
    results_dir = Path("results") / dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    prompt = Path("prompt.txt").read_text().strip().replace("{team1}", "Dark suits").replace("{team2}", "Dark sportswear")
    clips = sorted(data_dir.glob("*.mp4"))
    
    for clip in clips:
        base = clip.name.rsplit(".", 1)[0]
        path = results_dir / f"{base}.json"
        
        # Skip if already has qwen3-vl results
        if path.exists():
            data = json.loads(path.read_text())
            if MODEL in data.get("models", {}):
                print(f"SKIP {clip.name} (already has qwen3-vl results)")
                continue
        
        label_file = clip.with_suffix(".json")
        gt = json.loads(label_file.read_text()) if label_file.exists() else {}
        truth = "goal" if gt.get("label") == "goal" or gt.get("action") == "Goal" else "not_goal"
        truth_team = gt.get("team")
        
        print(f"\nProcessing {clip.name}...")
        try:
            result = run_qwen3vl(str(clip), prompt)
            parsed = parse_json(result["raw"])
            pred = "goal" if parsed.get("goal") else "not_goal"
            pred_team = parsed.get("team")
            print(f"  Raw: {result['raw'][:150]}...")
            print(f"  Pred: {pred}, team: {pred_team}, latency: {result['latency']:.1f}s")
            
            if path.exists():
                data = json.loads(path.read_text())
            else:
                data = {
                    "clip": clip.name,
                    "truth": truth,
                    "truth_team": truth_team,
                    "models": {},
                }
            
            goal_ok = pred == truth
            team_ok = (truth != "goal" or pred != "goal" or truth_team == pred_team)
            
            data["models"][MODEL] = {
                "pred": pred,
                "pred_team": pred_team,
                "raw": result["raw"],
                "goal_correct": goal_ok,
                "team_correct": team_ok,
                "latency": round(result["latency"], 1),
            }
            path.write_text(json.dumps(data, indent=2))
            print(f"  Saved")
        except Exception as e:
            print(f"  ERROR: {e}")
            
    print(f"\nDone. Run 'python3 compare_models.py' to see full comparison.")
