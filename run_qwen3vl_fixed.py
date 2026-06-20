#!/usr/bin/env python3
"""Run qwen3-vl:2b and qwen3-vl:8b with proper num_predict (500) on quarter dataset."""
import json, base64, subprocess, tempfile, os, time
from pathlib import Path
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DATASET = "9-8GT-right-quarter"
DATA_DIR = Path("data") / DATASET
RESULTS_DIR = Path("results") / DATASET
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = Path("prompt.txt").read_text().strip().replace("{team1}", "Dark suits").replace("{team2}", "Dark sportswear")

MODELS = [
    ("qwen3-vl:2b", 120, 512, 500, "qwen3vl2b_56px_fixed"),
    ("qwen3-vl:8b", 300, 512, 500, "qwen3vl8b_56px_fixed"),
]

def extract_frame(clip_path: str, ts: float, max_size: int = 56) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="fixed-"))
    frame = work_dir / "frame.jpg"
    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", clip_path, "-vframes", "1",
           "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    if not frame.exists():
        raise RuntimeError(f"Could not extract frame")
    return str(frame)

def get_duration(clip_path: str) -> float:
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
    dr = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        return float(dr.stdout.strip())
    except ValueError:
        return 0.0

def classify(model: str, clip_path: str, prompt: str, timeout: int, num_ctx: int, num_predict: int) -> dict:
    duration = get_duration(clip_path)
    ts = duration / 2
    frame = extract_frame(clip_path, ts, max_size=56)
    with open(frame, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": num_ctx, "num_predict": num_predict}
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
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

def load_gt(clip: Path) -> tuple:
    label_file = clip.with_suffix(".json")
    gt = json.loads(label_file.read_text()) if label_file.exists() else {}
    truth = "goal" if gt.get("label") == "goal" or gt.get("action") == "Goal" else "not_goal"
    return truth, gt.get("team")

def print_cm(results, title):
    tp = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
    fn = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "not_goal")
    fp = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "goal")
    tn = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "not_goal")
    errors = sum(1 for r in results if r["pred"] == "error")
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(results) if results else 0
    team_total = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
    team_correct = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal" and r.get("team_correct"))
    team_acc = team_correct / team_total if team_total else 0
    avg_lat = sum(r["latency"] for r in results if r.get("latency")) / len([r for r in results if r.get("latency")]) if any(r.get("latency") for r in results) else 0
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Confusion Matrix  ({len(results)} clips)")
    print(f"{' '*14}  Pred goal  Pred not-goal")
    print(f"  Truth goal        {tp:^8}      {fn:^8}")
    print(f"  Truth not-goal    {fp:^8}      {tn:^8}")
    if errors:
        print(f"  Errors: {errors}")
    print(f"-"*70)
    print(f"  Precision:  {prec*100:.1f}%")
    print(f"  Recall:     {rec*100:.1f}%")
    print(f"  F1:         {f1*100:.1f}%")
    print(f"  Accuracy:   {acc*100:.1f}%")
    print(f"  Team Acc:   {team_acc*100:.1f}% ({team_correct}/{team_total})")
    print(f"  Avg Latency: {avg_lat:.1f}s")

clips = sorted(DATA_DIR.glob("*.mp4"))
print(f"Dataset: {DATASET}")
print(f"Clips: {len(clips)}")
print(f"Strategy: 56px single frame, num_predict=500")
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

for model_name, timeout, num_ctx, num_predict, result_key in MODELS:
    print(f"\n\n{'='*70}")
    print(f"RUNNING: {model_name} (key={result_key})")
    print(f"{'='*70}")
    results = []
    
    for i, clip in enumerate(clips):
        truth, truth_team = load_gt(clip)
        base = clip.name.rsplit(".", 1)[0]
        path = RESULTS_DIR / f"{base}.json"
        
        # Check if already done
        if path.exists():
            data = json.loads(path.read_text())
            if result_key in data.get("models", {}):
                m = data["models"][result_key]
                print(f"[{i+1}/{len(clips)}] SKIP {clip.name} (already done, lat={m.get('latency',0)}s)")
                results.append({
                    "clip": clip.name, "truth": truth, "truth_team": truth_team,
                    "pred": m["pred"], "pred_team": m.get("pred_team"),
                    "team_correct": m.get("team_correct", False),
                    "latency": m.get("latency", 0),
                })
                continue
        
        print(f"[{i+1}/{len(clips)}] {clip.name} (truth: {truth}, team: {truth_team})", flush=True)
        try:
            result = classify(model_name, str(clip), PROMPT, timeout, num_ctx, num_predict)
            parsed = parse_json(result["raw"])
            pred = "goal" if parsed.get("goal") else "not_goal"
            pred_team = parsed.get("team")
            latency = result["latency"]
            
            goal_ok = pred == truth
            team_ok = (truth != "goal" or pred != "goal" or truth_team == pred_team)
            
            print(f"  -> {pred}, team={pred_team}, latency={latency:.1f}s")
            print(f"  -> raw: {result['raw'][:120]}")
            
            if path.exists():
                data = json.loads(path.read_text())
            else:
                data = {"clip": clip.name, "truth": truth, "truth_team": truth_team, "models": {}}
            
            data["models"][result_key] = {
                "pred": pred, "pred_team": pred_team, "raw": result["raw"],
                "goal_correct": goal_ok, "team_correct": team_ok,
                "latency": round(latency, 1),
            }
            path.write_text(json.dumps(data, indent=2))
            
            results.append({
                "clip": clip.name, "truth": truth, "truth_team": truth_team,
                "pred": pred, "pred_team": pred_team, "team_correct": team_ok,
                "latency": latency,
            })
        except Exception as e:
            print(f"  -> ERROR: {str(e)[:200]}")
            results.append({
                "clip": clip.name, "truth": truth, "truth_team": truth_team,
                "pred": "error", "pred_team": None, "team_correct": False,
            })
    
    print_cm(results, f"{model_name} — 56px FIXED — FINAL")

print(f"\n\nFinished all models: {time.strftime('%Y-%m-%d %H:%M:%S')}")
