#!/usr/bin/env python3
"""Run all models on quarter dataset with 56px images at 2fps (multi-frame)."""
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
    ("richardyoung/smolvlm2-2.2b-instruct:latest", 60, 512, "smolvlm2_56px_multi"),
    ("qwen3-vl:8b", 300, 512, "qwen3vl8b_56px_multi"),
    ("qwen3-vl:2b", 600, 512, "qwen3vl2b_56px_multi"),
]

def extract_frames(clip_path: str, fps: float = 2.0, max_size: int = 56) -> list:
    work_dir = Path(tempfile.mkdtemp(prefix="multi-"))
    duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", clip_path]
    dr = subprocess.run(duration_cmd, capture_output=True, text=True)
    try:
        duration = float(dr.stdout.strip())
    except ValueError:
        duration = 0.0
    
    frames = []
    interval = 1.0 / fps
    t = interval
    while t < duration:
        frame = work_dir / f"frame_{t:.2f}.jpg"
        cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", clip_path, "-vframes", "1",
               "-vf", f"scale='min({max_size},iw)':-1", "-q:v", "5", str(frame)]
        subprocess.run(cmd, capture_output=True, timeout=30)
        if frame.exists():
            frames.append(str(frame))
        t += interval
    return frames

def classify(model: str, clip_path: str, prompt: str, timeout: int, num_ctx: int) -> dict:
    frames = extract_frames(clip_path, fps=2.0, max_size=56)
    images_b64 = []
    for f in frames:
        with open(f, "rb") as fh:
            images_b64.append(base64.b64encode(fh.read()).decode("utf-8"))
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": num_ctx, "num_predict": 50}
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data["message"]["content"].strip()
    return {"raw": raw, "latency": t1 - t0, "n_frames": len(frames)}

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
    print(f"  Avg Frames: {sum(r.get('n_frames',0) for r in results)/len(results):.1f}")

clips = sorted(DATA_DIR.glob("*.mp4"))
print(f"Dataset: {DATASET}")
print(f"Clips: {len(clips)}")
print(f"Strategy: 56px images, 2fps multi-frame, 12s clips")
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n{'#'*70}")
print(f"# MODELS: {len(MODELS)}")
print(f"#"*70)

for model_name, timeout, num_ctx, result_key in MODELS:
    print(f"\n\n{'='*70}")
    print(f"RUNNING: {model_name} (key={result_key})")
    print(f"{'='*70}")
    results = []
    
    for i, clip in enumerate(clips):
        truth, truth_team = load_gt(clip)
        base = clip.name.rsplit(".", 1)[0]
        path = RESULTS_DIR / f"{base}.json"
        
        # Check if already done for this model key
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
                    "n_frames": m.get("n_frames", 0),
                })
                continue
        
        print(f"[{i+1}/{len(clips)}] {clip.name} (truth: {truth}, team: {truth_team})", flush=True)
        try:
            result = classify(model_name, str(clip), PROMPT, timeout, num_ctx)
            parsed = parse_json(result["raw"])
            pred = "goal" if parsed.get("goal") else "not_goal"
            pred_team = parsed.get("team")
            latency = result["latency"]
            n_frames = result["n_frames"]
            
            goal_ok = pred == truth
            team_ok = (truth != "goal" or pred != "goal" or truth_team == pred_team)
            
            print(f"  -> {pred}, team={pred_team}, frames={n_frames}, latency={latency:.1f}s")
            print(f"  -> raw: {result['raw'][:120]}")
            
            if path.exists():
                data = json.loads(path.read_text())
            else:
                data = {"clip": clip.name, "truth": truth, "truth_team": truth_team, "models": {}}
            
            data["models"][result_key] = {
                "pred": pred, "pred_team": pred_team, "raw": result["raw"],
                "goal_correct": goal_ok, "team_correct": team_ok,
                "latency": round(latency, 1), "n_frames": n_frames,
            }
            path.write_text(json.dumps(data, indent=2))
            
            results.append({
                "clip": clip.name, "truth": truth, "truth_team": truth_team,
                "pred": pred, "pred_team": pred_team, "team_correct": team_ok,
                "latency": latency, "n_frames": n_frames,
            })
        except Exception as e:
            print(f"  -> ERROR: {str(e)[:200]}")
            results.append({
                "clip": clip.name, "truth": truth, "truth_team": truth_team,
                "pred": "error", "pred_team": None, "team_correct": False,
            })
    
    print_cm(results, f"{model_name} — 56px 2fps — FINAL")

print(f"\n\nFinished all models: {time.strftime('%Y-%m-%d %H:%M:%S')}")
