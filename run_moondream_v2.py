#!/usr/bin/env python3
import base64, json, subprocess, tempfile, time, requests
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL = "moondream:latest"
DATASET = "9-8GT-right-quarter"
DATA_DIR = Path("data") / DATASET
RESULTS_DIR = Path("results") / DATASET
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Prompt that forces Moondream to produce output
PROMPT = "What is happening in this football image? Is it a goal? Describe and answer."

def extract_frame(clip_path: str, max_size: int = 224) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="moondream-"))
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
           "-vf", f"scale={max_size}:-1", "-q:v", "5", str(frame)]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return str(frame) if frame.exists() else None

def classify(clip_path: str) -> dict:
    frame = extract_frame(clip_path)
    with open(frame, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
    }
    t0 = time.time()
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip()
    return {"raw": raw, "latency": t1 - t0}

clips = sorted(DATA_DIR.glob("*.mp4"))
print(f"Clips: {len(clips)}\nStrategy: 224px single frame, descriptive prompt\nModel: {MODEL}\n")

total_results = []

for i, clip in enumerate(clips, 1):
    json_path = clip.with_suffix(".json")
    data = json.loads(json_path.read_text())
    if data.get("label") == "not_goal":
        truth = "not_goal"
    elif data.get("action") == "Goal":
        truth = "goal"
    else:
        truth = "not_goal"
    truth_team = data.get("team")
    
    result = classify(str(clip))
    raw = result["raw"]
    raw_lower = raw.lower()
    
    # Parse: if description contains "goal" in a positive context
    if "goal" in raw_lower and "not a goal" not in raw_lower and "no goal" not in raw_lower:
        pred = "goal"
    elif "goal" in raw_lower:
        pred = "goal"  # Conservative: if it mentions goal at all
    else:
        pred = "not_goal"
    
    pred_team = None
    if pred == "goal":
        if "suit" in raw_lower or "formal" in raw_lower or "jacket" in raw_lower or "shirt" in raw_lower:
            pred_team = "Dark suits"
        elif "sport" in raw_lower or "track" in raw_lower or "athletic" in raw_lower:
            pred_team = "Dark sportswear"
    
    team_correct = False
    if truth == "goal" and pred == "goal" and truth_team and pred_team:
        team_correct = truth_team.lower() == pred_team.lower()
    
    total_results.append({
        "clip": clip.name, "truth": truth, "truth_team": truth_team,
        "pred": pred, "pred_team": pred_team, "team_correct": team_correct,
        "raw": raw, "latency": result["latency"]
    })
    
    print(f"[{i}/{len(clips)}] {clip.name} (truth: {truth}, team: {truth_team})")
    print(f"  -> {pred}, team={pred_team}, latency={result['latency']:.1f}s")
    print(f"  -> raw: {raw[:120]}")

tp = sum(1 for r in total_results if r["truth"] == "goal" and r["pred"] == "goal")
fn = sum(1 for r in total_results if r["truth"] == "goal" and r["pred"] == "not_goal")
fp = sum(1 for r in total_results if r["truth"] == "not_goal" and r["pred"] == "goal")
tn = sum(1 for r in total_results if r["truth"] == "not_goal" and r["pred"] == "not_goal")
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
acc = (tp + tn) / len(total_results) if total_results else 0
team_correct = sum(1 for r in total_results if r.get("team_correct"))
team_total = sum(1 for r in total_results if r["truth"] == "goal" and r["pred"] == "goal")
team_acc = team_correct / team_total if team_total else 0
avg_lat = sum(r["latency"] for r in total_results) / len(total_results) if total_results else 0

print(f"\n{'='*70}")
print(f"moondream:1.8b — 224px v2 (descriptive prompt) — FINAL")
print(f"{'='*70}")
print(f"Confusion Matrix  ({len(total_results)} clips)")
print(f"{' '*14}  Pred goal  Pred not-goal")
print(f"  Truth goal        {tp:^8}      {fn:^8}")
print(f"  Truth not-goal    {fp:^8}      {tn:^8}")
print(f"-"*70)
print(f"  Precision:  {prec*100:.1f}%")
print(f"  Recall:     {rec*100:.1f}%")
print(f"  F1:         {f1*100:.1f}%")
print(f"  Accuracy:   {acc*100:.1f}%")
print(f"  Team Acc:   {team_acc*100:.1f}% ({team_correct}/{team_total})")
print(f"  Avg Latency: {avg_lat:.1f}s")

for r in total_results:
    base = r["clip"].rsplit(".", 1)[0]
    path = RESULTS_DIR / f"{base}.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"clip": r["clip"], "truth": r["truth"], "truth_team": r["truth_team"], "models": {}}
    data["models"]["moondream_224px_v2"] = {
        "pred": r["pred"], "pred_team": r["pred_team"], "raw": r["raw"],
        "goal_correct": r["truth"] == r["pred"], "team_correct": r.get("team_correct", False),
        "latency": r["latency"]
    }
    path.write_text(json.dumps(data, indent=2))

print(f"\nResults saved to {RESULTS_DIR}")
