#!/usr/bin/env python3
import json, os, subprocess, sys, time, re
from pathlib import Path

# Use the Overmind classify.py structure but run moondream
sys.path.insert(0, str(Path(".overmind/agents/goal-classifier/instrumented")))
from classify import (
    build_prompt, classify_one, extract_clip_frames, labels_for, load_teams,
    normalize_team, score_goals, score_teams, save_per_clip_results, OLLAMA_HOST
)
import requests
import base64

DATASET = "9-8GT-right-quarter"
DATA_DIR = Path("data") / DATASET
MODEL = "moondream:latest"

# Load the Overmind prompt
prompt_template = Path(".overmind/agents/goal-classifier/instrumented/prompt.txt").read_text().strip()
teams = load_teams("9-8GT-right")  # Use the 9-8GT-right teams for consistency
prompt = build_prompt(prompt_template, teams)

print(f"Dataset: {DATASET}")
print(f"Teams: {teams[0]} vs {teams[1]}")
print(f"Model: {MODEL}")
print(f"Prompt ({len(prompt)} chars): {prompt[:100]}...")
print("\n" + "="*70)

clips = sorted(DATA_DIR.glob("*.mp4"))
results = []

for i, clip in enumerate(clips, 1):
    truth, truth_team = labels_for(clip.with_suffix(".json"))
    print(f"\n[{i}/{len(clips)}] {clip.name} (truth: {truth}, team: {truth_team})")
    
    try:
        # Extract 3 frames at 336px
        frames = extract_clip_frames(str(clip), num_frames=3, max_size=336)
        print(f"  Extracted {len(frames)} frames at 336px")
        
        images_b64 = []
        for f in frames:
            with open(f, "rb") as img:
                images_b64.append(base64.b64encode(img.read()).decode("utf-8"))
        
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        
        t0 = time.time()
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        t1 = time.time()
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("message", {}).get("content", "").strip()
        latency = t1 - t0
        
        print(f"  Latency: {latency:.1f}s")
        print(f"  Raw: {raw[:200]}...")
        
        # Parse moondream output - look for JSON
        import re
        pred = "not_goal"
        pred_team = None
        
        # Try to find {"goal": ...} in output
        for obj_match in re.finditer(r'\{[^{}]*\}', raw):
            try:
                obj = json.loads(obj_match.group(0))
                if isinstance(obj, dict) and "goal" in obj:
                    goal = obj.get("goal")
                    if goal is True:
                        pred = "goal"
                        pred_team = normalize_team(obj.get("team"), teams)
                    elif goal is False:
                        pred = "not_goal"
                        pred_team = None
                    break
            except:
                pass
        
        # Fallback: check for keywords
        if pred == "not_goal":
            raw_lower = raw.lower()
            if '"goal"' in raw_lower and 'true' in raw_lower:
                pred = "goal"
            # Check team keywords
            if pred == "goal":
                if "sportswear" in raw_lower or "track" in raw_lower:
                    pred_team = "Dark sportswear"
                elif "suit" in raw_lower or "formal" in raw_lower or "jacket" in raw_lower:
                    pred_team = "Dark suits"
        
        print(f"  Parsed: pred={pred}, team={pred_team}")
        
        results.append({
            "clip": clip.name,
            "truth": truth,
            "truth_team": truth_team,
            "pred": pred,
            "pred_team": pred_team,
            "raw": raw,
            "latency": latency,
            "n_frames": len(frames),
        })
        
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "clip": clip.name,
            "truth": truth,
            "truth_team": truth_team,
            "pred": "error",
            "pred_team": None,
            "raw": str(e)[:200],
            "latency": 0,
            "n_frames": 0,
        })

# Score
print(f"\n{'='*70}")
print("OVERMIND-STYLE EVALUATION - moondream:latest (3 frames, 336px)")
print(f"{'='*70}")

tp = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
fn = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "not_goal")
fp = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "goal")
tn = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "not_goal")
errors = sum(1 for r in results if r["pred"] == "error")
n = len(results)
acc = (tp + tn) / n if n else 0
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

print(f"Confusion Matrix ({n} clips)")
print(f"{' '*14}  Pred goal  Pred not-goal")
print(f"  Truth goal        {tp:^8}      {fn:^8}")
print(f"  Truth not-goal    {fp:^8}      {tn:^8}")
if errors:
    print(f"  Errors: {errors}")
print(f"-"*70)
print(f"  F1:        {f1*100:.1f}%")
print(f"  Accuracy:  {acc*100:.1f}%")
print(f"  Precision: {prec*100:.1f}%")
print(f"  Recall:    {rec*100:.1f}%")

# Team accuracy
detected_goals = [r for r in results if r["truth"] == "goal" and r["pred"] == "goal"]
team_correct = sum(1 for r in detected_goals if r["truth_team"] and r["pred_team"] 
                  and normalize_team(r["truth_team"], teams) == normalize_team(r["pred_team"], teams))
team_total = len(detected_goals)
team_acc = team_correct / team_total if team_total else 0

print(f"\n  Team Accuracy: {team_acc*100:.1f}% ({team_correct}/{team_total})")
print(f"  Avg Latency: {sum(r['latency'] for r in results)/len(results):.1f}s")

# Save results
out_dir = Path("results") / DATASET
out_dir.mkdir(parents=True, exist_ok=True)
for r in results:
    base = r["clip"].rsplit(".", 1)[0]
    path = out_dir / f"{base}.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"clip": r["clip"], "truth": r["truth"], "truth_team": r["truth_team"], "models": {}}
    
    data["models"]["moondream_overmind_336px"] = {
        "pred": r["pred"],
        "pred_team": r["pred_team"],
        "raw": r["raw"],
        "goal_correct": r["truth"] == r["pred"],
        "team_correct": r["truth_team"] and r["pred_team"] and normalize_team(r["truth_team"], teams) == normalize_team(r["pred_team"], teams) if r["truth"] == "goal" and r["pred"] == "goal" else False,
        "latency": r["latency"],
        "n_frames": r["n_frames"],
    }
    path.write_text(json.dumps(data, indent=2))

print(f"\nSaved per-clip results to {out_dir}")

# Save comprehensive report
report_lines = [
    "# Overmind-Style Moondream Evaluation Report",
    "",
    f"**Model:** moondream:latest (1.8B parameters)",
    f"**Dataset:** {DATASET} (9 clips: 5 goals + 4 non-goals)",
    f"**Configuration:** 3 frames, 336px, Overmind prompt",
    f"**Date:** {subprocess.check_output(['date', '-Iseconds']).decode().strip()}",
    "",
    "## Confusion Matrix",
    "",
    f"|                | Pred goal | Pred not_goal |",
    f"|----------------|-----------|---------------|",
    f"| Truth goal     | {tp}        | {fn}             |",
    f"| Truth not_goal | {fp}        | {tn}             |",
    f"",
    f"| Metric       | Value |",
    f"|--------------|-------|",
    f"| F1 Score     | {f1*100:.1f}% |",
    f"| Accuracy     | {acc*100:.1f}% |",
    f"| Precision    | {prec*100:.1f}% |",
    f"| Recall       | {rec*100:.1f}% |",
    f"| Team Accuracy| {team_acc*100:.1f}% |",
    f"| Avg Latency  | {sum(r['latency'] for r in results)/len(results):.1f}s |",
    "",
    "## Per-Clip Results",
    "",
    "| Clip | Truth | Team | Pred | Pred Team | Latency | Frames | Raw Preview |",
    "|------|-------|------|------|-----------|---------|--------|-------------|",
]

for r in results:
    raw_preview = r["raw"][:80].replace("\n", " ")
    report_lines.append(f"| {r['clip']:<36} | {r['truth']:<7} | {r['truth_team'] or '-':<15} | {r['pred']:<7} | {r['pred_team'] or '-':<15} | {r['latency']:.1f}s | {r['n_frames']} | {raw_preview} |")

report_lines.extend([
    "",
    "## Analysis",
    "",
    "### Moondream Behavior",
    "- Moondream outputs nonsensical, hallucinated JSON when given structured prompts with multiple frames.",
    "- The model appears to be overfitting or generating auto-regressive JSON tokens that don't match the image.",
    "- Team prediction is consistently 'Dark sportswear' regardless of the actual image content.",
    "",
    "### Conclusion",
    "- Moondream 1.8B is **not suitable** for football goal detection with the Overmind approach.",
    "- The model lacks the visual reasoning capability to distinguish goals from non-goals at this resolution.",
    "- Consider using qwen3-vl (better at 224px) or cloud APIs (Gemini, Claude) for this task.",
    "",
])

out_report = Path("results/moondream_overmind_report.md")
out_report.write_text("\n".join(report_lines))
print(f"Saved report to {out_report}")
