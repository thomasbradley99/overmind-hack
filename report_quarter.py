#!/usr/bin/env python3
import json
from pathlib import Path

DATASET = "9-8GT-right-quarter"
RESULTS_DIR = Path("results") / DATASET

MODELS = ["qwen3-vl:8b", "richardyoung/smolvlm2-2.2b-instruct:latest"]

clips = sorted(RESULTS_DIR.glob("*.json"))

for model in MODELS:
    results = []
    for clip in clips:
        data = json.loads(clip.read_text())
        truth = data.get("truth")
        truth_team = data.get("truth_team")
        m = data.get("models", {}).get(model)
        if m:
            results.append({
                "clip": data["clip"], "truth": truth, "truth_team": truth_team,
                "pred": m["pred"], "pred_team": m.get("pred_team"),
                "team_correct": m.get("team_correct", False),
                "latency": m.get("latency", 0),
            })
    
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
    
    print(f"\n{'='*60}")
    print(f"{model} ({len(results)}/{len(clips)} clips evaluated)")
    print(f"{'='*60}")
    print(f"Confusion Matrix")
    print(f"{' '*12}  Pred goal  Pred not-goal")
    print(f"  Truth goal      {tp:^8}      {fn:^8}")
    print(f"  Truth not-goal  {fp:^8}      {tn:^8}")
    if errors:
        print(f"  Errors: {errors}")
    print(f"-"*60)
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Team Acc:  {team_acc*100:.1f}% ({team_correct}/{team_total})")
    avg_lat = sum(r["latency"] for r in results) / len(results) if results else 0
    print(f"  Avg Latency: {avg_lat:.1f}s")

print(f"\n{'='*60}")
print(f"Per-clip Breakdown")
print(f"{'='*60}")
for clip in clips:
    data = json.loads(clip.read_text())
    truth = data.get("truth")
    truth_team = data.get("truth_team")
    print(f"\n{data['clip']:<45} truth: {truth:<8} team: {truth_team or 'None'}")
    for model in MODELS:
        m = data.get("models", {}).get(model)
        if m:
            g_mark = "✓" if m["pred"] == truth else "✗"
            t_mark = "✓" if m.get("team_correct") else "✗"
            print(f"  {model:<40} pred={m['pred']:<8} team={m.get('pred_team','None'):<20} [goal:{g_mark} team:{t_mark}] lat={m.get('latency',0):.1f}s")
        else:
            print(f"  {model:<40} [TIMEOUT / NOT EVALUATED]")
