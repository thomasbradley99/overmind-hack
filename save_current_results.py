#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results/9-8GT-right-quarter")
clips = sorted(RESULTS_DIR.glob("*.json"))

MODELS = {
    "richardyoung/smolvlm2-2.2b-instruct:latest": "smolvlm2-2.2b (224px, 3 frames, orig prompt)",
    "richardyoung/smolvlm2-2.2b-instruct:latest_new_prompt": "smolvlm2-2.2b (224px, 3 frames, new prompt)",
    "qwen3-vl:8b": "qwen3-vl:8b (224px, 3 frames)",
    "smolvlm2_56px_multi": "smolvlm2-2.2b (56px, 23 frames, 2fps)",
    "qwen3vl8b_56px_multi": "qwen3-vl:8b (56px, 23 frames, 2fps)",
    "qwen3vl2b_56px_multi": "qwen3-vl:2b (56px, 23 frames, 2fps)",
    "qwen3vl2b_56px_fixed": "qwen3-vl:2b (56px, 1 frame, num_predict=500)",
    "qwen3vl8b_56px_fixed": "qwen3-vl:8b (56px, 1 frame, num_predict=500)",
    "qwen3vl4b_112px": "qwen3-vl:4b (112px, 1 frame, num_predict=500)",
    "moondream_224px": "moondream:1.8b (224px, 1 frame, direct prompt)",
    "moondream_224px_v2": "moondream:1.8b (224px, 1 frame, descriptive prompt)",
}

report = []
report.append("# Football Goal Detection - Baseline Results Report")
report.append(f"Generated: {datetime.now().isoformat()}")
report.append(f"Dataset: 9-8GT-right-quarter (9 clips: 5 goals + 4 non-goals)")
report.append("")
report.append("## Model Configurations Tested")
report.append("")
report.append("| # | Model | Resolution | Frames | Prompt |")
report.append("|---|-------|-----------|--------|--------|")
for i, (key, name) in enumerate(MODELS.items(), 1):
    report.append(f"| {i} | {name} |")
report.append("")
report.append("## Per-Clip Ground Truth")
report.append("")
report.append("| Clip | Truth | Team |")
report.append("|------|-------|------|")
for clip in clips:
    data = json.loads(clip.read_text())
    report.append(f"| {clip.stem} | {data.get('truth')} | {data.get('truth_team')} |")
report.append("")
report.append("## Aggregate Results Summary (sorted by F1)")
report.append("")
report.append("| Model | F1 | Accuracy | Precision | Recall | Team Acc | Latency |")
report.append("|-------|-----|----------|-----------|--------|----------|---------|")

rows = []
for model_key, model_name in MODELS.items():
    results = []
    for clip in clips:
        data = json.loads(clip.read_text())
        m = data.get("models", {}).get(model_key)
        if m:
            results.append({
                "truth": data.get("truth"),
                "pred": m["pred"],
                "team_correct": m.get("team_correct", False),
                "latency": m.get("latency", 0)
            })
    
    if results:
        tp = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
        fn = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "not_goal")
        fp = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "goal")
        tn = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "not_goal")
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        acc = (tp + tn) / len(results) if results else 0
        team_total = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
        team_correct = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal" and r["team_correct"])
        team_acc = team_correct / team_total if team_total else 0
        avg_lat = sum(r["latency"] for r in results if r["latency"]) / len([r for r in results if r["latency"]]) if any(r["latency"] for r in results) else 0
        rows.append((f1, model_name, acc, prec, rec, team_acc, avg_lat))

rows.sort(reverse=True)
for f1, model_name, acc, prec, rec, team_acc, avg_lat in rows:
    report.append(f"| {model_name} | {f1*100:.1f}% | {acc*100:.1f}% | {prec*100:.1f}% | {rec*100:.1f}% | {team_acc*100:.1f}% | {avg_lat:.1f}s |")

report.append("")
report.append("## Key Findings")
report.append("")
report.append("1. **smolvlm2-2.2b** always predicts 'goal' on every clip (100% recall, 0% precision on non-goals)")
report.append("2. **qwen3-vl** at 56/112px always predicts 'not_goal' (0% recall, 100% precision on non-goals)")
report.append("3. **qwen3-vl:8b at 224px** is the only config with balanced predictions (33.3% F1)")
report.append("4. **moondream v1** has 100% precision but only 20% recall (55% empty responses)")
report.append("5. **moondream v2** has 50% F1 but 42.9% precision (too many false positives)")
report.append("6. **Local CPU models are not viable** for accurate football goal detection at ≤224px")
report.append("")
report.append("## Next Steps")
report.append("- Try improved moondream prompts (more specific, chain-of-thought)")
report.append("- Try higher resolution frames (448px, 672px)")
report.append("- Try multiple frame extraction per clip (beginning, middle, end)")
report.append("- Consider cloud vision APIs if local models remain inadequate")
report.append("")

out = Path("results/baseline_report.md")
out.write_text("\n".join(report))
print(f"Saved to {out}")

# Also save as JSON
json_report = {
    "timestamp": datetime.now().isoformat(),
    "dataset": "9-8GT-right-quarter",
    "clips": 9,
    "goals": 5,
    "non_goals": 4,
    "models_tested": len(MODELS),
    "results": []
}

for f1, model_name, acc, prec, rec, team_acc, avg_lat in rows:
    json_report["results"].append({
        "model": model_name,
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "team_accuracy": round(team_acc, 3),
        "avg_latency": round(avg_lat, 1)
    })

out_json = Path("results/baseline_results.json")
out_json.write_text(json.dumps(json_report, indent=2))
print(f"Saved to {out_json}")
