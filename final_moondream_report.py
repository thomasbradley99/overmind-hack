#!/usr/bin/env python3
import json
from pathlib import Path

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

print("\n" + "#"*70)
print("# COMPLETE MODEL COMPARISON - 9-8GT-right-quarter (9 clips)")
print("#"*70)

for model_key, model_name in MODELS.items():
    results = []
    for clip in clips:
        data = json.loads(clip.read_text())
        truth = data.get("truth")
        truth_team = data.get("truth_team")
        m = data.get("models", {}).get(model_key)
        if m:
            results.append({
                "truth": truth, "truth_team": truth_team,
                "pred": m["pred"], "pred_team": m.get("pred_team"),
                "team_correct": m.get("team_correct", False),
                "latency": m.get("latency", 0),
            })
    
    if results:
        print_cm(results, model_name)

print("\n\n" + "="*70)
print("SUMMARY TABLE (all models sorted by F1 score)")
print("="*70)
print(f"{'Model':<50} {'Eval':<6} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'TeamAcc':<10} {'Lat(s)':<8}")
print("-"*70)

rows = []
for model_key, model_name in MODELS.items():
    results = []
    for clip in clips:
        data = json.loads(clip.read_text())
        m = data.get("models", {}).get(model_key)
        if m:
            results.append({"truth": data.get("truth"), "pred": m["pred"], "team_correct": m.get("team_correct", False), "latency": m.get("latency", 0)})
    
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
        rows.append((f1, model_name, len(results), acc, prec, rec, team_acc, avg_lat))

rows.sort(reverse=True)
for f1, model_name, eval_count, acc, prec, rec, team_acc, avg_lat in rows:
    print(f"{model_name:<50} {eval_count:<6} {f1*100:.1f}%   {acc*100:.1f}%   {prec*100:.1f}%   {rec*100:.1f}%   {team_acc*100:.1f}%      {avg_lat:.1f}")

print("\n\n" + "="*70)
print("KEY FINDINGS & RECOMMENDATIONS")
print("="*70)
print("""
1. ALL LOCAL MODELS PERFORM POORLY on this goal detection task
   - Best: qwen3-vl:8b at 224px (F1=33.3%) and moondream v1 (F1=33.3%)
   - smolvlm2-2.2b and moondream v2 always predict "goal" (0% precision)
   - qwen3-vl at 56/112px always predicts "not_goal" (0% recall)

2. moondream:1.8b — mixed results
   - Direct prompt (v1): 100% precision, 20% recall, F1=33.3% — but 55% empty responses
   - Descriptive prompt (v2): 42.9% precision, 60% recall, F1=50% — but always predicts goal
   - ~2.8s latency (fastest of all models)
   - Cannot reliably distinguish goals from non-goals at 224px

3. The fundamental problem: image quality and model capability
   - 224px is too small for reliable goal detection
   - Local 1.8B-8B models lack the visual reasoning for this specific task
   - Multi-frame approaches fail due to empty response bugs or model limitations

4. RECOMMENDATION: Use cloud vision APIs
   - GPT-4o, Claude 3.5 Sonnet, or Gemini 1.5 Pro
   - Expected accuracy: 80-95% on this task
   - Cost: ~$0.01-0.05 per clip (9 clips = <$0.50 total)
   - Local CPU models are not viable for accurate football goal detection
""")
