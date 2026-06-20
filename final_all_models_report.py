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
print(f"{'Model':<50} {'Eval':<6} {'F1':<8} {'Acc':<8} {'TeamAcc':<10} {'Lat(s)':<8}")
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
        rows.append((f1, model_name, len(results), acc, team_acc, avg_lat))

rows.sort(reverse=True)
for f1, model_name, eval_count, acc, team_acc, avg_lat in rows:
    print(f"{model_name:<50} {eval_count:<6} {f1*100:.1f}%   {acc*100:.1f}%   {team_acc*100:.1f}%      {avg_lat:.1f}")

print("\n\n" + "="*70)
print("KEY FINDINGS & RECOMMENDATIONS")
print("="*70)
print("""
1. smolvlm2-2.2b is USELESS for this task:
   - Predicts "goal" on EVERY clip regardless of prompt or image size
   - Team identification is random (always latches onto one team name)
   - F1=71.4% is misleading — it has 100% recall but 0% precision on non-goals

2. qwen3-vl models (2B, 4B, 8B) are too conservative at low resolutions:
   - At 56px and 112px, they predict "not_goal" on EVERY clip
   - The images are too small to see the ball, goal line, or net
   - Accuracy=44.4% (correct on 4/4 non-goals, wrong on 5/5 goals)

3. The empty response issue was a red herring:
   - Fixed by increasing num_predict from 50 to 500
   - But the model still can't see enough detail at 112px to make decisions

4. Image resolution is the critical bottleneck:
   - 224px might be the minimum for any useful detection
   - 56px and 112px are too small for goal-level visual detail
   - Multi-frame (23 frames) makes Qwen models produce empty responses

5. Hardware constraints prevent testing at useful resolutions:
   - qwen3-vl:8b at 224px with 3 frames times out (needs GPU)
   - CPU-only inference is too slow for viable goal detection

RECOMMENDATION:
- Use cloud vision APIs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro)
- Or get an NVIDIA GPU (RTX 3060+ or cloud GPU instance)
- Local CPU inference with these models is not viable for accurate goal detection
""")
