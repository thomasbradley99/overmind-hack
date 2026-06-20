#!/usr/bin/env python3
"""Generate final comparison report for all models on quarter dataset."""
import json
from pathlib import Path

RESULTS_DIR = Path("results/9-8GT-right-quarter")
clips = sorted(RESULTS_DIR.glob("*.json"))

MODELS = {
    "qwen3-vl:8b": "qwen3-vl:8b (112px, optimized)",
    "richardyoung/smolvlm2-2.2b-instruct:latest": "smolvlm2-2.2b-instruct (224px)",
    "richardyoung/smolvlm2-2.2b-instruct:latest_new_prompt": "smolvlm2-2.2b (new prompt)",
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
    print(f"Confusion Matrix  ({len(results)} clips evaluated)")
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
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "prec": prec, "rec": rec, "f1": f1, "acc": acc, "team_acc": team_acc, "avg_lat": avg_lat}

print("\n" + "#"*70)
print("# FINAL MODEL COMPARISON - 9-8GT-right-quarter (9 clips)")
print("#"*70)

all_model_results = {}
for model_key, model_name in MODELS.items():
    results = []
    for clip in clips:
        data = json.loads(clip.read_text())
        truth = data.get("truth")
        truth_team = data.get("truth_team")
        m = data.get("models", {}).get(model_key)
        if m:
            results.append({
                "clip": data["clip"], "truth": truth, "truth_team": truth_team,
                "pred": m["pred"], "pred_team": m.get("pred_team"),
                "team_correct": m.get("team_correct", False),
                "latency": m.get("latency", 0),
            })
    
    if results:
        stats = print_cm(results, model_name)
        stats["evaluated"] = len(results)
        all_model_results[model_name] = stats

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Model':<50} {'Eval':<6} {'F1':<8} {'Acc':<8} {'TeamAcc':<10} {'Lat(s)':<8}")
print("-"*70)
for name, stats in all_model_results.items():
    print(f"{name:<50} {stats['evaluated']:<6} {stats['f1']*100:.1f}%   {stats['acc']*100:.1f}%   {stats['team_acc']*100:.1f}%      {stats['avg_lat']:.1f}")

print("\n" + "="*70)
print("PER-CLIP BREAKDOWN")
print("="*70)
for clip in clips:
    data = json.loads(clip.read_text())
    truth = data.get("truth")
    truth_team = data.get("truth_team")
    print(f"\n{data['clip']:<45} truth={truth:<8} team={truth_team or 'None'}")
    for model_key, model_name in MODELS.items():
        m = data.get("models", {}).get(model_key)
        if m:
            g_mark = "✓" if m["pred"] == truth else "✗"
            t_mark = "✓" if m.get("team_correct") else "✗"
            print(f"  {model_name:<45} pred={m['pred']:<8} team={m.get('pred_team') or 'None':<20} [goal:{g_mark} team:{t_mark}] lat={m.get('latency',0):.1f}s")
        else:
            print(f"  {model_name:<45} [NOT EVALUATED]")
