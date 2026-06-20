#!/usr/bin/env python3
import json
from pathlib import Path

RESULTS_DIR = Path("results/9-8GT-right-quarter")
clips = sorted(RESULTS_DIR.glob("*.json"))

MODELS = {
    "qwen3-vl:8b": "qwen3-vl:8b (112px, reduced context)",
    "richardyoung/smolvlm2-2.2b-instruct:latest": "smolvlm2-2.2b (224px, original prompt)",
    "richardyoung/smolvlm2-2.2b-instruct:latest_new_prompt": "smolvlm2-2.2b (224px, new prompt)",
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

print("\n" + "#"*70)
print("# FINAL MODEL COMPARISON - 9-8GT-right-quarter (9 clips)")
print("#"*70)
print(f"\nNOTE: qwen3-vl:2b was attempted but timed out on ALL clips (>5 min each).")
print(f"      qwen3-vl:8b was successfully run with 112px images + reduced context.")

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
        print_cm(results, model_name)

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Model':<50} {'Eval':<6} {'F1':<8} {'Acc':<8} {'TeamAcc':<10} {'Lat(s)':<8}")
print("-"*70)

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
        print(f"{model_name:<50} {len(results):<6} {f1*100:.1f}%   {acc*100:.1f}%   {team_acc*100:.1f}%      {avg_lat:.1f}")

print("\n" + "="*70)
print("qwen3-vl:2b STATUS")
print("="*70)
print("Model: qwen3-vl:2b")
print("Clips evaluated: 0/9")
print("Status: TIMED OUT on all clips (>300s per clip)")
print("Issue: AMD HawkPoint1 integrated GPU cannot handle Qwen vision models")
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
            pred_team = m.get("pred_team") or "None"
            print(f"  {model_name:<45} pred={m['pred']:<8} team={pred_team:<20} [goal:{g_mark} team:{t_mark}] lat={m.get('latency',0):.1f}s")
        else:
            print(f"  {model_name:<45} [NOT EVALUATED]")
    print(f"  {'qwen3-vl:2b (2B param, 224px)':<45} [TIMEOUT - not evaluated]")
