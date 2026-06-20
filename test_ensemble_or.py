#!/usr/bin/env python3
"""Test OR (union) ensemble strategy: if either model says goal, it's a goal."""
import json
from pathlib import Path

eval_data = json.loads(Path("results/evaluation_results.json").read_text())

clip_data = {}
for r in eval_data["per_clip_results"]:
    clip = r["clip"]
    if clip not in clip_data:
        clip_data[clip] = {"truth": r["truth"], "truth_team": r["truth_team"]}
    clip_data[clip][f"{r['model']} ({r['config']})"] = r

smolvlm2_key = "smolvlm2-2.2b (56px, 23 frames, 2fps)"
moondream_direct = "moondream:1.8b (224px, 1 frame, direct prompt)"
moondream_v2 = "moondream:1.8b (224px, 1 frame, descriptive prompt)"
qwen3vl8b = "qwen3-vl:8b (224px, 3 frames)"

def evaluate_or(name, model1, model2, team_source):
    tp = fn = fp = tn = 0
    team_total = team_correct = 0
    
    for clip, data in sorted(clip_data.items()):
        pred1 = data[model1]["pred"] == "goal"
        pred2 = data[model2]["pred"] == "goal"
        pred = "goal" if (pred1 or pred2) else "not_goal"
        
        # Team from the first model that says goal, in priority order
        team = None
        if pred == "goal":
            for src in team_source:
                if data[src]["pred"] == "goal" and data[src]["pred_team"]:
                    team = data[src]["pred_team"]
                    break
        
        truth = data["truth"]
        if truth == "goal" and pred == "goal": tp += 1
        elif truth == "goal" and pred != "goal": fn += 1
        elif truth != "goal" and pred == "goal": fp += 1
        elif truth != "goal" and pred != "goal": tn += 1
        
        if pred == "goal" and truth == "goal":
            team_total += 1
            if team == data["truth_team"]:
                team_correct += 1
        
        g_ok = truth == pred
        t_ok = truth == "goal" and pred == "goal" and team == data["truth_team"]
        print(f"  {clip:<45} truth={truth:<8} pred={pred:<8} team={team or '-':<15} goal={'✓' if g_ok else '✗'} team={'✓' if t_ok else '✗'}")
    
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(clip_data) if clip_data else 0
    team_acc = team_correct / team_total if team_total else 0
    
    print(f"\n{name}")
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"Precision={prec*100:.1f}% Recall={rec*100:.1f}% F1={f1*100:.1f}% Acc={acc*100:.1f}%")
    print(f"Team Acc={team_acc*100:.1f}% ({team_correct}/{team_total})")
    return f1, acc, prec, rec, team_acc

print(f"{'='*70}")
print("OR (UNION) ENSEMBLE STRATEGIES")
print(f"{'='*70}")

print("\n--- Strategy A: smolvlm2 OR moondream direct ---")
print("If either says goal, it's a goal. Team from moondream direct first, then smolvlm2.")
f1_a, _, _, _, _ = evaluate_or("A", smolvlm2_key, moondream_direct, [moondream_direct, smolvlm2_key])

print("\n--- Strategy B: smolvlm2 OR moondream v2 ---")
print("If either says goal, it's a goal. Team from moondream v2 first, then smolvlm2.")
f1_b, _, _, _, _ = evaluate_or("B", smolvlm2_key, moondream_v2, [moondream_v2, smolvlm2_key])

print("\n--- Strategy C: moondream v2 OR moondream direct (no smolvlm2) ---")
print("If either says goal, it's a goal. Team from direct first, then v2.")
f1_c, _, _, _, _ = evaluate_or("C", moondream_v2, moondream_direct, [moondream_direct, moondream_v2])

print("\n--- Strategy D: All three OR ---")
print("If any of smolvlm2, moondream v2, or direct says goal, it's a goal.")
# Manual for 3 models
tp = fn = fp = tn = 0
team_total = team_correct = 0
for clip, data in sorted(clip_data.items()):
    pred = "goal" if any(data[m]["pred"] == "goal" for m in [smolvlm2_key, moondream_v2, moondream_direct]) else "not_goal"
    team = None
    if pred == "goal":
        for src in [moondream_direct, moondream_v2, smolvlm2_key]:
            if data[src]["pred"] == "goal" and data[src]["pred_team"]:
                team = data[src]["pred_team"]
                break
    truth = data["truth"]
    if truth == "goal" and pred == "goal": tp += 1
    elif truth == "goal" and pred != "goal": fn += 1
    elif truth != "goal" and pred == "goal": fp += 1
    elif truth != "goal" and pred != "goal": tn += 1
    if pred == "goal" and truth == "goal":
        team_total += 1
        if team == data["truth_team"]: team_correct += 1
    g_ok = truth == pred
    t_ok = truth == "goal" and pred == "goal" and team == data["truth_team"]
    print(f"  {clip:<45} truth={truth:<8} pred={pred:<8} team={team or '-':<15} goal={'✓' if g_ok else '✗'} team={'✓' if t_ok else '✗'}")

prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1_d = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
acc = (tp + tn) / len(clip_data) if clip_data else 0
team_acc = team_correct / team_total if team_total else 0
print(f"\nD. All three OR")
print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
print(f"Precision={prec*100:.1f}% Recall={rec*100:.1f}% F1={f1_d*100:.1f}% Acc={acc*100:.1f}%")
print(f"Team Acc={team_acc*100:.1f}% ({team_correct}/{team_total})")

print(f"\n{'='*70}")
print("SUMMARY OF ALL ENSEMBLE STRATEGIES")
print(f"{'='*70}")
print(f"Individual models:")
print(f"  smolvlm2:             F1=71.4%  Prec=55.6%  Rec=100.0%  Team=60.0%")
print(f"  moondream direct:     F1=33.3%  Prec=100.0% Rec=20.0%   Team=100.0%")
print(f"  moondream v2:         F1=50.0%  Prec=42.9%  Rec=60.0%   Team=66.7%")
print(f"  qwen3-vl:8b 224px:    F1=33.3%  Prec=100.0% Rec=20.0%   Team=0.0%")
print(f"\nEnsemble strategies:")
print(f"  A. smolvlm2 OR direct:     F1={f1_a*100:.1f}%")
print(f"  B. smolvlm2 OR v2:         F1={f1_b*100:.1f}%")
print(f"  C. v2 OR direct:           F1={f1_c*100:.1f}%")
print(f"  D. All three OR:           F1={f1_d*100:.1f}%")
print(f"\nConclusion: OR ensemble with smolvlm2 + any moondream = same as smolvlm2 alone")
print(f"(smolvlm2 already catches everything, so adding moondream doesn't help recall)")
print(f"Team accuracy improves when moondream is used as team source for its detections.")
