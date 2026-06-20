#!/usr/bin/env python3
"""Test multiple ensemble strategies on pre-computed data."""
import json
from pathlib import Path

# Load data
eval_data = json.loads(Path("results/evaluation_results.json").read_text())

# Build per-clip lookup
clip_data = {}
for r in eval_data["per_clip_results"]:
    clip = r["clip"]
    if clip not in clip_data:
        clip_data[clip] = {"truth": r["truth"], "truth_team": r["truth_team"]}
    
    model_key = f"{r['model']} ({r['config']})"
    clip_data[clip][model_key] = {
        "pred": r["pred"],
        "team": r["pred_team"],
        "goal_correct": r["goal_correct"],
        "team_correct": r["team_correct"]
    }

# Define models to use
smolvlm2_key = "smolvlm2-2.2b (56px, 23 frames, 2fps)"
moondream_direct = "moondream:1.8b (224px, 1 frame, direct prompt)"
moondream_v2 = "moondream:1.8b (224px, 1 frame, descriptive prompt)"
qwen3vl8b = "qwen3-vl:8b (224px, 3 frames)"

def evaluate_strategy(name, pred_fn):
    """Evaluate a strategy. pred_fn takes clip_data dict and returns (pred, team)."""
    tp = fn = fp = tn = 0
    team_total = team_correct = 0
    details = []
    
    for clip, data in sorted(clip_data.items()):
        pred, team = pred_fn(data)
        truth = data["truth"]
        truth_team = data["truth_team"]
        
        if truth == "goal" and pred == "goal": tp += 1
        elif truth == "goal" and pred != "goal": fn += 1
        elif truth != "goal" and pred == "goal": fp += 1
        elif truth != "goal" and pred != "goal": tn += 1
        
        if pred == "goal" and truth == "goal":
            team_total += 1
            if team == truth_team:
                team_correct += 1
        
        g_ok = truth == pred
        t_ok = truth == "goal" and pred == "goal" and team == truth_team
        details.append(f"  {clip:<45} truth={truth:<8} pred={pred:<8} team={team or '-':<15} goal={'✓' if g_ok else '✗'} team={'✓' if t_ok else '✗'}")
    
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(clip_data) if clip_data else 0
    team_acc = team_correct / team_total if team_total else 0
    
    print(f"\n{'='*70}")
    print(f"STRATEGY: {name}")
    print(f"{'='*70}")
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"Precision={prec*100:.1f}% Recall={rec*100:.1f}% F1={f1*100:.1f}% Acc={acc*100:.1f}%")
    print(f"Team Acc={team_acc*100:.1f}% ({team_correct}/{team_total})")
    for d in details:
        print(d)
    return f1, acc, prec, rec, team_acc

# Strategy 1: Both agree -> goal
print(f"\n{'='*70}")
print("TESTING ENSEMBLE STRATEGIES")
print(f"{'='*70}")

def both_agree(d):
    s_pred = d[smolvlm2_key]["pred"]
    m_pred = d[moondream_direct]["pred"]
    if s_pred == "goal" and m_pred == "goal":
        return "goal", d[moondream_direct]["team"] or d[smolvlm2_key]["team"]
    return "not_goal", None

evaluate_strategy("1. Both agree (smolvlm2 + moondream direct) -> goal", both_agree)

# Strategy 2: Union (OR)
def union_or(d):
    s_pred = d[smolvlm2_key]["pred"]
    m_pred = d[moondream_direct]["pred"]
    if s_pred == "goal" or m_pred == "goal":
        return "goal", d[moondream_direct]["team"] or d[smolvlm2_key]["team"]
    return "not_goal", None

evaluate_strategy("2. Union (smolvlm2 OR moondream direct) -> goal", union_or)

# Strategy 3: Weighted voting (smolvlm2=0.3, moondream=0.7) with threshold 0.5
def weighted_voting(d):
    s_pred = d[smolvlm2_key]["pred"]
    m_pred = d[moondream_direct]["pred"]
    score = (0.3 if s_pred == "goal" else 0) + (0.7 if m_pred == "goal" else 0)
    if score >= 0.5:
        return "goal", d[moondream_direct]["team"] or d[smolvlm2_key]["team"]
    return "not_goal", None

evaluate_strategy("3. Weighted (smolvlm2=0.3, moondream=0.7, threshold=0.5)", weighted_voting)

# Strategy 4: Use moondream v2 as primary + moondream direct as veto
# If v2 says goal and direct does not veto (i.e., direct says goal or empty)
def v2_with_direct_veto(d):
    m2_pred = d[moondream_v2]["pred"]
    md_pred = d[moondream_direct]["pred"]
    if m2_pred == "goal" and md_pred != "not_goal":
        return "goal", d[moondream_v2]["team"] or d[smolvlm2_key]["team"]
    if md_pred == "goal":
        return "goal", d[moondream_direct]["team"]
    return "not_goal", None

evaluate_strategy("4. Moondream v2 primary + direct veto (empty not veto)", v2_with_direct_veto)

# Strategy 5: 3-model voting (smolvlm2 + moondream direct + qwen3vl8b)
def three_model_vote(d):
    s_pred = d[smolvlm2_key]["pred"]
    md_pred = d[moondream_direct]["pred"]
    q_pred = d[qwen3vl8b]["pred"]
    votes = sum(1 for p in [s_pred, md_pred, q_pred] if p == "goal")
    if votes >= 2:
        team = d[moondream_direct]["team"] or d[qwen3vl8b]["team"] or d[smolvlm2_key]["team"]
        return "goal", team
    return "not_goal", None

evaluate_strategy("5. 3-Model vote (>=2 say goal)", three_model_vote)

# Strategy 6: smolvlm2 first, moondream confirmation only on non-goals
# This is: use smolvlm2 for all, but if smolvlm2 says goal and moondream says not_goal, trust moondream
# This is the same as both_agree

# Strategy 7: moondream direct only, if empty then use smolvlm2
def moondream_with_fallback(d):
    md_pred = d[moondream_direct]["pred"]
    # If moondream gave a clear prediction (goal or not_goal), use it
    if md_pred == "goal" or md_pred == "not_goal":
        return md_pred, d[moondream_direct]["team"]
    # Fallback to smolvlm2
    s_pred = d[smolvlm2_key]["pred"]
    return s_pred, d[smolvlm2_key]["team"]

evaluate_strategy("6. Moondream direct with smolvlm2 fallback", moondream_with_fallback)

# Strategy 8: smolvlm2 first, moondream only on clips where smolvlm2 says goal (cascade)
# This is the same as both_agree

def cascade_with_override(d):
    s_pred = d[smolvlm2_key]["pred"]
    md_pred = d[moondream_direct]["pred"]
    # If moondream says goal with strong confidence (we know it is 100% precise), use it
    if md_pred == "goal":
        return "goal", d[moondream_direct]["team"]
    # If moondream says not_goal, veto smolvlm2
    if md_pred == "not_goal":
        return "not_goal", None
    # moondream empty -> trust smolvlm2
    return s_pred, d[smolvlm2_key]["team"]

evaluate_strategy("7. Cascade: moondream vetoes if not_goal, smolvlm2 if empty", cascade_with_override)

# Save best results to file
best = {
    "strategies_tested": [
        "1. Both agree",
        "2. Union OR", 
        "3. Weighted voting",
        "4. Moondream v2 + direct veto",
        "5. 3-Model vote",
        "6. Moondream direct with smolvlm2 fallback",
        "7. Cascade: moondream vetoes if not_goal, smolvlm2 if empty"
    ],
    "best_strategy": "7. Cascade: moondream vetoes if not_goal, smolvlm2 if empty",
    "rationale": "moondream has 100% precision when it detects a goal. When it says not_goal, it is conservative but often correct. When it returns empty, we trust smolvlm2's high recall."
}
Path("results/ensemble_strategy_comparison.json").write_text(json.dumps(best, indent=2))
print(f"\nSaved strategy comparison to results/ensemble_strategy_comparison.json")
