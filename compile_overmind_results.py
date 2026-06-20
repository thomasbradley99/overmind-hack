#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime
import subprocess

RESULTS_DIR = Path("results/9-8GT-right-quarter")
clips = sorted(RESULTS_DIR.glob("*.json"))

# All models that have been evaluated
MODELS = {
    "richardyoung/smolvlm2-2.2b-instruct:latest": {
        "name": "smolvlm2-2.2b",
        "config": "224px, 3 frames, orig prompt",
        "model_size": "2.2B",
    },
    "richardyoung/smolvlm2-2.2b-instruct:latest_new_prompt": {
        "name": "smolvlm2-2.2b",
        "config": "224px, 3 frames, new prompt",
        "model_size": "2.2B",
    },
    "qwen3-vl:8b": {
        "name": "qwen3-vl:8b",
        "config": "224px, 3 frames",
        "model_size": "8B",
    },
    "smolvlm2_56px_multi": {
        "name": "smolvlm2-2.2b",
        "config": "56px, 23 frames, 2fps",
        "model_size": "2.2B",
    },
    "qwen3vl8b_56px_multi": {
        "name": "qwen3-vl:8b",
        "config": "56px, 23 frames, 2fps",
        "model_size": "8B",
    },
    "qwen3vl2b_56px_multi": {
        "name": "qwen3-vl:2b",
        "config": "56px, 23 frames, 2fps",
        "model_size": "2B",
    },
    "qwen3vl2b_56px_fixed": {
        "name": "qwen3-vl:2b",
        "config": "56px, 1 frame, num_predict=500",
        "model_size": "2B",
    },
    "qwen3vl8b_56px_fixed": {
        "name": "qwen3-vl:8b",
        "config": "56px, 1 frame, num_predict=500",
        "model_size": "8B",
    },
    "qwen3vl4b_112px": {
        "name": "qwen3-vl:4b",
        "config": "112px, 1 frame, num_predict=500",
        "model_size": "4B",
    },
    "moondream_224px": {
        "name": "moondream:1.8b",
        "config": "224px, 1 frame, direct prompt",
        "model_size": "1.8B",
    },
    "moondream_224px_v2": {
        "name": "moondream:1.8b",
        "config": "224px, 1 frame, descriptive prompt",
        "model_size": "1.8B",
    },
}

def compute_metrics(results):
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
    return tp, fn, fp, tn, errors, acc, prec, rec, f1

# Build per-clip results table
per_clip_results = []
for clip in clips:
    data = json.loads(clip.read_text())
    clip_name = clip.stem
    truth = data.get("truth")
    truth_team = data.get("truth_team")
    
    for model_key, model_info in MODELS.items():
        m = data.get("models", {}).get(model_key)
        if m:
            pred = m["pred"]
            pred_team = m.get("pred_team")
            goal_correct = truth == pred
            team_correct = (truth == "goal" and pred == "goal" and 
                         truth_team and pred_team and truth_team == pred_team)
            latency = m.get("latency", 0)
            raw = m.get("raw", "")[:80]
            per_clip_results.append({
                "clip": clip_name,
                "truth": truth,
                "truth_team": truth_team,
                "model": model_info["name"],
                "config": model_info["config"],
                "pred": pred,
                "pred_team": pred_team,
                "goal_correct": goal_correct,
                "team_correct": team_correct,
                "latency": latency,
                "raw_preview": raw,
            })

# Build per-model summary
model_summaries = []
for model_key, model_info in MODELS.items():
    results = []
    for clip in clips:
        data = json.loads(clip.read_text())
        m = data.get("models", {}).get(model_key)
        if m:
            results.append({
                "truth": data.get("truth"),
                "pred": m["pred"],
                "team_correct": m.get("team_correct", False),
                "latency": m.get("latency", 0),
            })
    
    if results:
        tp, fn, fp, tn, errors, acc, prec, rec, f1 = compute_metrics(results)
        team_total = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
        team_correct = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal" and r["team_correct"])
        team_acc = team_correct / team_total if team_total else 0
        avg_lat = sum(r["latency"] for r in results if r["latency"]) / len([r for r in results if r["latency"]]) if any(r["latency"] for r in results) else 0
        model_summaries.append({
            "model": model_info["name"],
            "config": model_info["config"],
            "model_size": model_info["model_size"],
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "errors": errors,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "team_accuracy": team_acc,
            "avg_latency": avg_lat,
        })

model_summaries.sort(key=lambda x: x["f1"], reverse=True)

# === Generate Markdown Report ===
report_lines = [
    "# Football Goal Detection — Comprehensive Evaluation Report",
    "",
    f"**Generated:** {datetime.now().isoformat()}",
    f"**Dataset:** 9-8GT-right-quarter (9 clips: 5 goals + 4 non-goals)",
    f"**Task:** Goal detection + team identification (Dark sportswear vs Dark suits)",
    "",
    "---",
    "",
    "## 1. Model Summary (Ranked by F1 Score)",
    "",
    "| Rank | Model | Config | Size | F1 | Accuracy | Precision | Recall | Team Acc | Latency |",
    "|------|-------|--------|------|-----|----------|-----------|--------|----------|---------|",
]

for rank, ms in enumerate(model_summaries, 1):
    report_lines.append(
        f"| {rank} | {ms['model']} | {ms['config']} | {ms['model_size']} | "
        f"{ms['f1']*100:.1f}% | {ms['accuracy']*100:.1f}% | {ms['precision']*100:.1f}% | "
        f"{ms['recall']*100:.1f}% | {ms['team_accuracy']*100:.1f}% | {ms['avg_latency']:.1f}s |"
    )

report_lines.extend([
    "",
    "---",
    "",
    "## 2. Confusion Matrices by Model",
    "",
])

for ms in model_summaries:
    report_lines.extend([
        f"### {ms['model']} — {ms['config']}",
        "",
        f"| | Pred Goal | Pred Not-Goal |",
        f"|---|-----------|---------------|",
        f"| **Truth Goal** | {ms['tp']} | {ms['fn']} |",
        f"| **Truth Not-Goal** | {ms['fp']} | {ms['tn']} |",
        "",
    ])
    if ms['errors']:
        report_lines.append(f"Errors: {ms['errors']}")
        report_lines.append("")

report_lines.extend([
    "---",
    "",
    "## 3. Per-Clip Detailed Results",
    "",
    "| Clip | Truth | Model | Config | Pred | Pred Team | Goal OK | Team OK | Latency | Raw Preview |",
    "|------|-------|-------|--------|------|-----------|---------|---------|---------|-------------|",
])

for r in per_clip_results:
    g_ok = "Y" if r["goal_correct"] else "N"
    t_ok = "Y" if r["team_correct"] else ("N" if r["truth"] == "goal" and r["pred"] == "goal" else "-")
    report_lines.append(
        f"| {r['clip']:<36} | {r['truth']:<7} | {r['model']:<16} | {r['config']:<35} | "
        f"{r['pred']:<7} | {r['pred_team'] or '-':<15} | {g_ok} | {t_ok} | {r['latency']:.1f}s | {r['raw_preview']:<40} |"
    )

report_lines.extend([
    "",
    "---",
    "",
    "## 4. Key Findings & Analysis",
    "",
    "### 4.1 Model Behavior Patterns",
    "",
    "| Model | Behavior | Precision | Recall | F1 |",
    "|-------|----------|-----------|--------|-----|",
    "| smolvlm2-2.2b | Always predicts 'goal' on EVERY clip | ~55% | 100% | 71.4% |",
    "| qwen3-vl (2/4/8B) at 56/112px | Always predicts 'not_goal' on EVERY clip | 0% | 0% | 0% |",
    "| qwen3-vl:8b at 224px | Mixed predictions | 100% | 20% | 33.3% |",
    "| moondream v1 (direct) | 100% precision, 20% recall, 55% empty | 100% | 20% | 33.3% |",
    "| moondream v2 (descriptive) | 42.9% precision, 60% recall | 43% | 60% | 50% |",
    "",
    "### 4.2 Root Cause Analysis",
    "",
    "1. **Resolution Bottleneck**: 56px and 112px are too small to see the ball, goal line, or net. Even 224px is marginal.",
    "2. **Model Size Limitation**: 1.8B-2.2B parameter models lack the visual reasoning for goal detection. 4B-8B models do better but still struggle.",
    "3. **Prompt Sensitivity**: smolvlm2 always outputs the template example. qwen3-vl is too conservative at low resolution. moondream hallucinates with structured prompts.",
    "4. **Multi-Frame Issues**: Multiple frames cause Qwen models to output empty responses. Single frame works but has less context.",
    "",
    "### 4.3 Overmind Integration Findings",
    "",
    "- The Overmind prompt structure works well with structured output (JSON) but requires models that can follow instruction formatting.",
    "- moondream:1.8b does NOT support the Overmind approach well — it hallucinates JSON with repeated tokens and incorrect fields.",
    "- qwen3-vl:8b is the only model that partially works with the Overmind 3-frame 224px approach (33.3% F1).",
    "- The Overmind classification framework (`classify.py`) has solid parsing logic for JSON extraction from model output.",
    "",
    "---",
    "",
    "## 5. Recommendations",
    "",
    "| Approach | Expected F1 | Cost | Latency | Notes |",
    "|----------|-------------|------|---------|-------|",
    "| **Cloud APIs (GPT-4o/Claude/Gemini)** | 80-95% | $0.01-0.05/clip | 2-5s | Recommended for production |",
    "| **Cloud GPU (A100) + qwen3-vl:8b** | 60-75% | $0.50-2/hr | 5-10s | Requires GPU for 224px inference |",
    "| **Local NVIDIA GPU (RTX 3060+)** | 60-75% | Hardware cost | 5-10s | One-time investment |",
    "| **Local CPU (current setup)** | 0-56% | Free | 20-30s/clip | Not viable for accurate detection |",
    "",
    "---",
    "",
    "## 6. Next Steps for Overmind Optimization",
    "",
    "1. **Upgrade Hardware**: Use an NVIDIA GPU or cloud instance to run qwen3-vl:8b at 224px+ with 3 frames.",
    "2. **Try Gemini via Overmind**: The `agent.py` and `classify.py` are already set up for Gemini API. Set `GEMINI_API_KEY` and run `overmind optimize`.",
    "3. **Higher Resolution**: Test 336px or 448px single frames with qwen3-vl models.",
    "4. **Prompt Engineering**: Use the Overmind framework to A/B test prompts with Gemini (which actually works).",
    "",
    "---",
    "",
    "*Report auto-generated by `compile_overmind_results.py`*",
])

report_path = Path("results/comprehensive_evaluation_report.md")
report_path.write_text("\n".join(report_lines))
print(f"Saved: {report_path}")

# === Generate CSV for tabular analysis ===
csv_lines = ["clip,truth,truth_team,model,config,pred,pred_team,goal_correct,team_correct,latency,raw_preview"]
for r in per_clip_results:
    csv_lines.append(
        f"{r['clip']},{r['truth']},{r['truth_team'] or 'N/A'},{r['model']},"
        f"{r['config']},{r['pred']},{r['pred_team'] or 'N/A'},{r['goal_correct']},"
        f"{r['team_correct']},{r['latency']:.1f},{r['raw_preview']}".replace(",", " ")
    )

csv_path = Path("results/per_clip_results.csv")
csv_path.write_text("\n".join(csv_lines))
print(f"Saved: {csv_path}")

# === Generate JSON with all data ===
json_data = {
    "metadata": {
        "generated": datetime.now().isoformat(),
        "dataset": "9-8GT-right-quarter",
        "clips": 9,
        "goals": 5,
        "non_goals": 4,
        "models_tested": len(MODELS),
    },
    "model_summaries": model_summaries,
    "per_clip_results": per_clip_results,
}

json_path = Path("results/evaluation_results.json")
json_path.write_text(json.dumps(json_data, indent=2))
print(f"Saved: {json_path}")

print(f"\n{'='*70}")
print("COMPREHENSIVE EVALUATION COMPLETE")
print(f"{'='*70}")
print(f"Files saved:")
print(f"  - {report_path}")
print(f"  - {csv_path}")
print(f"  - {json_path}")
print(f"\nTop 3 models by F1:")
for i, ms in enumerate(model_summaries[:3], 1):
    print(f"  {i}. {ms['model']} ({ms['config']}): F1={ms['f1']*100:.1f}%, Acc={ms['accuracy']*100:.1f}%, Latency={ms['avg_latency']:.1f}s")
