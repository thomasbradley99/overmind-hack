#!/usr/bin/env python3
"""
Evaluate AI Output Against Ground Truth

Compares detected events to ground truth and outputs metrics.
Completely independent of how the AI output was generated.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def evaluate_goals(ai_output: dict, ground_truth: dict, tolerance: float = 30.0) -> dict:
    """
    Compare AI-detected goals to ground truth goals.
    
    Args:
        ai_output: AI output with events list
        ground_truth: Ground truth with events list
        tolerance: Seconds tolerance for matching (default 30s)
    
    Returns:
        Dict with metrics and detailed comparison
    """
    # Extract goals from both
    gt_events = ground_truth.get("events", [])
    gt_goals = [e for e in gt_events if e.get("action") == "Goal"]
    
    ai_events = ai_output.get("events", [])
    ai_goals = [e for e in ai_events if e.get("type") == "Goal" or e.get("action") == "Goal"]
    
    # Match goals
    matched_gt = set()
    matched_ai = set()
    matches = []
    
    for i, gt in enumerate(gt_goals):
        gt_time = gt.get("time", 0)
        for j, ai in enumerate(ai_goals):
            if j in matched_ai:
                continue
            ai_time = ai.get("time", 0)
            if abs(gt_time - ai_time) <= tolerance:
                matched_gt.add(i)
                matched_ai.add(j)
                matches.append({
                    "gt_time": gt_time,
                    "ai_time": ai_time,
                    "gt_team": gt.get("team", "?"),
                    "ai_team": ai.get("team", "?"),
                    "diff_seconds": round(ai_time - gt_time, 1)
                })
                break
    
    # Find misses and false positives
    missed = []
    for i, gt in enumerate(gt_goals):
        if i not in matched_gt:
            missed.append({
                "time": gt.get("time", 0),
                "team": gt.get("team", "?"),
                "description": gt.get("description", "")
            })
    
    false_positives = []
    for j, ai in enumerate(ai_goals):
        if j not in matched_ai:
            false_positives.append({
                "time": ai.get("time", 0),
                "team": ai.get("team", "?"),
                "description": ai.get("description", "")
            })
    
    # Calculate metrics
    tp = len(matched_gt)
    fp = len(false_positives)
    fn = len(missed)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "metrics": {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        },
        "ground_truth_count": len(gt_goals),
        "ai_detected_count": len(ai_goals),
        "matches": matches,
        "missed": missed,
        "false_positives": false_positives,
        "tolerance_seconds": tolerance
    }


def print_report(result: dict, ai_path: str, gt_path: str):
    """Print a human-readable evaluation report."""
    m = result["metrics"]
    
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"AI Output:     {ai_path}")
    print(f"Ground Truth:  {gt_path}")
    print(f"Tolerance:     {result['tolerance_seconds']}s")
    print()
    
    print("-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Ground Truth Goals: {result['ground_truth_count']}")
    print(f"AI Detected Goals:  {result['ai_detected_count']}")
    print()
    print(f"Precision: {m['precision']}%  (of AI detections, how many were real)")
    print(f"Recall:    {m['recall']}%  (of real goals, how many were found)")
    print(f"F1 Score:  {m['f1']}%")
    print()
    
    print("-" * 40)
    print("MATCHED GOALS")
    print("-" * 40)
    if result["matches"]:
        for match in result["matches"]:
            gt_t = format_time(match["gt_time"])
            ai_t = format_time(match["ai_time"])
            diff = match["diff_seconds"]
            sign = "+" if diff >= 0 else ""
            print(f"  OK  {gt_t} (GT) ~ {ai_t} (AI)  [{sign}{diff}s]  {match['gt_team']}")
    else:
        print("  (none)")
    print()
    
    print("-" * 40)
    print("MISSED GOALS (in GT, not detected by AI)")
    print("-" * 40)
    if result["missed"]:
        for miss in result["missed"]:
            t = format_time(miss["time"])
            print(f"  MISS  {t}  {miss['team']}  {miss['description'][:50]}")
    else:
        print("  (none)")
    print()
    
    print("-" * 40)
    print("FALSE POSITIVES (AI detected, not in GT)")
    print("-" * 40)
    if result["false_positives"]:
        for fp in result["false_positives"]:
            t = format_time(fp["time"])
            print(f"  FP  {t}  {fp['team']}  {fp.get('description', '')[:50]}")
    else:
        print("  (none)")
    print()
    
    print("=" * 60)
    print(f"FINAL F1: {m['f1']}%")
    print("=" * 60)


def save_report(result: dict, output_path: str, ai_path: str, gt_path: str):
    """Save evaluation result to JSON and TXT."""
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save TXT (human readable - unified timeline view)
    txt_path = output_path.replace('.json', '.txt')
    m = result["metrics"]
    
    # Build unified timeline
    events = []
    
    # Add matches (green)
    for match in result["matches"]:
        events.append({
            "time": match["gt_time"],
            "gt": f"{match['gt_team']}",
            "ai": f"{format_time(match['ai_time'])} {match['ai_team']}",
            "status": "✅",
            "type": "match"
        })
    
    # Add misses (red)
    for miss in result["missed"]:
        events.append({
            "time": miss["time"],
            "gt": f"{miss['team']}: {miss['description'][:30]}",
            "ai": "-",
            "status": "❌",
            "type": "miss"
        })
    
    # Add false positives (orange)
    for fp in result["false_positives"]:
        events.append({
            "time": fp["time"],
            "gt": "-",
            "ai": f"{fp['team']}: {fp.get('description', '')[:30]}",
            "status": "🟠",
            "type": "fp"
        })
    
    # Sort by time
    events.sort(key=lambda x: x["time"])
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Tolerance: {result['tolerance_seconds']}s\n\n")
        
        f.write(f"GT Goals: {result['ground_truth_count']}    ")
        f.write(f"AI Goals: {result['ai_detected_count']}    ")
        f.write(f"Matched: {m['true_positives']}\n\n")
        
        f.write(f"Precision: {m['precision']}%    ")
        f.write(f"Recall: {m['recall']}%    ")
        f.write(f"F1: {m['f1']}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TIMELINE\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Time':<8} {'Status':<4} {'Ground Truth':<30} {'AI Detection':<30}\n")
        f.write("-" * 80 + "\n")
        
        for e in events:
            t = format_time(e["time"])
            f.write(f"{t:<8} {e['status']:<4} {e['gt']:<30} {e['ai']:<30}\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"\n✅ = Match    ❌ = Missed (in GT, not detected)    🟠 = False Positive (AI hallucination)\n")
        f.write(f"\nFINAL F1: {m['f1']}%\n")
    
    print(f"\nSaved to: {output_path}")
    print(f"Saved to: {txt_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate AI output against ground truth")
    parser.add_argument('--ai-output', required=True, help='Path to AI output JSON (stage4_events.json)')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON')
    parser.add_argument('--tolerance', type=float, default=5.0, help='Match tolerance in seconds (default: 5)')
    parser.add_argument('--output', type=str, help='Save results to JSON file (default: auto-save next to ai-output)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    args = parser.parse_args()
    
    # Load files
    ai_output = load_json(args.ai_output)
    ground_truth = load_json(args.gt)
    
    # Evaluate
    result = evaluate_goals(ai_output, ground_truth, args.tolerance)
    
    # Print report
    print_report(result, args.ai_output, args.gt)
    
    # Auto-save by default (next to ai-output file)
    if not args.no_save:
        if args.output:
            output_path = args.output
        else:
            # Auto-save next to the AI output
            ai_dir = Path(args.ai_output).parent
            output_path = ai_dir / "eval_results.json"
        save_report(result, str(output_path), args.ai_output, args.gt)


if __name__ == '__main__':
    main()
