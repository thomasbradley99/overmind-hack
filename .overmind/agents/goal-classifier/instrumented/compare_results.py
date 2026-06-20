#!/usr/bin/env python3
"""Compare model predictions across all clip result files."""
import json
from pathlib import Path
import sys


def compare_dataset(dataset: str = "9-8GT-right"):
    results_dir = Path("results") / dataset
    if not results_dir.exists():
        print(f"No results found at {results_dir}")
        sys.exit(1)

    clip_files = sorted(results_dir.glob("*.json"))
    all_models = set()
    clips = []
    for cf in clip_files:
        data = json.loads(cf.read_text())
        models = data.get("models", {})
        all_models.update(models.keys())
        clips.append({
            "clip": data["clip"],
            "truth": data["truth"],
            "truth_team": data.get("truth_team"),
            "models": models,
        })

    all_models = sorted(all_models)
    # Split into per-model (all clips) and overlapping-clips metrics
    overlapping_clips = [c for c in clips if all(m in c["models"] for m in all_models)]
    n_overlap = len(overlapping_clips)

    print(f"\nDataset: {dataset}")
    print(f"Total clips in data: {len(clip_files)}")
    print(f"Models compared: {len(all_models)}  ({', '.join(all_models)})")
    print(f"Overlapping clips (all models present): {n_overlap}")

    # Helper to compute metrics for a model over a specific clip list
    def _model_metrics(model, clip_list):
        tp = fn = fp = tn = 0
        team_correct = team_total = 0
        for c in clip_list:
            m = c["models"].get(model)
            if not m:
                continue
            truth = c["truth"]
            pred = m["pred"]
            if truth == "goal" and pred == "goal":
                tp += 1
            elif truth == "goal" and pred == "not_goal":
                fn += 1
            elif truth == "not_goal" and pred == "goal":
                fp += 1
            elif truth == "not_goal" and pred == "not_goal":
                tn += 1
            if pred == "goal" and truth == "goal":
                team_total += 1
                if m.get("team_correct"):
                    team_correct += 1
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        team_acc = team_correct / team_total if team_total else 0
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0
        return {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "prec": prec, "rec": rec, "f1": f1, "acc": acc,
            "team_total": team_total, "team_correct": team_correct, "team_acc": team_acc,
        }

    def _print_cm(title, metrics):
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        print(f"Confusion Matrix")
        print(f"{' '*16}  Pred goal  Pred not-goal")
        print(f"  Truth goal        {metrics['tp']:^8}      {metrics['fn']:^8}")
        print(f"  Truth not-goal    {metrics['fp']:^8}      {metrics['tn']:^8}")
        print(f"-"*70)
        print(f"  Goal Detection")
        print(f"    Precision : {metrics['prec']*100:.1f}%")
        print(f"    Recall    : {metrics['rec']*100:.1f}%")
        print(f"    F1        : {metrics['f1']*100:.1f}%")
        print(f"    Accuracy  : {metrics['acc']*100:.1f}%")
        print(f"  Team Assignment")
        print(f"    Correct on detected goals: {metrics['team_correct']}/{metrics['team_total']}")
        print(f"    Accuracy  : {metrics['team_acc']*100:.1f}%")

    # Per-model on ALL clips that model has data for
    for model in all_models:
        model_clips = [c for c in clips if model in c["models"]]
        m = _model_metrics(model, model_clips)
        _print_cm(f"Model: {model}  (all clips evaluated: {len(model_clips)})", m)

    # Overlapping comparison table
    # Overlapping comparison table with confusion matrices and clip names
    if n_overlap > 0 and len(all_models) > 1:
        print(f"\n{'='*70}")
        print(f"FAIR COMPARISON — {n_overlap} OVERLAPPING CLIPS")
        print(f"{'='*70}")
        print("Clips evaluated:")
        for c in overlapping_clips:
            print(f"  - {c['clip']}")

        for model in all_models:
            m = _model_metrics(model, overlapping_clips)
            print(f"\n{'='*70}")
            print(f"Model: {model}")
            print(f"{'='*70}")
            print(f"Confusion Matrix (on overlapping clips only)")
            print(f"{' '*16}  Pred goal  Pred not-goal")
            print(f"  Truth goal        {m['tp']:^8}      {m['fn']:^8}")
            print(f"  Truth not-goal    {m['fp']:^8}      {m['tn']:^8}")
            print(f"-"*70)
            print(f"  Goal Detection")
            print(f"    Precision : {m['prec']*100:.1f}%")
            print(f"    Recall    : {m['rec']*100:.1f}%")
            print(f"    F1        : {m['f1']*100:.1f}%")
            print(f"    Accuracy  : {m['acc']*100:.1f}%")
            print(f"  Team Assignment")
            print(f"    Correct on detected goals: {m['team_correct']}/{m['team_total']}")
            print(f"    Accuracy  : {m['team_acc']*100:.1f}%")

        print(f"\n{'='*70}")
        print(f"Summary Table (on overlapping clips only)")
        print(f"{'='*70}")
        print(f"{'Model':<50} {'Goal F1':>8} {'Team Acc':>8} {'Accuracy':>8}")
        print(f"{'='*70}")
        for model in all_models:
            m = _model_metrics(model, overlapping_clips)
            print(f"{model:<50} {m['f1']*100:>7.1f}% {m['team_acc']*100:>7.1f}% {m['acc']*100:>7.1f}%")

    # Per-clip breakdown (showing only overlapping if >1 model)
    display_clips = overlapping_clips if n_overlap > 0 and len(all_models) > 1 else clips
    print(f"\n{'='*70}")
    if len(all_models) > 1 and n_overlap > 0:
        print(f"Per-clip Breakdown (overlapping clips only)")
    else:
        print("Per-clip Breakdown")
    print(f"{'='*70}")
    for c in display_clips:
        truth = c["truth"]
        t_team = c.get("truth_team", "-")
        print(f"\n{c['clip']:<45}  truth: {truth:<8} team: {t_team}")
        for model in all_models:
            m = c["models"].get(model)
            if not m:
                print(f"  {model:<45}  [NO DATA]")
                continue
            pred = m["pred"]
            p_team = m.get("pred_team", "-")
            g_ok = m.get("goal_correct", False)
            t_ok = m.get("team_correct", False)
            g_mark = "✓" if g_ok else "✗"
            t_mark = "✓" if t_ok else "✗"
            print(f"  {model:<45}  pred: {pred:<8} team: {p_team}  [goal:{g_mark} team:{t_mark}]")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="9-8GT-right")
    args = p.parse_args()
    compare_dataset(args.dataset)
