#!/usr/bin/env python3
"""Compare model predictions against ground truth from /data folder."""
import json
from pathlib import Path
import sys


def compare_dataset(dataset: str = "9-8GT-right"):
    data_dir = Path("data") / dataset
    results_dir = Path("results") / dataset

    if not data_dir.exists():
        print(f"No data found at {data_dir}")
        sys.exit(1)

    # Load all ground truth clips from /data
    all_clips = sorted(data_dir.glob("*.mp4"))
    gt_data = {}
    for clip in all_clips:
        label_file = clip.with_suffix(".json")
        if label_file.exists():
            gt = json.loads(label_file.read_text())
            truth = "goal" if gt.get("label") == "goal" or gt.get("action") == "Goal" else "not_goal"
            gt_data[clip.name] = {
                "truth": truth,
                "truth_team": gt.get("team"),
            }

    print(f"\nDataset: {dataset}")
    print(f"Total clips in /data: {len(all_clips)}")
    print(f"Clips with ground truth: {len(gt_data)}")

    # Load results from per-clip JSON files
    clip_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []
    all_models = set()
    model_results = {}  # model -> {clip_name -> result}
    for cf in clip_files:
        data = json.loads(cf.read_text())
        clip_name = data.get("clip")
        models = data.get("models", {})
        all_models.update(models.keys())
        for model, result in models.items():
            if model not in model_results:
                model_results[model] = {}
            model_results[model][clip_name] = result

    all_models = sorted(all_models)
    if not all_models:
        print("No model results found.")
        return

    print(f"Models compared: {len(all_models)}  ({', '.join(all_models)})")

    def _compute_metrics(model, clip_names):
        """Compute confusion matrix metrics for a model on specific clips."""
        tp = fn = fp = tn = 0
        team_correct = team_total = 0
        evaluated_clips = []

        for clip_name in clip_names:
            if clip_name not in gt_data:
                continue
            gt = gt_data[clip_name]
            result = model_results.get(model, {}).get(clip_name)
            if not result:
                continue

            evaluated_clips.append(clip_name)
            truth = gt["truth"]
            pred = result["pred"]
            truth_team = gt.get("truth_team")
            pred_team = result.get("pred_team")

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
                if pred_team == truth_team:
                    team_correct += 1

        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0
        team_acc = team_correct / team_total if team_total else 0

        return {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "prec": prec, "rec": rec, "f1": f1, "acc": acc,
            "team_total": team_total, "team_correct": team_correct, "team_acc": team_acc,
            "evaluated_clips": evaluated_clips,
        }

    def _print_cm(model, metrics, title_suffix=""):
        print(f"\n{'='*70}")
        print(f"Model: {model} {title_suffix}")
        print(f"{'='*70}")
        print(f"Clips evaluated ({len(metrics['evaluated_clips'])}):")
        for clip in metrics["evaluated_clips"]:
            gt = gt_data[clip]
            print(f"  - {clip:<45} truth: {gt['truth']:<8} team: {gt.get('truth_team', '-')}")
        print(f"\nConfusion Matrix")
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

    # Per-model metrics on ALL clips that model was evaluated on
    for model in all_models:
        clip_names = sorted(model_results[model].keys())
        metrics = _compute_metrics(model, clip_names)
        _print_cm(model, metrics, f"(all clips evaluated: {len(clip_names)})")

    # Find overlapping clips (all models have results)
    overlapping = set(model_results[all_models[0]].keys())
    for model in all_models[1:]:
        overlapping &= set(model_results[model].keys())
    overlapping = sorted(overlapping)
    n_overlap = len(overlapping)

    if n_overlap > 0 and len(all_models) > 1:
        print(f"\n{'='*70}")
        print(f"FAIR COMPARISON — {n_overlap} OVERLAPPING CLIPS")
        print(f"{'='*70}")
        print("Overlapping clips:")
        for clip in overlapping:
            gt = gt_data[clip]
            print(f"  - {clip:<45} truth: {gt['truth']:<8} team: {gt.get('truth_team', '-')}")

        for model in all_models:
            metrics = _compute_metrics(model, overlapping)
            _print_cm(model, metrics, "(overlapping clips)")

        print(f"\n{'='*70}")
        print(f"Summary Table (on overlapping clips only)")
        print(f"{'='*70}")
        print(f"{'Model':<50} {'Goal F1':>8} {'Team Acc':>8} {'Accuracy':>8}")
        print(f"{'='*70}")
        for model in all_models:
            metrics = _compute_metrics(model, overlapping)
            print(f"{model:<50} {metrics['f1']*100:>7.1f}% {metrics['team_acc']*100:>7.1f}% {metrics['acc']*100:>7.1f}%")

    # Per-clip breakdown
    display_clips = overlapping if n_overlap > 0 and len(all_models) > 1 else sorted(gt_data.keys())
    print(f"\n{'='*70}")
    print(f"Per-clip Breakdown")
    print(f"{'='*70}")
    for clip_name in display_clips:
        if clip_name not in gt_data:
            continue
        gt = gt_data[clip_name]
        print(f"\n{clip_name:<45}  truth: {gt['truth']:<8} team: {gt.get('truth_team', '-')}")
        for model in all_models:
            result = model_results.get(model, {}).get(clip_name)
            if not result:
                print(f"  {model:<45}  [NOT EVALUATED]")
                continue
            pred = result["pred"]
            pred_team = result.get("pred_team", "-")
            truth = gt["truth"]
            truth_team = gt.get("truth_team")
            g_ok = pred == truth
            t_ok = (truth != "goal" or pred != "goal" or pred_team == truth_team)
            g_mark = "✓" if g_ok else "✗"
            t_mark = "✓" if t_ok else "✗"
            print(f"  {model:<45}  pred: {pred:<8} team: {pred_team}  [goal:{g_mark} team:{t_mark}]")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="9-8GT-right")
    args = p.parse_args()
    compare_dataset(args.dataset)
