#!/usr/bin/env python3
"""
Ensemble evaluation: Combine smolvlm2 (recall) + moondream (precision/team ID)

Strategy: Cascading ensemble
1. Run smolvlm2 first (fast, ~1s) — catches all goals but many false positives
2. Run moondream on clips where smolvlm2 says "goal" — filters false positives + provides team ID
3. Final decision:
   - Both say "goal" → CONFIRMED goal, use moondream's team (100% team accuracy)
   - smolvlm2 says "goal", moondream says "not_goal" → REJECTED (smolvlm2 hallucinating)
   - smolvlm2 says "goal", moondream empty → LOW confidence, flag for review
   - Either says "not_goal" → not a goal

This is designed to run distributed: smolvlm2 on local machine, moondream on remote.
"""
import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import base64
from PIL import Image
import io

OLLAMA_URL = "http://localhost:11434"

def encode_image(path: str) -> str:
    """Encode image to base64 for Ollama API."""
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def ollama_chat(model: str, prompt: str, image_path: Optional[str] = None, 
                remote_url: Optional[str] = None, timeout: int = 120) -> str:
    """Call Ollama chat API. Can use remote URL for distributed setup."""
    url = (remote_url or OLLAMA_URL) + "/api/chat"
    messages = [{"role": "user", "content": prompt}]
    if image_path:
        b64 = encode_image(image_path)
        messages[0]["images"] = [b64]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 500}
    }
    
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]

def parse_goal_response(text: str) -> Tuple[str, Optional[str]]:
    """Extract goal bool and team from model response."""
    t = text.lower()
    
    # JSON extraction
    if '"goal"' in t or '"goal":' in text.lower():
        try:
            import re
            m = re.search(r'\{[^}]*"goal"[^}]*\}', text, re.DOTALL | re.IGNORECASE)
            if m:
                j = json.loads(m.group())
                g = j.get("goal", j.get("is_goal", False))
                pred = "goal" if g else "not_goal"
                team = j.get("team", j.get("team_name", j.get("scoring_team")))
                return pred, team
        except Exception:
            pass
    
    # Keyword fallback
    if "goal" in t and "not_goal" not in t and "no goal" not in t and "not a goal" not in t:
        # Check for team names
        if "dark sportswear" in t: return "goal", "Dark sportswear"
        if "dark suits" in t: return "goal", "Dark suits"
        return "goal", None
    if "not_goal" in t or "no goal" in t or "not a goal" in t or ("goal" not in t and len(t) < 50):
        return "not_goal", None
    return "not_goal", None


def run_ensemble_local(clip_dir: str = "data/9-8GT-right-quarter", 
                       results_dir: str = "results/9-8GT-right-quarter",
                       moondream_remote: Optional[str] = None) -> Dict:
    """
    Run cascading ensemble locally.
    
    Args:
        clip_dir: Directory with clips and JSON labels
        results_dir: Directory to save results
        moondream_remote: If set, URL of remote Ollama instance (e.g., "http://192.168.1.100:11434")
    
    Returns:
        Dict with ensemble results
    """
    clip_path = Path(clip_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load clips
    clips = sorted(clip_path.glob("*.mp4"))
    print(f"Found {len(clips)} clips in {clip_dir}")
    
    ensemble_results = {
        "strategy": "cascading_ensemble",
        "models": ["smolvlm2-2.2b", "moondream:1.8b"],
        "moondream_remote": moondream_remote,
        "clips": [],
        "aggregate": {}
    }
    
    # Prepare prompts
    smolvlm2_prompt = """Analyze this football clip and determine if a goal was scored. 
Return JSON: {"goal": true/false, "team": "team name"}
"""
    
    moondream_prompt = """You are a football referee analyzing a video clip. Determine if a goal was scored and identify the scoring team.

Look for these signs of a goal:
- Ball crossing the goal line into the net
- Players celebrating (arms raised, jumping, hugging)
- Goalkeeper retrieving ball from net
- Ball hitting the back of the net

Teams:
- "Dark sportswear" — players in dark athletic wear, shorts, jerseys
- "Dark suits" — players in dark formal suits, blazers, trousers

Return ONLY a JSON object with no extra text or markdown:
{"goal": true/false, "team": "team name or null"}
"""

    for clip in clips:
        label_file = clip.with_suffix(".json")
        if not label_file.exists():
            continue
            
        gt = json.loads(label_file.read_text())
        truth = "goal" if gt.get("action") == "Goal" or gt.get("label") == "goal" else "not_goal"
        truth_team = gt.get("team", gt.get("truth_team"))
        
        print(f"\n{'='*60}")
        print(f"Clip: {clip.name}")
        print(f"Truth: {truth} | Team: {truth_team}")
        
        # Stage 1: Extract frame and run smolvlm2 (fast, local)
        print("  Stage 1: smolvlm2-2.2b...")
        frame_path = extract_frame(str(clip))
        t0 = time.time()
        smolvlm2_raw = ollama_chat("richardyoung/smolvlm2-2.2b-instruct:latest", 
                                    smolvlm2_prompt, frame_path)
        smolvlm2_time = time.time() - t0
        smolvlm2_pred, smolvlm2_team = parse_goal_response(smolvlm2_raw)
        print(f"    -> {smolvlm2_pred} | Team: {smolvlm2_team} | {smolvlm2_time:.1f}s")
        
        # Stage 2: Run moondream (precision/team, can be remote)
        print(f"  Stage 2: moondream:1.8b (remote={moondream_remote is not None})...")
        t0 = time.time()
        moondream_raw = ollama_chat("moondream:1.8b", moondream_prompt, frame_path,
                                     remote_url=moondream_remote)
        moondream_time = time.time() - t0
        moondream_pred, moondream_team = parse_goal_response(moondream_raw)
        print(f"    -> {moondream_pred} | Team: {moondream_team} | {moondream_time:.1f}s")
        
        # Stage 3: Ensemble decision
        if smolvlm2_pred == "goal" and moondream_pred == "goal":
            # Both agree — HIGH confidence, use moondream team
            final_pred = "goal"
            final_team = moondream_team or smolvlm2_team
            confidence = "high"
        elif smolvlm2_pred == "goal" and moondream_pred == "not_goal":
            # Disagreement — moondream says no goal, smolvlm2 hallucinating
            final_pred = "not_goal"
            final_team = None
            confidence = "medium"
        elif smolvlm2_pred == "goal" and moondream_pred == "not_goal":
            # moondream empty / no response
            final_pred = "goal"  # Trust smolvlm2 recall
            final_team = smolvlm2_team
            confidence = "low"
        else:
            final_pred = "not_goal"
            final_team = None
            confidence = "high" if moondream_pred == "not_goal" else "medium"
        
        print(f"  ENSEMBLE: {final_pred} | Team: {final_team} | Confidence: {confidence}")
        
        result = {
            "clip": clip.name,
            "truth": truth,
            "truth_team": truth_team,
            "smolvlm2": {"pred": smolvlm2_pred, "team": smolvlm2_team, "latency": smolvlm2_time, "raw": smolvlm2_raw[:200]},
            "moondream": {"pred": moondream_pred, "team": moondream_team, "latency": moondream_time, "raw": moondream_raw[:200], "remote": moondream_remote is not None},
            "ensemble": {"pred": final_pred, "team": final_team, "confidence": confidence}
        }
        ensemble_results["clips"].append(result)
        
        # Save per-clip result
        out_file = results_path / f"ensemble_{clip.stem}.json"
        out_file.write_text(json.dumps(result, indent=2))
    
    # Compute aggregate metrics
    tp = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] == "goal")
    fn = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] != "goal")
    fp = sum(1 for r in ensemble_results["clips"] if r["truth"] != "goal" and r["ensemble"]["pred"] == "goal")
    tn = sum(1 for r in ensemble_results["clips"] if r["truth"] != "goal" and r["ensemble"]["pred"] != "goal")
    
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(ensemble_results["clips"]) if ensemble_results["clips"] else 0
    
    team_total = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] == "goal")
    team_correct = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] == "goal" and r["ensemble"]["team"] == r["truth_team"])
    team_acc = team_correct / team_total if team_total else 0
    
    ensemble_results["aggregate"] = {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "team_total": team_total, "team_correct": team_correct, "team_accuracy": team_acc
    }
    
    # Save full results
    (results_path / "ensemble_results.json").write_text(json.dumps(ensemble_results, indent=2))
    
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Confusion Matrix:")
    print(f"  TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
    print(f"Metrics:")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Team Acc:  {team_acc*100:.1f}%")
    
    return ensemble_results


def extract_frame(video_path: str, output_path: Optional[str] = None) -> str:
    """Extract a representative frame from video."""
    import subprocess
    if output_path is None:
        output_path = video_path.replace(".mp4", "_frame.jpg")
    
    # Get middle frame
    cmd = ["ffmpeg", "-i", video_path, "-vf", "select='eq(n,0)'", 
           "-vframes", "1", "-y", "-q:v", "2", output_path]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except Exception:
        # Fallback: just take first frame
        cmd = ["ffmpeg", "-ss", "00:00:01", "-i", video_path, 
               "-vframes", "1", "-y", "-q:v", "2", output_path]
        subprocess.run(cmd, capture_output=True, timeout=30)
    
    return output_path


def run_ensemble_simulated(clip_dir: str = "data/9-8GT-right-quarter",
                           results_dir: str = "results/9-8GT-right-quarter") -> Dict:
    """
    Simulate ensemble using pre-computed results from evaluation_results.json.
    This is much faster and doesn't require running the models again.
    """
    results_path = Path(results_dir)
    
    # Load pre-computed results
    eval_file = Path("results/evaluation_results.json")
    if not eval_file.exists():
        print(f"Pre-computed results not found at {eval_file}")
        print("Run run_ensemble_local() to generate actual results")
        return {}
    
    eval_data = json.loads(eval_file.read_text())
    
    # Build lookup from per-clip results
    # We need to map the model keys to smolvlm2 and moondream configs
    smolvlm2_key = "smolvlm2_56px_multi"  # or "richardyoung/smolvlm2-2.2b-instruct:latest"
    moondream_direct_key = "moondream_224px"
    moondream_desc_key = "moondream_224px_v2"
    
    # Try to find matching keys
    available_models = set()
    for clip_result in eval_data.get("per_clip_results", []):
        available_models.add(f"{clip_result['model']} ({clip_result['config']})")
    
    print(f"Available model configs in pre-computed data:")
    for m in sorted(available_models):
        print(f"  - {m}")
    
    # Simulate with specific model configs
    # smolvlm2: 56px, 23 frames, 2fps (fastest config with same behavior)
    # moondream: direct prompt (descriptive has too many false positives)
    
    ensemble_results = {
        "strategy": "cascading_ensemble_simulated",
        "models": ["smolvlm2-2.2b (56px, 23 frames, 2fps)", "moondream:1.8b (224px, 1 frame, direct prompt)"],
        "clips": [],
        "aggregate": {}
    }
    
    # Group per-clip results by clip name
    clip_results = {}
    for r in eval_data.get("per_clip_results", []):
        clip = r["clip"]
        if clip not in clip_results:
            clip_results[clip] = {}
        model_key = f"{r['model']} ({r['config']})"
        clip_results[clip][model_key] = r
    
    # For each clip, apply ensemble logic
    for clip_name, models in sorted(clip_results.items()):
        # Find smolvlm2 result (any config works, they all behave same)
        smolvlm2_results = [v for k, v in models.items() if "smolvlm2" in k]
        # Find moondream result (direct prompt preferred)
        moondream_results = [v for k, v in models.items() if "moondream" in k and "direct" in k]
        if not moondream_results:
            moondream_results = [v for k, v in models.items() if "moondream" in k]
        
        if not smolvlm2_results or not moondream_results:
            continue
        
        smolvlm2 = smolvlm2_results[0]
        moondream = moondream_results[0]
        
        truth = smolvlm2["truth"]
        truth_team = smolvlm2["truth_team"]
        
        # Ensemble logic
        smolvlm2_pred = smolvlm2["pred"]
        moondream_pred = moondream["pred"]
        
        if smolvlm2_pred == "goal" and moondream_pred == "goal":
            final_pred = "goal"
            final_team = moondream["pred_team"] or smolvlm2["pred_team"]
            confidence = "high"
        elif smolvlm2_pred == "goal" and moondream_pred == "not_goal":
            final_pred = "not_goal"
            final_team = None
            confidence = "medium"
        else:
            final_pred = "not_goal" if moondream_pred == "not_goal" else smolvlm2_pred
            final_team = moondream["pred_team"] or smolvlm2["pred_team"]
            confidence = "medium"
        
        result = {
            "clip": clip_name,
            "truth": truth,
            "truth_team": truth_team,
            "smolvlm2": {"pred": smolvlm2_pred, "team": smolvlm2["pred_team"]},
            "moondream": {"pred": moondream_pred, "team": moondream["pred_team"]},
            "ensemble": {"pred": final_pred, "team": final_team, "confidence": confidence}
        }
        ensemble_results["clips"].append(result)
    
    # Compute metrics
    tp = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] == "goal")
    fn = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] != "goal")
    fp = sum(1 for r in ensemble_results["clips"] if r["truth"] != "goal" and r["ensemble"]["pred"] == "goal")
    tn = sum(1 for r in ensemble_results["clips"] if r["truth"] != "goal" and r["ensemble"]["pred"] != "goal")
    
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / len(ensemble_results["clips"]) if ensemble_results["clips"] else 0
    
    team_total = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] == "goal")
    team_correct = sum(1 for r in ensemble_results["clips"] if r["truth"] == "goal" and r["ensemble"]["pred"] == "goal" and r["ensemble"]["team"] == r["truth_team"])
    team_acc = team_correct / team_total if team_total else 0
    
    ensemble_results["aggregate"] = {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "team_total": team_total, "team_correct": team_correct, "team_accuracy": team_acc
    }
    
    (results_path / "ensemble_simulated_results.json").write_text(json.dumps(ensemble_results, indent=2))
    
    print(f"\n{'='*60}")
    print("SIMULATED ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Models: smolvlm2-2.2b + moondream:1.8b (direct)")
    print(f"Strategy: Both agree → goal; moondream rejects → not_goal")
    print(f"{'='*60}")
    print(f"Confusion Matrix:")
    print(f"  TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
    print(f"Metrics:")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Team Acc:  {team_acc*100:.1f}%")
    print(f"{'='*60}")
    print(f"Per-clip breakdown:")
    for r in ensemble_results["clips"]:
        g_ok = r["truth"] == r["ensemble"]["pred"]
        t_ok = r["truth"] == "goal" and r["ensemble"]["pred"] == "goal" and r["ensemble"]["team"] == r["truth_team"]
        print(f"  {r['clip']:<45} truth={r['truth']:<8} ensemble={r['ensemble']['pred']:<8} conf={r['ensemble']['confidence']:<8} goal={'✓' if g_ok else '✗'} team={'✓' if t_ok else '✗'}")
    
    return ensemble_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local", "simulated"], default="simulated")
    p.add_argument("--moondream-remote", default=None, help="URL of remote Ollama for moondream (e.g., http://192.168.1.100:11434)")
    p.add_argument("--clip-dir", default="data/9-8GT-right-quarter")
    p.add_argument("--results-dir", default="results/9-8GT-right-quarter")
    args = p.parse_args()
    
    if args.mode == "simulated":
        run_ensemble_simulated(args.clip_dir, args.results_dir)
    else:
        run_ensemble_local(args.clip_dir, args.results_dir, args.moondream_remote)
