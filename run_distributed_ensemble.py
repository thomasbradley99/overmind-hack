#!/usr/bin/env python3
"""
Distributed Ensemble: Two-Model Inference Across Two Machines

Architecture:
- Laptop A (Local): smolvlm2-2.2b (fast, high recall, ~1s per clip)
- Laptop B (Remote): moondream:1.8b (precise, high team accuracy, ~1.3s per clip)

Ensemble Modes:
  or:        Union — if either model says goal, it's a goal (best team accuracy)
  and:       Intersection — only if both agree (best precision, low recall)
  cascade:   smolvlm2 first, moondream confirms (balanced)
  smolvlm2:  Baseline comparison
  moondream: Baseline comparison

Connection: Ollama HTTP API over local network (port 11434)
"""
import json
import time
import requests
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import base64
from PIL import Image
import io
import argparse
import re

OLLAMA_LOCAL = "http://localhost:11434"
OLLAMA_REMOTE = None

MODELS = {
    "smolvlm2": {
        "name": "richardyoung/smolvlm2-2.2b-instruct:latest",
        "prompt": """Analyze this football clip and determine if a goal was scored.
Return ONLY JSON: {"goal": true/false, "team": "team name"}
""",
        "location": "local"
    },
    "moondream": {
        "name": "moondream:1.8b",
        "prompt": """You are a football referee analyzing a video clip. Determine if a goal was scored and identify the scoring team.

Signs of a goal:
- Ball crossing the goal line into the net
- Players celebrating (arms raised, jumping, hugging)
- Goalkeeper retrieving ball from net
- Ball hitting the back of the net

Teams:
- "Dark sportswear" — players in dark athletic wear, shorts, jerseys
- "Dark suits" — players in dark formal suits, blazers, trousers

Return ONLY a JSON object with no extra text or markdown:
{"goal": true/false, "team": "team name or null"}
""",
        "location": "remote"
    }
}

def get_ollama_url(remote: Optional[str] = None) -> str:
    return remote or OLLAMA_LOCAL

def encode_image(path: str, max_size: int = 224) -> str:
    img = Image.open(path)
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def ollama_chat(model: str, prompt: str, image_path: str, 
                remote_url: Optional[str] = None, timeout: int = 120) -> Tuple[str, float]:
    """Call Ollama chat API. Returns (text, latency)."""
    url = get_ollama_url(remote_url) + "/api/chat"
    b64 = encode_image(image_path)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 500}
    }
    
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    latency = time.time() - t0
    return r.json()["message"]["content"], latency

def extract_frame(video_path: str, output_path: str, timestamp: str = "00:00:01") -> str:
    cmd = ["ffmpeg", "-ss", timestamp, "-i", video_path, "-vframes", "1", "-y", "-q:v", "2", output_path]
    subprocess.run(cmd, capture_output=True, timeout=30)
    return output_path

def parse_goal_response(text: str) -> Tuple[str, Optional[str]]:
    t = text.lower()
    if '"goal"' in t or '"goal":' in text.lower():
        try:
            m = re.search(r'\{[^}]*"goal"[^}]*\}', text, re.DOTALL | re.IGNORECASE)
            if m:
                j = json.loads(m.group())
                g = j.get("goal", j.get("is_goal", False))
                pred = "goal" if g else "not_goal"
                team = j.get("team", j.get("team_name", j.get("scoring_team")))
                return pred, team
        except Exception:
            pass
    if "goal" in t and "not_goal" not in t and "no goal" not in t and "not a goal" not in t:
        if "dark sportswear" in t: return "goal", "Dark sportswear"
        if "dark suits" in t: return "goal", "Dark suits"
        return "goal", None
    if "not_goal" in t or "no goal" in t or "not a goal" in t or (len(t) < 10):
        return "not_goal", None
    return "not_goal", None

def run_parallel_inference(frame_path: str, remote_url: Optional[str] = None) -> Dict:
    """Run both models in parallel on two machines."""
    results = {}
    threads = []
    
    def infer_model(key, config, remote):
        try:
            text, latency = ollama_chat(config["name"], config["prompt"], frame_path, remote)
            pred, team = parse_goal_response(text)
            results[key] = {
                "pred": pred, "team": team, "latency": latency,
                "raw": text[:200], "success": True, "remote": remote is not None
            }
        except Exception as e:
            results[key] = {"error": str(e), "success": False, "remote": remote is not None}
    
    # Start smolvlm2 on local
    t1 = threading.Thread(target=infer_model, args=("smolvlm2", MODELS["smolvlm2"], None))
    t1.start()
    threads.append(t1)
    
    # Start moondream on remote (or local if no remote)
    t2 = threading.Thread(target=infer_model, args=("moondream", MODELS["moondream"], remote_url))
    t2.start()
    threads.append(t2)
    
    # Wait for both
    for t in threads:
        t.join()
    
    return results

def ensemble_decision(smolvlm2_result: Dict, moondream_result: Dict, mode: str) -> Tuple[str, Optional[str], str, str]:
    """Apply ensemble strategy. Returns (pred, team, confidence, rationale)."""
    s_pred = smolvlm2_result.get("pred", "not_goal")
    m_pred = moondream_result.get("pred", "not_goal")
    s_team = smolvlm2_result.get("team")
    m_team = moondream_result.get("team")
    
    if mode == "or":
        # Union: if either says goal, it's a goal
        if s_pred == "goal" or m_pred == "goal":
            pred = "goal"
            # Use moondream's team if available, else smolvlm2's
            team = m_team if m_pred == "goal" and m_team else s_team
            confidence = "high" if (s_pred == "goal" and m_pred == "goal") else "medium"
            rationale = "At least one model detected a goal. Union strategy maximizes recall."
        else:
            pred = "not_goal"
            team = None
            confidence = "high"
            rationale = "Both models agree: no goal detected."
    
    elif mode == "and":
        # Intersection: only if both agree
        if s_pred == "goal" and m_pred == "goal":
            pred = "goal"
            team = m_team or s_team
            confidence = "high"
            rationale = "Both models agree on goal. High confidence detection."
        else:
            pred = "not_goal"
            team = None
            confidence = "high"
            rationale = "Models disagree or both say no goal. Conservative approach."
    
    elif mode == "cascade":
        # Cascade: smolvlm2 first, moondream confirms
        if s_pred == "goal" and m_pred == "goal":
            pred = "goal"
            team = m_team or s_team
            confidence = "high"
            rationale = "smolvlm2 detected goal, moondream confirmed. Highest confidence."
        elif s_pred == "goal" and m_pred != "goal":
            # moondream vetoed or empty
            pred = "not_goal"
            team = None
            confidence = "medium"
            rationale = "smolvlm2 detected goal but moondream did not confirm. Vetoed."
        else:
            pred = "not_goal"
            team = None
            confidence = "high"
            rationale = "smolvlm2 did not detect a goal. No need for moondream check."
    
    elif mode == "smolvlm2":
        pred = s_pred
        team = s_team
        confidence = "medium"
        rationale = "Using smolvlm2 only (baseline)."
    
    elif mode == "moondream":
        pred = m_pred
        team = m_team
        confidence = "medium"
        rationale = "Using moondream only (baseline)."
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return pred, team, confidence, rationale

def run_ensemble(clip_dir: str = "data/9-8GT-right-quarter",
                 results_dir: str = "results/9-8GT-right-quarter",
                 remote_url: Optional[str] = None,
                 mode: str = "or",
                 dry_run: bool = False) -> Dict:
    clip_path = Path(clip_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    clips = sorted(clip_path.glob("*.mp4"))
    print(f"Found {len(clips)} clips in {clip_dir}")
    print(f"Mode: {mode.upper()}")
    print(f"Remote: {remote_url if remote_url else 'None (local only)'}")
    
    # Load pre-computed data if dry_run
    precomputed = None
    if dry_run:
        eval_file = Path("results/evaluation_results.json")
        if eval_file.exists():
            precomputed = {}
            for r in json.loads(eval_file.read_text()).get("per_clip_results", []):
                clip = r["clip"]
                if clip not in precomputed:
                    precomputed[clip] = {}
                precomputed[clip][f"{r['model']} ({r['config']})"] = r
            print(f"Loaded pre-computed data for {len(precomputed)} clips")
    
    ensemble_results = {
        "strategy": f"distributed_{mode}",
        "models": ["smolvlm2-2.2b", "moondream:1.8b"],
        "moondream_remote": remote_url,
        "mode": mode,
        "clips": [],
        "aggregate": {}
    }
    
    for clip in clips:
        label_file = clip.with_suffix(".json")
        if not label_file.exists():
            continue
            
        gt = json.loads(label_file.read_text())
        truth = "goal" if gt.get("action") == "Goal" or gt.get("label") == "goal" else "not_goal"
        truth_team = gt.get("team", gt.get("truth_team"))
        clip_stem = clip.stem
        
        print(f"\n{'='*60}")
        print(f"Clip: {clip_stem}")
        print(f"Truth: {truth} | Team: {truth_team}")
        
        if dry_run and precomputed and clip_stem in precomputed:
            # Use pre-computed data
            smolvlm2_r = precomputed[clip_stem].get("smolvlm2-2.2b (56px, 23 frames, 2fps)")
            moondream_r = precomputed[clip_stem].get("moondream:1.8b (224px, 1 frame, direct prompt)")
            
            if not smolvlm2_r or not moondream_r:
                print(f"  [SKIP] Missing pre-computed data")
                continue
            
            smolvlm2_result = {
                "pred": smolvlm2_r["pred"], "team": smolvlm2_r["pred_team"],
                "latency": 1.0, "raw": smolvlm2_r.get("raw_preview", ""), "success": True, "remote": False
            }
            moondream_result = {
                "pred": moondream_r["pred"], "team": moondream_r["pred_team"],
                "latency": 1.3, "raw": moondream_r.get("raw_preview", ""), "success": True, "remote": True
            }
            print(f"  [DRY RUN] Using pre-computed data")
        else:
            # Extract frame
            frame_path = str(clip.with_suffix("_frame.jpg"))
            extract_frame(str(clip), frame_path)
            
            # Run parallel inference
            print(f"  Running parallel inference...")
            t0 = time.time()
            results = run_parallel_inference(frame_path, remote_url)
            total_time = time.time() - t0
            
            if not results.get("smolvlm2", {}).get("success"):
                print(f"  [ERROR] smolvlm2 failed: {results.get('smolvlm2', {}).get('error')}")
                continue
            if not results.get("moondream", {}).get("success"):
                print(f"  [ERROR] moondream failed: {results.get('moondream', {}).get('error')}")
                # Fallback: use smolvlm2 only
                results["moondream"] = {"pred": "not_goal", "team": None, "latency": 0, "raw": "", "success": True, "remote": remote_url is not None}
            
            smolvlm2_result = results["smolvlm2"]
            moondream_result = results["moondream"]
            print(f"  Parallel inference time: {total_time:.2f}s")
        
        print(f"  smolvlm2:  {smolvlm2_result['pred']:<8} | Team: {smolvlm2_result.get('team', '-') or '-':<15} | {smolvlm2_result['latency']:.1f}s")
        print(f"  moondream: {moondream_result['pred']:<8} | Team: {moondream_result.get('team', '-') or '-':<15} | {moondream_result['latency']:.1f}s")
        
        # Ensemble decision
        pred, team, confidence, rationale = ensemble_decision(smolvlm2_result, moondream_result, mode)
        
        print(f"  ENSEMBLE:  {pred:<8} | Team: {team or '-':<15} | Confidence: {confidence}")
        
        result = {
            "clip": clip_stem,
            "truth": truth,
            "truth_team": truth_team,
            "smolvlm2": smolvlm2_result,
            "moondream": moondream_result,
            "ensemble": {"pred": pred, "team": team, "confidence": confidence, "rationale": rationale}
        }
        ensemble_results["clips"].append(result)
        
        out_file = results_path / f"ensemble_{mode}_{clip_stem}.json"
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
    
    (results_path / f"ensemble_{mode}_results.json").write_text(json.dumps(ensemble_results, indent=2))
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS — Mode: {mode.upper()}")
    print(f"{'='*60}")
    print(f"Confusion Matrix: TP={tp} | FN={fn} | FP={fp} | TN={tn}")
    print(f"Precision: {prec*100:.1f}%")
    print(f"Recall:    {rec*100:.1f}%")
    print(f"F1:        {f1*100:.1f}%")
    print(f"Accuracy:  {acc*100:.1f}%")
    print(f"Team Acc:  {team_acc*100:.1f}%")
    print(f"\nPer-clip breakdown:")
    for r in ensemble_results["clips"]:
        g_ok = r["truth"] == r["ensemble"]["pred"]
        t_ok = r["truth"] == "goal" and r["ensemble"]["pred"] == "goal" and r["ensemble"]["team"] == r["truth_team"]
        print(f"  {r['clip']:<45} truth={r['truth']:<8} ensemble={r['ensemble']['pred']:<8} conf={r['ensemble']['confidence']:<8} goal={'✓' if g_ok else '✗'} team={'✓' if t_ok else '✗'}")
    
    return ensemble_results

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Distributed Ensemble: smolvlm2 + moondream")
    p.add_argument("--clip-dir", default="data/9-8GT-right-quarter")
    p.add_argument("--results-dir", default="results/9-8GT-right-quarter")
    p.add_argument("--remote-ip", default=None, help="Remote laptop IP for moondream (e.g., 192.168.1.100)")
    p.add_argument("--remote-port", default="11434", help="Remote Ollama port")
    p.add_argument("--mode", default="or", choices=["or", "and", "cascade", "smolvlm2", "moondream"])
    p.add_argument("--dry-run", action="store_true", help="Simulate with pre-computed data")
    args = p.parse_args()
    
    remote_url = f"http://{args.remote_ip}:{args.remote_port}" if args.remote_ip else None
    run_ensemble(args.clip_dir, args.results_dir, remote_url, args.mode, args.dry_run)
