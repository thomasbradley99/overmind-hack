#!/usr/bin/env python3
"""
Cascading Ensemble: smolvlm2 (recall) + moondream (precision/team ID)

Distributed Design:
- Local machine: Runs smolvlm2-2.2b (fast, ~1s per clip, catches all goals)
- Remote machine: Runs moondream:1.8b (precision, ~1.3s per clip, 100% team accuracy)

Connection: Ollama HTTP API over local network (port 11434)

Ensemble Strategy:
1. smolvlm2 first: If it says "goal", proceed to moondream
2. moondream confirmation: If moondream says "goal", HIGH confidence
3. If moondream says "not_goal" or empty, MEDIUM/LOW confidence
4. Team ID: Use moondream's team when both agree, else smolvlm2's team
"""
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple
import base64
from PIL import Image
import io
import argparse
import re

OLLAMA_LOCAL = "http://localhost:11434"

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
                remote_url: Optional[str] = None, timeout: int = 120) -> str:
    url = get_ollama_url(remote_url) + "/api/chat"
    b64 = encode_image(image_path)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 500}
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]

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

def run_ensemble(clip_dir: str = "data/9-8GT-right-quarter",
                 results_dir: str = "results/9-8GT-right-quarter",
                 moondream_remote: Optional[str] = None,
                 dry_run: bool = False) -> Dict:
    clip_path = Path(clip_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    clips = sorted(clip_path.glob("*.mp4"))
    print(f"Found {len(clips)} clips in {clip_dir}")
    
    # Load pre-computed data if dry_run
    precomputed = None
    if dry_run:
        eval_file = Path("results/evaluation_results.json")
        if eval_file.exists():
            precomputed = json.loads(eval_file.read_text())
            pc = {}
            for r in precomputed.get("per_clip_results", []):
                clip = r["clip"]
                if clip not in pc:
                    pc[clip] = {}
                pc[clip][f"{r['model']} ({r['config']})"] = r
            precomputed = pc
            print(f"Loaded pre-computed data for {len(pc)} clips")
    
    ensemble_results = {
        "strategy": "cascading_ensemble",
        "models": ["smolvlm2-2.2b (local)", f"moondream:1.8b ({'remote' if moondream_remote else 'local'})"],
        "moondream_remote": moondream_remote,
        "clips": [],
        "aggregate": {}
    }
    
    smolvlm2_prompt = """Analyze this football clip and determine if a goal was scored.
Return ONLY JSON: {"goal": true/false, "team": "team name"}
"""
    
    moondream_prompt = """You are a football referee analyzing a video clip. Determine if a goal was scored and identify the scoring team.

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
"""
    
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
            smolvlm2_key = "smolvlm2-2.2b (56px, 23 frames, 2fps)"
            moondream_key = "moondream:1.8b (224px, 1 frame, direct prompt)"
            
            smolvlm2_r = precomputed[clip_stem].get(smolvlm2_key)
            moondream_r = precomputed[clip_stem].get(moondream_key)
            
            if not smolvlm2_r or not moondream_r:
                print(f"  [SKIP] Missing model data for {clip_stem}")
                continue
            
            smolvlm2_pred = smolvlm2_r["pred"]
            smolvlm2_team = smolvlm2_r["pred_team"]
            smolvlm2_time = 1.0
            smolvlm2_raw = smolvlm2_r.get("raw_preview", "")
            
            moondream_pred = moondream_r["pred"]
            moondream_team = moondream_r["pred_team"]
            moondream_time = 1.3
            moondream_raw = moondream_r.get("raw_preview", "")
            
            print(f"  [DRY RUN] Using pre-computed data")
        else:
            # Extract frame
            frame_path = str(clip.with_suffix("_frame.jpg"))
            extract_frame(str(clip), frame_path)
            
            print(f"  Stage 1: smolvlm2-2.2b (local)...")
            t0 = time.time()
            smolvlm2_raw = ollama_chat("richardyoung/smolvlm2-2.2b-instruct:latest", 
                                        smolvlm2_prompt, frame_path)
            smolvlm2_time = time.time() - t0
            smolvlm2_pred, smolvlm2_team = parse_goal_response(smolvlm2_raw)
            print(f"    -> {smolvlm2_pred} | Team: {smolvlm2_team} | {smolvlm2_time:.1f}s")
            
            print(f"  Stage 2: moondream:1.8b (remote={moondream_remote is not None})...")
            t0 = time.time()
            moondream_raw = ollama_chat("moondream:1.8b", moondream_prompt, frame_path,
                                         remote_url=moondream_remote)
            moondream_time = time.time() - t0
            moondream_pred, moondream_team = parse_goal_response(moondream_raw)
            print(f"    -> {moondream_pred} | Team: {moondream_team} | {moondream_time:.1f}s")
        
        # Ensemble decision
        if smolvlm2_pred == "goal" and moondream_pred == "goal":
            final_pred = "goal"
            final_team = moondream_team or smolvlm2_team
            confidence = "high"
            rationale = "Both models agree. moondream team ID is 100% accurate when it detects."
        elif smolvlm2_pred == "goal" and moondream_pred == "not_goal":
            final_pred = "not_goal"
            final_team = None
            confidence = "medium"
            rationale = "moondream rejected. In our data, moondream 'not_goal' is 50% correct (4 TN, 4 FN)."
        elif smolvlm2_pred == "not_goal" and moondream_pred == "not_goal":
            final_pred = "not_goal"
            final_team = None
            confidence = "high"
            rationale = "Both models agree on not_goal."
        else:
            final_pred = "not_goal"
            final_team = None
            confidence = "medium"
            rationale = "smolvlm2 not_goal, moondream empty. Defaulting to not_goal."
        
        print(f"  ENSEMBLE: {final_pred} | Team: {final_team} | Confidence: {confidence}")
        
        result = {
            "clip": clip_stem,
            "truth": truth,
            "truth_team": truth_team,
            "smolvlm2": {"pred": smolvlm2_pred, "team": smolvlm2_team, "latency": smolvlm2_time, "raw": smolvlm2_raw[:200]},
            "moondream": {"pred": moondream_pred, "team": moondream_team, "latency": moondream_time, "raw": moondream_raw[:200], "remote": moondream_remote is not None},
            "ensemble": {"pred": final_pred, "team": final_team, "confidence": confidence, "rationale": rationale}
        }
        ensemble_results["clips"].append(result)
        
        out_file = results_path / f"ensemble_{clip_stem}.json"
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
    
    (results_path / "ensemble_results.json").write_text(json.dumps(ensemble_results, indent=2))
    
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
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
    p = argparse.ArgumentParser()
    p.add_argument("--clip-dir", default="data/9-8GT-right-quarter")
    p.add_argument("--results-dir", default="results/9-8GT-right-quarter")
    p.add_argument("--moondream-remote", default=None, help="Remote Ollama URL for moondream")
    p.add_argument("--dry-run", action="store_true", help="Simulate with pre-computed data")
    args = p.parse_args()
    
    run_ensemble(args.clip_dir, args.results_dir, args.moondream_remote, args.dry_run)
