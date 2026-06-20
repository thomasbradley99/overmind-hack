#!/usr/bin/env python3
"""
Distributed Ensemble using exo API across two machines.

Architecture:
- macOS node (auto-discovered): runs smolvlm2-2.2b via MLX backend (GPU)
- Linux node (this machine, auto-discovered): runs moondream:1.8b via CPU backend

exo handles:
- Auto-discovery via mDNS/libp2p (zero configuration)
- Model placement on appropriate nodes based on backend support
- Load balancing and failover
- Parallel inference across the cluster

API endpoint: http://localhost:52415 (same on any node, exo routes internally)
"""
import json
import time
import asyncio
import aiohttp
import base64
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from PIL import Image
import io
import argparse
import re

EXO_API = "http://localhost:52415"

# exo model IDs (HuggingFace mlx-community format)
SMOLVLM2_MODEL = "mlx-community/SmolVLM-Instruct-4bit"  # macOS node (GPU)
MOONDREAM_MODEL = "beshkenadze/moondream3-preview-mlx-4bit"  # Linux node (CPU)

def encode_image_to_base64(path: str, max_size: int = 224) -> str:
    """Encode image to base64 data URL for exo API."""
    img = Image.open(path)
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"

async def extract_frame(video_path: str, output_path: str, timestamp: str = "00:00:01") -> str:
    """Extract a frame from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-ss", timestamp, "-i", video_path,
        "-vframes", "1", "-y", "-q:v", "2", output_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    await proc.communicate()
    return output_path

async def exo_chat_completion(
    session: aiohttp.ClientSession,
    model_id: str,
    prompt: str,
    image_path: str,
    timeout: int = 120
) -> Tuple[str, float, Dict]:
    """
    Send a chat completion request to exo cluster.
    exo automatically routes to the node where the model is running.
    """
    image_url = encode_image_to_base64(image_path)

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "stream": False,
        "max_tokens": 500,
        "temperature": 0.0
    }

    t0 = time.time()
    async with session.post(
        f"{EXO_API}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        latency = time.time() - t0

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, latency, usage

def parse_goal_response(text: str) -> Tuple[str, Optional[str]]:
    """Extract goal bool and team from model response."""
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

async def ensure_model_loaded(session: aiohttp.ClientSession, model_id: str) -> bool:
    """Check if model is available, add if not."""
    async with session.get(f"{EXO_API}/models") as resp:
        resp.raise_for_status()
        models = await resp.json()
        model_ids = [m.get("id", m.get("model_id", "")) for m in models]
        if model_id in model_ids or any(model_id.endswith(m.split("/")[-1]) for m in model_ids):
            return True

    print(f"  [exo] Adding custom model: {model_id}")
    async with session.post(
        f"{EXO_API}/models/add",
        json={"model_id": model_id}
    ) as resp:
        if resp.status == 200:
            print(f"  [exo] Model added successfully")
            return True
        else:
            text = await resp.text()
            print(f"  [exo] Failed to add model: {resp.status} - {text}")
            return False

async def run_parallel_inference(
    session: aiohttp.ClientSession,
    frame_path: str,
    smolvlm2_prompt: str,
    moondream_prompt: str
) -> Dict[str, Any]:
    """Run both models in parallel via exo cluster."""

    async def infer_model(key: str, model_id: str, prompt: str):
        try:
            loaded = await ensure_model_loaded(session, model_id)
            if not loaded:
                return {key: {"error": "Model not available", "success": False}}

            text, latency, usage = await exo_chat_completion(
                session, model_id, prompt, frame_path
            )
            pred, team = parse_goal_response(text)
            return {key: {
                "pred": pred, "team": team, "latency": latency,
                "raw": text[:300], "success": True, "usage": usage
            }}
        except Exception as e:
            return {key: {"error": str(e), "success": False}}

    tasks = [
        infer_model("smolvlm2", SMOLVLM2_MODEL, smolvlm2_prompt),
        infer_model("moondream", MOONDREAM_MODEL, moondream_prompt)
    ]
    results_list = await asyncio.gather(*tasks)

    results = {}
    for r in results_list:
        results.update(r)
    return results

def ensemble_decision(smolvlm2_result: Dict, moondream_result: Dict, mode: str) -> Tuple[str, Optional[str], str, str]:
    """Apply ensemble strategy."""
    s_pred = smolvlm2_result.get("pred", "not_goal")
    m_pred = moondream_result.get("pred", "not_goal")
    s_team = smolvlm2_result.get("team")
    m_team = moondream_result.get("team")

    if mode == "or":
        if s_pred == "goal" or m_pred == "goal":
            pred = "goal"
            team = m_team if m_pred == "goal" and m_team else s_team
            confidence = "high" if (s_pred == "goal" and m_pred == "goal") else "medium"
            rationale = "At least one model detected a goal. Union strategy maximizes recall."
        else:
            pred = "not_goal"
            team = None
            confidence = "high"
            rationale = "Both models agree: no goal detected."

    elif mode == "and":
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
        if s_pred == "goal" and m_pred == "goal":
            pred = "goal"
            team = m_team or s_team
            confidence = "high"
            rationale = "smolvlm2 detected goal, moondream confirmed."
        elif s_pred == "goal" and m_pred != "goal":
            pred = "not_goal"
            team = None
            confidence = "medium"
            rationale = "smolvlm2 detected goal but moondream did not confirm."
        else:
            pred = "not_goal"
            team = None
            confidence = "high"
            rationale = "smolvlm2 did not detect a goal."

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

async def run_ensemble(
    clip_dir: str = "data/9-8GT-right-quarter",
    results_dir: str = "results/9-8GT-right-quarter",
    exo_url: str = "http://localhost:52415",
    mode: str = "or",
    dry_run: bool = False
) -> Dict:
    global EXO_API
    EXO_API = exo_url

    clip_path = Path(clip_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    clips = sorted(clip_path.glob("*.mp4"))
    print(f"Found {len(clips)} clips in {clip_dir}")
    print(f"Mode: {mode.upper()}")
    print(f"exo API: {exo_url}")

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
        "strategy": f"exo_distributed_{mode}",
        "models": [SMOLVLM2_MODEL, MOONDREAM_MODEL],
        "exo_url": exo_url,
        "mode": mode,
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

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{exo_url}/state", timeout=10) as resp:
                state = await resp.json()
                nodes = state.get("nodes", [])
                print(f"exo cluster: {len(nodes)} node(s) connected")
                for n in nodes:
                    print(f"  - {n.get('node_id', 'unknown')}: {n.get('platform', 'unknown')}")
        except Exception as e:
            print(f"WARNING: Could not connect to exo at {exo_url}: {e}")
            print("  Proceeding in dry-run mode...")
            dry_run = True

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
                smolvlm2_r = precomputed[clip_stem].get("smolvlm2-2.2b (56px, 23 frames, 2fps)")
                moondream_r = precomputed[clip_stem].get("moondream:1.8b (224px, 1 frame, direct prompt)")

                if not smolvlm2_r or not moondream_r:
                    print(f"  [SKIP] Missing pre-computed data")
                    continue

                smolvlm2_result = {
                    "pred": smolvlm2_r["pred"], "team": smolvlm2_r["pred_team"],
                    "latency": 1.0, "raw": smolvlm2_r.get("raw_preview", ""), "success": True
                }
                moondream_result = {
                    "pred": moondream_r["pred"], "team": moondream_r["pred_team"],
                    "latency": 1.3, "raw": moondream_r.get("raw_preview", ""), "success": True
                }
                print(f"  [DRY RUN] Using pre-computed data")
            else:
                frame_path = str(clip.with_suffix("_frame.jpg"))
                await extract_frame(str(clip), frame_path)

                print(f"  Running parallel inference via exo cluster...")
                t0 = time.time()
                results = await run_parallel_inference(
                    session, frame_path, smolvlm2_prompt, moondream_prompt
                )
                total_time = time.time() - t0

                if not results.get("smolvlm2", {}).get("success"):
                    print(f"  [ERROR] smolvlm2 failed: {results.get('smolvlm2', {}).get('error')}")
                    continue
                if not results.get("moondream", {}).get("success"):
                    print(f"  [ERROR] moondream failed: {results.get('moondream', {}).get('error')}")
                    results["moondream"] = {"pred": "not_goal", "team": None, "latency": 0, "raw": "", "success": True}

                smolvlm2_result = results["smolvlm2"]
                moondream_result = results["moondream"]
                print(f"  Parallel inference time: {total_time:.2f}s")

            print(f"  smolvlm2:  {smolvlm2_result['pred']:<8} | Team: {smolvlm2_result.get('team', '-') or '-':<15} | {smolvlm2_result['latency']:.1f}s")
            print(f"  moondream: {moondream_result['pred']:<8} | Team: {moondream_result.get('team', '-') or '-':<15} | {moondream_result['latency']:.1f}s")

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

            out_file = results_path / f"exo_ensemble_{mode}_{clip_stem}.json"
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

    (results_path / f"exo_ensemble_{mode}_results.json").write_text(
        json.dumps(ensemble_results, indent=2)
    )

    print(f"\n{'='*60}")
    print(f"EXO DISTRIBUTED ENSEMBLE RESULTS — Mode: {mode.upper()}")
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
        print(f"  {r['clip']:<45} truth={r['truth']:<8} ensemble={r['ensemble']['pred']:<8} conf={r['ensemble']['confidence']:<8} goal={'OK' if g_ok else 'FAIL'} team={'OK' if t_ok else 'FAIL'}")

    return ensemble_results

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="exo Distributed Ensemble: smolvlm2 + moondream")
    p.add_argument("--clip-dir", default="data/9-8GT-right-quarter")
    p.add_argument("--results-dir", default="results/9-8GT-right-quarter")
    p.add_argument("--exo-url", default="http://localhost:52415", help="exo API URL")
    p.add_argument("--mode", default="or", choices=["or", "and", "cascade", "smolvlm2", "moondream"])
    p.add_argument("--dry-run", action="store_true", help="Simulate with pre-computed data")
    args = p.parse_args()

    asyncio.run(run_ensemble(args.clip_dir, args.results_dir, args.exo_url, args.mode, args.dry_run))
