#!/usr/bin/env python3
"""
Goal classifier — the core experiment loop.

Runs every clip in a dataset through a VLM and asks one yes/no question
("did a goal get scored?"), then scores the predictions against the known
labels and writes a results file.

The two knobs worth optimizing (this is what Overmind tunes) are:
  - the PROMPT   (--prompt, default prompt.txt)
  - the MODEL    (--model, default $CLASSIFIER_MODEL or gemini-3.5-flash)

Usage:
  python3 classify.py                                  # default dataset, prompt.txt
  python3 classify.py --prompt prompt.txt --model gemini-3.5-flash
  python3 classify.py --dataset 9-8GT-right --limit 6  # quick smoke test

Programmatic (e.g. from an Overmind experiment):
  from classify import classify_dataset
  metrics = classify_dataset(prompt_text, model="gemini-3.5-flash")
  score = metrics["f1"]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.generativeai as genai

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Load .env (KEY=VALUE), without overriding real env vars
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def label_for(clip_json: Path) -> str:
    """A clip is a positive ('goal') if its sibling json says so, else 'not_goal'."""
    data = json.loads(clip_json.read_text())
    if data.get("label"):
        return data["label"]
    return "goal" if data.get("action") == "Goal" else "not_goal"


def classify_one(clip: Path, prompt: str, model_name: str, api_key: str) -> dict:
    """Classify a single clip. Sends the video inline (no Files API)."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name, generation_config=genai.GenerationConfig(temperature=0.0)
    )
    truth = label_for(clip.with_suffix(".json"))
    try:
        resp = model.generate_content(
            [{"mime_type": "video/mp4", "data": clip.read_bytes()}, prompt]
        )
        raw = resp.text.strip()
        txt = raw
        if txt.startswith("```"):
            txt = txt.split("```")[1].lstrip("json").strip()
        pred = "goal" if json.loads(txt).get("goal") else "not_goal"
        return {"clip": clip.name, "truth": truth, "pred": pred, "raw": raw}
    except Exception as e:
        return {"clip": clip.name, "truth": truth, "pred": "error", "raw": str(e)[:160]}


def score(results: list[dict]) -> dict:
    """Confusion matrix + metrics, positive class = goal."""
    tp = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "goal")
    fn = sum(1 for r in results if r["truth"] == "goal" and r["pred"] == "not_goal")
    tn = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "not_goal")
    fp = sum(1 for r in results if r["truth"] == "not_goal" and r["pred"] == "goal")
    errors = sum(1 for r in results if r["pred"] == "error")
    n = tp + fn + tn + fp
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "errors": errors,
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def classify_dataset(prompt: str, dataset: str = "9-8GT-right",
                     model: str | None = None, workers: int = 8,
                     limit: int | None = None, verbose: bool = False) -> dict:
    """Run the whole dataset and return metrics. Importable for Overmind."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (export it or put it in .env)")
    model = model or os.environ.get("CLASSIFIER_MODEL", "gemini-3.5-flash")

    clips = sorted((DATA_DIR / dataset).glob("*.mp4"))
    if limit:
        clips = clips[:limit]
    if not clips:
        raise RuntimeError(f"No clips in {DATA_DIR / dataset}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(classify_one, c, prompt, model, api_key) for c in clips]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if verbose:
                mark = "OK " if r["pred"] == r["truth"] else ("ERR" if r["pred"] == "error" else "X  ")
                print(f"  {mark} {r['clip']:<40} truth={r['truth']:<9} pred={r['pred']}")

    results.sort(key=lambda r: r["clip"])
    metrics = score(results)
    metrics["model"] = model
    metrics["dataset"] = dataset
    metrics["n_clips"] = len(clips)
    metrics["results"] = results
    return metrics


def main():
    p = argparse.ArgumentParser(description="VLM goal classifier + eval")
    p.add_argument("--dataset", default="9-8GT-right", help="Folder under data/")
    p.add_argument("--prompt", default=str(ROOT / "prompt.txt"), help="Prompt file")
    p.add_argument("--model", default=None, help="Gemini model (default $CLASSIFIER_MODEL)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Only classify first N clips")
    args = p.parse_args()

    prompt_text = Path(args.prompt).read_text().strip()
    print(f"Dataset: {args.dataset}   Prompt: {args.prompt}")
    m = classify_dataset(prompt_text, dataset=args.dataset, model=args.model,
                         workers=args.workers, limit=args.limit, verbose=True)

    print("\n" + "=" * 50)
    print(f"CONFUSION MATRIX (positive = goal)  model={m['model']}")
    print("=" * 50)
    print("                 pred goal   pred not_goal")
    print(f"  truth goal        {m['tp']:^9}   {m['fn']:^11}")
    print(f"  truth not_goal    {m['fp']:^9}   {m['tn']:^11}")
    if m["errors"]:
        print(f"  (errors: {m['errors']})")
    print("-" * 50)
    print(f"  Accuracy : {m['accuracy']*100:5.1f}%")
    print(f"  Precision: {m['precision']*100:5.1f}%")
    print(f"  Recall   : {m['recall']*100:5.1f}%")
    print(f"  F1       : {m['f1']*100:5.1f}%")
    print("=" * 50)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.dataset}_{m['model']}.json"
    out.write_text(json.dumps(m, indent=2))
    print(f"\nSaved: {out}")
    print(f"SCORE f1={m['f1']:.4f} accuracy={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
