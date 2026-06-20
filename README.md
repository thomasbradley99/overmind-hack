# Overmind Hack — VLM Goal Classifier

Local football analysis: a VLM watches short clips and decides **goal vs not-goal**.
This repo is the clean experiment loop — built to be optimized with
[Overmind](https://github.com/overmind-core) (which tunes the **prompt** and **model**).

## The loop

```
data/<dataset>/*.mp4   (labeled clips: goals + non-goals)
        │
        ▼
   classify.py  ──uses──►  prompt.txt  +  model        ← the two knobs Overmind tunes
        │
        ▼
   results/<dataset>_<model>.json   (confusion matrix + accuracy / precision / recall / F1)
```

## Quick start

```bash
git lfs install                       # videos are stored in Git LFS
git clone <repo-url> && cd overmind-hack
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # paste your GEMINI_API_KEY

python3 classify.py                   # run the whole dataset
python3 classify.py --limit 6         # quick/cheap smoke test
python3 classify.py --verbose-raw   # print each Gemini API response live
```

Output prints a confusion matrix and a final line like:

```
SCORE f1=0.7586 accuracy=0.7879
```

and saves the full per-clip breakdown to `results/`.

## The two knobs (what Overmind optimizes)

| Knob | Where | How to change |
|------|-------|---------------|
| **Prompt** | `prompt.txt` | edit the file, or `--prompt other.txt` |
| **Model**  | `.env` `CLASSIFIER_MODEL` | or `--model gemini-3.5-flash` |

`classify.py` also exposes a function for programmatic experiments:

```python
from classify import classify_dataset
metrics = classify_dataset(open("prompt.txt").read(), model="gemini-3.5-flash")
print(metrics["f1"])          # <- the score to optimize
```

## Repo layout

```
overmind-hack/
├── classify.py          # THE script: clips → predictions → score
├── prompt.txt           # the goal/not-goal prompt (optimize this)
├── data/                # labeled clip datasets
│   └── 9-8GT-right/     # 17 goal clips + 17 non-goal clips (+ .json labels)
├── results/             # eval output (gitignored)
├── scripts/             # dataset builders
│   ├── make_clips.py        # cut goal clips from a game's ground truth
│   └── make_negatives.py    # cut random non-goal clips
├── games/               # source full match (video in LFS) + ground truth
├── goals/               # 59 human-labeled goal candidates (TP/FP) — reference
└── legacy/              # older full-game multi-stage pipeline (not used by the loop)
```

## Datasets

Each clip has a sibling `.json`. A clip is a **positive** (`goal`) if its json has
`action: "Goal"`, otherwise a **negative** (`not_goal`). Build more:

```bash
python3 scripts/make_clips.py --game 9-8GT-right        # goal clips
python3 scripts/make_negatives.py --game 9-8GT-right     # random non-goal clips
```

To add a new game: drop `games/<name>/{video.mp4, info.json, ground-truth.json}`,
then run the two scripts with `--game <name>`.

## Notes

- Clips are sent to the model **inline** (not via the Files API), so small/ephemeral
  API keys work and there's no upload step.
- Calls run in a thread pool (`--workers`, default 8) to stay under rate limits.
- `legacy/` holds the original 3-stage narrative pipeline (`run.py` + `pipeline/`)
  and full-game `eval.py`; kept for reference, not part of the classifier loop.
