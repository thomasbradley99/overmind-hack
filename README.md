# Overmind Hack — VLM Football Goal Detection

A self-contained repo for detecting football events (Goals, Near Misses, Big
Hits) in 5-a-side match video using a **VLM (Google Gemini)**, plus evaluation
against human-labeled ground truth.

Everything needed to run is in this one repo — videos included (via Git LFS).

## How it works

```
video.mp4
   │
   ├─ 1. spot       chop into 60s chunks → Gemini describes each chunk (plain text)
   ├─ 2. synthesize Gemini reads all descriptions → narrative + identified events
   └─ 3. extract    Gemini converts narrative → structured JSON {events, metadata}
   │
   └─→ stage3_events.json   ← final output
                │
                └─ eval.py compares detected Goals vs ground-truth.json (P / R / F1)
```

## Layout

```
overmind-hack/
├── run.py                  # run the pipeline on a game OR a single clip
├── eval.py                 # score detected goals vs ground truth
├── pipeline/
│   ├── spot.py             # stage 1: describe video chunks
│   ├── synthesize.py       # stage 2: narrative + event identification
│   └── extract.py          # stage 3: narrative → structured JSON
├── games/                  # full matches (video + ground truth)
│   └── 9-8GT-right/
│       ├── video.mp4        (720p, ~160 MB, Git LFS)
│       ├── info.json        teams + metadata
│       └── ground-truth.json
├── clips/                  # labeled goal clips cut from the games (Git LFS)
│   └── 9-8GT-right/
│       ├── goal_01_283s_Dark-sportswear.mp4
│       ├── goal_01_283s_Dark-sportswear.json   # answer: {time, team, action, ...}
│       └── ...                                  # one pair per goal
├── scripts/
│   └── make_clips.py       # regenerate clips from any game's ground truth
└── goals/                  # 59 labeled goal candidates (TP/FP) — see goals/README.md
```

## Getting the repo (teammates start here)

This repo uses **Git LFS** for video files. Install it once, then clone:

```bash
# 1. install git-lfs (once per machine)
#    mac:    brew install git-lfs
#    ubuntu: sudo apt-get install git-lfs
git lfs install

# 2. clone — LFS videos download automatically
git clone <repo-url>
cd overmind-hack
```

If you cloned before installing LFS, run `git lfs pull` to fetch the videos.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # google-generativeai
cp .env.example .env                      # then paste your GEMINI_API_KEY
```

Also requires `ffmpeg` / `ffprobe` on PATH (used to chunk video).

## Run

```bash
# Cheapest first test: a single 12s goal clip (one Gemini call)
python3 run.py --video clips/9-8GT-right/goal_01_283s_Dark-sportswear.mp4

# Quick partial game: first 5 minutes only
python3 run.py --game 9-8GT-right --minutes 5

# Full game, then evaluate against ground truth
python3 run.py --game 9-8GT-right --eval
```

Outputs:
- `--game`  → `games/<game>/runs/<timestamp>/`
- `--video` → `outputs/<clip-name>/runs/<timestamp>/`

| File | What it is |
|------|-----------|
| `stage1_observations.txt` | raw per-chunk descriptions from the VLM |
| `stage2_narrative.txt` | synthesized match narrative + events |
| `stage3_events.json` | **final structured events** |
| `timing.json` | seconds per stage |

## Evaluate separately

```bash
python3 eval.py \
  --ai-output games/9-8GT-right/runs/<timestamp>/stage3_events.json \
  --gt games/9-8GT-right/ground-truth.json \
  --tolerance 30
```

Prints precision / recall / F1 for **Goals** (a detection matches a GT goal if
within `tolerance` seconds), plus a timeline of matches, misses, false positives.

## Clips

`clips/<game>/` holds short clips cut around each ground-truth goal, with a
matching `.json` answer file (time, team, action — no description). They're for
fast/cheap iteration: one ~12s clip = a single Gemini call.

Regenerate them for any game (defaults: Goals only, 8s before / 4s after):

```bash
python3 scripts/make_clips.py --game 9-8GT-right
python3 scripts/make_clips.py --game 9-8GT-right --actions Goal "Near Miss" --before 10 --after 5
```

## Add another game

Create `games/<name>/` with:
- `video.mp4` (LFS handles the size automatically)
- `info.json`  → `{"teams": ["A", "B"]}`
- `ground-truth.json` → `{"events": [{"time": <sec>, "action": "Goal", "team": "...", ...}]}`

Then `python3 scripts/make_clips.py --game <name>` to cut its clips.

## Cost / time

~$0.50–0.80 and a few minutes for a full ~37-min game (Gemini API). Use
`--minutes N` or the `clips/` for fast, cheap iteration.

## Config

Defaults live in code; override in `.env`:
- `GEMINI_API_KEY` (required)
- `STAGE2_MODEL` / `STAGE3_MODEL` / `STAGE4_MODEL` (default `gemini-2.5-flash` / `flash` / `pro`)
- `CHUNK_DURATION` / `CHUNK_OVERLAP` (default 60 / 15 seconds)
