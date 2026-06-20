# Overmind performance — goal-classifier (`9-8GT-right`)

34 clips · `gemini-3.5-flash` · onestage `prompt.txt`

## Summary

| Approach | Goal F1 | Accuracy | Team (on detected goals) | Notes |
|----------|---------|----------|--------------------------|-------|
| **Overmind `optimize` (native)** | — | 77.7/100 composite | goal 42.9 · team 18.3 (dims) | 5 iters, **0 candidates applied** |
| **Before hybrid loop** | 91.4% | 91.2% | 62.5% | Starting prompt |
| **After hybrid loop (best kept)** | **94.1%** | **94.1%** | **62.5%** | `scripts/optimize_loop.py` |

Hybrid loop = Overmind baseline/traces + **Gemini coding agent** edits `prompt.txt` + `classify.py` re-score (see `OVERMIND.md`).

## Best run (confusion matrix)

```
                 pred goal   pred not_goal
  truth goal           16            1
  truth not_goal        1           16
```

- **F1:** 94.1% · **Precision:** 94.1% · **Recall:** 94.1%
- **Team accuracy** (16 detected goals): 62.5% (10/16 correct)
- **Spurious goals** on non-goal clips: 1

Full per-clip JSON: `results/9-8GT-right_gemini-3.5-flash.json` (gitignored; regenerate with `python classify.py`).

Machine-readable summary: `.overmind/agents/goal-classifier/experiments/overmind_results.json`

## What worked / didn’t

| Component | Result |
|-----------|--------|
| Agent register + eval spec + dataset | ✅ |
| Baseline score + per-case traces | ✅ (77.7/100) |
| `overmind optimize` auto-codegen | ❌ Gemini analyzer → truncated JSON, no file edits |
| `scripts/gemini_prompt_optimize.py` | ✅ +2.7 pp F1 on goal detection |
| Team assignment via prompt-only | ⚠️ Flat ~62–64% across iterations |

## Reproduce

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

# Score current prompt.txt
python classify.py

# Full optimize loop (3 iters, keeps best prompt)
python scripts/optimize_loop.py

# Overmind baseline only
overmind optimize-step baseline --state .overmind/agents/goal-classifier/experiments/skill_state.json
```
