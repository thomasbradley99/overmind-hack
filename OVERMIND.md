# Overmind integration

This repo plugs a **Gemini VLM goal classifier** into [Overmind OSS](https://github.com/overmind-core/overmind).

## What Overmind is good for here

| Use Overmind for | Works? | Notes |
|------------------|--------|--------|
| Register agent + validate one clip | Yes | `overmind agent register` / `validate` |
| Eval spec + dataset + policies | Yes | `.overmind/agents/goal-classifier/setup_spec/` |
| **Baseline score** on train clips | Yes | `overmind optimize` Phase 1 or `optimize-step baseline` |
| Per-case **traces** + failure clusters | Yes | `.overmind/agents/goal-classifier/experiments/traces/` |
| Dimension scores (goal / team / structure) | Yes | e.g. 77.7/100 baseline |
| Holdout + rollback gates | Yes | After candidates exist |
| Cloud dashboard / job telemetry | Yes | `OVERMIND_API_KEY` in `.overmind/.env` |

## What does not work well (Gemini-only)

| Use Overmind for | Works? | Why |
|------------------|--------|-----|
| `overmind optimize` **auto-editing** `prompt.txt` | **No** (reliably) | Analyzer returns long truncated JSON, not `### FILE: prompt.txt` |
| Gemini as `ANALYZER_MODEL` codegen | **Fragile** | 4k output cap + strict parser |
| Fast iteration | **Slow** | Each eval case = full video API call |
| `tool_description` focus | **Mismatch** | No tools; prompt-only agent |

**Claude/GPT as `ANALYZER_MODEL`** fixes auto-codegen for many agents. We only have **Gemini**.

## Recommended hybrid (Gemini-only)

1. **Overmind** — measure and diagnose  
2. **Gemini coding agent** — edit `prompt.txt` (small steps, not one giant codegen message)  
3. **`classify.py`** — quick local score  

```bash
cd overmind-hack
source .venv/bin/activate   # or your env with google-generativeai

# Optional: Overmind baseline on train set
export $(grep -v '^#' .overmind/.env | xargs)
overmind optimize-step init goal-classifier --overwrite --settings '{"iterations":1,"candidates_per_iteration":1,"holdout_enforcement":false}'
overmind optimize-step baseline --state .overmind/agents/goal-classifier/experiments/skill_state.json

# Local score + failure list
python classify.py --dataset 9-8GT-right

# Hybrid: Gemini revises prompt.txt from failures (uses Overmind's coding_agent)
~/.local/share/uv/tools/overmind/bin/python scripts/gemini_prompt_optimize.py --from-overmind --apply --eval

# Team assignment (edits prompt_team.txt; score with twostage classifier)
~/.local/share/uv/tools/overmind/bin/python scripts/gemini_prompt_optimize.py \\
  --focus team --apply --eval --iterations 3 --keep-best \\
  --classifier-mode twostage

## Two-stage VLM (optional experiment)

`python classify.py --mode twostage` uses `prompt_goal.txt` + team prompts. Default is **onestage** (`prompt.txt` only).
```

Or in Cursor: `/overmind-optimize-agent goal-classifier` (Overmind scores; Cursor edits `prompt.txt`).

## Do not rely on

```bash
overmind optimize goal-classifier --fast   # auto-codegen skips with Gemini analyzer
```

Use it only if you add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for `ANALYZER_MODEL`.

## Agent contract

`agent.py` → `run(input_data)`:

```python
# input:  { "clip_path": "data/9-8GT-right/goal_01_....mp4", "dataset": "9-8GT-right" }
# output: { "goal": true/false, "team": "Dark sportswear" | null, "raw": "..." }
```

Only **`prompt.txt`** is in optimizable scope (`eval_spec.json`).

## Env vars (`.overmind/.env`)

| Var | Role |
|-----|------|
| `GEMINI_API_KEY` | Clip classifier (video) + coding-agent prompt edits |
| `CLASSIFIER_MODEL` | e.g. `gemini-3.5-flash` |
| `ANALYZER_MODEL` | e.g. `gemini/gemini-2.5-pro` for diagnose / coding agent |
| `OVERMIND_API_KEY` | Overmind CLI + telemetry (`ovr_...`) |

## Hack narrative

> We plugged a Gemini goal classifier into Overmind OSS. **Evaluation and traces worked** on our clip dataset. **Automatic prompt codegen** (`overmind optimize`) did not with Gemini-as-analyzer. We used **Overmind baseline + `scripts/gemini_prompt_optimize.py`** (Gemini coding agent) to revise `prompt.txt` and re-scored with `classify.py`.

## Artifacts

| Path | Content |
|------|---------|
| `setup_spec/eval_spec.json` | Scoring weights, optimizable `prompt.txt` only |
| `setup_spec/dataset.json` | 34 labeled clips |
| `experiments/traces/baseline/` | Per-case runs |
| `experiments/results.tsv` | Score history |
| `results/*_gemini*.json` | Local `classify.py` metrics |
