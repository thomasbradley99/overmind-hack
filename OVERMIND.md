# Overmind integration

This repo is set up to be optimized with [Overmind](https://github.com/overmind-core/overmind).

## What Overmind will tune

| Artifact | Role |
|----------|------|
| `prompt.txt` | VLM prompt (goal + team: sportswear vs suits) |
| `classify.py` | Parsing, model config, temperature |
| `agent.py` | Entrypoint wrapper (usually thin) |

**Score to improve:** weighted match on `goal` + `team` in eval spec (F1-style aggregate).

## One-time setup

```bash
# install Overmind CLI
uv tool install overmind
# or: pipx install overmind

cd overmind-hack
overmind init                    # API keys → .overmind/.env

# export labeled clips to Overmind dataset format
python3 scripts/export_overmind_dataset.py --dataset 9-8GT-right
```

In Cursor / Claude Code (recommended), run skills in order:

```text
/overmind-register-agent agent.py
/overmind-generate-spec-and-dataset goal-classifier
/overmind-optimize-agent goal-classifier
```

Or CLI-style (from repo root):

```bash
overmind agent register goal-classifier agent:run
overmind agent validate goal-classifier --data data/seed.json
overmind setup goal-classifier --data data/seed.json
overmind optimize goal-classifier
```

## Agent contract

`agent.py` exposes:

```python
def run(input_data: dict) -> dict:
    # input:  { "clip_path": "data/9-8GT-right/goal_01_....mp4", "dataset": "9-8GT-right" }
    # output: { "goal": true/false, "team": "Dark sportswear" | null, "raw": "..." }
```

`data/seed.json` rows look like:

```json
{
  "input": { "clip_path": "data/9-8GT-right/goal_05_606s_Dark-suits.mp4", "dataset": "9-8GT-right" },
  "expected_output": { "goal": true, "team": "Dark suits" }
}
```

## Policy hints (for policies.md)

Suggested rules for the optimizer:

- A goal requires the ball to fully cross the line into the net.
- Saves, blocks, post hits, and shots wide are **not** goals.
- Team assignment: sportswear/tracksuits vs suits (both wear dark kits — use clothing style).
- If `goal` is false, `team` must be null.

## After optimization

Results land in `.overmind/agents/goal-classifier/experiments/`:

- `best_agent.py` — highest-scoring prompt/code version
- `traces/` — full LLM call traces per test case
- `report.md` — score history and diffs

Copy winning `prompt.txt` changes back to the repo root when done.

## Env vars

| Var | Used by |
|-----|---------|
| `GEMINI_API_KEY` | `agent.py` / `classify.py` (repo `.env`) |
| Overmind analyzer keys | `.overmind/.env` from `overmind init` |

Both may need the same Gemini key depending on your Overmind provider setup.
