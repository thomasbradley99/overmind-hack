# Agent Policy: Agent

## 1. Domain Knowledge

### 1.1 Purpose & Context
The agent classifies a single football clip to determine whether a goal was scored and, if so, which team scored. It operates locally using a vision‑language model (Ollama) and returns a structured result for downstream evaluation.

### 1.2 Domain Rules
- The agent must accept a clip path (`clip_path` or `video_path`) and an optional dataset name; default dataset is **9‑8GT‑right**. *(inferred)*
- It must load the two team names from the dataset’s `info.json`; if unavailable, default to **“Team A”** and **“Team B”**. *(inferred)*
- The output must contain a boolean `goal`, a string or `null` `team`, a non‑empty `raw` string, and an optional `error`. *(inferred)*
- If `goal` is `true`, `team` must be a non‑null value; if `goal` is `false`, `team` must be `null`. *(inferred)*
- The agent must use the environment variable `CLASSIFIER_MODEL` (or the Ollama default) and `OLLAMA_HOST` for the model call. *(inferred)*
- The agent must extract three evenly spaced frames from the clip, encode them in base64, and send them to the model. *(inferred)*

### 1.3 Domain Edge Cases
- **Missing clip path**: return `goal: false`, `team: null`, `raw: "missing clip_path"`, and an `error`. *(inferred)*
- **File not found**: return `goal: false`, `team: null`, `raw: "file not found: …"`, and an `error`. *(inferred)*
- **No API key**: return `goal: false`, `team: null`, `raw: "GEMINI_API_KEY not set"`, and an `error`. *(inferred)*
- **Frame extraction failure**: return `goal: false`, `team: null`, `raw: error message`, and an `error`. *(inferred)*
- **Model output cannot be parsed**: default to `goal: false`, `team: null`, and include the raw text for debugging. *(inferred)*

### 1.4 Terminology & Definitions
- **Goal**: A successful score by either team.
- **Team**: The club that scored the goal.
- **Raw**: The verbatim response from the vision‑language model.
- **Error**: A human‑readable message indicating why classification failed.

## 2. Agent Behavior

### 2.1 Output Constraints
- `goal`: boolean, never omitted.
- `team`: string or `null`; must align with `goal` per consistency rule.
- `raw`: non‑empty string containing the model’s full response.
- `error`: optional; present only when processing fails.

### 2.2 Tool Usage
The agent uses no external tools beyond local functions: `load_teams`, `build_prompt`, `classify_one`, and `normalize_team`. All I/O is performed locally.

### 2.3 Decision Mapping
1. Load clip and team names.  
2. Build prompt with `{team1}`/`{team2}` placeholders.  
3. Call `classify_one` to obtain `raw`.  
4. Parse `raw` via `_extract_prediction` to get `goal` (bool) and `pred_team`.  
5. Normalize `pred_team` against the dataset’s team list.  
6. Return structured output.

### 2.4 Quality Expectations
- **Accuracy**: Correctly detect goals and assign the proper team.  
- **Robustness**: Gracefully handle missing files, API keys, or parsing errors.  
- **Traceability**: Provide the raw model output for debugging.  
- **Consistency**: Enforce the goal‑team correlation rule.  
- **Performance**: Extract a small number of frames (3) to keep inference fast.
