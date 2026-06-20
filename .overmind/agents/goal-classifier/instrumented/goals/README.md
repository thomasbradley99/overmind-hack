# goals/ — labeled goal candidates

Each subfolder is one goal candidate from a game, named:

```
<game>_<seconds>s_<TP|FP>
```

- `TP` = true positive (a real goal at that timestamp)
- `FP` = false positive (something that looked like a goal but wasn't)

Every folder contains an `info.json`:

```json
{
  "game": "9-8GT-right",
  "goal_time_in_video": 1406,     // seconds into the full game
  "goal_time_in_clip": 10.0,      // seconds into the (optional) clip
  "team": "Dark suits",
  "teams": ["Dark sportswear", "Dark suits"],
  "is_real_goal": true,
  "type": "TP",
  "description": "Low finish into the bottom corner from the right side of the box",
  "clip_start": 1396.0,
  "clip_duration": 20.0
}
```

These are the human-verified labels. Use them as ground truth for goal
detection, or to cut clips from the full game in `games/`.
