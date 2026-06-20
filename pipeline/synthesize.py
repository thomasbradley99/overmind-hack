"""
Stage 3: Synthesize / Interpret
- Takes text observations from Stage 2 (what AI saw in each chunk)
- Interprets: identifies goals, near misses, big hits
- Deduplicates events from overlapping chunks
- Output: Narrative text with confirmed events
"""

import os
import google.generativeai as genai


def run(observations: str, teams: list[str], video_duration: float, api_key: str) -> str:
    """
    Interpret observations and create narrative with identified events.
    
    Args:
        observations: Combined text observations from all chunks (Stage 2 output)
        teams: ["Team A", "Team B"]
        video_duration: Total video duration in seconds
        api_key: Gemini API key
    
    Returns:
        Narrative text with confirmed events (goals, near misses, big hits)
    """
    print("🧠 Stage 3: Interpreting observations...")
    
    if not observations or len(observations.strip()) < 50:
        print("   ⚠️ No observations to interpret")
        return f"""MATCH SUMMARY:
{teams[0]} vs {teams[1]}
Duration: {video_duration/60:.0f} minutes

No significant action observed in this game.
"""
    
    # Check for custom prompt from optimizer
    prompt_file = os.environ.get("STAGE3_PROMPT_FILE")
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file) as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("{team1}", teams[0])
        prompt = prompt.replace("{team2}", teams[1])
        prompt = prompt.replace("{duration_min}", str(int(video_duration // 60)))
        prompt = prompt.replace("{observations}", observations)
        print(f"   Using custom prompt: {prompt_file}")
    else:
        # Default prompt - INTERPRET observations
        prompt = f"""You are analyzing observations from a 5-a-side football game.

An AI watched the video in chunks and described what it saw. Your job is to:
1. READ the observations carefully
2. IDENTIFY actual events (goals, near misses, big hits)
3. DEDUPLICATE - if same event described twice (overlapping chunks), count once
4. CREATE a match narrative

Teams: {teams[0]} vs {teams[1]}
Duration: {video_duration/60:.0f} minutes

OBSERVATIONS FROM VIDEO:
{observations}

---

YOUR TASK: Interpret these observations and identify events.

HOW TO IDENTIFY:
- GOAL: Ball clearly goes into the net + usually celebrations follow
  - "ball goes in net" / "scores" / "players celebrate" = GOAL
  - Don't count saves, shots wide, shots blocked

- NEAR MISS: Shot that almost scored but didn't
  - "keeper saves" / "hits post" / "just wide" / "blocked" = NEAR MISS
  - Also nice skills that created chances

- BIG HIT: Physical challenges, tackles, fouls
  - "strong tackle" / "player down" / "collision" / "foul" = BIG HIT

DEDUPLICATION:
- If the same moment appears in multiple chunks (similar timestamp ±5s, same description), count it ONCE
- Use the most accurate timestamp

OUTPUT FORMAT:

---
MATCH SUMMARY:
[2-3 entertaining sentences about the game. Be a fun commentator!]

CONFIRMED EVENTS:
[List each event with timestamp, type, and description]
MM:SS - GOAL - [Team]: [What happened]
MM:SS - NEAR MISS - [Team]: [What happened]  
MM:SS - BIG HIT - [Team]: [What happened]

KEY PLAYERS:
[Any players who stood out from the observations - "the tall guy", "their keeper", etc.]

POSSESSION ESTIMATE:
{teams[0]}: XX%
{teams[1]}: YY%

FINAL SCORE:
{teams[0]} X - Y {teams[1]}
---

IMPORTANT:
- Only count actual GOALS - ball must go in net
- Be conservative - if unclear whether it was a goal or save, call it a near miss
- Timestamps should be in MM:SS format
- List events chronologically
"""

    # Configure Gemini
    genai.configure(api_key=api_key)
    model_name = os.environ.get("STAGE3_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(temperature=0.0)
    )
    print(f"   Using model: {model_name}")
    
    print(f"   Processing {len(observations)} chars of observations...")
    response = model.generate_content(prompt)
    
    narrative = response.text.strip()
    
    # Count identified events
    lines = narrative.lower()
    goal_count = lines.count("- goal -")
    miss_count = lines.count("- near miss -")
    hit_count = lines.count("- big hit -")
    
    print(f"   ✓ Narrative generated")
    print(f"   ✓ Identified: {goal_count} goals, {miss_count} near misses, {hit_count} big hits")
    
    return narrative
