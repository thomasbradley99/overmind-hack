"""
Stage 4: Extract JSON
- Takes narrative from stage 3
- Converts to structured JSON matching our schema
- Output: { events: [...], metadata: {...} }

⚠️ SCHEMA CONTRACT: Output MUST match /shared/EVENT_SCHEMA.ts
   - GameEvent: id, time, team, action, description, player?
   - EventAction: "Goal" | "Near Miss" | "Big Hit"
   - GameMetadata: teams, narrative, intensity_score, possession, team_intensity

DO NOT MODIFY without checking EVENT_SCHEMA.ts compatibility!
"""

import json
import os
import google.generativeai as genai

# =============================================================================
# SCHEMA DEFINITION (must match /shared/EVENT_SCHEMA.ts)
# =============================================================================

VALID_ACTIONS = ["Goal", "Near Miss", "Big Hit"]

def validate_event(event: dict, teams: list[str]) -> tuple[bool, list[str]]:
    """
    Validate a single event against EVENT_SCHEMA.ts
    Returns (is_valid, list_of_errors)
    """
    errors = []
    
    # Required: id (string)
    if not isinstance(event.get("id"), str):
        errors.append("Missing or invalid 'id' (must be string)")
    
    # Required: time (number, seconds from video start)
    if not isinstance(event.get("time"), (int, float)):
        errors.append("Missing or invalid 'time' (must be number)")
    elif event["time"] < 0:
        errors.append(f"Invalid 'time': {event['time']} (must be >= 0)")
    
    # Required: team (string, must match one of the teams)
    if not isinstance(event.get("team"), str):
        errors.append("Missing or invalid 'team' (must be string)")
    elif event["team"] not in teams:
        errors.append(f"Invalid 'team': '{event['team']}' not in {teams}")
    
    # Required: action (must be one of VALID_ACTIONS)
    action = event.get("action")
    if action not in VALID_ACTIONS:
        errors.append(f"Invalid 'action': '{action}' not in {VALID_ACTIONS}")
    
    # Required: description (non-empty string)
    if not isinstance(event.get("description"), str) or len(event.get("description", "")) == 0:
        errors.append("Missing or empty 'description'")
    
    # Optional: player (string if present)
    if "player" in event and not isinstance(event["player"], str):
        errors.append("Invalid 'player' (must be string if present)")
    
    return len(errors) == 0, errors


def validate_output(result: dict, teams: list[str]) -> tuple[bool, list[str]]:
    """
    Validate full output against EVENT_SCHEMA.ts
    Returns (is_valid, list_of_errors)
    """
    all_errors = []
    
    # Must have events array
    if not isinstance(result.get("events"), list):
        all_errors.append("Missing or invalid 'events' array")
        return False, all_errors
    
    # Validate each event
    for i, event in enumerate(result["events"]):
        is_valid, errors = validate_event(event, teams)
        if not is_valid:
            for error in errors:
                all_errors.append(f"Event {i}: {error}")
    
    # Must have metadata
    if not isinstance(result.get("metadata"), dict):
        all_errors.append("Missing 'metadata' object")
    
    return len(all_errors) == 0, all_errors


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run(narrative: str, teams: list[str], video_duration: float, api_key: str) -> dict:
    """
    Extract structured JSON from narrative
    
    Args:
        narrative: Text narrative from stage 3
        teams: ["Team A", "Team B"]
        video_duration: Total video duration in seconds
        api_key: Gemini API key
    
    Returns:
        {
            "events": [GameEvent, ...],  # Matches EVENT_SCHEMA.ts
            "metadata": {...}
        }
    """
    print("📊 Stage 4: Extracting JSON...")
    
    # Configure Gemini - use same model as rest of optimized pipeline
    genai.configure(api_key=api_key)
    model_name = os.environ.get("STAGE4_MODEL", "gemini-2.5-pro")
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(temperature=0.0)
    )
    
    prompt = f"""Convert this match narrative into structured JSON.

NARRATIVE:
{narrative}

OUTPUT FORMAT (valid JSON only):
{{
    "events": [
        {{
            "time": 383,
            "action": "Goal",
            "team": "{teams[0]}",
            "description": "Close range finish after good passing move",
            "player": "#7"
        }}
    ],
    "metadata": {{
        "teams": ["{teams[0]}", "{teams[1]}"],
        "narrative": "Copy the MATCH SUMMARY - keep the entertaining style",
        "key_players": "Copy the KEY PLAYERS section if present, otherwise empty string",
        "intensity_score": 65,
        "possession": {{
            "{teams[0]}": 55,
            "{teams[1]}": 45
        }},
        "team_intensity": {{
            "{teams[0]}": "High pressing",
            "{teams[1]}": "Counter-attacking"
        }},
        "final_score": {{
            "{teams[0]}": 3,
            "{teams[1]}": 2
        }}
    }}
}}

RULES:
1. "time" must be in SECONDS (convert MM:SS to seconds, e.g., 06:23 = 383)
2. "action" must be EXACTLY one of: "Goal", "Near Miss", "Big Hit"
3. "team" must be EXACTLY "{teams[0]}" or "{teams[1]}"
4. "description" - keep it FACTUAL, copy from narrative accurately
5. "player" - include if mentioned (e.g., "#7", "keeper", "tall guy")
6. "narrative" - copy the match summary with its personality
7. "intensity_score" is 0-100 based on how exciting the game was
8. "possession" - extract from POSSESSION section (must add to 100)
9. "team_intensity" - extract team style descriptions
10. Include ALL events from the narrative

Output ONLY valid JSON, no markdown, no explanation.
"""

    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Clean up JSON (remove markdown code blocks if present)
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"   ⚠️ Failed to parse JSON: {e}")
        print(f"   Raw response: {response_text[:500]}")
        # Return minimal valid structure
        result = {
            "events": [],
            "metadata": {
                "teams": teams,
                "narrative": "Analysis completed but JSON extraction failed.",
                "intensity_score": 50
            }
        }
    
    # ==========================================================================
    # NORMALIZE AND VALIDATE EVENTS
    # ==========================================================================
    
    events = result.get("events", [])
    valid_events = []
    
    for i, event in enumerate(events):
        # Skip if missing required time field
        if "time" not in event:
            print(f"   ⚠️ Skipping event {i}: missing 'time'")
            continue
        
        # Normalize action field (handle legacy "type" field)
        event_action = event.get("action") or event.get("type", "")
        
        # Map to valid actions
        if event_action not in VALID_ACTIONS:
            action_lower = event_action.lower()
            if "goal" in action_lower:
                event_action = "Goal"
            elif any(x in action_lower for x in ["hit", "tackle", "foul", "collision"]):
                event_action = "Big Hit"
            else:
                event_action = "Near Miss"  # Default fallback
        
        event["action"] = event_action
        event.pop("type", None)  # Remove legacy field
        
        # Add event ID
        event["id"] = f"event_{len(valid_events)+1:03d}"
        
        # Ensure team is valid (default to first team if invalid)
        if event.get("team") not in teams:
            print(f"   ⚠️ Event {i}: invalid team '{event.get('team')}', defaulting to '{teams[0]}'")
            event["team"] = teams[0]
        
        # Ensure description exists
        if not event.get("description"):
            event["description"] = f"{event_action} by {event.get('team', 'unknown team')}"
        
        valid_events.append(event)
    
    result["events"] = valid_events
    
    # ==========================================================================
    # ENSURE METADATA
    # ==========================================================================
    
    if "metadata" not in result:
        result["metadata"] = {}
    
    result["metadata"]["teams"] = teams
    
    # Intensity score (0-100)
    intensity = result["metadata"].get("intensity_score", 50)
    result["metadata"]["intensity_score"] = max(0, min(100, int(intensity)))
    
    # Key players
    if "key_players" not in result["metadata"]:
        result["metadata"]["key_players"] = ""
    
    # Possession (defaults to 50-50)
    if "possession" not in result["metadata"]:
        result["metadata"]["possession"] = {teams[0]: 50, teams[1]: 50}
    
    # Team intensity descriptions
    if "team_intensity" not in result["metadata"]:
        result["metadata"]["team_intensity"] = {teams[0]: "", teams[1]: ""}
    
    # ==========================================================================
    # FINAL VALIDATION
    # ==========================================================================
    
    is_valid, errors = validate_output(result, teams)
    if not is_valid:
        print(f"   ⚠️ Schema validation errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"      - {error}")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more")
    else:
        print(f"   ✓ Schema validation passed")
    
    print(f"   ✓ Extracted {len(valid_events)} events")
    print(f"   ✓ Intensity score: {result['metadata'].get('intensity_score', 'N/A')}")
    print(f"   ✓ Possession: {result['metadata'].get('possession', {})}")
    
    return result
