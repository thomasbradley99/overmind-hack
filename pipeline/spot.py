"""
Stage 2: Spot / Observe
- Split video into chunks
- Send each chunk to Gemini for PURE OBSERVATION
- Output: Plain text descriptions of what happened (no classification)
- The AI just describes what it sees - interpretation happens in Stage 3
"""

import subprocess
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict
import google.generativeai as genai


def _orientation_filter(orientation: Optional[Dict]) -> Optional[str]:
    if not orientation:
        return None
    rotation = orientation.get("rotation_degrees", 0)
    if rotation == 90:
        return "transpose=1"
    if rotation == 270:
        return "transpose=2"
    if rotation == 180:
        return "hflip,vflip"
    return None


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip()) if result.stdout.strip() else 0


def chunk_video(
    video_path: str,
    work_dir: Path,
    chunk_duration: int = None,
    chunk_overlap: int = None,
    orientation: Optional[Dict] = None,
    max_duration: Optional[float] = None
) -> list[dict]:
    """
    Split video into chunks with optional overlap
    Returns list of chunk info: [{"path": Path, "start": 0, "end": 120}, ...]
    
    Args:
        max_duration: If provided, only chunk up to this duration (for partial analysis)
    """
    # Read from env vars if not provided
    if chunk_duration is None:
        chunk_duration = int(os.environ.get("CHUNK_DURATION", "60"))
    if chunk_overlap is None:
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "0"))
    
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    duration = get_video_duration(video_path)
    analysis_duration = min(duration, max_duration) if max_duration else duration
    print(f"   Video duration: {duration:.0f}s ({duration/60:.1f} min)")
    if max_duration and max_duration < duration:
        print(f"   ⚠️  Limiting analysis to first {analysis_duration:.0f}s ({analysis_duration/60:.1f} min)")
    print(f"   Chunk settings: {chunk_duration}s duration, {chunk_overlap}s overlap")
    
    chunks = []
    start = 0
    chunk_num = 0
    
    # Calculate step size (chunk_duration - overlap)
    step = max(chunk_duration - chunk_overlap, 1)
    
    orientation_filter = _orientation_filter(orientation)

    while start < analysis_duration:
        end = min(start + chunk_duration, duration)
        chunk_path = chunks_dir / f"chunk_{chunk_num:03d}.mp4"
        
        if orientation_filter:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', str(video_path),
                '-t', str(chunk_duration),
                '-vf', orientation_filter,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                str(chunk_path)
            ]
        else:
            # Stream copy - MUCH faster (no re-encoding)
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', str(video_path),
                '-t', str(chunk_duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                str(chunk_path)
            ]
        
        subprocess.run(cmd, capture_output=True, timeout=30)
        
        if chunk_path.exists():
            chunks.append({
                "path": chunk_path,
                "start": start,
                "end": end,
                "chunk_num": chunk_num
            })
            print(f"   ✓ Chunk {chunk_num}: {start//60}:{start%60:02d} - {end//60:.0f}:{end%60:02.0f}")
        
        start += step
        chunk_num += 1
    
    return chunks


def observe_chunk(chunk: dict, teams: list[str], total_duration: float, api_key: str) -> dict:
    """
    Observe a single chunk - PURE DESCRIPTION, no classification.
    Returns: {"chunk_num": N, "start": S, "end": E, "description": "text..."}
    """
    chunk_num = chunk["chunk_num"]
    chunk_start = chunk["start"]
    chunk_end = chunk["end"]
    chunk_path = chunk["path"]
    
    # Configure Gemini
    temperature = float(os.environ.get("STAGE2_TEMPERATURE", "0.0"))
    model_name = os.environ.get("STAGE2_MODEL", "gemini-2.5-flash")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(temperature=temperature)
    )
    
    start_min = chunk_start // 60
    start_sec = chunk_start % 60
    end_min = chunk_end // 60
    end_sec = chunk_end % 60
    total_min = total_duration // 60
    
    team1 = teams[0] if len(teams) > 0 else "Team A"
    team2 = teams[1] if len(teams) > 1 else "Team B"
    
    # Check for custom prompt from optimizer
    prompt_file = os.environ.get("STAGE2_PROMPT_FILE")
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file) as f:
            prompt_template = f.read()
        # Fill in template variables
        prompt = prompt_template.replace("{start_min}", str(start_min))
        prompt = prompt.replace("{start_sec}", str(int(start_sec)))
        prompt = prompt.replace("{end_min}", str(int(end_min)))
        prompt = prompt.replace("{end_sec}", str(int(end_sec)))
        prompt = prompt.replace("{total_min}", str(int(total_min)))
        prompt = prompt.replace("{team1}", team1)
        prompt = prompt.replace("{team2}", team2)
        if chunk_num == 0:
            print(f"   Using custom prompt: {prompt_file}")
    else:
        # Default prompt - PURE OBSERVATION
        # Calculate chunk duration for the prompt
        chunk_duration_sec = int(chunk_end - chunk_start)
        chunk_duration_min = chunk_duration_sec // 60
        chunk_duration_remaining_sec = chunk_duration_sec % 60
        
        prompt = f"""You are watching a 5-a-side football game.
This is a {chunk_duration_sec}-second clip from a {total_min:.0f}-minute game.
Teams: {team1} vs {team2}

CONTEXT: This clip is from {start_min}:{start_sec:02.0f} to {end_min}:{end_sec:02.0f} in the full video.
But YOUR timestamps should be RELATIVE TO THIS CLIP (starting from 0:00).

YOUR TASK: Describe what happens in this clip. Be a neutral observer.

DESCRIBE:
- Any shots on goal (who shot, what happened - saved, scored, missed, hit post)
- Any tackles, fouls, or physical challenges
- Any notable skills, passes, or plays
- Any celebrations (they indicate something happened!)

FORMAT: Write plain text, chronologically. For each notable moment:
[MM:SS] What happened

Example:
[00:15] Player in dark shoots from edge of box, keeper dives right and saves
[00:34] Strong tackle by player in light, wins the ball cleanly
[00:58] Quick passing move, shot goes just wide of far post

CRITICAL - TIMESTAMPS:
- Use timestamps RELATIVE TO THIS CLIP starting from [00:00]
- This clip is {chunk_duration_sec} seconds long, so timestamps should be between [00:00] and [{chunk_duration_min}:{chunk_duration_remaining_sec:02d}]
- Example: If something happens 30 seconds into this clip, write [00:30]
- DO NOT use the full video timestamp - we will add that later

OTHER:
- Describe WHAT YOU SEE, not what you think it means
- If ball goes in net + celebrations = describe that ("ball goes in net, players celebrate")
- Don't classify as "goal" vs "near miss" - just describe what happened
- If nothing notable happens, write "No significant action in this segment"
"""

    try:
        # Upload video to Gemini
        video_file = genai.upload_file(chunk_path, mime_type="video/mp4")
        
        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            print(f"   ⚠️ Chunk {chunk_num}: Video processing failed")
            return {
                "chunk_num": chunk_num,
                "start": chunk_start,
                "end": chunk_end,
                "description": "[Video processing failed]",
                "error": "Video processing failed"
            }
        
        # Generate content
        response = model.generate_content([video_file, prompt])
        
        # Clean up uploaded file
        genai.delete_file(video_file.name)
        
        description = response.text.strip()
        
        # Log what we found
        if "no significant" in description.lower():
            print(f"   ○ Chunk {chunk_num}: No action")
        else:
            line_count = len([l for l in description.split('\n') if l.strip().startswith('[')])
            print(f"   ★ Chunk {chunk_num}: {line_count} moment(s) described")
        
        return {
            "chunk_num": chunk_num,
            "start": chunk_start,
            "end": chunk_end,
            "description": description,
            "prompt_used": prompt[:300] + "..." if len(prompt) > 300 else prompt
        }
        
    except Exception as e:
        print(f"   ⚠️ Chunk {chunk_num}: Error - {e}")
        return {
            "chunk_num": chunk_num,
            "start": chunk_start,
            "end": chunk_end,
            "description": f"[Error: {str(e)}]",
            "error": str(e)
        }


def run(
    video_path: str,
    work_dir: Path,
    teams: list[str],
    api_key: str,
    orientation: Optional[Dict] = None,
    max_duration: Optional[float] = None
) -> dict:
    """
    Observe video chunks (parallel) - outputs TEXT descriptions, not JSON flags.
    
    Returns:
        {
            "chunk_observations": [...],  # Text description per chunk
            "combined_observations": "...",  # All descriptions combined
            "config": {...}
        }
    """
    print("👁️ Stage 2: Observing video...")
    
    # Log config from env vars
    chunk_duration = int(os.environ.get("CHUNK_DURATION", "60"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "0"))
    temperature = float(os.environ.get("STAGE2_TEMPERATURE", "0.0"))
    model_name = os.environ.get("STAGE2_MODEL", "gemini-2.5-flash")
    print(f"   Config: chunks={chunk_duration}s, overlap={chunk_overlap}s, temp={temperature}, model={model_name}")
    
    # Get video duration
    duration = get_video_duration(video_path)
    
    # Chunk video (respect max_duration if provided)
    print(f"   Chunking video into {chunk_duration}s segments (overlap={chunk_overlap}s)...")
    chunks = chunk_video(video_path, work_dir, chunk_duration=chunk_duration, chunk_overlap=chunk_overlap, orientation=orientation, max_duration=max_duration)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # Observe chunks in parallel
    print("   Observing chunks (parallel)...")
    all_results = []
    
    max_workers = min(len(chunks), 25)  # Limit concurrent uploads to avoid SSL errors
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(observe_chunk, chunk, teams, duration, api_key): chunk
            for chunk in chunks
        }
        
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
    
    # Sort by chunk number
    all_results.sort(key=lambda x: x["chunk_num"])
    
    # Combine all observations into one text with GLOBAL timestamps
    import re
    combined = ""
    
    # Get chunk duration from config (default 60s)
    chunk_duration = int(os.environ.get("CHUNK_DURATION", "60"))
    
    for obs in all_results:
        chunk_start = obs['start']  # Global start time in seconds
        chunk_end = obs['end']
        chunk_header = f"\n--- Chunk {obs['chunk_num']} ({chunk_start//60:.0f}:{chunk_start%60:02.0f} - {chunk_end//60:.0f}:{chunk_end%60:02.0f}) ---\n"
        
        description = obs["description"]
        
        def convert_timestamp(match):
            """
            Convert timestamps to global time.
            Gemini sometimes ignores the prompt and gives global timestamps anyway.
            If timestamp > chunk_duration, assume it's already global and don't add offset.
            """
            time_str = match.group(1)
            parts = time_str.split(':')
            if len(parts) == 2:
                timestamp_seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                timestamp_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return match.group(0)  # Return unchanged if can't parse
            
            # If timestamp is within chunk bounds (0 to chunk_duration+buffer), 
            # treat as chunk-relative and add offset
            # Otherwise, Gemini gave us a global timestamp - use as-is
            if timestamp_seconds <= chunk_duration + 15:  # +15s buffer for overlap
                global_seconds = chunk_start + timestamp_seconds
            else:
                # Already a global timestamp, don't add offset
                global_seconds = timestamp_seconds
            
            global_min = int(global_seconds // 60)
            global_sec = int(global_seconds % 60)
            return f"[{global_min:02d}:{global_sec:02d}]"
        
        # Replace timestamps like [00:45] with global time
        description_global = re.sub(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', convert_timestamp, description)
        
        combined += chunk_header + description_global + "\n"
    
    # Count chunks with action
    action_chunks = sum(1 for obs in all_results if "no significant" not in obs.get("description", "").lower())
    
    print(f"   ✓ Observed {len(chunks)} chunks ({action_chunks} with notable action)")
    
    return {
        "chunk_observations": all_results,
        "combined_observations": combined,
        "config": {
            "chunk_duration": chunk_duration,
            "chunk_overlap": chunk_overlap,
            "temperature": temperature,
            "model": model_name
        }
    }
