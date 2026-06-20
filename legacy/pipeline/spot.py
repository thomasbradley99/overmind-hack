"""
Stage 2: Spot / Observe
- Split video into chunks
- Extract frames from each chunk and send to local vision model
- Output: Plain text descriptions of what happened (no classification)
- The AI just describes what it sees - interpretation happens in Stage 3
"""

import subprocess
import json
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict

from .local_model import generate_with_images


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


def extract_frames(chunk_path: str, output_dir: Path, num_frames: int = 4) -> list[str]:
    """Extract evenly spaced frames from a video chunk."""
    chunk_path = Path(chunk_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    duration = get_video_duration(str(chunk_path))
    if duration <= 0:
        return []

    frames = []
    for i in range(num_frames):
        timestamp = duration * i / max(num_frames - 1, 1)
        frame_path = output_dir / f"frame_{i:03d}.jpg"
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', str(chunk_path),
            '-vframes', '1',
            '-q:v', '2',
            str(frame_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=15)
        if frame_path.exists():
            frames.append(str(frame_path))
    
    return frames


def observe_chunk(chunk: dict, teams: list[str], total_duration: float) -> dict:
    """
    Observe a single chunk by extracting frames and feeding them to the local vision model.
    Returns: {"chunk_num": N, "start": S, "end": E, "description": "text..."}
    """
    chunk_num = chunk["chunk_num"]
    chunk_start = chunk["start"]
    chunk_end = chunk["end"]
    chunk_path = chunk["path"]
    
    start_min = chunk_start // 60
    start_sec = chunk_start % 60
    end_min = chunk_end // 60
    end_sec = chunk_end % 60
    total_min = total_duration // 60
    
    team1 = teams[0] if len(teams) > 0 else "Team A"
    team2 = teams[1] if len(teams) > 1 else "Team B"
    
    # Extract frames from the chunk
    frames_dir = chunk_path.parent / f"chunk_{chunk_num:03d}_frames"
    frames = extract_frames(str(chunk_path), frames_dir, num_frames=4)
    
    if not frames:
        print(f"   ⚠️ Chunk {chunk_num}: No frames extracted")
        return {
            "chunk_num": chunk_num,
            "start": chunk_start,
            "end": chunk_end,
            "description": "[No frames extracted - unable to analyze]",
            "error": "Frame extraction failed"
        }

    # Check for custom prompt from optimizer
    prompt_file = os.environ.get("STAGE2_PROMPT_FILE")
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file) as f:
            prompt_template = f.read()
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
        chunk_duration_sec = int(chunk_end - chunk_start)
        chunk_duration_min = chunk_duration_sec // 60
        chunk_duration_remaining_sec = chunk_duration_sec % 60
        
        prompt = f"""You are watching a 5-a-side football game.
This is a {chunk_duration_sec}-second clip from a {total_min:.0f}-minute game.
Teams: {team1} vs {team2}

CONTEXT: This clip is from {start_min}:{start_sec:02.0f} to {end_min}:{end_sec:02.0f} in the full video.
But YOUR timestamps should be RELATIVE TO THIS CLIP (starting from 0:00).

YOUR TASK: Describe what happens in this clip. Be a neutral observer.

You are given {len(frames)} frames from this video clip, in chronological order.
Look at them carefully and describe the action.

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
        description = generate_with_images(
            prompt=prompt,
            image_paths=frames,
            max_tokens=512,
            temperature=0.0,
        ).strip()
        
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
    api_key: str = None,  # Kept for API compatibility but unused
    orientation: Optional[Dict] = None,
    max_duration: Optional[float] = None
) -> dict:
    """
    Observe video chunks using local vision model - outputs TEXT descriptions, not JSON flags.
    
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
    model_name = os.environ.get("STAGE2_MODEL", "local/SmolVLM")
    print(f"   Config: chunks={chunk_duration}s, overlap={chunk_overlap}s, temp={temperature}, model={model_name}")
    
    # Get video duration
    duration = get_video_duration(video_path)
    
    # Chunk video (respect max_duration if provided)
    print(f"   Chunking video into {chunk_duration}s segments (overlap={chunk_overlap}s)...")
    chunks = chunk_video(video_path, work_dir, chunk_duration=chunk_duration, chunk_overlap=chunk_overlap, orientation=orientation, max_duration=max_duration)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # Observe chunks sequentially (local model is CPU-bound, parallelism is limited)
    print("   Observing chunks (local model - sequential)...")
    all_results = []
    
    for chunk in chunks:
        result = observe_chunk(chunk, teams, duration)
        all_results.append(result)
    
    # Sort by chunk number (already in order, but just to be safe)
    all_results.sort(key=lambda x: x["chunk_num"])
    
    # Combine all observations into one text with GLOBAL timestamps
    combined = ""
    
    for obs in all_results:
        chunk_start = obs['start']  # Global start time in seconds
        chunk_end = obs['end']
        chunk_header = f"\n--- Chunk {obs['chunk_num']} ({chunk_start//60:.0f}:{chunk_start%60:02.0f} - {chunk_end//60:.0f}:{chunk_end%60:02.0f}) ---\n"
        
        description = obs["description"]
        
        def convert_timestamp(match):
            """
            Convert timestamps to global time.
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
            
            if timestamp_seconds <= chunk_duration + 15:  # +15s buffer for overlap
                global_seconds = chunk_start + timestamp_seconds
            else:
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
