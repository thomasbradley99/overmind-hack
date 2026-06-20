# Distributed Model Inference Setup

## Goal
Run two models in parallel across two laptops to reduce total inference time and combine their strengths:
- **Laptop A (this machine)**: smolvlm2-2.2b — fast goal detection (high recall, low precision)
- **Laptop B (other machine)**: moondream:1.8b — precise team identification (high precision, low recall)

## Architecture

```
Laptop A (Local)                Laptop B (Remote)
+----------------------------+  +----------------------------+
|  smolvlm2-2.2b (Ollama)    |  |  moondream:1.8b (Ollama)  |
|  Port: 11434                 |  |  Port: 11434               |
|  Runs locally, fast (~1s)    |  |  Runs on remote, ~1.3s     |
|  High recall, catches all    |  |  High precision, team ID   |
|  goals, many false positives |  |  when it detects           |
+----------------------------+  +----------------------------+
              |                              |
              |   HTTP API (JSON + base64)     |
              +----------> Ensemble Script <---+
                           (on Laptop A)
```

## Ensemble Strategies

| Strategy | Goal Detection | Team ID | F1 | Precision | Recall | Team Acc |
|----------|---------------|---------|-----|-----------|--------|----------|
| smolvlm2 alone | Always "goal" | Random | 71.4% | 55.6% | 100% | 60% |
| moondream direct | Only when sure | 100% accurate | 33.3% | 100% | 20% | 100% |
| moondream v2 | More aggressive | 66.7% | 50.0% | 42.9% | 60% | 66.7% |
| **OR (Union)** | If either says goal | From moondream if available | **71.4%** | **55.6%** | **100%** | **80%** |
| AND (Intersection) | Only if both agree | moondream team | 33.3% | 100% | 20% | 100% |
| **Cascade** | smolvlm2 first, moondream confirms | moondream if confirmed | 33.3% | 100% | 20% | 100% |

**Best strategy for this dataset:** OR (Union) — same F1 as smolvlm2 but better team accuracy (80% vs 60%)

## Setup Instructions

### Laptop A (This Machine - Local)

Already configured with Ollama running on localhost:11434.

### Laptop B (Remote Machine)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull moondream:1.8b
ollama pull moondream:1.8b

# 3. Configure Ollama to listen on all interfaces
# Edit the Ollama service:
sudo systemctl edit ollama
# Add:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"
# Then:
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 4. Verify it's accessible from Laptop A
curl http://<LAPTOP_B_IP>:11434/api/tags
```

### Firewall (if needed)

```bash
# On Laptop B, allow port 11434
sudo ufw allow 11434/tcp  # Ubuntu/Debian
sudo firewall-cmd --add-port=11434/tcp --permanent  # Fedora/RHEL
```

## Running the Ensemble

```bash
# Find Laptop B's IP address
python3 run_distributed_ensemble.py --remote-ip <LAPTOP_B_IP> --mode or
```

Modes:
- `or`: Union — if either model says goal, it's a goal (best team accuracy)
- `and`: Intersection — only if both agree (best precision, low recall)
- `cascade`: smolvlm2 first, moondream confirms (balanced)
- `smolvlm2-only`: Baseline comparison
- `moondream-only`: Baseline comparison

## Network Topology (exo-style)

While exo labs is designed for macOS with MLX, we replicate its distributed pattern using Ollama's HTTP API:

- **Automatic discovery**: Manual IP configuration (exo's auto-discovery requires specific networking)
- **API compatibility**: Ollama's `/api/chat` endpoint (same format as OpenAI, compatible with exo's API layer)
- **Load balancing**: Run models on separate machines, parallel inference
- **Fault tolerance**: If remote fails, falls back to local model

## Performance

| Setup | Time per clip | Total 9 clips | Bottleneck |
|-------|--------------|---------------|------------|
| Single machine, sequential | ~2.5s | ~23s | Sequential |
| Distributed, parallel | ~1.3s | ~12s | Network latency |
| Distributed with batching | ~1.0s | ~9s | Max(smolvlm2, moondream) |

