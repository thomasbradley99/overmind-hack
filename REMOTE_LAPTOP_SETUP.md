# Remote Laptop Setup — Run moondream:1.8b for Distributed Ensemble

## Quick Start (5 minutes)

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull moondream:1.8b

```bash
ollama pull moondream:1.8b
```

### 3. Allow Network Access

Ollama by default only listens on localhost. To make it accessible from the other laptop, edit the systemd service:

```bash
sudo systemctl edit ollama
```

Add these lines:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 4. Check Firewall

```bash
# Ubuntu/Debian
sudo ufw allow 11434/tcp

# Fedora/RHEL
sudo firewall-cmd --add-port=11434/tcp --permanent
sudo firewall-cmd --reload
```

### 5. Find Your IP Address

```bash
ip addr show | grep "inet " | grep -v "127.0.0.1"
# Or simpler:
ip route get 1 | awk '{print $7; exit}'
```

Note this IP — you'll use it on the local laptop.

### 6. Test From Local Laptop

From the local laptop, run:

```bash
curl http://<REMOTE_IP>:11434/api/tags
```

You should see moondream in the list.

## What This Does

This laptop will run **moondream:1.8b** for football goal detection. The local laptop will send frames to this machine via HTTP, and moondream will analyze them.

moondream is:
- **Small** (1.7GB model, 1.8B parameters)
- **Fast** (~1.3 seconds per frame on CPU)
- **Precise** — when it detects a goal, it's 100% correct and identifies the team accurately

## Architecture

```
Local Laptop                          Remote Laptop (This Machine)
+----------------------------+        +-----------------------------+
| smolvlm2-2.2b (Ollama)     |        | moondream:1.8b (Ollama)     |
| Catches all goals (~1s)    |        | Precise when detects (~1.3s)|
| Many false positives       |        | 100% precision, 100% team   |
| 55.6% precision            |        | accuracy on detected goals  |
+----------------------------+        +-----------------------------+
       |                                         |
       |       HTTP POST to /api/chat             |
       +----------------------------------------->|
       |       Returns JSON with analysis         |
       |<-----------------------------------------+
       |                                         |
       +------> Ensemble Decision (OR/AND)       |
```

## Ensemble Results

| Strategy | F1 | Precision | Recall | Team Accuracy |
|----------|-----|-----------|--------|---------------|
| smolvlm2 alone | 71.4% | 55.6% | 100% | 60% |
| moondream alone | 33.3% | 100% | 20% | 100% |
| **OR (Union)** | **71.4%** | 55.6% | 100% | **80%** |
| AND (Intersection) | 33.3% | 100% | 20% | 100% |
| Cascade | 33.3% | 100% | 20% | 100% |

**Best strategy: OR** — same F1 as smolvlm2 but better team accuracy (80% vs 60%)

## Troubleshooting

### "Connection refused"
- Check Ollama is running: `sudo systemctl status ollama`
- Check it's listening on all interfaces: `netstat -tlnp | grep 11434` (should show 0.0.0.0:11434)
- Check firewall allows port 11434

### Model not found
```bash
ollama list  # Should show moondream:1.8b
ollama pull moondream:1.8b  # If not present
```

### Slow responses
- First inference after model load is slow (~10-20s)
- Subsequent inferences are fast (~1-2s)
- If the model was unloaded, it needs to reload (memory pressure)

## Advanced: Using exo labs Instead

If you want true exo-style distributed inference with automatic device discovery, install exo:

```bash
# Install prerequisites
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Clone and build
git clone https://github.com/exo-explore/exo.git
cd exo/dashboard && npm install && npm run build && cd ..

# Run (CPU-only on Linux)
EXO_ENABLE_IMAGE_MODELS=true uv run exo
```

Note: exo is primarily designed for macOS with Apple Silicon + MLX. On Linux it runs on CPU only and may not support vision models well. The Ollama approach above is more reliable for this use case.
