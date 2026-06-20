# exo Distributed Ensemble Setup — macOS + Linux

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     exo Cluster (Auto-Discovery)                    │
│  ┌──────────────────────┐          ┌──────────────────────────┐  │
│  │  macOS Laptop (Node A) │          │  Linux Laptop (Node B)     │  │
│  │  ┌────────────────────┐│          │  ┌──────────────────────┐  │  │
│  │  │ MLX Backend (GPU)  ││          │  │  CPU Backend           │  │  │
│  │  │                    ││          │  │  (Linux MLX-CPU)       │  │  │
│  │  │ smolvlm2-2.2b      ││          │  │                      │  │  │
│  │  │ (Recall: 100%)     ││          │  │ moondream:1.8b       │  │  │
│  │  │ 1.2GB, ~0.5s/frame ││          │  │ (Precision: 100%)    │  │  │
│  │  │                    ││          │  │ 1.7GB, ~1.3s/frame   │  │  │
│  │  └────────────────────┘│          │  └──────────────────────┘  │  │
│  │  Automatic discovery   │          │  Automatic discovery       │  │
│  │  via libp2p (mDNS)   │◄────────►│  via libp2p (mDNS)       │  │
│  │  + Thunderbolt/RDMA   │          │  + TCP fallback          │  │
│  └──────────────────────┘          └──────────────────────────┘  │
│                                                                     │
│  Cluster API: http://localhost:52415 (on any node)                  │
│  ├─ /v1/chat/completions — OpenAI-compatible                        │
│  ├─ /v1/images/generations — Image generation (exo only)            │
│  ├─ /ollama/api/chat — Ollama-compatible                            │
│  └─ /instance/previews — Model placement optimization               │
│                                                                     │
│  exo automatically shards the model across nodes based on:          │
│  - Memory availability                                              │
│  - Network latency (Thunderbolt 5 / RDMA on macOS)                │
│  - Topology-aware placement (pipeline vs tensor parallelism)        │
│                                                                     │
│  For our ensemble: we run two separate model instances, not shards │
│  - Each model is small enough for a single device                   │
│  - We route requests to the appropriate model node                  │
│  - exo handles load balancing and failover automatically            │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Differences from Ollama Approach

| Feature | exo | Ollama |
|---------|-----|--------|
| Auto-discovery | mDNS/libp2p, zero config | Manual IP config |
| Cross-platform | macOS + Linux (same binary) | Different binaries |
| Backend | MLX (GPU on macOS, CPU on Linux) | Various (llama.cpp, etc.) |
| Sharding | Automatic tensor/pipeline | N/A |
| API | OpenAI + Claude + Ollama compatible | Ollama only |
| Vision models | `EXO_ENABLE_IMAGE_MODELS` | Varies by model |
| RDMA | Thunderbolt 5 (macOS 26.2+) | N/A |

## macOS Setup (smolvlm2-2.2b Node)

### Prerequisites

- macOS Tahoe 26.2 or later (for RDMA, optional for older versions)
- Xcode (for Metal toolchain)
- Apple Silicon (M1 or newer)
- Homebrew

### Step-by-Step Installation

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install dependencies
brew install uv node

# 3. Install Rust (nightly required for exo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup toolchain install nightly

# 4. Install macmon (for hardware monitoring on Apple Silicon)
# Note: Use pinned fork, not Homebrew (Homebrew 0.6.1 crashes on M5)
cargo install --git https://github.com/vladkens/macmon \
  --rev a1cd06b6cc0d5e61db24fd8832e74cd992097a7d \
  macmon --force

# 5. Clone exo
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo

# 6. Build dashboard
cd dashboard && npm install && npm run build && cd ..

# 7. Enable image models (required for vision models like smolvlm2)
export EXO_ENABLE_IMAGE_MODELS=true

# 8. Run exo
uv run exo
```

### Verify exo is running

```bash
# Check dashboard
curl http://localhost:52415/state | jq

# Check available models
curl http://localhost:52415/models
```

### Add smolvlm2 model (custom from HuggingFace)

```bash
# smolvlm2 is available as mlx-community/SmolVLM-Instruct-4bit
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "mlx-community/SmolVLM-Instruct-4bit"}'

# Wait for download to complete
curl -N "http://localhost:52415/instance/await?model_id=mlx-community/SmolVLM-Instruct-4bit&timeout_seconds=300"
```

### Create smolvlm2 instance for inference

```bash
# Preview placement
curl "http://localhost:52415/instance/previews?model_id=smolvlm-instruct-4bit" | jq

# Create instance (use first valid placement)
INSTANCE=$(curl -s "http://localhost:52415/instance/previews?model_id=smolvlm-instruct-4bit" | \
  jq -r '.previews[] | select(.error == null) | .instance' | head -n1)

curl -X POST http://localhost:52415/instance \
  -H 'Content-Type: application/json' \
  -d "{\"instance\": $INSTANCE}"
```

### Test smolvlm2 inference

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/SmolVLM-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Analyze this image and determine if a goal was scored. Return ONLY JSON: {\"goal\": true/false, \"team\": \"team name\"}"}
    ],
    "stream": false
  }'
```

---

## Linux Setup (moondream Node)

### Prerequisites

- Linux (Ubuntu/Debian/Fedora)
- Python 3.13
- uv, node, rust

### Step-by-Step Installation

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env 2>/dev/null || true

# 2. Install Node.js
sudo apt update
sudo apt install -y nodejs npm

# 3. Install Rust (nightly)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup toolchain install nightly

# 4. Clone exo
cd ~
git clone https://github.com/exo-explore/exo.git
cd exo

# 5. Build dashboard
cd dashboard && npm install && npm run build && cd ..

# 6. Enable image models (required for vision models)
export EXO_ENABLE_IMAGE_MODELS=true

# 7. Run exo (CPU on Linux)
uv run exo
```

**Note:** On Linux, exo runs on CPU. GPU support is under development. For our small models (1.2-1.7GB), CPU is sufficient.

### Verify exo is running

```bash
curl http://localhost:52415/state | jq
curl http://localhost:52415/models
```

### Add moondream model (custom from HuggingFace)

```bash
# moondream is available as beshkenadze/moondream3-preview-mlx-4bit
curl -X POST http://localhost:52415/models/add \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "beshkenadze/moondream3-preview-mlx-4bit"}'

# Wait for download
curl -N "http://localhost:52415/instance/await?model_id=beshkenadze/moondream3-preview-mlx-4bit&timeout_seconds=300"
```

### Test moondream inference

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "beshkenadze/moondream3-preview-mlx-4bit",
    "messages": [
      {"role": "user", "content": "You are a football referee. Determine if a goal was scored and identify the team. Return JSON: {\"goal\": true/false, \"team\": \"team name\"}"}
    ],
    "stream": false
  }'
```

---

## Cluster Discovery and Verification

### How exo Auto-Discovery Works

1. **mDNS Multicast**: Both machines broadcast their presence on the local network
2. **libp2p**: Uses peer-to-peer networking for device discovery
3. **No manual IP configuration**: Devices find each other automatically

### Verify Cluster Formation

On **either** machine:

```bash
# Check cluster state — should show both nodes
curl http://localhost:52415/state | jq '.nodes'

# Expected output:
# [
#   {"node_id": "mac-node-...", "platform": "darwin", "devices": [...]},
#   {"node_id": "linux-node-...", "platform": "linux", "devices": [...]}
# ]
```

### Custom Namespace (Optional)

If multiple exo clusters on the same network, use a namespace to isolate:

```bash
# On BOTH machines
export EXO_LIBP2P_NAMESPACE=football-ensemble-cluster
uv run exo
```

---

## Ensemble Script for exo

```bash
# Run the ensemble using exo's distributed API
python3 run_exo_ensemble.py \
  --exo-url http://localhost:52415 \
  --mode or \
  --clip-dir data/9-8GT-right-quarter \
  --results-dir results/9-8GT-right-quarter
```

The script (`run_exo_ensemble.py`) uses exo's `/v1/chat/completions` API to send inference requests to both models. exo automatically routes to the node where each model is running.

---

## Troubleshooting

### macOS: "Cannot find mlx module"

```bash
# Ensure MLX is installed (should be automatic with uv)
uv pip install mlx mlx-lm
```

### Linux: "CPU only, no GPU"

This is expected. exo runs on CPU on Linux. For our 1.2-1.7GB models, CPU is sufficient.

### "Models not found"

```bash
# Check if models are loaded
curl http://localhost:52415/models

# Check instances
curl http://localhost:52415/state | jq '.instances'
```

### "Nodes not discovering each other"

- Check both machines are on the same network
- Check firewall allows mDNS (port 5353 UDP) and libp2p ports
- Try explicit namespace: `EXO_LIBP2P_NAMESPACE=...`

### "RDMA not working"

- Requires macOS 26.2+ and Thunderbolt 5
- Run `rdma_ctl enable` in Recovery mode
- See README.md for caveats

---

## Performance Expectations

| Setup | smolvlm2 | moondream | Parallel Total |
|-------|----------|-----------|----------------|
| macOS (GPU) | ~0.5s/frame | N/A | ~0.5s |
| Linux (CPU) | N/A | ~1.3s/frame | ~1.3s |
| Distributed | ~0.5s | ~1.3s | ~1.3s (parallel) |

**Key advantage:** With exo, both models run in parallel across the cluster. Total time = max(smolvlm2_time, moondream_time) = ~1.3s per clip, not sequential ~1.8s.

