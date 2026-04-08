# CyberForge

**A local-first AI model workbench for cybersecurity professionals.**  
Profile your hardware, discover and recommend models, quantize/prune/distill, benchmark, compare, chat — all from one UI served on a single port.

---

## What It Does

| Capability | Status |
|---|---|
| Hardware profiling (GPU, CPU, RAM, backends) | **Working** |
| Model discovery (local Ollama + Hugging Face search) | **Working** |
| Smart recommendation engine (task-aware, hardware-gated) | **Working** |
| Quantization (Ollama GGUF Q4/Q5/Q8, BnB, AWQ, GPTQ) | **Working** (Ollama path fully tested) |
| Pruning (magnitude / structured via Transformers) | **Working** (CPU + GPU) |
| Distillation (teacher→student via Transformers) | **Working** (CPU + GPU) |
| Model editing (rank-one ROME-style via Transformers) | **Working** (CPU + GPU) |
| Benchmarking (latency, throughput, TTFT) | **Working** (Ollama path) |
| Compare & report (side-by-side, HTML export) | **Working** |
| Chat interface | **Working** (Ollama backend) |
| Artifact lifecycle & cleanup | **Working** |
| Backend capability detection & preflight checks | **Working** |
| Cyber IDS pipeline (NSL-KDD, CICIDS, etc.) | **Scaffolded** — loaders wired, models not shipped |
| LoRA / QLoRA fine-tuning | **Scaffolded** — requires `[gpu]` extras |
| vLLM / TensorRT-LLM serving | **Scaffolded** — code present, not tested in demo |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/dragonscypher/CyberForge.git
cd CyberForge

# 2. Create venv & install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -e ".[dev]"

# 3. (Optional) GPU support
pip install -e ".[gpu]"

# 4. Install & start Ollama  (required for chat/benchmark/quantize)
# https://ollama.com/download
ollama serve                    # in a separate terminal

# 5. Launch CyberForge
uvicorn apps.api.main:app --reload
```

Open **http://localhost:8000** for the full UI, or **/docs** for the API explorer.

See [RUNNING.md](RUNNING.md) for detailed setup instructions.

## Configuration

All config is optional. The app ships with sensible defaults.

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `HF_TOKEN` | Hugging Face token for gated model downloads | *(none — public models work without it)* |
| `OPENROUTER_API_KEY` | Cloud inference via OpenRouter | *(none — local-only mode)* |
| `CYBERFORGE_CONFIG_DIR` | Where to store `config.yaml` | `data` |

Copy `.env.example` → `.env` to set these.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Browser UI                     │
│               http://localhost:8000               │
└────────┬────────────────────────────────────────┘
         │  in-process ASGI calls (httpx)
┌────────▼────────────────────────────────────────┐
│  FastAPI  (/api/hardware, /api/models, ...)      │
│           (/api/bench, /api/optimize, ...)        │
│           (/api/recommend, /api/reports, ...)     │
├──────────────────────────────────────────────────┤
│ packages/hardware/  – profiler, capability        │
│ packages/models/    – registry, discovery, quant  │
│ packages/bench/     – harness, metrics            │
│ packages/core/      – config, recommend, scoring  │
│ packages/serve/     – Ollama, OpenRouter clients  │
│ packages/reports/   – charts, HTML generator      │
│ packages/cyber/     – IDS datasets, verifiers     │
│ packages/train/     – LoRA/QLoRA (optional)       │
└──────────────────────────────────────────────────┘
```

## Repository Layout

```
apps/api/          FastAPI backend (main.py, routers, workers)
packages/core/     Config, recommendation engine, scoring, lifecycle
packages/hardware/ Hardware profiler & capability matrix
packages/models/   Model registry, HF discovery, quantization, pruning, distillation
packages/serve/    Ollama / OpenRouter / vLLM / TensorRT-LLM clients
packages/bench/    Benchmarking harness & metrics
packages/reports/  Chart generation & HTML report builder
packages/cyber/    Cyber datasets, prompt packs, verifiers
packages/train/    PEFT / QLoRA training loops
data/              Runtime data (cache, saved models, DB) — gitignored
reports/           Generated benchmark reports — gitignored
tests/             Capability & integration tests
ui.py              Full web UI (single-file, mounted in-process)
```

## System Requirements

- **Python** 3.10+
- **Ollama** (for chat, benchmark, GGUF quantization)
- **NVIDIA GPU** recommended but not required — CPU-only mode works for all transformers-based features
- ~4 GB free disk for a small quantized model + cache

## License

MIT — see [LICENSE](LICENSE).
