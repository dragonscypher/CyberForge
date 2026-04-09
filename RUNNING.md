# Running CyberForge — Detailed Setup

## Prerequisites

| Requirement | Version                   | Notes                                                      |
| ----------- | ------------------------- | ---------------------------------------------------------- |
| Python      | 3.10+                     | 3.12 recommended                                           |
| Ollama      | Any recent                | Required for chat, benchmark, and GGUF quantization        |
| NVIDIA GPU  | Optional                  | CPU-only mode works; GPU accelerates quantization/training |
| OS          | Windows 10+, Linux, macOS | Tested on Windows; Linux/macOS should work                 |

## Step-by-Step

### 1. Clone the repo

```bash
git clone https://github.com/dragonscypher/CyberForge.git
cd CyberForge
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Core (CPU-safe, no PyTorch)
pip install -e ".[dev]"

# Optional: GPU support (requires NVIDIA GPU + CUDA)
pip install -e ".[gpu]"

# Optional: Cyber/IDS pipeline
pip install -e ".[cyber]"
```

### 4. Install Ollama

Download from https://ollama.com/download and start it:

```bash
ollama serve
```

Pull a model to get started:

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

### 5. (Optional) Configure environment

```bash
cp .env.example .env
# Edit .env to set HF_TOKEN, OLLAMA_HOST, etc.
```

### 6. Start CyberForge

```bash
uvicorn apps.api.main:app --reload
```

Open http://localhost:8000 in your browser.

## What You'll See

1. **Hardware Profile** — auto-detected GPU, CPU, RAM, and backend availability
2. **Model Discovery** — browse local Ollama models and search Hugging Face
3. **Recommendations** — task-aware model suggestions gated by your hardware
4. **Optimize** — quantize (GGUF), prune, distill, or edit models
5. **Benchmark** — measure latency, throughput, and TTFT
6. **Compare** — side-by-side benchmark comparison with HTML reports
7. **Chat** — interactive chat with any Ollama model

## Hugging Face Token

A token is **only** needed to:
- Download gated models (Llama, Mistral, etc.)
- Use `[gpu]` quantization methods that download from HF (AWQ, GPTQ)

Public models and all Ollama operations work without a token.

Get a token at: https://huggingface.co/settings/tokens

Set it via:
- Environment: `HF_TOKEN=hf_...`
- Or: enter it in the UI's onboarding/settings page

## Troubleshooting

| Problem                      | Solution                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| "Ollama not reachable"       | Run `ollama serve` in another terminal                                                     |
| "PyTorch CUDA not available" | Install CUDA torch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| "bitsandbytes not found"     | `pip install bitsandbytes` (Linux) or use WSL (Windows support is limited)                 |
| Slow operations on CPU       | Expected — GPU is recommended for training/distillation                                    |
| Port 8000 in use             | `uvicorn apps.api.main:app --reload --port 8001`                                           |

## Running Tests

```bash
python -m pytest tests/ -v
```

Currently 76 tests covering capability detection, preflight checks, and machine-class gating across 5 system tiers.
