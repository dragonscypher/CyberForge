# Release Notes — CyberForge v0.1.0 (Public Demo)

**Released:** April 2026  
**Branch:** `release/demo-v0.1.0`  
**License:** MIT

---

## Working Features

### Hardware & Capability Detection
- Automatic GPU/CPU/RAM profiling via `nvidia-smi` + `psutil`
- Backend availability detection: Ollama, PyTorch CUDA, transformers, bitsandbytes, vLLM, TensorRT-LLM
- TTL-cached profiler (30s hardware / 5s services) to avoid redundant subprocess calls
- `OLLAMA_HOST` environment variable support (both profiler and quantization)
- Per-backend capability matrix with 5-state availability model
- Machine-class tier system: `cpu_only`, `low_vram`, `mid_vram`, `high_vram`
- Universal preflight checks before every operation (chat, benchmark, quantize, train)

### Model Discovery & Recommendation
- Local model listing from Ollama
- Hugging Face model search with task/size filtering
- Task-aware recommendation engine (general, coding, cyber)
- Hardware-gated: only recommends what your system can actually run

### Quantization
- **Ollama GGUF** (q4_0, q4_k_m, q5_k_m, q8_0) — fully tested end-to-end
- **BnB 4-bit / 8-bit** — requires PyTorch CUDA + bitsandbytes
- **AWQ** — requires PyTorch CUDA + autoawq
- **GPTQ** — requires PyTorch CUDA + auto-gptq
- Method availability detection with actionable install messages
- Quantization comparison across methods

### Optimization (Transformers-based)
- **Pruning** — magnitude and structured pruning via PyTorch
- **Distillation** — teacher→student knowledge distillation
- **Model editing** — rank-one model editing (ROME-style)
- All three work on CPU (`device_map="cpu"`) with GPU acceleration optional
- Iterative prune with preflight validation

### Benchmarking
- Latency, throughput (tokens/sec), time-to-first-token (TTFT)
- Ollama backend fully tested
- Benchmark history stored as JSON reports

### Compare & Report
- Side-by-side benchmark comparison
- HTML report generation with charts
- Report listing and retrieval via API

### Chat
- Interactive chat with any Ollama model
- Streaming responses

### Lifecycle Management
- Volatile artifact tracking and cleanup
- Auto-cleanup on startup/shutdown
- Manual cleanup endpoint

### Web UI
- Single-page app served on the same port as the API (`:8000`)
- In-process ASGI calls (zero network overhead between UI and API)
- Pages: Hardware Profile, Model Discovery, Recommend, Optimize, Benchmark, Compare, Chat, Reports, Settings

---

## Gated / Scaffolded Features

These features have code present but are not wired into the demo flow or require additional setup:

| Feature                                 | Status                         | What's Needed                                                        |
| --------------------------------------- | ------------------------------ | -------------------------------------------------------------------- |
| Cyber IDS pipeline                      | Loaders + prompt packs present | Raw datasets (NSL-KDD, CICIDS) not shipped; install `[cyber]` extras |
| LoRA / QLoRA fine-tuning                | Training loops scaffolded      | Install `[gpu]` extras; requires CUDA GPU                            |
| vLLM serving                            | Client code present            | Install vLLM separately; configure endpoint                          |
| TensorRT-LLM serving                    | Client code present            | Install TensorRT-LLM; configure endpoint                             |
| OpenRouter cloud inference              | Client code present            | Set `OPENROUTER_API_KEY`                                             |
| Cyber verifiers (YARA, Sigma, Suricata) | Prompt packs + verifier stubs  | Not yet integrated into scoring                                      |

---

## System Requirements

| Component | Minimum                 | Recommended                |
| --------- | ----------------------- | -------------------------- |
| Python    | 3.10                    | 3.12                       |
| RAM       | 8 GB                    | 16+ GB                     |
| GPU       | None (CPU works)        | NVIDIA with 6+ GB VRAM     |
| Disk      | 4 GB free               | 20+ GB for multiple models |
| Ollama    | Required for demo paths | Latest version             |

---

## Known Limitations

1. **CORS is permissive** — `allow_origins=["*"]` for local development; tighten before any public deployment
2. **Single-user** — no authentication; designed for local use
3. **SQLite** — sufficient for single-user; would need PostgreSQL for multi-user
4. **AWQ RAM check** uses total RAM (16 GB minimum) rather than model-specific estimation
5. **VRAM estimates** in machine-class tiers are conservative approximations (Q4 quantized models)
6. **HF token** set process-wide via `os.environ` — single-user only
7. **Benchmark scores** are relative to the local machine — not comparable across hardware
8. **Windows tested** — Linux/macOS should work but are not CI-verified

---

## Test Coverage

76 tests across:
- 5 system-class tiers (cpu_only, low_vram, mid_vram, high_vram, ollama_only)
- Backend alias mapping (bnb_4bit/bnb_8bit → bitsandbytes)
- Transformers training without peft
- Production readiness gates per feature
- MachineClass allowed_optimizations and allowed_quant_backends
- Profiler caching (TTL, force refresh, invalidation)

```bash
python -m pytest tests/ -v
```
