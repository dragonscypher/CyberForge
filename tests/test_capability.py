"""Simulated tests for per-backend capability detection across system classes.

Each test constructs a mock HardwareProfile representing a real machine class,
then asserts that SystemCapability.from_profile() and preflight_check() produce
correct per-backend availability states.

System classes tested:
  1. cpu_only       — No GPU, 16 GB RAM
  2. low_vram       — 4 GB VRAM, 16 GB RAM
  3. mid_vram       — 6 GB VRAM (RTX 3060 Laptop), 40 GB RAM, pytorch_cuda=False (CPU torch)
  4. high_vram      — 24 GB VRAM (RTX 3090), 64 GB RAM, full stack
  5. ollama_only    — GPU visible, pytorch_cuda=False, ollama=True (Ollama sees CUDA, PyTorch doesn't)
"""

import pytest

from packages.hardware.capability import (BackendDetail, PreflightResult,
                                          SystemCapability, preflight_check)
from packages.hardware.profiler import BackendStatus, GPUInfo, HardwareProfile

# ── Fixtures ─────────────────────────────────────────────────────


def _profile(
    *,
    gpus: list[GPUInfo] | None = None,
    ram_total_mb: int = 16_000,
    ollama: bool = False,
    pytorch_cuda: bool = False,
    bitsandbytes: bool = False,
    vllm: bool = False,
    tensorrt_llm: bool = False,
    transformers: bool = True,
    peft: bool = False,
) -> HardwareProfile:
    return HardwareProfile(
        os="Linux",
        os_version="6.5",
        cpu_model="Test CPU",
        cpu_cores=8,
        cpu_threads=16,
        ram_total_mb=ram_total_mb,
        ram_available_mb=int(ram_total_mb * 0.7),
        gpus=gpus or [],
        disk_free_mb=100_000,
        disk_path="/tmp",
        backends=BackendStatus(
            ollama=ollama,
            pytorch_cuda=pytorch_cuda,
            bitsandbytes=bitsandbytes,
            vllm=vllm,
            tensorrt_llm=tensorrt_llm,
            transformers=transformers,
            peft=peft,
        ),
    )


def _gpu(
    vram_total_mb: int = 6144,
    vram_free_mb: int | None = None,
    name: str = "NVIDIA GeForce RTX 3060 Laptop GPU",
    cuda_version: str = "12.4",
) -> GPUInfo:
    return GPUInfo(
        index=0,
        name=name,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb if vram_free_mb is not None else int(vram_total_mb * 0.85),
        cuda_version=cuda_version,
    )


# ── 1. CPU-only (no GPU, 16 GB RAM) ─────────────────────────────


class TestCpuOnly:
    @pytest.fixture()
    def sc(self):
        return SystemCapability.from_profile(_profile(ram_total_mb=16_000))

    def test_tier(self, sc: SystemCapability):
        assert sc.tier == "cpu_only"
        assert sc.has_nvidia_gpu is False

    def test_ollama_unavailable_by_service(self, sc: SystemCapability):
        bd = sc.backend("ollama")
        assert bd is not None
        assert bd.status == "unavailable_by_service"

    def test_pytorch_cuda_unavailable_by_hardware(self, sc: SystemCapability):
        bd = sc.backend("pytorch_cuda")
        assert bd is not None
        assert bd.status == "unavailable_by_hardware"

    def test_pytorch_cpu_always_available(self, sc: SystemCapability):
        bd = sc.backend("pytorch_cpu")
        assert bd is not None
        assert bd.status == "available"
        assert bd.can_infer is True

    def test_bitsandbytes_unavailable_by_hardware(self, sc: SystemCapability):
        bd = sc.backend("bitsandbytes")
        assert bd is not None
        assert bd.status == "unavailable_by_hardware"

    def test_transformers_available(self, sc: SystemCapability):
        bd = sc.backend("transformers")
        assert bd is not None
        assert bd.status == "available"

    def test_any_inference(self, sc: SystemCapability):
        assert sc.any_inference_available is True  # pytorch_cpu always available

    def test_preflight_chat_ollama_blocked(self, sc: SystemCapability):
        pf = preflight_check("chat", "ollama", 1.5, 0, sc)
        assert pf.allowed is False
        assert pf.backend_status == "unavailable_by_service"

    def test_preflight_benchmark_pytorch_cpu_small(self, sc: SystemCapability):
        pf = preflight_check("benchmark", "pytorch_cpu", 1.5, 1000, sc)
        assert pf.allowed is True
        assert pf.category == "cpu_only"

    def test_preflight_benchmark_pytorch_cpu_too_large(self, sc: SystemCapability):
        pf = preflight_check("benchmark", "pytorch_cpu", 30.0, 25_000, sc)
        assert pf.allowed is False
        assert pf.category == "too_large"


# ── 2. Low VRAM (4 GB VRAM, PyTorch CUDA, 16 GB RAM) ────────────


class TestLowVram:
    @pytest.fixture()
    def sc(self):
        return SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=4096, name="GTX 1060 3GB")],
                ram_total_mb=16_000,
                pytorch_cuda=True,
                transformers=True,
                ollama=True,
            )
        )

    def test_tier(self, sc: SystemCapability):
        assert sc.tier == "low_vram"

    def test_ollama_available(self, sc: SystemCapability):
        assert sc.backend("ollama").status == "available"

    def test_pytorch_cuda_available(self, sc: SystemCapability):
        assert sc.backend("pytorch_cuda").status == "available"
        assert sc.backend("pytorch_cuda").can_train is True

    def test_bitsandbytes_unavailable_by_config(self, sc: SystemCapability):
        bd = sc.backend("bitsandbytes")
        assert bd.status == "unavailable_by_config"
        assert "not installed" in bd.reason

    def test_preflight_benchmark_fits(self, sc: SystemCapability):
        pf = preflight_check("benchmark", "ollama", 1.5, 1000, sc)
        assert pf.allowed is True

    def test_preflight_benchmark_too_large(self, sc: SystemCapability):
        pf = preflight_check("benchmark", "ollama", 14.0, 10_000, sc)
        assert pf.allowed is False or pf.category in ("partial_offload", "too_large")


# ── 3. Mid VRAM (6 GB, RTX 3060 Laptop, PyTorch CPU-only torch) ─


class TestMidVramNoCuda:
    """The user's actual machine: GPU visible via nvidia-smi but PyTorch CUDA=False."""

    @pytest.fixture()
    def sc(self):
        return SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=6144)],
                ram_total_mb=40_000,
                pytorch_cuda=False,
                transformers=True,
                ollama=False,
            )
        )

    def test_tier(self, sc: SystemCapability):
        assert sc.tier == "mid_vram"

    def test_pytorch_cuda_unavailable_by_config_not_hardware(self, sc: SystemCapability):
        """Must NOT say 'No NVIDIA GPU detected' — GPU IS detected."""
        bd = sc.backend("pytorch_cuda")
        assert bd.status == "unavailable_by_config"
        assert "GPU detected" in bd.reason
        assert "No NVIDIA GPU" not in bd.reason

    def test_bitsandbytes_unavailable_by_config(self, sc: SystemCapability):
        bd = sc.backend("bitsandbytes")
        assert bd.status == "unavailable_by_config"

    def test_pytorch_cpu_available(self, sc: SystemCapability):
        bd = sc.backend("pytorch_cpu")
        assert bd.status == "available"

    def test_ollama_unavailable_by_service(self, sc: SystemCapability):
        assert sc.backend("ollama").status == "unavailable_by_service"

    def test_has_nvidia_gpu_true(self, sc: SystemCapability):
        assert sc.has_nvidia_gpu is True

    def test_preflight_quantize_ollama_blocked(self, sc: SystemCapability):
        pf = preflight_check("quantize", "ollama", 3.0, 0, sc)
        assert pf.allowed is False
        assert pf.backend_status == "unavailable_by_service"

    def test_preflight_quantize_bitsandbytes_blocked(self, sc: SystemCapability):
        pf = preflight_check("quantize", "bitsandbytes", 3.0, 0, sc)
        assert pf.allowed is False


# ── 4. High VRAM (24 GB, RTX 3090, full stack) ──────────────────


class TestHighVram:
    @pytest.fixture()
    def sc(self):
        return SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=24_576, name="NVIDIA GeForce RTX 3090")],
                ram_total_mb=64_000,
                pytorch_cuda=True,
                bitsandbytes=True,
                transformers=True,
                peft=True,
                ollama=True,
                vllm=True,
            )
        )

    def test_tier(self, sc: SystemCapability):
        assert sc.tier == "high_vram"

    def test_all_major_backends_available(self, sc: SystemCapability):
        for name in ("ollama", "pytorch_cuda", "pytorch_cpu", "bitsandbytes", "transformers"):
            assert sc.backend(name).status == "available", f"{name} should be available"

    def test_vllm_available(self, sc: SystemCapability):
        assert sc.backend("vllm").status == "available"

    def test_any_flags(self, sc: SystemCapability):
        assert sc.any_inference_available is True
        assert sc.any_quantize_available is True
        assert sc.any_train_available is True

    def test_preflight_quantize_bitsandbytes_allowed(self, sc: SystemCapability):
        pf = preflight_check("quantize", "bitsandbytes", 7.0, 5000, sc)
        assert pf.allowed is True

    def test_preflight_benchmark_fits(self, sc: SystemCapability):
        pf = preflight_check("benchmark", "ollama", 7.0, 5000, sc)
        assert pf.allowed is True
        assert pf.category == "fits"

    def test_preflight_partial_offload_70b(self, sc: SystemCapability):
        # 70B at 40GB: partial offload allowed (24GB VRAM + 64GB RAM)
        pf = preflight_check("benchmark", "ollama", 70.0, 40_000, sc)
        assert pf.allowed is True
        assert pf.category == "partial_offload"

    def test_preflight_too_large_200b(self, sc: SystemCapability):
        pf = preflight_check("benchmark", "ollama", 200.0, 120_000, sc)
        assert pf.allowed is False
        assert pf.category == "too_large"


# ── 5. Ollama-only CUDA (GPU visible, pytorch_cuda=False, ollama running) ──


class TestOllamaOnlyCuda:
    """Machine where Ollama sees the GPU but PyTorch doesn't (CPU-only torch install)."""

    @pytest.fixture()
    def sc(self):
        return SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=8192, name="RTX 4060")],
                ram_total_mb=32_000,
                pytorch_cuda=False,
                ollama=True,
                transformers=True,
            )
        )

    def test_ollama_available_for_inference(self, sc: SystemCapability):
        bd = sc.backend("ollama")
        assert bd.status == "available"
        assert bd.can_infer is True

    def test_pytorch_cuda_unavailable_by_config(self, sc: SystemCapability):
        bd = sc.backend("pytorch_cuda")
        assert bd.status == "unavailable_by_config"

    def test_no_global_no_cuda_message(self, sc: SystemCapability):
        """The system must NOT globally say 'No CUDA' if Ollama can see the GPU."""
        bd = sc.backend("pytorch_cuda")
        assert "No NVIDIA GPU" not in bd.reason

    def test_preflight_chat_ollama_allowed(self, sc: SystemCapability):
        pf = preflight_check("chat", "ollama", 3.0, 2000, sc)
        assert pf.allowed is True

    def test_preflight_quantize_bnb_blocked(self, sc: SystemCapability):
        pf = preflight_check("quantize", "bitsandbytes", 7.0, 5000, sc)
        assert pf.allowed is False
        # bitsandbytes=False & pytorch_cuda=False but GPU detected → unavailable_by_config
        assert pf.backend_status in ("unavailable_by_config", "unavailable_by_hardware")

    def test_available_backends_for_infer(self, sc: SystemCapability):
        infers = sc.available_backends_for("infer")
        names = {bd.name for bd in infers}
        assert "ollama" in names
        assert "pytorch_cpu" in names
        assert "transformers" in names
        # pytorch_cuda should NOT be in available infer backends
        assert "pytorch_cuda" not in names


# ── 6. Edge cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_unknown_backend_preflight(self):
        sc = SystemCapability.from_profile(_profile())
        pf = preflight_check("chat", "nonexistent_backend", 1.0, 0, sc)
        assert pf.allowed is False
        assert pf.category == "no_backend"

    def test_unsupported_action_for_backend(self):
        sc = SystemCapability.from_profile(
            _profile(ollama=True, ram_total_mb=16_000)
        )
        pf = preflight_check("train", "ollama", 1.0, 0, sc)
        assert pf.allowed is False
        assert "does not support" in pf.reason

    def test_backend_detail_lookup(self):
        sc = SystemCapability.from_profile(_profile())
        assert sc.backend("pytorch_cpu") is not None
        assert sc.backend("does_not_exist") is None

    def test_available_backends_for_invalid_action(self):
        sc = SystemCapability.from_profile(_profile())
        assert sc.available_backends_for("fly") == []


# ── 7. Production-readiness: backend alias mapping ───────────────


class TestBackendAlias:
    """bnb_4bit and bnb_8bit must resolve to the bitsandbytes backend in preflight."""

    def test_bnb_4bit_alias_blocked_when_no_cuda(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu(vram_total_mb=6144)], pytorch_cuda=False)
        )
        pf = preflight_check("quantize", "bnb_4bit", 7.0, 5000, sc)
        assert pf.allowed is False
        assert pf.backend_status in ("unavailable_by_config", "unavailable_by_hardware")

    def test_bnb_8bit_alias_blocked_when_no_cuda(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu(vram_total_mb=6144)], pytorch_cuda=False)
        )
        pf = preflight_check("quantize", "bnb_8bit", 7.0, 5000, sc)
        assert pf.allowed is False

    def test_bnb_4bit_alias_allowed_when_available(self):
        sc = SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=24_576, name="RTX 3090")],
                ram_total_mb=64_000,
                pytorch_cuda=True,
                bitsandbytes=True,
                transformers=True,
            )
        )
        pf = preflight_check("quantize", "bnb_4bit", 7.0, 5000, sc)
        assert pf.allowed is True

    def test_bnb_8bit_alias_allowed_when_available(self):
        sc = SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=24_576, name="RTX 3090")],
                ram_total_mb=64_000,
                pytorch_cuda=True,
                bitsandbytes=True,
                transformers=True,
            )
        )
        pf = preflight_check("quantize", "bnb_8bit", 7.0, 5000, sc)
        assert pf.allowed is True


# ── 8. Production-readiness: transformers can_train without peft ─


class TestTransformersTrainWithoutPeft:
    """Pruning, distillation, and editing use transformers training but NOT peft.
    Preflight must allow 'train' action on transformers when peft is missing."""

    @pytest.fixture()
    def sc_no_peft(self):
        return SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=6144)],
                ram_total_mb=40_000,
                pytorch_cuda=False,
                transformers=True,
                peft=False,
            )
        )

    @pytest.fixture()
    def sc_with_peft(self):
        return SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=24_576, name="RTX 3090")],
                ram_total_mb=64_000,
                pytorch_cuda=True,
                transformers=True,
                peft=True,
            )
        )

    def test_transformers_can_train_true_without_peft(self, sc_no_peft: SystemCapability):
        bd = sc_no_peft.backend("transformers")
        assert bd is not None
        assert bd.can_train is True

    def test_transformers_can_train_true_with_peft(self, sc_with_peft: SystemCapability):
        bd = sc_with_peft.backend("transformers")
        assert bd is not None
        assert bd.can_train is True

    def test_preflight_train_transformers_allowed_without_peft(self, sc_no_peft: SystemCapability):
        """doPrune/doDistill/doEdit send action='train', backend='transformers'."""
        pf = preflight_check("train", "transformers", 0, 0, sc_no_peft)
        assert pf.allowed is True

    def test_preflight_train_transformers_cpu_warning(self, sc_no_peft: SystemCapability):
        """When pytorch_cuda unavailable, warn about CPU-only execution."""
        pf = preflight_check("train", "transformers", 0, 0, sc_no_peft)
        assert pf.category == "cpu_only"
        assert "CPU" in pf.reason

    def test_preflight_train_transformers_large_model_blocked(self, sc_no_peft: SystemCapability):
        """Model too large for RAM-only execution should be blocked."""
        pf = preflight_check("train", "transformers", 70.0, 50_000, sc_no_peft)
        assert pf.allowed is False
        assert pf.category == "too_large"


# ── 9. Production-readiness: per-feature per-system-class gates ──


class TestProductionReadiness:
    """Prove each feature is either safely allowed or properly blocked
    for each system class. One happy path + one blocked path per feature."""

    # ── Ollama Quantization ──

    def test_ollama_quant_allowed_when_ollama_running(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], ollama=True, ram_total_mb=40_000)
        )
        pf = preflight_check("quantize", "ollama", 3.0, 2000, sc)
        assert pf.allowed is True

    def test_ollama_quant_blocked_when_ollama_down(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], ollama=False, ram_total_mb=40_000)
        )
        pf = preflight_check("quantize", "ollama", 3.0, 2000, sc)
        assert pf.allowed is False
        assert pf.backend_status == "unavailable_by_service"

    # ── bitsandbytes Quantization (via bnb_4bit alias) ──

    def test_bnb_quant_blocked_cpu_only(self):
        sc = SystemCapability.from_profile(_profile(ram_total_mb=16_000))
        pf = preflight_check("quantize", "bnb_4bit", 3.0, 0, sc)
        assert pf.allowed is False

    def test_bnb_quant_blocked_no_cuda_torch(self):
        """User's machine: GPU exists but pytorch_cuda=False."""
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], pytorch_cuda=False, ram_total_mb=40_000)
        )
        pf = preflight_check("quantize", "bnb_4bit", 3.0, 0, sc)
        assert pf.allowed is False

    # ── AWQ Quantization ──

    def test_awq_blocked_no_cuda(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], pytorch_cuda=False, ram_total_mb=40_000)
        )
        pf = preflight_check("quantize", "awq", 7.0, 0, sc)
        assert pf.allowed is False

    # ── GPTQ Quantization ──

    def test_gptq_blocked_no_cuda(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], pytorch_cuda=False, ram_total_mb=40_000)
        )
        pf = preflight_check("quantize", "gptq", 7.0, 0, sc)
        assert pf.allowed is False

    # ── Pruning (HF Transformers) ──

    def test_prune_allowed_transformers_cpu(self):
        """Pruning works on CPU with just transformers."""
        sc = SystemCapability.from_profile(
            _profile(ram_total_mb=40_000, transformers=True)
        )
        pf = preflight_check("train", "transformers", 3.0, 2000, sc)
        assert pf.allowed is True
        assert pf.category == "cpu_only"

    def test_prune_blocked_no_transformers(self):
        sc = SystemCapability.from_profile(
            _profile(ram_total_mb=16_000, transformers=False)
        )
        pf = preflight_check("train", "transformers", 3.0, 2000, sc)
        assert pf.allowed is False

    # ── Distillation (HF Transformers) ──

    def test_distill_allowed_with_cuda(self):
        sc = SystemCapability.from_profile(
            _profile(
                gpus=[_gpu(vram_total_mb=24_576, name="RTX 3090")],
                ram_total_mb=64_000,
                pytorch_cuda=True,
                transformers=True,
            )
        )
        pf = preflight_check("train", "transformers", 7.0, 5000, sc)
        assert pf.allowed is True
        assert pf.category == "fits"

    def test_distill_cpu_warns_slow(self):
        sc = SystemCapability.from_profile(
            _profile(ram_total_mb=40_000, transformers=True, pytorch_cuda=False)
        )
        pf = preflight_check("train", "transformers", 3.0, 2000, sc)
        assert pf.allowed is True
        assert pf.category == "cpu_only"
        assert "CPU" in pf.reason

    # ── Model Editor (HF Transformers) ──

    def test_edit_allowed_transformers(self):
        sc = SystemCapability.from_profile(
            _profile(ram_total_mb=40_000, transformers=True)
        )
        pf = preflight_check("train", "transformers", 3.0, 2000, sc)
        assert pf.allowed is True

    # ── Chat (Ollama) ──

    def test_chat_allowed_ollama_running(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], ollama=True, ram_total_mb=40_000)
        )
        pf = preflight_check("chat", "ollama", 3.0, 2000, sc)
        assert pf.allowed is True

    def test_chat_blocked_ollama_down(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], ollama=False, ram_total_mb=40_000)
        )
        pf = preflight_check("chat", "ollama", 3.0, 2000, sc)
        assert pf.allowed is False
        assert pf.backend_status == "unavailable_by_service"

    # ── Benchmark (Ollama) ──

    def test_benchmark_allowed_ollama(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu()], ollama=True, ram_total_mb=40_000)
        )
        pf = preflight_check("benchmark", "ollama", 3.0, 2000, sc)
        assert pf.allowed is True

    def test_benchmark_blocked_too_large(self):
        sc = SystemCapability.from_profile(
            _profile(gpus=[_gpu(vram_total_mb=4096)], ollama=True, ram_total_mb=16_000)
        )
        pf = preflight_check("benchmark", "ollama", 70.0, 50_000, sc)
        assert pf.allowed is False
        assert pf.category == "too_large"


# ── 10. MachineClass allowed_optimizations & allowed_quant_backends ──


class TestMachineClassGating:
    """Verify MachineClass gates are accurate for all system tiers."""

    def test_cpu_only_transformers_all_hf_opts_allowed(self):
        """CPU + transformers: prune, distill, edit all work via device_map='cpu'."""
        from packages.hardware.capability import MachineClass

        mc = MachineClass.classify(
            _profile(ram_total_mb=32_000, transformers=True)
        )
        assert mc.tier == "cpu_only"
        for opt in ("quantize", "prune", "distill", "edit"):
            assert opt in mc.allowed_optimizations, f"{opt} must be allowed on CPU+transformers"

    def test_cpu_only_no_transformers_only_quantize(self):
        """CPU without transformers: only quantize (via Ollama GGUF)."""
        from packages.hardware.capability import MachineClass

        mc = MachineClass.classify(
            _profile(ram_total_mb=16_000, transformers=False)
        )
        assert mc.allowed_optimizations == ["quantize"]

    def test_gpu_cuda_transformers_all_opts_allowed(self):
        """Full GPU stack: all optimizations available."""
        from packages.hardware.capability import MachineClass

        mc = MachineClass.classify(
            _profile(
                gpus=[_gpu(vram_total_mb=24_576, name="RTX 3090")],
                ram_total_mb=64_000,
                pytorch_cuda=True,
                transformers=True,
                bitsandbytes=True,
            )
        )
        for opt in ("quantize", "prune", "distill", "edit"):
            assert opt in mc.allowed_optimizations

    def test_quant_backends_exclude_ollama_when_not_running(self):
        """allowed_quant_backends should NOT include ollama when it's down."""
        from packages.hardware.capability import MachineClass

        mc = MachineClass.classify(
            _profile(gpus=[_gpu()], ollama=False, ram_total_mb=40_000)
        )
        assert "ollama" not in mc.allowed_quant_backends

    def test_quant_backends_include_ollama_when_running(self):
        from packages.hardware.capability import MachineClass

        mc = MachineClass.classify(
            _profile(gpus=[_gpu()], ollama=True, ram_total_mb=40_000)
        )
        assert "ollama" in mc.allowed_quant_backends

    def test_quant_backends_include_bnb_with_full_stack(self):
        from packages.hardware.capability import MachineClass

        mc = MachineClass.classify(
            _profile(
                gpus=[_gpu(vram_total_mb=24_576)],
                ram_total_mb=64_000,
                pytorch_cuda=True,
                bitsandbytes=True,
                transformers=True,
            )
        )
        assert "bnb_4bit" in mc.allowed_quant_backends
        assert "bnb_8bit" in mc.allowed_quant_backends
        assert "awq" in mc.allowed_quant_backends
        assert "gptq" in mc.allowed_quant_backends

    def test_quant_backends_no_bnb_without_cuda(self):
        from packages.hardware.capability import MachineClass

        mc = MachineClass.classify(
            _profile(
                gpus=[_gpu()],
                ram_total_mb=40_000,
                pytorch_cuda=False,
                bitsandbytes=True,  # installed but no CUDA
                transformers=True,
            )
        )
        assert "bnb_4bit" not in mc.allowed_quant_backends
        assert "bnb_8bit" not in mc.allowed_quant_backends


# ── 11. Profiler caching ─────────────────────────────────────────


class TestProfilerCaching:
    """Verify HardwareProfiler TTL caching avoids redundant probes."""

    def test_second_call_returns_cached(self):
        """Two immediate calls should return the same object (cached)."""
        from packages.hardware.profiler import HardwareProfiler

        profiler = HardwareProfiler()
        p1 = profiler.profile()
        p2 = profiler.profile()
        assert p1 is p2  # same object — no re-scan

    def test_force_returns_fresh(self):
        """force=True must return a new scan."""
        from packages.hardware.profiler import HardwareProfiler

        profiler = HardwareProfiler()
        p1 = profiler.profile()
        p2 = profiler.profile(force=True)
        assert p1 is not p2  # different object

    def test_invalidate_causes_rescan(self):
        from packages.hardware.profiler import HardwareProfiler

        profiler = HardwareProfiler()
        p1 = profiler.profile()
        profiler.invalidate()
        p2 = profiler.profile()
        assert p1 is not p2
