"""Training router — LoRA / QLoRA adapter training endpoints + optimizer lab."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from packages.train.lora import LoraConfig, LoraTrainer, TrainResult
from packages.train.optimizer_lab import (OptimizerLabConfig, OptimizerType,
                                          list_available_optimizers,
                                          validate_optimizer_config)

router = APIRouter()


class TrainRequest(BaseModel):
    base_model: str
    dataset_path: str
    adapter_name: str = "cyber-lora"
    task_mode: str = "cyber"  # general | coding | cyber
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    temporary: bool = True
    run_benchmark_after: bool = False


class TrainResponse(BaseModel):
    adapter_path: str = ""
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    epochs_completed: int = 0
    success: bool = True
    error: Optional[str] = None


@router.post("/lora", response_model=TrainResponse)
async def train_lora(req: TrainRequest):
    """Start LoRA adapter training."""
    config = LoraConfig(
        base_model=req.base_model,
        adapter_name=req.adapter_name,
        r=req.lora_r,
        lora_alpha=req.lora_alpha,
        lora_dropout=req.lora_dropout,
        target_modules=req.target_modules,
        load_in_4bit=False,
        optimizer=req.optimizer,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        batch_size=req.batch_size,
        max_seq_length=req.max_seq_length,
        dataset_path=req.dataset_path,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
    )
    trainer = LoraTrainer(config)
    result = await trainer.train()
    return TrainResponse(
        adapter_path=result.adapter_path,
        train_loss=result.train_loss,
        eval_loss=result.eval_loss,
        epochs_completed=result.epochs_completed,
        success=result.success,
        error=result.error,
    )


@router.post("/qlora", response_model=TrainResponse)
async def train_qlora(req: TrainRequest):
    """Start QLoRA adapter training (4-bit quantized base)."""
    config = LoraConfig(
        base_model=req.base_model,
        adapter_name=req.adapter_name,
        r=req.lora_r,
        lora_alpha=req.lora_alpha,
        lora_dropout=req.lora_dropout,
        target_modules=req.target_modules,
        load_in_4bit=True,
        optimizer=req.optimizer,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        batch_size=req.batch_size,
        max_seq_length=req.max_seq_length,
        dataset_path=req.dataset_path,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
    )
    trainer = LoraTrainer(config)
    result = await trainer.train()
    return TrainResponse(
        adapter_path=result.adapter_path,
        train_loss=result.train_loss,
        eval_loss=result.eval_loss,
        epochs_completed=result.epochs_completed,
        success=result.success,
        error=result.error,
    )


class MergeRequest(BaseModel):
    base_model: str
    adapter_name: str = "cyber-lora"
    output_dir: str = "data/cache/merged"


class MergeResponse(BaseModel):
    output_path: str = ""
    success: bool = True
    error: Optional[str] = None


@router.post("/merge", response_model=MergeResponse)
async def merge_adapter(req: MergeRequest):
    """Merge a trained LoRA adapter into the base model."""
    config = LoraConfig(base_model=req.base_model, adapter_name=req.adapter_name)
    trainer = LoraTrainer(config)
    try:
        path = await trainer.merge_and_export(req.output_dir)
        return MergeResponse(output_path=path, success=True)
    except Exception as e:
        return MergeResponse(success=False, error=str(e))


# ── Optimizer Lab (TRAIN-003) ────────────────────────────────────


class OptimizerLabRequest(BaseModel):
    optimizer: str = "adamw"  # adamw | adafactor | muon
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    adafactor_scale_parameter: bool = True
    adafactor_relative_step: bool = False
    muon_momentum: float = 0.95
    muon_nesterov: bool = True


class OptimizerLabResponse(BaseModel):
    optimizer_type: str
    hf_optim_string: str
    optimizer_kwargs: dict
    warnings: list[str] = []
    available_optimizers: list[dict] = []


@router.get("/optimizers")
async def get_optimizers(advanced_mode: bool = False):
    """List available optimizers. Set advanced_mode=true to see experimental options."""
    return {"optimizers": list_available_optimizers(advanced_mode=advanced_mode)}


@router.post("/optimizer-lab/validate", response_model=OptimizerLabResponse)
async def validate_optimizer(req: OptimizerLabRequest):
    """Validate and preview optimizer configuration before training."""
    try:
        opt_type = OptimizerType(req.optimizer.lower())
    except ValueError:
        return OptimizerLabResponse(
            optimizer_type=req.optimizer,
            hf_optim_string="adamw_torch",
            optimizer_kwargs={},
            warnings=[f"Unknown optimizer '{req.optimizer}', falling back to adamw"],
            available_optimizers=list_available_optimizers(advanced_mode=True),
        )

    config = OptimizerLabConfig(
        optimizer=opt_type,
        learning_rate=req.learning_rate,
        weight_decay=req.weight_decay,
        warmup_ratio=req.warmup_ratio,
        scale_parameter=req.adafactor_scale_parameter,
        relative_step=req.adafactor_relative_step,
        muon_momentum=req.muon_momentum,
        muon_nesterov=req.muon_nesterov,
    )

    validation = validate_optimizer_config(config)
    from packages.train.optimizer_lab import (build_optimizer_kwargs,
                                              get_training_args_optim)

    return OptimizerLabResponse(
        optimizer_type=config.optimizer.value,
        hf_optim_string=get_training_args_optim(config),
        optimizer_kwargs=build_optimizer_kwargs(config),
        warnings=validation.warnings,
        available_optimizers=list_available_optimizers(advanced_mode=True),
    )
