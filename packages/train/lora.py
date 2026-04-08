"""LoRA / QLoRA adapter training via PEFT + TRL.

Requires optional GPU dependencies: torch, transformers, peft, trl, bitsandbytes.
Install with: pip install -e ".[gpu]"
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


def _trust_remote() -> bool:
    return os.environ.get("CYBERFORGE_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")


def _check_gpu_deps() -> None:
    """Raise ImportError with helpful message if GPU deps are missing."""
    missing: list[str] = []
    for pkg in ("torch", "transformers", "peft", "trl"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"GPU dependencies not installed: {', '.join(missing)}. "
            'Install with: pip install -e ".[gpu]"'
        )


class LoraConfig(BaseModel):
    base_model: str
    adapter_name: str = "cyber-lora"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    task_type: str = "CAUSAL_LM"
    load_in_4bit: bool = True
    optimizer: str = "adamw"  # adamw | adafactor
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 2048
    output_dir: str = "data/cache/adapters"
    dataset_path: Optional[str] = None
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    fp16: bool = False
    bf16: bool = True


class TrainResult(BaseModel):
    adapter_path: str = ""
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    epochs_completed: int = 0
    success: bool = True
    error: Optional[str] = None


class LoraTrainer:
    """Adapter training via PEFT + TRL."""

    def __init__(self, config: LoraConfig):
        self._config = config

    def _build_bnb_config(self):
        """Build BitsAndBytesConfig for 4-bit quantization (QLoRA)."""
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self._config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def _build_peft_config(self):
        """Build PEFT LoraConfig."""
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType

        task_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }
        return PeftLoraConfig(
            r=self._config.r,
            lora_alpha=self._config.lora_alpha,
            lora_dropout=self._config.lora_dropout,
            target_modules=self._config.target_modules,
            task_type=task_map.get(self._config.task_type, TaskType.CAUSAL_LM),
            bias="none",
        )

    def _train_sync(self) -> TrainResult:
        """Blocking training — runs in thread pool."""
        _check_gpu_deps()

        import torch
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  TrainingArguments)
        from trl import SFTTrainer

        cfg = self._config
        trc = _trust_remote()
        output_dir = Path(cfg.output_dir) / cfg.adapter_name
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Loading tokenizer: %s", cfg.base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=trc,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("Loading base model: %s (4bit=%s)", cfg.base_model, cfg.load_in_4bit)
        model_kwargs: dict = {
            "token": os.environ.get("HF_TOKEN"),
            "trust_remote_code": trc,
            "device_map": "auto",
        }
        if cfg.load_in_4bit:
            model_kwargs["quantization_config"] = self._build_bnb_config()

        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        peft_config = self._build_peft_config()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Load dataset
        from datasets import load_dataset as hf_load_dataset

        if cfg.dataset_path:
            if cfg.dataset_path.endswith((".json", ".jsonl")):
                dataset = hf_load_dataset("json", data_files=cfg.dataset_path, split="train")
            elif cfg.dataset_path.endswith(".csv"):
                dataset = hf_load_dataset("csv", data_files=cfg.dataset_path, split="train")
            else:
                dataset = hf_load_dataset(cfg.dataset_path, split="train")
        else:
            raise ValueError("dataset_path is required for training")

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=2,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            optim="adamw_torch" if cfg.optimizer == "adamw" else "adafactor",
            report_to="none",
            remove_unused_columns=False,
        )

        # Detect text column
        cols = dataset.column_names
        text_col = "text" if "text" in cols else cols[0]

        def formatting_func(example):
            return [example[text_col]]

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            peft_config=peft_config,
            args=training_args,
            formatting_func=formatting_func,
            max_seq_length=cfg.max_seq_length,
        )

        log.info("Starting training...")
        train_output = trainer.train()

        # Save adapter
        adapter_path = str(output_dir / "final")
        trainer.save_model(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return TrainResult(
            adapter_path=adapter_path,
            train_loss=train_output.training_loss,
            epochs_completed=cfg.epochs,
            success=True,
        )

    async def train(self) -> TrainResult:
        """Run training in background thread to keep event loop free."""
        try:
            return await asyncio.to_thread(self._train_sync)
        except ImportError as e:
            return TrainResult(success=False, error=str(e))
        except Exception as e:
            log.exception("Training failed")
            return TrainResult(success=False, error=str(e))

    def _merge_sync(self, output_dir: str) -> str:
        """Merge adapter into base model and export."""
        _check_gpu_deps()

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self._config
        adapter_path = str(Path(cfg.output_dir) / cfg.adapter_name / "final")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        log.info("Loading base + adapter for merge")
        trc = _trust_remote()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=trc,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=trc,
            device_map="cpu",
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

        model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        log.info("Merged model saved to %s", out)
        return str(out)

    async def merge_and_export(self, output_dir: str) -> str:
        try:
            return await asyncio.to_thread(self._merge_sync, output_dir)
        except ImportError as e:
            raise ImportError(str(e)) from e
