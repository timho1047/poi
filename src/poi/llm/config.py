from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig

from .. import settings


@dataclass
class LLMConfig:
    # Settings
    model_id: str = "mlabonne/Meta-Llama-3-8B"
    run_name: str = "llama3-1"
    logging_steps: int = 4
    resume_from_checkpoint: bool = True
    max_length: int = 2048
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "warning"

    # Training parameters
    lr: float = 1e-5
    lr_scheduler_type: str = "constant"
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 8
    quantization_bits: Literal[4, 8] = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    warmup_steps: int = 20

    # Inferred configs, no need to provide during initialization
    checkpoint_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    bnb_config: BitsAndBytesConfig = field(init=False)
    lora_config: LoraConfig = field(init=False)
    training_args: SFTConfig = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)

    def __post_init__(self):
        self.checkpoint_dir = settings.CHECKPOINTS_DIR / "llm" / self.run_name
        self.model_dir = self.checkpoint_dir / self.run_name
        self.log_dir = settings.LOGS_DIR / "llm" / self.run_name

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dirs = [
            d
            for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        self.resume_from_checkpoint = (
            len(checkpoint_dirs) > 0 and self.resume_from_checkpoint
        )

        self.bnb_config = (
            BNB_CONFIG_8BIT if self.quantization_bits == 8 else BNB_CONFIG_4BIT
        )
        self.lora_config = LORA_CONFIG
        self.lora_config.r = self.lora_rank
        self.lora_config.lora_alpha = self.lora_alpha
        self.training_args = SFTConfig(
            # Settings
            output_dir=self.checkpoint_dir,
            logging_dir=self.log_dir,
            num_train_epochs=self.num_epochs,
            report_to="tensorboard",
            logging_steps=self.logging_steps,
            log_level=self.log_level,
            run_name=self.run_name,
            eval_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            eval_on_start=False,
            save_strategy="epoch",
            max_length=self.max_length,
            # Training parameters
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.lr,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_steps=self.warmup_steps,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            seed=settings.RANDOM_STATE,
            data_seed=settings.RANDOM_STATE,
            bf16=True,
            fp16=False,
            # Some speed optimization
            gradient_checkpointing=True,  # Should be default, but make it explicit
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Faster variant
            # packing=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


LORA_CONFIG = LoraConfig(
    r=16,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    bias="none",  # Bias type
    task_type="CAUSAL_LM",  # Task type
    target_modules=[  # Which modules to apply LoRA to
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

BNB_CONFIG_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

BNB_CONFIG_8BIT = BitsAndBytesConfig(
    load_in_8bit=True,
)
