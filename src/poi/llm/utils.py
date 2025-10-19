from poi.llm.config import LLMConfig
from peft import PeftModel


def print_trainable_parameters(model: PeftModel):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def print_training_configuration(config: LLMConfig):
    print("Training Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}"
    )
    print(f"  Lr scheduler type: {config.lr_scheduler_type}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Quantization bits: {config.quantization_bits}")
    print(f"  Lora rank: {config.lora_rank}")
    print(f"  Lora alpha: {config.lora_alpha}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Checkpoint directory: {config.checkpoint_dir}")
    print(f"  Log directory: {config.log_dir}")