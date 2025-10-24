from peft import PeftModel

from poi.llm.config import LLMConfig


def print_trainable_parameters(model: PeftModel):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")


def print_training_configuration(config: LLMConfig):
    print("Training Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Lr scheduler type: {config.lr_scheduler_type}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Quantization bits: {config.quantization_bits}")
    print(f"  Lora rank: {config.lora_rank}")
    print(f"  Lora alpha: {config.lora_alpha}")
    print(f"  Lora dropout: {config.lora_config.lora_dropout}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Checkpoint directory: {config.output_dir}")
    print(f"  Log directory: {config.log_dir}")


def generate_model_card(config: LLMConfig) -> str:
    """Generate a model card in markdown format with training configuration details."""
    model_card = f"""---
base_model: {config.model_id}
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:{config.model_id}
- lora
- sft
- transformers
- trl
- unsloth
---

# {config.run_name}

This model is a fine-tuned version of [{config.model_id}](https://huggingface.co/{config.model_id}) using LoRA (Low-Rank Adaptation) and quantization techniques.

## Model Details

- **Base Model:** {config.model_id}
- **Fine-tuned Model:** {config.hub_id}
- **Training Run:** {config.run_name}
- **Device:** {config.device}

## Training Configuration

### Hyperparameters

- **Number of Epochs:** {config.num_epochs}
- **Batch Size:** {config.batch_size}
- **Gradient Accumulation Steps:** {config.gradient_accumulation_steps}
- **Effective Batch Size:** {config.batch_size * config.gradient_accumulation_steps}
- **Learning Rate:** {config.lr}
- **Learning Rate Scheduler:** {config.lr_scheduler_type}
- **Warmup Steps:** {config.warmup_steps}
- **Max Sequence Length:** {config.max_length}
- **Optimizer:** {config.optimizer}
- **Max Gradient Norm:** {config.max_grad_norm}
- **Random Seed:** {config.training_args.seed}

### LoRA Configuration

- **LoRA Rank (r):** {config.lora_rank}
- **LoRA Alpha:** {config.lora_alpha}
- **LoRA Dropout:** {config.lora_config.lora_dropout}
- **Target Modules:** {", ".join(config.lora_config.target_modules)}
- **Task Type:** {config.lora_config.task_type}

### Quantization

- **Quantization Bits:** {config.quantization_bits}-bit


## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{config.model_id}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{config.hub_id}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{config.model_id}")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length={config.max_length})
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Framework Versions

- Transformers
- PEFT
- TRL
- PyTorch
- BitsAndBytes
"""

    return model_card
