import warnings

from huggingface_hub import upload_file
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel

from datasets import Dataset

from .. import settings
from ..llm import LLMConfig
from .utils import generate_model_card, print_trainable_parameters, print_training_configuration

# Suppress harmless warnings
warnings.filterwarnings("ignore", message=".*use_reentrant parameter.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def train_llm_fast(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset, push_to_hub: bool = False):
    print_training_configuration(config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_length,
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=config.quantization_bits == 4,
        load_in_8bit=config.quantization_bits == 8,
        attn_implementation="flash_attention_2",
        device_map="balanced",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_config.r,
        target_modules=config.lora_config.target_modules,
        lora_alpha=config.lora_config.lora_alpha,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        use_gradient_checkpointing=config.training_args.gradient_checkpointing,
        random_state=settings.RANDOM_STATE,
    )

    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    model_card = generate_model_card(config)
    
    if push_to_hub:
        trainer.push_to_hub(commit_message=f"Training completed for {config.run_name}")
        upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=config.hub_id,
            token=settings.HF_TOKEN,
        )
    else:
        trainer.save_model(config.checkpoint_dir)

    config.model_card_path.write_text(model_card)
    return trainer


def train_llm(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset, push_to_hub: bool = False):
    print_training_configuration(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config.lora_config)

    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=config.tokenizer,
        args=config.training_args,
    )

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    model_card = generate_model_card(config)

    if push_to_hub:
        trainer.push_to_hub(commit_message=f"Training completed for {config.run_name}")
        upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=config.hub_id,
            token=settings.HF_TOKEN,
        )
    else:
        trainer.save_model(config.checkpoint_dir)

    config.model_card_path.write_text(model_card)
    return trainer
