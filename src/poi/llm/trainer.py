import warnings

from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
)
from trl.trainer.sft_trainer import SFTTrainer

from datasets import Dataset

from .config import LLMConfig
from .utils import print_trainable_parameters, print_training_configuration

# Suppress harmless warnings
warnings.filterwarnings("ignore", message=".*use_reentrant parameter.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def train_llm(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset):
    print_training_configuration(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
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

    model.save_pretrained(config.model_dir)
    config.tokenizer.save_pretrained(config.model_dir)

    return trainer
