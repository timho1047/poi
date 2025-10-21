import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from unsloth import FastLanguageModel

from .config import LLMConfig


def load_pretrained_model(config: LLMConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def load_inference_model(config: LLMConfig, from_hub: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, config.hub_id if from_hub else config.checkpoint_dir.as_posix())
    model.eval()
    return model

def load_fast_inference_model(config: LLMConfig, from_hub: bool = False):
    model, _ = FastLanguageModel.from_pretrained(
        model_name=config.hub_id if from_hub else config.checkpoint_dir.as_posix(),
        max_seq_length=config.max_length,
        dtype=None,
        load_in_4bit=config.quantization_bits == 4,
        load_in_8bit=config.quantization_bits == 8,
    )
    model = FastLanguageModel.for_inference(model)
    return model

@torch.inference_mode()
def inference(config: LLMConfig, model: PeftModel, prompt: str):
    inputs = config.tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=config.tokenizer.eos_token_id,
    )
    response = config.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt) :]
    return response
