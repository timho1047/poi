import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .config import LLMConfig


def load_pretrained_model(config: LLMConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def load_inference_model(config: LLMConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, config.model_dir)
    model.eval()
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
