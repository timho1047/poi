from pathlib import Path
from typing import TypedDict

from transformers import PreTrainedTokenizer

from datasets import Dataset


class LLJsonRecord(TypedDict):
    instruction: str
    input: str
    output: str


def format_prompt_completion(ex: LLJsonRecord) -> dict[str, str]:
    prompt = ex["instruction"] + "\n" + ex["input"] + " "
    completion = ex["output"]
    return {"prompt": prompt, "completion": completion}


def add_eos(example, eos_token):
    if not example["completion"].endswith(eos_token):
        example["completion"] = example["completion"] + eos_token
    return example


def tokenize_prompt_completion(ex, tokenizer):
    prompt_ids = tokenizer(text=ex["prompt"])["input_ids"]
    prompt_completion_ids = tokenizer(text=ex["prompt"] + ex["completion"])["input_ids"]
    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    return {"input_ids": prompt_completion_ids, "completion_mask": completion_mask}


def load_prompt_completion_llm_dataset(json_file_path: Path | str, max_examples: int = None) -> Dataset:
    """Load JSON data and convert to Hugging Face Dataset

    Args:
        json_file_path: Path to the JSON file
        max_examples: Maximum number of examples to load, None for all examples
    Returns:
        Hugging Face Dataset
    """
    if isinstance(json_file_path, Path):
        json_file_path = json_file_path.as_posix()

    ds = Dataset.from_json(json_file_path)
    if max_examples is not None:
        ds = ds.select(range(0, max_examples))
    ds = ds.map(format_prompt_completion, remove_columns=["instruction", "input", "output"], desc="Formatting prompt and completion")
    return ds


def load_tokenized_llm_dataset(json_file_path: Path | str, tokenizer: PreTrainedTokenizer, max_examples: int = None) -> Dataset:
    """Load JSON data and convert to Hugging Face Dataset

    Args:
        json_file_path: Path to the JSON file
        max_examples: Maximum number of examples to load, None for all examples
    Returns:
        Hugging Face Dataset
    """
    # Attribution: https://github.com/unslothai/unsloth/issues/3399#issuecomment-3442779585
    # Attribution: https://github.com/huggingface/trl/blob/f23543fc966bcc5c4252411d1098da2d4dc2d453/trl/trainer/sft_trainer.py#L661
    ds = (
        load_prompt_completion_llm_dataset(json_file_path, max_examples)
        .map(add_eos, fn_kwargs={"eos_token": tokenizer.eos_token}, desc="Adding EOS token")
        .map(tokenize_prompt_completion, fn_kwargs={"tokenizer": tokenizer}, remove_columns=["prompt", "completion"], desc="Tokenizing prompt and completion")
    )

    return ds
