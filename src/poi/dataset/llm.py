from pathlib import Path
from typing import TypedDict

from datasets import Dataset

class LLJsonRecord(TypedDict):
    instruction: str
    input: str
    output: str


def format_prompt_completion(ex: LLJsonRecord) -> dict[str, str]:
    prompt = ex["instruction"] + "\n" + ex["input"] + " "
    completion = ex["output"]
    return {"prompt": prompt, "completion": completion}


def tokenize_prompt_completion(ex, config):
    # make sure completion end with EOS
    full_text = ex["prompt"] + ex["completion"] + config.tokenizer.eos_token
    tokenized = config.tokenizer(
        full_text,
        truncation=True,
        max_length=config.max_length,
        return_offsets_mapping=True,
        padding=False,
        return_tensors=None,
    )
    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]
    prompt_end_char = len(ex["prompt"])
    labels = [-100 if start < prompt_end_char else token_id for (start, _), token_id in zip(offsets, input_ids)]
    return {"input_ids": input_ids, "labels": labels}


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


def load_tokenized_llm_dataset(json_file_path: Path | str, config, max_examples: int = None) -> Dataset:
    """Load JSON data and convert to Hugging Face Dataset

    Args:
        json_file_path: Path to the JSON file
        config: Configuration object
        max_examples: Maximum number of examples to load, None for all examples
    Returns:
        Hugging Face Dataset
    """
    # Attribution: https://github.com/unslothai/unsloth/issues/3399#issuecomment-3442779585
    # Attribution: https://github.com/huggingface/trl/blob/f23543fc966bcc5c4252411d1098da2d4dc2d453/trl/trainer/sft_trainer.py#L661
    ds = load_prompt_completion_llm_dataset(json_file_path, max_examples).map(
        tokenize_prompt_completion, fn_kwargs={"config": config}, remove_columns=["prompt", "completion"], desc="Tokenizing prompt and completion"
    )

    return ds
