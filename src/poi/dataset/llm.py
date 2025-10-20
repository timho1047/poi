import json
from pathlib import Path
from typing import TypedDict

from datasets import Dataset

from ..llm.config import LLMConfig


class LLJsonRecord(TypedDict):
    instruction: str
    input: str
    output: str


def load_llm_dataset(json_file_path: Path | str, config: LLMConfig) -> Dataset:
    """Load JSON data and convert to Hugging Face Dataset"""
    if not isinstance(json_file_path, Path):
        json_file_path = Path(json_file_path)

    data = json.loads(json_file_path.read_text())
    bos = config.tokenizer.bos_token
    eos = config.tokenizer.eos_token

    def format_chat_template(example: LLJsonRecord):
        text = (
            bos
            + example["instruction"]
            + "\n"
            + example["input"]
            + " "
            + example["output"]
            + eos
        )
        return {"text": text}

    # Create Hugging Face Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_chat_template)

    return dataset
