import json
from pathlib import Path
from typing import TypedDict

from datasets import Dataset


class LLJsonRecord(TypedDict):
    instruction: str
    input: str
    output: str


def load_llm_dataset(json_file_path: Path | str, max_examples: int = None) -> Dataset:
    """Load JSON data and convert to Hugging Face Dataset
    
    Args:
        json_file_path: Path to the JSON file
        max_examples: Maximum number of examples to load, None for all examples
    Returns:
        Hugging Face Dataset
    """
    if not isinstance(json_file_path, Path):
        json_file_path = Path(json_file_path)

    data = json.loads(json_file_path.read_text())
    if max_examples is not None:
        data = data[:max_examples]

    def format_chat_template(example: LLJsonRecord):
        text = example["instruction"] + "\n" + example["input"] + " " + example["output"]
        return {"text": text}

    # Create Hugging Face Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_chat_template)

    return dataset
