from pathlib import Path
from typing import TypedDict

from datasets import Dataset


class LLJsonRecord(TypedDict):
    instruction: str
    input: str
    output: str


def format_example(ex: LLJsonRecord) -> dict[str, str]:
    prompt = ex["instruction"] + "\n" + ex["input"] + " "
    completion = ex["output"]
    return {"prompt": prompt, "completion": completion}


def load_llm_dataset(json_file_path: Path | str, max_examples: int = None) -> Dataset:
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
    ds = ds.map(format_example).remove_columns(["instruction", "input", "output"])

    return ds
