from collections import Counter

from datasets import Dataset
from transformers import AutoTokenizer


def check_sequence_lengths(dataset: Dataset, tokenizer: AutoTokenizer, name: str = "Train") -> None:
    lengths = []
    for example in dataset:
        tokens = tokenizer(example["text"], return_tensors="pt")
        length = tokens["input_ids"].shape[1]
        lengths.append(length)

    print(f"\n{name} sequence length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths) / len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths) // 2]}")
    print("\n  Distribution:")
    counter = Counter([l // 100 * 100 for l in lengths])  # bucket by 100s
    for bucket in sorted(counter.keys()):
        print(f"    {bucket}-{bucket + 99}: {counter[bucket]} examples")
