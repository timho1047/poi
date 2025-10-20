from poi import settings
from poi.dataset.llm import load_llm_dataset
from poi.llm import LLMConfig
from poi.llm.trainer import train_llm_fast

if __name__ == "__main__":
    config = LLMConfig(run_name="llama3-nyc-1")

    DATASET_DIR = settings.DATASETS_DIR / "NYC"
    max_examples = 20

    train_dataset = load_llm_dataset(DATASET_DIR / "train_codebook.json", max_examples=max_examples)
    eval_dataset = load_llm_dataset(DATASET_DIR / "test_codebook.json", max_examples=max_examples)

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Eval dataset size: {len(eval_dataset)} examples")

    train_llm_fast(config, train_dataset, eval_dataset)
