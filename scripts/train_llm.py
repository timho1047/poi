from poi.dataset.llm import load_llm_dataset
from poi.llm import LLMConfig, train_llm
from poi.settings import DATASETS_DIR

if __name__ == "__main__":
    config = LLMConfig(run_name="llama3-nyc-1")

    # Create dataset
    train_dataset = load_llm_dataset(DATASETS_DIR / "NYC" / "train_codebook.json", config)
    eval_dataset = load_llm_dataset(DATASETS_DIR / "NYC" / "test_codebook.json", config)

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Eval dataset size: {len(eval_dataset)} examples")

    train_llm(config, train_dataset, eval_dataset)
