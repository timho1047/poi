from poi import settings
from poi.dataset.llm import load_llm_dataset
from poi.llm import LLMConfig
from poi.llm.trainer import train_llm_fast

if __name__ == "__main__":
    config = LLMConfig(run_name="llama3-nyc-test", num_epochs=3, batch_size=2, gradient_accumulation_steps=4, do_eval=True)

    DATASET_DIR = settings.DATASETS_DIR / "NYC"
    max_examples = 5

    train_dataset = load_llm_dataset(DATASET_DIR / "LLM Dataset" / "train_codebook.json", max_examples=max_examples)
    eval_dataset = load_llm_dataset(DATASET_DIR / "LLM Dataset" / "test_codebook.json", max_examples=max_examples)

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Eval dataset size: {len(eval_dataset)} examples")

    # if push_to_hub is True, the model will be pushed to the Hugging Face hub repo, otherwise it will be saved to the local directory
    train_llm_fast(config, train_dataset, eval_dataset, push_to_hub=False)
