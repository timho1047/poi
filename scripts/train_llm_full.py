from poi import settings
from poi.dataset.llm import load_tokenized_llm_dataset
from poi.llm import LLMConfig
from poi.llm.trainer import train_full_llm_fast

if __name__ == "__main__":
    config = LLMConfig(
        run_name="llama3-nyc-test-full-fintune", num_epochs=8, batch_size=4, gradient_accumulation_steps=16, do_eval=True, resume_from_checkpoint=False
    )

    DATASET_DIR = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "paper"
    max_examples = None

    train_dataset = load_tokenized_llm_dataset(DATASET_DIR / "train_codebook.json", config=config, max_examples=max_examples)
    eval_dataset = load_tokenized_llm_dataset(DATASET_DIR / "test_codebook.json", config=config, max_examples=max_examples)

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Eval dataset size: {len(eval_dataset)} examples")

    # if push_to_hub is True, the model will be pushed to the Hugging Face hub repo, otherwise it will be saved to the local directory
    train_full_llm_fast(config, train_dataset, eval_dataset, push_to_hub=True)
