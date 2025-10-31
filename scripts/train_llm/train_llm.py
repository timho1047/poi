from poi import settings
from poi.dataset.llm import load_tokenized_llm_dataset
from poi.llm import LLMConfig
from poi.llm.trainer import train_llm_fast

if __name__ == "__main__":
    config = LLMConfig(
        run_name="llama3-nyc-test-no-sid", num_epochs=8, batch_size=8, gradient_accumulation_steps=8, do_eval=True, resume_from_checkpoint=False
    )

    DATASET_DIR = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "paper"
    max_examples = None

    train_dataset = load_tokenized_llm_dataset(
        DATASET_DIR / "train_id.json", config=config, max_examples=max_examples
    )
    eval_dataset = load_tokenized_llm_dataset(
        DATASET_DIR / "test_id.json", config=config, max_examples=max_examples
    )

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Eval dataset size: {len(eval_dataset)} examples")

    # if push_to_hub is True, the model will be pushed to the Hugging Face hub repo, otherwise it will be saved to the local directory
    train_llm_fast(config, train_dataset, eval_dataset, push_to_hub=True)
