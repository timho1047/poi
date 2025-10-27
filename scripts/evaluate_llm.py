from poi import settings
from poi.dataset.llm import load_tokenized_llm_dataset
from poi.llm import LLMConfig, evaluate_model, load_fast_inference_model

if __name__ == "__main__":
    # Configuration
    config = LLMConfig(run_name="llama3-nyc-test", num_epochs=8, batch_size=4, gradient_accumulation_steps=4, do_eval=True)

    # Load the evaluation dataset (must be tokenized)
    DATASET_DIR = settings.DATASETS_DIR / "NYC" / "LLM Dataset" / "test_codebook.json"
    eval_dataset = load_tokenized_llm_dataset(DATASET_DIR, config=config)

    # Load the trained model
    print("Loading model...")
    model = load_fast_inference_model(config)

    # Compute evaluation metrics
    print("Computing evaluation metrics...")
    metrics = evaluate_model(config, model, eval_dataset)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("=" * 50)
