"""
Cleaner version of multi-model sequential training using memory utility functions.
This demonstrates best practices for training multiple models with proper cleanup.
"""

import os

from poi import settings
from poi.dataset.llm import load_llm_dataset
from poi.llm import LLMConfig, cleanup_memory, print_memory_summary
from poi.llm.trainer import train_llm_fast_multi_gpu


def train_single_model(config: LLMConfig, train_dataset, eval_dataset, push_to_hub: bool = False):
    """
    Train a single model and return the trainer.

    Args:
        config: LLM configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        push_to_hub: Whether to push model to Hugging Face Hub

    Returns:
        trainer: The trained model's trainer object
    """
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Training: {config.run_name}")
        print(f"{'=' * 70}\n")

    trainer = train_llm_fast_multi_gpu(config, train_dataset, eval_dataset, push_to_hub)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Completed: {config.run_name}")
        print(f"{'=' * 70}\n")
        print_memory_summary()

    return trainer


def main():
    """Main training loop for multiple models."""
    rank = int(os.environ.get("RANK", 0))

    # Load datasets
    DATASET_DIR = settings.DATASETS_DIR / "NYC"

    if rank == 0:
        print("Loading datasets...")

    train_dataset = load_llm_dataset(DATASET_DIR / "LLM Dataset" / "train_codebook.json", max_examples=4)
    eval_dataset = load_llm_dataset(DATASET_DIR / "LLM Dataset" / "test_codebook.json", max_examples=4)

    # Define model configurations
    configs = [
        LLMConfig(
            run_name="llama3-nyc-model-1",
            num_epochs=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            resume_from_checkpoint=False,
        ),
        LLMConfig(
            run_name="llama3-nyc-model-2",
            num_epochs=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            resume_from_checkpoint=False,
            lr=2e-5,
        ),
    ]

    if rank == 0:
        print(f"\nTraining {len(configs)} models sequentially...\n")

    # Train each model
    for idx, config in enumerate(configs, 1):
        if rank == 0:
            print(f"\n{'#' * 70}")
            print(f"# Model {idx}/{len(configs)}")
            print(f"{'#' * 70}\n")

        try:
            # Train model
            trainer = train_single_model(
                config=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                push_to_hub=False,
            )

            # Clean up
            del trainer
            cleanup_memory()

        except Exception as e:
            if rank == 0:
                print(f"\nERROR training {config.run_name}: {str(e)}")
            raise

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
