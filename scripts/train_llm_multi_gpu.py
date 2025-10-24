from poi import settings
from poi.dataset.llm import load_llm_dataset
from poi.llm import LLMConfig
from poi.llm.trainer import train_llm_fast_ddp

if __name__ == "__main__":
    config = LLMConfig(run_name="llama3-nyc-multi-gpu-5", num_epochs=2, batch_size=2, gradient_accumulation_steps=1, resume_from_checkpoint=False, do_ddp=True, do_eval=False)

    DATASET_DIR = settings.DATASETS_DIR / "NYC"

    train_dataset = load_llm_dataset(DATASET_DIR / "LLM Dataset" / "train_codebook.json", max_examples=4)
    eval_dataset = load_llm_dataset(DATASET_DIR / "LLM Dataset" / "test_codebook.json", max_examples=4)

    # if push_to_hub is True, the model will be pushed to the Hugging Face hub repo, otherwise it will be saved to the local directory
    train_llm_fast_ddp(config, train_dataset, None, push_to_hub=False)
