from poi import settings
from poi.llm import LLMConfig
from poi.llm.trainer import TrainLLMRun, train_llm_fast_ddp_batch

if __name__ == "__main__":
    DATASET_DIR = settings.DATASETS_DIR / "NYC"
    runs = [
        TrainLLMRun(
            config=LLMConfig(
                run_name="llama3-nyc-model-1",
                num_epochs=2,
                batch_size=2,
                gradient_accumulation_steps=1,
                resume_from_checkpoint=False,
                do_ddp=True,
                do_eval=False,
            ),
            train_dataset_path=DATASET_DIR / "LLM Dataset" / "train_codebook.json",
            eval_dataset_path=None,
            max_examples=5,
            push_to_hub=True,
            force_push=False,
        ),
        TrainLLMRun(
            config=LLMConfig(
                run_name="llama3-nyc-model-2",
                num_epochs=2,
                batch_size=2,
                gradient_accumulation_steps=1,
                resume_from_checkpoint=False,
                do_ddp=True,
                do_eval=False,
            ),
            train_dataset_path=DATASET_DIR / "LLM Dataset" / "train_codebook.json",
            eval_dataset_path=None,
            max_examples=5,
            push_to_hub=True,
            force_push=False,
        ),
 
    ]

    train_llm_fast_ddp_batch(runs)
