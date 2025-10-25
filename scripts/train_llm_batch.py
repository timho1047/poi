"""
To run this script,
1. provide HF_TOKEN in .env
2. source .venv/bin/activate
3. torchrun --nproc_per_node=2 scripts/train_llm_batch.py
"""


from poi import settings
from poi.llm import LLMConfig
from poi.llm.trainer import TrainLLMRun, train_llm_fast_ddp_batch

if __name__ == "__main__":
    NYC_DATASET_DIR = settings.DATASETS_DIR / "NYC" / "LLM Dataset"
    runs = [
        TrainLLMRun(
            config=LLMConfig(
                run_name="llama3-nyc-base",
                num_epochs=8,
                batch_size=2,
                gradient_accumulation_steps=1,
                resume_from_checkpoint=False,
                do_ddp=True,
                do_eval=False,
            ),
            train_dataset_path=NYC_DATASET_DIR / "rqvae-nyc-div0.25-commit0.5-lr5e-5" / "train_codebook.json",
            eval_dataset_path=NYC_DATASET_DIR / "rqvae-nyc-div0.25-commit0.5-lr5e-5" / "val_codebook.json",
            max_examples=10,
            push_to_hub=True,
            force_push=True,
        ),
        TrainLLMRun(
            config=LLMConfig(
                run_name="llama3-nyc-no-time",
                num_epochs=8,
                batch_size=2,
                gradient_accumulation_steps=1,
                resume_from_checkpoint=False,
                do_ddp=True,
                do_eval=False,
            ),
            train_dataset_path=NYC_DATASET_DIR / "ablation without Time" / "train_codbook_notime.json",
            eval_dataset_path=NYC_DATASET_DIR / "ablation without Time" / "val_codebook_notime.json",
            max_examples=100,
            push_to_hub=True,
            force_push=True,
        ),
 
    ]

    train_llm_fast_ddp_batch(runs)
