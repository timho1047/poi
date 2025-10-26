"""
Run command:
source .venv/bin/activate
torchrun --nproc_per_node=2 scripts/train_llm_batch.py

"""

from poi import settings
from poi.llm import LLMConfig
from poi.llm.trainer import TrainLLMRun, train_llm_fast_ddp_batch

if __name__ == "__main__":
    DATASET_DIR = settings.DATASETS_DIR / "NYC" / "LLM Dataset"
    runs = [
        TrainLLMRun(
            config=LLMConfig(
                run_name="llama3-nyc-no-time-multitest",
                num_epochs=8,
                batch_size=2,
                gradient_accumulation_steps=1,
                resume_from_checkpoint=False,
                do_ddp=True,
                do_eval=True,
            ),
            train_dataset_path=DATASET_DIR / "ablation without Time" / "train_codebook_notime.json",
            eval_dataset_path=DATASET_DIR / "ablation without Time" / "val_codebook_notime.json",
            max_examples=10,
            push_to_hub=True,
            force_push=False,
        ),
        TrainLLMRun(
            config=LLMConfig(
                run_name="llama3-nyc-no-div-multitest",
                num_epochs=8,
                batch_size=2,
                gradient_accumulation_steps=1,
                resume_from_checkpoint=False,
                do_ddp=True,
                do_eval=True,
            ),
            train_dataset_path=DATASET_DIR / "rqvae-nyc-div0.0-commit0.5-lr5e-5" / "train_codebook.json",
            eval_dataset_path=DATASET_DIR / "rqvae-nyc-div0.0-commit0.5-lr5e-5" / "val_codebook.json",
            max_examples=10,
            push_to_hub=True,
            force_push=False,
        ),
 
    ]

    train_llm_fast_ddp_batch(runs)
