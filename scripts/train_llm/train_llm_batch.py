"""
Run command:
source .venv/bin/activate
torchrun --nproc_per_node=8 scripts/train_llm_batch.py

"""

from pathlib import Path
from typing import TypedDict

from poi import settings
from poi.llm import LLMConfig
from poi.llm.trainer import TrainLLMRun, train_llm_fast_ddp_batch


class CreateRunBatchItem(TypedDict):
    run_name: str
    dataset_dir: Path


def create_run_batch(
    run_items: list[CreateRunBatchItem],
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    do_ddp: bool,
    do_eval: bool,
    max_examples: int | None,
    push_to_hub: bool,
    force_push: bool,
) -> list[TrainLLMRun]:
    train_llm_runs = []
    for item in run_items:
        # infer dataset path
        if not item["dataset_dir"].exists():
            raise ValueError(f"Dataset path {item['train_dataset_path']} or {item['eval_dataset_path']} does not exist")

        train_dataset_path = None
        eval_dataset_path = None
        for file in item["dataset_dir"].glob("*.json"):
            if file.name.startswith("train"):
                train_dataset_path = file
            elif file.name.startswith("val"):
                eval_dataset_path = file
        if train_dataset_path is None or eval_dataset_path is None:
            raise ValueError(f"No train or eval dataset found in {item['dataset_dir']}")

        train_llm_runs.append(
            TrainLLMRun(
                config=LLMConfig(
                    run_name=item["run_name"],
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    resume_from_checkpoint=False,
                    do_ddp=do_ddp,
                    do_eval=do_eval,
                ),
                train_dataset_path=train_dataset_path,
                eval_dataset_path=eval_dataset_path,
                max_examples=max_examples,
                push_to_hub=push_to_hub,
                force_push=force_push,
            )
        )
    return train_llm_runs


if __name__ == "__main__":
    NYC_DS = settings.DATASETS_DIR / "NYC" / "LLM Dataset"
    TKY_DS = settings.DATASETS_DIR / "TKY" / "LLM Dataset"

    NYC_BASE_DS = NYC_DS / "rqvae-nyc-div0.25-commit0.5-lr5e-5"
    NYC_NO_TIME_DS = NYC_DS / "ablation without Time"
    NYC_NO_DIV_DS = NYC_DS / "rqvae-nyc-div0.0-commit0.5-lr5e-5"
    NYC_NO_SID_DS = NYC_DS / "ablation without SID"
    NYC_NO_QUANT_DS = NYC_DS / "rqvae-nyc-div0.25-commit0.5-lr5e-5-without_L_quant"
    NYC_DIV_0_5_DS = NYC_DS / "rqvae-nyc-div0.5-commit0.5-lr5e-5"
    NYC_DIV_0_75_DS = NYC_DS / "rqvae-nyc-div0.75-commit0.5-lr5e-5"

    TKY_BASE_DS = TKY_DS / "rqvae-tky-div0.25-commit0.25-lr5e-5"
    TKY_NO_TIME_DS = TKY_DS / "ablation without Time"
    TKY_NO_DIV_DS = TKY_DS / "rqvae-tky-div0.0-commit0.25-lr5e-5"
    TKY_NO_SID_DS = TKY_DS / "ablation without SID"
    TKY_NO_QUANT_DS = TKY_DS / "rqvae-tky-div0.25-commit0.25-lr5e-5-without_L_quant"
    TKY_DIV_0_5_DS = TKY_DS / "rqvae-tky-div0.5-commit0.25-lr5e-5"
    TKY_DIV_0_75_DS = TKY_DS / "rqvae-tky-div0.75-commit0.25-lr5e-5"


    NUM_EPOCHS = 8
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 2
    DO_DDP = True
    DO_EVAL = True
    MAX_EXAMPLES = None
    PUSH_TO_HUB = True
    FORCE_PUSH = False

    run_items: list[CreateRunBatchItem] = [
        ########### 
        # NYC 
        ##########
        CreateRunBatchItem(
            run_name="llama3-nyc-base",
            dataset_dir=NYC_BASE_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-nyc-no-time",
            dataset_dir=NYC_NO_TIME_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-nyc-no-div",
            dataset_dir=NYC_NO_DIV_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-nyc-no-sid",
            dataset_dir=NYC_NO_SID_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-nyc-no-quant",
            dataset_dir=NYC_NO_QUANT_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-nyc-div-0.5",
            dataset_dir=NYC_DIV_0_5_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-nyc-div-0.75",
            dataset_dir=NYC_DIV_0_75_DS,
        ),
        ########### 
        # TKY 
        ##########
        CreateRunBatchItem(
            run_name="llama3-tky-base",
            dataset_dir=TKY_BASE_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-tky-no-time",
            dataset_dir=TKY_NO_TIME_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-tky-no-div",
            dataset_dir=TKY_NO_DIV_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-tky-no-sid",
            dataset_dir=TKY_NO_SID_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-tky-no-quant",
            dataset_dir=TKY_NO_QUANT_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-tky-div-0.5",
            dataset_dir=TKY_DIV_0_5_DS,
        ),
        CreateRunBatchItem(
            run_name="llama3-tky-div-0.75",
            dataset_dir=TKY_DIV_0_75_DS,
        ),
    ]

    runs = create_run_batch(run_items, NUM_EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, DO_DDP, DO_EVAL, MAX_EXAMPLES, PUSH_TO_HUB, FORCE_PUSH)
    
    print(f"Created {len(runs)} runs")
    for run in runs:
        print(f"Run: {run['config'].run_name}")
        print(f"Train dataset: {run['train_dataset_path']}")
        print(f"Eval dataset: {run['eval_dataset_path']}")
        print(f"Max examples: {run['max_examples']}")
        print(f"Push to hub: {run['push_to_hub']}")
        print(f"Force push: {run['force_push']}")
        print("-" * 70)

    # train_llm_fast_ddp_batch(runs)
