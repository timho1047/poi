import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import torch

from .config import LLMConfig


@contextmanager
def ddp_context():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    # Initialize DDP
    if is_dist:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print(f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size} device={device}")

    yield local_rank, rank, world_size

    # Graceful DDP cleanup
    if is_dist and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


class TrainLLMRun(TypedDict):
    config: LLMConfig
    train_dataset_path: Path
    eval_dataset_path: Path | None
    max_examples: int | None
    push_to_hub: bool
    force_push: bool

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


def serialize_run(run: TrainLLMRun) -> dict:
    return {
        "config": {
            "run_name": run["config"].run_name,
            "num_epochs": run["config"].num_epochs,
            "batch_size": run["config"].batch_size,
            "gradient_accumulation_steps": run["config"].gradient_accumulation_steps,
            "resume_from_checkpoint": run["config"].resume_from_checkpoint,
            "do_ddp": run["config"].do_ddp,
            "do_eval": run["config"].do_eval,
        },
        "train_dataset_path": str(run["train_dataset_path"]),
        "eval_dataset_path": str(run["eval_dataset_path"]) if run["eval_dataset_path"] else None,
        "max_examples": run["max_examples"],
        "push_to_hub": run["push_to_hub"],
        "force_push": run["force_push"],
    }


def load_run_config(config_path: Path):
    with open(config_path) as f:
        data = json.load(f)

    config_dict = data["config"]
    config = LLMConfig(
        run_name=config_dict["run_name"],
        num_epochs=config_dict["num_epochs"],
        batch_size=config_dict["batch_size"],
        gradient_accumulation_steps=config_dict["gradient_accumulation_steps"],
        resume_from_checkpoint=config_dict["resume_from_checkpoint"],
        do_ddp=config_dict["do_ddp"],
        do_eval=config_dict["do_eval"],
    )

    return {
        "config": config,
        "train_dataset_path": Path(data["train_dataset_path"]),
        "eval_dataset_path": Path(data["eval_dataset_path"]) if data["eval_dataset_path"] else None,
        "max_examples": data["max_examples"],
        "push_to_hub": data["push_to_hub"],
        "force_push": data["force_push"],
    }
