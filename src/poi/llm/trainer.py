import json
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import torch
from huggingface_hub import create_repo, repo_exists, upload_file, upload_folder
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    TrainerCallback,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel

from datasets import Dataset

from .. import settings
from ..dataset.llm import load_tokenized_llm_dataset
from ..llm import LLMConfig
from ..llm.ddp_utils import TrainLLMRun, ddp_context, serialize_run
from .utils import generate_model_card, print_trainable_parameters, print_training_configuration

# Suppress harmless warnings
warnings.filterwarnings("ignore", message=".*use_reentrant parameter.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def prepare_model(config: LLMConfig, rank: int = 0):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Load model on the specific device for this rank
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_length,
        load_in_4bit=config.quantization_bits == 4,
        load_in_8bit=config.quantization_bits == 8,
        full_finetuning=False,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_config.r,
        target_modules=config.lora_config.target_modules,
        lora_alpha=config.lora_config.lora_alpha,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=settings.RANDOM_STATE,
    )
    return model, tokenizer


class SaveBestModelCallback(TrainerCallback):
    """Track the best model and save. Only used when in DDP with unsloth."""

    def __init__(self, trainer: SFTTrainer, early_stopping_patience: int = 2, rank: int = 0):
        self.trainer = trainer
        self.rank = rank
        self.best_eval_loss = float("inf")
        self.best_epoch = 0
        self.early_stopping_patience = early_stopping_patience

    def on_evaluate(self, args, state, control, **kwargs):
        logs = state.log_history[-1]
        current_eval_loss = logs.get("eval_loss", float("inf"))

        if current_eval_loss < self.best_eval_loss:
            if self.rank == 0:
                print(f"\n✓ Eval loss improved from {self.best_eval_loss:.4f} to {current_eval_loss:.4f}, will save the model\n")
            self.best_eval_loss = current_eval_loss
            self.trainer.save_model(self.trainer.args.output_dir)
            self.best_epoch = state.epoch
        else:
            if self.rank == 0:
                print(f"\nCurrent eval loss: {current_eval_loss:.4f} > best eval loss: {self.best_eval_loss:.4f}\n")

        if state.epoch - self.best_epoch > self.early_stopping_patience:
            if self.rank == 0:
                print(f"\nEarly stopping at epoch {state.epoch}\n")
            control.should_training_stop = True


def train_llm_fast(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset | None = None, push_to_hub: bool = False, rank: int = 0):
    if rank == 0:
        print_training_configuration(config)

    model, tokenizer = prepare_model(config, rank)

    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    if rank == 0:
        print("Start training on main process...")
    if config.do_eval:
        trainer.add_callback(SaveBestModelCallback(trainer, rank=rank))
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Push to hub
    if rank == 0:
        if not config.do_eval:
            trainer.save_model(config.output_dir)
        model_card = generate_model_card(config)
        config.model_card_path.write_text(model_card)

        if push_to_hub:
            print(f"Uploading model to {config.hub_id}...")
            if not repo_exists(config.hub_id, token=settings.HF_TOKEN):
                create_repo(repo_id=config.hub_id, token=settings.HF_TOKEN, private=False)

            upload_folder(
                folder_path=str(config.output_dir),
                repo_id=config.hub_id,
                token=settings.HF_TOKEN,
                commit_message=f"Training completed for {config.run_name}",
                ignore_patterns=["checkpoint-*"],
            )

    return trainer, model, tokenizer


def train_full_llm_fast(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset | None = None, push_to_hub: bool = False):
    print_training_configuration(config)

    # Load model on the specific device for this rank
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        attn_implementation="sdpa",
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_config.r,
        target_modules=config.lora_config.target_modules,
        lora_alpha=config.lora_config.lora_alpha,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=settings.RANDOM_STATE,
    )

    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(SaveBestModelCallback(trainer, rank=0))

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    model_card = generate_model_card(config)
    config.model_card_path.write_text(model_card)

    if push_to_hub:
        print(f"Uploading model to {config.hub_id}...")
        if not repo_exists(config.hub_id, token=settings.HF_TOKEN):
            create_repo(repo_id=config.hub_id, token=settings.HF_TOKEN, private=False)

        upload_folder(
            folder_path=str(config.output_dir),
            repo_id=config.hub_id,
            token=settings.HF_TOKEN,
            commit_message=f"Training completed for {config.run_name}",
            ignore_patterns=["checkpoint-*"],
        )

    return trainer, model, tokenizer


def train_llm_fast_ddp(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset, push_to_hub: bool = False):
    """An simple implementation of DDP training for unsloth. Only works for training one model."""
    # Attribution: https://github.com/unslothai/unsloth/issues/2435#issuecomment-3436555056

    # Wrap the training in a DDP context, will fallback to single GPU training if not using DDP
    with ddp_context() as (_, rank, _):
        trainer, model, tokenizer = train_llm_fast(config, train_dataset, eval_dataset, push_to_hub, rank)

    return trainer, model, tokenizer


def train_llm_ddp_single_run(run: TrainLLMRun):
    """A single run of training a LLM model in DDP."""

    with ddp_context() as (_, rank, world_size):
        if rank == 0:
            start_time = time.time()
            print(f"\n{'=' * 70}")
            print(f"Training: {run['config'].run_name}")
            print(f"{'=' * 70}\n")
            if run["push_to_hub"]:
                assert settings.HF_TOKEN is not None, "HF_TOKEN is not set"

        train_dataset = None
        eval_dataset = None

        try:
            # Load the datasets on each rank on by one to avoid deadlock
            for i in range(world_size):
                if i == rank:
                    train_dataset = load_tokenized_llm_dataset(run["train_dataset_path"], config=run["config"], max_examples=run["max_examples"])
                    if run["eval_dataset_path"]:
                        eval_dataset = load_tokenized_llm_dataset(run["eval_dataset_path"], config=run["config"], max_examples=run["max_examples"])
                if world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.barrier()

            should_skip = False
            if rank == 0:
                should_skip = not run["force_push"] and repo_exists(run["config"].hub_id)

            if world_size > 1 and torch.distributed.is_initialized():
                should_skip_tensor = torch.tensor([should_skip], dtype=torch.bool, device=f"cuda:{rank}")
                torch.distributed.broadcast(should_skip_tensor, src=0)
                should_skip = should_skip_tensor.item()

            if should_skip:
                if rank == 0:
                    print(f"\n{'=' * 70}")
                    print(f"Repo {run['config'].hub_id} already exists, skipping...")
                    print(f"{'=' * 70}\n")
            else:
                train_llm_fast(run["config"], train_dataset, eval_dataset, run["push_to_hub"], rank=rank)

        except Exception as e:
            if rank == 0:
                print(f"\n{'=' * 70}")
                print(f"Error training {run['config'].run_name}")
                print(f"{'=' * 70}\n")
                print(f"Error: {str(e)}")

            if world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.barrier()

            raise

        finally:
            if rank == 0:
                print(f"\n{'=' * 70}")
                print(f"Completed: {run['config'].run_name}")
                print(f"Time taken: {(time.time() - start_time) / 3600:.2f} hours")
                print(f"{'=' * 70}\n")


def train_llm_ddp_batch(runs: list[TrainLLMRun], script_path: Path, nproc_per_node: int = 8):
    """
    Train a batch of LLM models in DDP. Using subprocess to run the training script to avoid deadlocks and memory cleaning.
    """

    if not script_path.exists():
        raise FileNotFoundError(f"Script path {script_path.absolute().as_posix()} does not exist")

    global_start_time = time.time()

    for idx, run in enumerate(runs, 1):
        print(f"\n{'=' * 70}")
        print(f"Starting run {idx}/{len(runs)}: {run['config'].run_name}")
        print(f"{'=' * 70}\n")

        local_start_time = time.time()

        # Use temporary file to avoid generating too many files in the current directory
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name
            json.dump(serialize_run(run), f)

        try:
            result = subprocess.run(
                ["torchrun", f"--nproc_per_node={nproc_per_node}", script_path.absolute().as_posix(), "--config", config_path], check=False
            )

            if result.returncode == 0:
                print(f"\n{'=' * 70}")
                print(f"Successfully completed: {run['config'].run_name}")
                print(f"Time taken: {(time.time() - local_start_time) / 3600:.2f} hours")
                print(f"{'=' * 70}\n")
            else:
                print(f"\n{'=' * 70}")
                print(f"Failed: {run['config'].run_name} (exit code: {result.returncode})")
                print(f"Time taken: {(time.time() - local_start_time) / 3600:.2f} hours")
                print("Continuing with next run...")
                print(f"{'=' * 70}\n")

        finally:
            # Clean up the temporary file
            Path(config_path).unlink(missing_ok=True)

    print(f"\n{'=' * 70}")
    print("All runs completed!")
    print(f"Total time: {(time.time() - global_start_time) / 3600:.2f} hours")
    print(f"{'=' * 70}\n")



def train_llm(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset, push_to_hub: bool = False):
    print_training_configuration(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config.lora_config)

    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=config.tokenizer,
        args=config.training_args,
    )

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    model_card = generate_model_card(config)

    if push_to_hub:
        trainer.push_to_hub(commit_message=f"Training completed for {config.run_name}")
        upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=config.hub_id,
            token=settings.HF_TOKEN,
        )
    else:
        trainer.save_model(config.output_dir)

    config.model_card_path.write_text(model_card)
    return trainer
