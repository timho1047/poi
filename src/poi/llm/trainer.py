import os
import shutil
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import torch
from huggingface_hub import create_repo, repo_exists, upload_file
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel
from transformers import TrainerCallback

from datasets import Dataset

from .. import settings
from ..dataset.llm import load_tokenized_llm_dataset
from ..llm import LLMConfig
from .memory_utils import cleanup_memory
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


def train_llm_fast(
    config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset, push_to_hub: bool = False, rank: int = 0, manual_load_best_at_end: bool = False
):
    if rank == 0:
        print_training_configuration(config)

    model, tokenizer = prepare_model(config, rank)

    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not manual_load_best_at_end else None,
        tokenizer=tokenizer,
    )

    if rank == 0:
        print("Start training on main process...")

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Only rank 0 saves the model
    if rank == 0:
        print("Training complete. Saving model...")
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
        print("Model saved successfully.")

    return trainer, model, tokenizer


# Stop training after each epoch
class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


class SaveBestModelCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        logs = state.log_history[-1]
        current_eval_loss = logs.get("eval_loss", float("inf"))
        best_eval_loss = state.best_metric
        if current_eval_loss >= best_eval_loss:
            control.should_save = True

def train_llm_fast_manual_load_best_at_end(
    config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset, push_to_hub: bool = False, rank: int = 0
):
    EARLY_STOPPING_PATIENCE = 2

    if config.do_eval:
        raise ValueError("Please set do_eval to False when using manual load best at end.")

    if rank == 0:
        print_training_configuration(config)

    # clear checkpoint directories
    checkpoint_dirs = [d for d in config.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    for checkpoint_dir in checkpoint_dirs:
        shutil.rmtree(checkpoint_dir)

    model, tokenizer = prepare_model(config, rank)
    
    
    
    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[SaveBestModelCallback()],
    )
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # best_eval_loss = float("inf")
    # best_model_dir = Path(config.output_dir) / "best_model"
    # best_epoch = 0


    # for epoch in range(int(config.training_args.num_train_epochs)):
    #     if rank == 0:
    #         print(f"\n{'=' * 50}")
    #         print(f"Training Epoch {epoch + 1}/{int(config.training_args.num_train_epochs)}")
    #         print(f"{'=' * 50}")

    #     # Train for one epoch
    #     trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

        # Evaluate after this epoch
        # eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        # current_eval_loss = eval_results.get("eval_loss", float("inf"))

        # if rank == 0:
        #     print(f"\nEpoch {epoch + 1} - Eval Loss: {current_eval_loss:.4f}")
        #     print(f"Best Eval Loss so far: {best_eval_loss:.4f}")

        #     # Save model if it's the best so far
        #     if current_eval_loss < best_eval_loss:
        #         print(f"✓ Eval loss improved from {best_eval_loss:.4f} to {current_eval_loss:.4f}")
        #         print(f"Saving best model to {best_model_dir}")
        #         best_eval_loss = current_eval_loss
        #         best_epoch = epoch
        #         trainer.save_model(str(best_model_dir))
        #     else:
        #         print(f"✗ Eval loss did not improve from {best_eval_loss:.4f}")

        #     if epoch - best_epoch > EARLY_STOPPING_PATIENCE:
        #         print(f"Early stopping at epoch {epoch}")
        #         break

    # Load the best model at the end (only on rank 0 to avoid DDP issues)
    if rank == 0:
        # print(f"\n{'=' * 50}")
        # print(f"Loading best model from {best_model_dir}")
        # print(f"Best eval loss: {best_eval_loss:.4f}")
        # print(f"{'=' * 50}")
        # # Copy best model to final output directory
        # for item in best_model_dir.iterdir():
        #     dest = config.output_dir / item.name
        #     if item.is_file():
        #         shutil.copy2(item, dest)
        #     elif item.is_dir():
        #         if dest.exists():
        #             shutil.rmtree(dest)
        #         shutil.copytree(item, dest)

        model_card = generate_model_card(config)
        config.model_card_path.write_text(model_card)

        if push_to_hub:
            from huggingface_hub import upload_folder

            print(f"Uploading model to {config.hub_id}...")
            if not repo_exists(config.hub_id, token=settings.HF_TOKEN):
                create_repo(repo_id=config.hub_id, token=settings.HF_TOKEN, private=False)

            upload_folder(
                folder_path=str(config.output_dir),
                repo_id=config.hub_id,
                token=settings.HF_TOKEN,
                commit_message=f"Training completed for {config.run_name}",
                ignore_patterns=["checkpoint-*", "best_model"],
            )

    return trainer, model, tokenizer


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


def train_llm_fast_ddp(config: LLMConfig, train_dataset: Dataset, eval_dataset: Dataset, push_to_hub: bool = False):
    # Attribution: https://github.com/unslothai/unsloth/issues/2435#issuecomment-3436555056

    # Wrap the training in a DDP context, will fallback to single GPU training if not using DDP
    with ddp_context() as (_, rank, _):
        trainer, model, tokenizer = train_llm_fast(config, train_dataset, eval_dataset, push_to_hub, rank)

    return trainer, model, tokenizer


class TrainLLMRun(TypedDict):
    config: LLMConfig
    train_dataset_path: Path
    eval_dataset_path: Path | None
    max_examples: int | None
    push_to_hub: bool
    force_push: bool


def train_llm_fast_ddp_batch(runs: list[TrainLLMRun]):
    with ddp_context() as (_, rank, world_size):
        if rank == 0:
            global_start_time = time.time()
            if any(run["push_to_hub"] for run in runs):
                assert settings.HF_TOKEN is not None, "HF_TOKEN is not set"

        for idx, run in enumerate(runs, 1):
            if rank == 0:
                print(f"\n{'=' * 70}")
                local_start_time = time.time()
                print(f"Training: {run['config'].run_name}")
                print(f"{'=' * 70}\n")

            train_dataset = None
            eval_dataset = None
            trainer = None
            model = None
            tokenizer = None
            try:
                train_dataset = load_tokenized_llm_dataset(run["train_dataset_path"], config=run["config"], max_examples=run["max_examples"])
                eval_dataset = (
                    load_tokenized_llm_dataset(run["eval_dataset_path"], config=run["config"], max_examples=run["max_examples"])
                    if run["eval_dataset_path"] is not None
                    else None
                )

                if not run["force_push"] and repo_exists(run["config"].hub_id):
                    if rank == 0:
                        print(f"\n{'=' * 70}")
                        print(f"Repo {run['config'].hub_id} already exists, skipping...")
                        print(f"{'=' * 70}\n")
                else:
                    trainer, model, tokenizer = train_llm_fast_manual_load_best_at_end(
                        run["config"], train_dataset, eval_dataset, run["push_to_hub"], rank=rank
                    )

            except Exception as e:
                if rank == 0:
                    print(f"\n{'=' * 70}")
                    print(f"Error training {run['config'].run_name}, skipping...")
                    print(f"{'=' * 70}\n")
                    print(f"Error: {str(e)}")

                # Synchronize all ranks after exception to prevent hanging
                if world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.barrier()

            finally:
                # 1. Delete optimizer state from trainer
                if trainer is not None:
                    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                        del trainer.optimizer
                    if hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler is not None:
                        del trainer.lr_scheduler
                    # Remove model reference from trainer
                    if hasattr(trainer, "model"):
                        trainer.model = None
                    del trainer

                # 2. Delete model (largest memory consumer)
                if model is not None:
                    # Move model to CPU first to free GPU memory
                    if hasattr(model, "cpu"):
                        model.cpu()
                    del model

                # 3. Delete tokenizer
                if tokenizer is not None:
                    del tokenizer

                # 4. Delete datasets
                if eval_dataset is not None:
                    del eval_dataset
                if train_dataset is not None:
                    del train_dataset

                # 5. Synchronize before cleanup to ensure all ranks are done
                if world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.barrier()

                # 6. Clean up memory
                cleanup_memory()

                if rank == 0:
                    print(f"\n{'=' * 70}")
                    print(f"Completed: {run['config'].run_name}")
                    print(f"Time taken for this run: {(time.time() - local_start_time) / 3600:.2f} hours")
                    print(f"Time taken for all runs: {(time.time() - global_start_time) / 3600:.2f} hours")
                    print(f"Completed {idx}/{len(runs)} runs")
                    print(f"{'=' * 70}\n")

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"All runs completed in {(time.time() - global_start_time) / 3600:.2f} hours!")
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
