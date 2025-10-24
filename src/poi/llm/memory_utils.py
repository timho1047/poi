"""
Utility functions for memory management during multi-GPU training.

These utilities help prevent OOM (Out Of Memory) errors when training
multiple models sequentially.
"""

import gc
import os

import torch


def get_rank_info():
    """
    Get distributed training rank information.

    Returns:
        tuple: (local_rank, rank, world_size, is_distributed)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    return local_rank, rank, world_size, is_distributed


def cleanup_memory(verbose: bool = True):
    """
    Comprehensive memory cleanup to avoid OOM errors between model training runs.

    This function:
    1. Forces Python garbage collection
    2. Clears PyTorch CUDA cache
    3. Synchronizes CUDA operations
    4. Reports memory usage (on rank 0)

    Args:
        verbose: Whether to print memory cleanup information (default: True)
    """
    local_rank, rank, world_size, is_distributed = get_rank_info()

    if verbose and rank == 0:
        print("\n" + "=" * 50)
        print("Starting memory cleanup...")
        print("=" * 50)

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Print memory stats on rank 0
        if verbose and rank == 0:
            allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
            reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(local_rank) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(local_rank) / 1024**3

            print(f"\nGPU {local_rank} Memory Stats:")
            print(f"  Current - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            print(f"  Peak    - Allocated: {max_allocated:.2f} GB, Reserved: {max_reserved:.2f} GB")

            # Reset peak memory stats for next model
            torch.cuda.reset_peak_memory_stats(local_rank)

    if verbose and rank == 0:
        print("Memory cleanup complete.")
        print("=" * 50 + "\n")


def get_memory_summary(device: int = 0) -> dict:
    """
    Get a summary of CUDA memory usage.

    Args:
        device: CUDA device index (default: 0)

    Returns:
        dict: Memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
            "free_gb": 0.0,
            "total_gb": 0.0,
        }

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    # Get total GPU memory
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free = total - allocated

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "max_reserved_gb": max_reserved,
        "free_gb": free,
        "total_gb": total,
    }


def print_memory_summary(device: int = 0):
    """
    Print a formatted summary of CUDA memory usage.

    Args:
        device: CUDA device index (default: 0)
    """
    local_rank, rank, world_size, is_distributed = get_rank_info()

    if rank != 0:
        return  # Only print on rank 0

    if not torch.cuda.is_available():
        print("CUDA not available - no GPU memory to report")
        return

    stats = get_memory_summary(device)

    print(f"\nGPU {device} Memory Summary:")
    print(f"  Total GPU Memory: {stats['total_gb']:.2f} GB")
    print(f"  Free Memory:      {stats['free_gb']:.2f} GB")
    print(f"  Allocated:        {stats['allocated_gb']:.2f} GB")
    print(f"  Reserved:         {stats['reserved_gb']:.2f} GB")
    print(f"  Peak Allocated:   {stats['max_allocated_gb']:.2f} GB")
    print(f"  Peak Reserved:    {stats['max_reserved_gb']:.2f} GB")
    print()


def cleanup_trainer(trainer):
    """
    Clean up a trainer object and associated resources.

    Args:
        trainer: The trainer object to clean up
    """
    if trainer is not None:
        # Explicitly delete model and tokenizer from trainer
        if hasattr(trainer, "model"):
            del trainer.model
        if hasattr(trainer, "tokenizer"):
            del trainer.tokenizer

        # Delete the trainer itself
        del trainer

    # Run memory cleanup
    cleanup_memory(verbose=False)


def ensure_clean_state():
    """
    Ensure a clean state before starting training.
    Useful when training multiple models sequentially.
    """
    local_rank, rank, world_size, is_distributed = get_rank_info()

    # Clean up any existing distributed process group
    if is_distributed and torch.distributed.is_initialized():
        if rank == 0:
            print("Warning: Distributed process group already initialized. Destroying...")
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    # Clean up memory
    cleanup_memory(verbose=False)
