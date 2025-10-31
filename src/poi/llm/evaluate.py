from pathlib import Path

import pandas as pd
from tqdm import tqdm

from datasets import Dataset

from ..dataset.llm import load_prompt_completion_llm_dataset
from ..llm import load_fast_inference_model
from .config import LLMConfig
from .inference import inference


def top_one_accuracy(config: LLMConfig, model, ds: Dataset, is_sid: bool = True):
    total = len(ds)
    correct = 0
    prefix = "<a_" if is_sid else "<"

    for i in tqdm(range(total)):
        res = ""
        retry_count = 0
        while retry_count < 5 and res == "":
            res = inference(config, model, ds[i]["prompt"] + prefix).strip()  # provide <a_ as prefix hint
            retry_count += 1
        res = prefix + res
        if ds[i]["completion"] in res:
            correct += 1
    return correct / total


def evaluate_model(config: LLMConfig, ds_dir: Path, from_hub: bool = True):
    if not ds_dir.exists():
        raise ValueError(f"Dataset path {ds_dir} does not exist")

    train_ds_path = None
    val_ds_path = None
    test_all_ds_path = None
    test_ds_path = None
    for file in ds_dir.glob("*.json"):
        if file.name.startswith("train"):
            train_ds_path = file
        elif file.name.startswith("val"):
            val_ds_path = file
        elif file.name.startswith("test_all"):
            test_all_ds_path = file
        elif file.name.startswith("test"):
            test_ds_path = file
    if not all([train_ds_path, test_ds_path]):
        raise ValueError(f"No train or test dataset found in {ds_dir}")

    train_ds = load_prompt_completion_llm_dataset(train_ds_path)
    val_ds = load_prompt_completion_llm_dataset(val_ds_path) if val_ds_path else None
    test_all_ds = load_prompt_completion_llm_dataset(test_all_ds_path) if test_all_ds_path else None
    test_ds = load_prompt_completion_llm_dataset(test_ds_path)

    is_sid = "codebook" in train_ds_path.name.lower()

    model = load_fast_inference_model(config, from_hub=from_hub)

    train_acc = top_one_accuracy(config, model, train_ds, is_sid)
    val_acc = top_one_accuracy(config, model, val_ds, is_sid) if val_ds else None
    test_all_acc = top_one_accuracy(config, model, test_all_ds, is_sid) if test_all_ds else None
    test_acc = top_one_accuracy(config, model, test_ds, is_sid)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc if val_acc else None,
        "test_all_acc": test_all_acc if test_all_acc else None,
        "test_acc": test_acc,
    }


def evaluate_all_and_save(config_ds_dir_pairs: list[tuple[LLMConfig, Path]], save_path: Path, from_hub: bool = True):
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("run_name,train_acc,val_acc,test_all_acc,test_acc\n")

    pbar = tqdm(config_ds_dir_pairs)
    for config, ds_dir in pbar:
        df = pd.read_csv(save_path)
        if config.run_name in df["run_name"].values:
            pbar.write(f"Skipping {config.run_name} as it already evaluated")
            pbar.update(1)
            continue
        pbar.set_description(f"Evaluating {config.run_name}")
        
        metrics = evaluate_model(config, ds_dir, from_hub)
        train_acc = f"{metrics['train_acc']:.4f}"
        test_acc = f"{metrics['test_acc']:.4f}"
        val_acc = f"{metrics['val_acc']:.4f}" if metrics['val_acc'] else "-"
        test_all_acc = f"{metrics['test_all_acc']:.4f}" if metrics['test_all_acc'] else "-"
        
        with save_path.open("a") as f:
            f.write(
                f"{config.run_name},{train_acc},{val_acc},{test_all_acc},{test_acc}\n"
            )
        pbar.write(f"Evaluated {config.run_name} with metrics: train_acc={train_acc}, val_acc={val_acc}, test_all_acc={test_all_acc}, test_acc={test_acc}")
        pbar.update(1)
