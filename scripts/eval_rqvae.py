import argparse
import json
from pathlib import Path

import torch
from poi.rqvae import RQVAEConfig
from poi.rqvae.inference import load_inference_model
from poi.dataset.rqvae import get_dataloaders

LOSS_TERMS = ["reconstruction", "quantization", "utilization", "compactness"]


def evaluate_on_loader(model, loader, device, loss_weights):
    model.eval()
    total = {k: 0.0 for k in LOSS_TERMS}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            _, step_loss, _ = model(x)
            for k in LOSS_TERMS:
                total[k] += float(step_loss.get(k, torch.tensor(0.0, device=device)).detach())
            n_batches += 1
    # 均值（按 batch 计）；如需样本加权均值，可在 dataset 返回样本数时做更精细统计
    avg = {k: (v / max(1, n_batches)) for k, v in total.items()}
    total_loss = sum(avg[k] * loss_weights[k] for k in LOSS_TERMS)
    return avg, float(total_loss)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RQ-VAE on test split")
    parser.add_argument("--dataset", dest="dataset_name", type=str, default="GWL", choices=["NYC", "TKY", "GWL"], help="Dataset name")
    parser.add_argument("--run", dest="run_name", type=str, default="rqvae-4", help="Run name for checkpoints/logs")
    parser.add_argument("--from-hub", dest="from_hub", action="store_true", help="Load best checkpoint from Hugging Face Hub")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Override batch size for evaluation")
    parser.add_argument("--workers", dest="num_workers", type=int, default=None, help="Override num_dataloader_workers")
    parser.add_argument("--seed", dest="split_seed", type=int, default=None, help="Override split seed")
    args = parser.parse_args()

    cfg = RQVAEConfig(run_name=args.run_name, dataset_name=args.dataset_name)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_dataloader_workers = args.num_workers
    if args.split_seed is not None:
        cfg.split_seed = args.split_seed
    cfg.use_splits = True  # ensure we use persisted splits

    loaders = get_dataloaders(
        dataset_path=cfg.dataset_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dataloader_workers,
        device=cfg.device,
        ratios=cfg.split_ratios,
        seed=cfg.split_seed,
    )
    test_loader = loaders["test"]

    model = load_inference_model(cfg, from_hub=args.from_hub)

    avg_losses, total_loss = evaluate_on_loader(model, test_loader, cfg.device, cfg.loss_weights)
    result = {
        "dataset": cfg.dataset_name,
        "run_name": cfg.run_name,
        "device": cfg.device,
        "losses": avg_losses,
        "total_loss": total_loss,
        "loss_weights": cfg.loss_weights,
        "num_batches": len(test_loader),
    }

    out_dir = cfg.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_test.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"Saved evaluation to {out_path}")


if __name__ == "__main__":
    main()
