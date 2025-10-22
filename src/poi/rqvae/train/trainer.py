import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard日志记录
from tqdm import tqdm

from ...dataset.rqvae import get_dataloader, get_dataloaders  # 导入自定义 DataLoader 工厂函数
from ...rqvae.model import RQVAE
from .config import RQVAEConfig
from .hf import generate_model_card, upload_to_hf

LOSS_TERMS = ["reconstruction", "quantization", "utilization", "compactness"]


def train_rqvae(config: RQVAEConfig, push_to_hub: bool = False):
    writer = SummaryWriter(log_dir=config.log_dir)  # 初始化TensorBoard日志目录

    print(f"Using device: {config.device}")
    rqvae_model = RQVAE(
        embedding_dim=config.embedding_dim,
        vae_hidden_dims=config.vae_hidden_dims,
        vector_dim=config.vector_dim,
        vector_num=config.vector_num,
        codebook_num=config.codebook_num,
        commitment_weight=config.commitment_weight,
        random_state=config.random_state,
    ).to(config.device)

    optimizer = optim.Adam(
        rqvae_model.parameters(),
        lr=config.lr,
        # betas=(0.9, 0.98),
        # eps=1e-9
    )

    # ===== 创建 DataLoader =====
    # 使用自定义 DataLoader 模块加载数据，支持 shuffle、多进程、pin_memory 等
    if getattr(config, "use_splits", False):
        loaders = get_dataloaders(
            dataset_path=config.dataset_path,
            batch_size=config.batch_size,
            num_workers=config.num_dataloader_workers,
            device=config.device,
            ratios=getattr(config, "split_ratios", (0.8, 0.1, 0.1)),
            seed=getattr(config, "split_seed", config.random_state),
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        print(f"DataLoaders ready: train={len(train_loader)}, val={len(val_loader)}, test={len(loaders['test'])}\n")
    else:
        train_loader = get_dataloader(
            dataset_path=config.dataset_path,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_dataloader_workers,
            device=config.device,
            drop_last=False,
        )
        val_loader = None
        print(f"DataLoader ready: {len(train_loader)} batches per epoch\n")

    ## Training

    rqvae_model.train()
    # ==== 断点续训：尝试加载 checkpoint ====
    start_epoch = 0
    best_loss = float("inf")
    code_indices_log = []
    if config.checkpoint_path.exists():
        print(f"Found checkpoint {config.checkpoint_path}, resuming training...")
        checkpoint = torch.load(config.checkpoint_path)
        rqvae_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        code_indices_log = checkpoint.get("code_indices_log", [])
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss}")
    else:
        print("No checkpoint found, starting from scratch.")

    # ==== 正常训练循环 ====
    with tqdm(total=config.epoch_num, desc="Training") as epoch_pbar:
        epoch_pbar.update(start_epoch)  # resume progress bar
        for epoch in tqdm(range(start_epoch, config.epoch_num), desc="Training"):
            total_loss_dict = {k: 0.0 for k in LOSS_TERMS}

            batch_loss = 0.0

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch}", leave=False) as pbar:
                for step, batch in enumerate(train_loader):
                    if epoch == 0 and step == 0:
                        # Initialize quantizer via k-means with the first batch only
                        rqvae_model.initialize(batch.to(config.device, non_blocking=True))
                        continue

                    x = batch.to(config.device, non_blocking=True)
                    optimizer.zero_grad()
                    quantized, step_loss_dict, all_indices = rqvae_model(x)

                    # Compute weighted loss according to config
                    loss = sum(
                        config.loss_weights[k]
                        * step_loss_dict.get(k, torch.tensor(0.0, device=config.device))
                        for k in LOSS_TERMS
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(rqvae_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    # 累加到 total_loss_dict
                    for k in LOSS_TERMS:
                        total_loss_dict[k] += float(step_loss_dict.get(k, torch.tensor(0.0)).detach())
                    if step % 50 == 0:  # 每50个 batch 打印一次
                        batch_loss = sum(
                            step_loss_dict.get(k, torch.tensor(0.0, device=config.device)) * config.loss_weights[k]
                            for k in LOSS_TERMS
                        )
                        pbar.update(50)
                        pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

                    # 只记录最后一批的 code indices 作为示例
                    if step == len(train_loader) - 1:
                        code_indices_log.append([inds.detach().cpu().numpy().tolist() for inds in all_indices])
            total_loss = sum([total_loss_dict[k] * config.loss_weights[k] for k in LOSS_TERMS])

            # Optional: validation
            val_total_loss = None
            if 'val_loader' in locals() and val_loader is not None:
                rqvae_model.eval()
                with torch.no_grad():
                    val_sum = {k: 0.0 for k in LOSS_TERMS}
                    for vbatch in val_loader:
                        vx = vbatch.to(config.device, non_blocking=True)
                        _, vloss_dict, _ = rqvae_model(vx)
                        for k in LOSS_TERMS:
                            val_sum[k] += float(vloss_dict.get(k, torch.tensor(0.0)).detach())
                    val_total_loss = sum(val_sum[k] * config.loss_weights[k] for k in LOSS_TERMS)
                rqvae_model.train()
            epoch_pbar.update(1)
            epoch_pbar.set_postfix(
                {
                    "total_loss": round(float(total_loss), 4),
                    **{k: round(float(total_loss_dict[k]), 4) for k in LOSS_TERMS},
                }
            )
            # TensorBoard 日志记录
            writer.add_scalar("Loss/total", total_loss, epoch)
            for k in LOSS_TERMS:
                writer.add_scalar(f"Loss/{k}", total_loss_dict[k], epoch)
            if val_total_loss is not None:
                writer.add_scalar("ValLoss/total", val_total_loss, epoch)
                for k in LOSS_TERMS:
                    writer.add_scalar(f"ValLoss/{k}", val_sum[k], epoch)
            # 保存最优模型
            # Select metric to track best
            metric_for_best = float(val_total_loss) if val_total_loss is not None else float(total_loss)
            if metric_for_best < best_loss:
                best_loss = metric_for_best
                torch.save(
                    {
                        "model_state_dict": rqvae_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": float(metric_for_best),
                    },
                    config.checkpoint_best_path,
                )
                which = "val" if val_total_loss is not None else "train"
                epoch_pbar.write(f"Saved new best model at epoch {epoch} with {which} loss {metric_for_best:.4f}")
            # 每个epoch都保存断点，便于中断后恢复
            torch.save(
                {
                    "model_state_dict": rqvae_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "code_indices_log": code_indices_log,
                },
                config.checkpoint_path,
            )
    # 保存 codebook indices log
    torch.save(code_indices_log, config.code_indices_log_path)
    print(f"Training finished. Artifacts: {config.checkpoint_best_path}, {config.code_indices_log_path}")
    writer.close()  # 关闭TensorBoard日志

    model_card = generate_model_card(config)
    config.model_card_path.write_text(model_card)

    if push_to_hub:
        upload_to_hf(config)
