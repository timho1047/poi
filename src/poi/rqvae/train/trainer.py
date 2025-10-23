import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard日志记录
from tqdm import tqdm

from ...dataset.rqvae import get_dataloader  # 导入自定义 DataLoader 工厂函数
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
    train_loader = get_dataloader(
        dataset_path=config.dataset_path,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_dataloader_workers,
        device=config.device,
        drop_last=False,
    )
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

                    # loss = sum([step_loss_dict[k]*LOSS_WEIGHTS[k] for k in LOSS_TERMS])
                    recon = step_loss_dict["reconstruction"]
                    quant = step_loss_dict["quantization"]
                    div = step_loss_dict.get("utilization", 0.0)  # or 'diversity' if in model
                    loss = recon + 1.0 * quant + 0.25 * div
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(rqvae_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    # 累加到 total_loss_dict
                    for k in LOSS_TERMS:
                        total_loss_dict[k] += float(step_loss_dict.get(k, torch.tensor(0.0)).detach())
                    if step % 50 == 0:  # 每50个 batch 打印一次
                        batch_loss = sum([step_loss_dict[k] * config.loss_weights[k] for k in LOSS_TERMS])
                        pbar.update(50)
                        pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

                    # 只记录最后一批的 code indices 作为示例
                    if step == len(train_loader) - 1:
                        code_indices_log.append([inds.detach().cpu().numpy().tolist() for inds in all_indices])
            total_loss = sum([total_loss_dict[k] * config.loss_weights[k] for k in LOSS_TERMS])
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
            # 保存最优模型
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(
                    {
                        "model_state_dict": rqvae_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": float(best_loss),
                    },
                    config.checkpoint_best_path,
                )
                epoch_pbar.write(f"Saved new best model at epoch {epoch} with loss {best_loss:.4f}")
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
    config.model_card_path.write_bytes(model_card)

    if push_to_hub:
        upload_to_hf(config)
