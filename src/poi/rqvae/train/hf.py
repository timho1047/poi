from datetime import UTC, datetime

from huggingface_hub import create_repo, upload_file

from ... import settings
from .config import RQVAEConfig


def generate_model_card(config: RQVAEConfig) -> bytes:
    """生成简易的模型卡片README内容，包含数据集和模型超参信息。"""
    lines = []
    lines.append("---\ntags:\n- poi\n- rqvae\n---\n")
    lines.append(f"# RQ-VAE ({config.dataset_name})\n")
    lines.append(f"Generated at: {datetime.now(UTC).isoformat()}Z\n")
    lines.append("## Dataset\n")
    lines.append(f"- Name: {config.dataset_name}")
    if config.metadata:
        lines.append(f"- POIs: {config.metadata.get('num_pois', 'N/A')}")
        lines.append(f"- Input dim: {config.metadata.get('total_dim', 'N/A')}")
        fd = config.metadata.get("feature_dims", {})
        if fd:
            lines.append(f"- Feature dims: {fd}")
    lines.append("\n## Model\n")
    lines.append(f"- VAE hidden dims: {config.vae_hidden_dims}")
    lines.append(f"- Codebooks: {config.codebook_num} x {config.vector_num} x {config.vector_dim}")
    lines.append(f"- Commitment weight: {config.commitment_weight}")
    lines.append(f"- Loss weights: {config.loss_weights}\n")
    lines.append("## Training\n")
    lines.append(f"- Batch size: {config.batch_size}")
    lines.append(f"- Epochs: {config.epoch_num}")
    lines.append(f"- LR: {config.lr}")
    
    content = "\n".join(lines) + "\n"
    return content.encode("utf-8")


def upload_to_hf(
    config: RQVAEConfig,
):
    # 创建或复用仓库
    try:
        create_repo(config.hub_id, private=False, exist_ok=True, token=settings.HF_TOKEN)
        print(f"[HF] Repo ready: https://huggingface.co/{config.hub_id}")
    except Exception as e:
        print(f"[HF] 创建/获取仓库失败: {e}")
        return

    for f in [config.checkpoint_best_path, config.code_indices_log_path, config.checkpoint_path, config.model_card_path]:
        upload_file(
            path_or_fileobj=f,
            path_in_repo=f.name,
            repo_id=config.hub_id,
            token=settings.HF_TOKEN,
        )
