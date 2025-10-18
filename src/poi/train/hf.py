import datetime
from pathlib import Path

from poi.train.config import RQVAEConfig

# 可选：Hugging Face 上传所需依赖
try:
    from huggingface_hub import HfApi, create_repo, upload_file

    HAVE_HF_HUB = True
except Exception:
    HAVE_HF_HUB = False


def generate_model_card(config: RQVAEConfig) -> str:
    """生成简易的模型卡片README内容，包含数据集和模型超参信息。"""
    lines = []
    lines.append(f"# RQ-VAE ({config.dataset_name})\n")
    lines.append(f"Generated at: {datetime.utcnow().isoformat()}Z\n")
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
    lines.append(
        f"- Codebooks: {config.codebook_num} × {config.vector_num} × {config.vector_dim}"
    )
    lines.append(f"- Commitment weight: {config.commitment_weight}")
    lines.append(f"- Loss weights: {config.loss_weights}\n")
    lines.append("## Training\n")
    lines.append(f"- Batch size: {config.batch_size}")
    lines.append(f"- Epochs: {config.epoch_num}")
    lines.append(f"- LR: {config.lr}")
    return "\n".join(lines) + "\n"


def upload_to_hf(
    repo_id: str,
    files: list[Path],
    config: RQVAEConfig,
    private: bool = True,
    token: str | None = None,
):
    """
    上传指定文件到 Hugging Face Hub。若仓库不存在则自动创建。

    Args:
        repo_id: '<username>/<repo_name>'
        files: 本地文件路径列表
        metadata_dict: 用于生成README的元信息
        private: 是否将仓库设为私有
        token: HF访问令牌
    """
    if not HAVE_HF_HUB:
        print(
            "[HF] huggingface_hub 未安装，跳过上传。可运行: pip install huggingface_hub"
        )
        return
    if not repo_id:
        print("[HF] 未设置 HF_REPO，跳过上传。")
        return
    api = HfApi()
    # 创建或复用仓库
    try:
        create_repo(repo_id, private=private, exist_ok=True, token=token)
        print(f"[HF] Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[HF] 创建/获取仓库失败: {e}")
        return
    # 上传README (model card)
    try:
        readme_content = generate_model_card(config)
        upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        print("[HF] 上传 README.md 成功")
    except Exception as e:
        print(f"[HF] 上传 README.md 失败: {e}")
    # 上传其它文件
    for f in files:
        if not f.exists():
            print(f"[HF] 文件不存在，跳过: {f}")
            continue
        try:
            upload_file(
                path_or_fileobj=f,
                path_in_repo=f.name,
                repo_id=repo_id,
                token=token,
            )
            print(f"[HF] 上传成功: {f}")
        except Exception as e:
            print(f"[HF] 上传失败 {f}: {e}")
