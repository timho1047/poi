import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ... import settings


@dataclass
class RQVAEConfig:
    # Training parameters
    dataset_name: Literal["NYC", "TKY"] = "TKY"
    batch_size: int = 128
    epoch_num: int = 3000
    diversity_start_epoch: int = 1000
    lr: float = 1e-3
    run_name: str = "rqvae-1"

    num_dataloader_workers: int = 4  # 数据加载并行进程数，可根据 CPU 核数调整（2~8）
    device: Literal["cpu", "cuda", "mps"] = settings.DEVICE

    # Model parameters
    codebook_num: int = 3
    vector_num: int = 64
    vector_dim: int = 64
    vae_hidden_dims: list[int] = field(default_factory=lambda: [128, 256, 512])

    quant_weight: float = 1.0
    div_weight: float = 0.25
    commitment_weight: float = 0.25
    recon_weight: float = 1.0
    use_kl_divergence: bool = False
    random_state: int = settings.RANDOM_STATE
    dropout_rate: float = 0.1

    # Inferred configs, no need to provide during initialization
    dataset_path: Path = field(init=False)
    hub_id: str = field(init=False)
    model_card_path: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    checkpoint_best_path: Path = field(init=False)
    code_indices_log_path: Path = field(init=False)
    log_dir: Path = field(init=False)
    metadata: dict[str, Any] = field(init=False)
    vector_num: int = field(init=False)
    embedding_dim: int = field(init=False)
    loss_weights: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.dataset_path = settings.DATASETS_DIR / self.dataset_name / "RQVAE Dataset"
        self.metadata = json.loads((self.dataset_path / "metadata.json").read_text())
        self.embedding_dim = self.metadata["total_dim"]
        self.log_dir = settings.OUTPUT_DIR / "logs" / "rqvae" / self.run_name
        self.checkpoint_dir = (
            settings.OUTPUT_DIR / "checkpoints" / "rqvae" / self.run_name
        )
        self.checkpoint_path = self.checkpoint_dir / "rqvae_checkpoint.pt"
        self.checkpoint_best_path = self.checkpoint_dir / "rqvae_best.pt"
        self.code_indices_log_path = self.checkpoint_dir / "code_indices_log.pt"
        self.hub_id = f"{settings.HF_ORG}/{self.run_name}"
        self.model_card_path = self.checkpoint_dir / "README.md"

        # According to the paper:
        # - NYC: 3 layers × 32 codewords × 64 dims
        # - TKY/GWL: 3 layers × 64 codewords × 64 dims
        self.vector_num = 32 if self.dataset_name == "NYC" else 64
        
        self.loss_weights = {
            "reconstruction": self.recon_weight,
            "quantization": self.quant_weight
        }
        if not self.use_kl_divergence:
            self.loss_weights["utilization"] = self.div_weight
            self.loss_weights["compactness"] = self.div_weight
        else:
            self.loss_weights["kl_divergence"] = self.div_weight
        self.loss_terms = list(self.loss_weights.keys())         
        self.diversity_terms = ["utilization", "compactness", "kl_divergence"]

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
