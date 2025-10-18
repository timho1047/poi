import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .. import settings


@dataclass
class RQVAEConfig:
    # Training parameters
    dataset_name: Literal["NYC", "TKY", "GWL"] = "GWL"
    batch_size: int = 128
    epoch_num: int = 50
    lr: float = 1e-5
    run_name: str = "rqvae-1"
    
    num_dataloader_workers: int = 4  # 数据加载并行进程数，可根据 CPU 核数调整（2~8）
    device: Literal["cpu", "cuda", "mps"] = settings.DEVICE

    # Model parameters
    codebook_num: int = 3
    vector_num: int = 64
    vector_dim: int = 64
    vae_hidden_dims: list[int] = field(default_factory=lambda: [128, 512, 1024])

    quant_weight: float = 1.0
    div_weight: float = 0.25
    commitment_weight: float = 0.5
    random_state: int = settings.RANDOM_STATE

    # Inferred configs, no need to provide during initialization
    dataset_path: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    checkpoint_best_path: Path = field(init=False)
    code_indices_log_path: Path = field(init=False)
    log_dir: Path = field(init=False)
    metadata: dict[str, Any] = field(init=False)
    vector_num: int = field(init=False)
    embedding_dim: int = field(init=False)

    def __post_init__(self):
        self.dataset_path = settings.DATASET_DIR / self.dataset_name
        self.metadata = json.loads((self.dataset_path / "metadata.json").read_text())
        self.embedding_dim = self.metadata["total_dim"]
        self.log_dir = settings.OUTPUT_DIR / "runs" / "rqvae" / self.run_name
        self.checkpoint_dir = settings.OUTPUT_DIR / "checkpoint" / "rqvae" / self.run_name
        self.checkpoint_path = self.checkpoint_dir / "rqvae_checkpoint.pt"
        self.checkpoint_best_path = self.checkpoint_dir / "rqvae_best.pt"
        self.code_indices_log_path = self.checkpoint_dir / "code_indices_log.pt"

        # According to the paper:
        # - NYC: 3 layers × 32 codewords × 64 dims
        # - TKY/GWL: 3 layers × 64 codewords × 64 dims
        self.vector_num = 32 if self.dataset_name == "NYC" else 64
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
