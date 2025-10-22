import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ... import settings


@dataclass
class RQVAEConfig:
    # Training parameters
    dataset_name: Literal["NYC", "TKY", "GWL"] = "NYC"
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
    vae_hidden_dims: list[int] = field(default_factory=lambda: [128, 256, 512])

    quant_weight: float = 1.0
    div_weight: float = 0.25
    commitment_weight: float = 0.5
    random_state: int = settings.RANDOM_STATE

    # Split parameters
    use_splits: bool = True
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = settings.RANDOM_STATE

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
    embedding_dim: int = field(init=False)
    loss_weights: dict[str, float] = field(init=False)

    def __post_init__(self):
        self.dataset_path = settings.DATASETS_DIR / self.dataset_name
        self.metadata = json.loads((self.dataset_path / "metadata.json").read_text())
        self.embedding_dim = self.metadata["total_dim"]
        self._refresh_paths()

        # According to the paper:
        # - NYC: 3 layers × 32 codewords × 64 dims
        # - TKY/GWL: 3 layers × 64 codewords × 64 dims
        self.vector_num = 32 if self.dataset_name == "NYC" else 64
        self.loss_weights = {
            "reconstruction": 1.0,
            "quantization": self.quant_weight,
            "utilization": self.div_weight,
            "compactness": self.div_weight,
        }

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _refresh_paths(self):
        """Refresh paths that depend on run_name and ensure hub id/model card path are consistent."""
        self.log_dir = settings.OUTPUT_DIR / "logs" / "rqvae" / self.run_name
        self.checkpoint_dir = settings.OUTPUT_DIR / "checkpoints" / "rqvae" / self.run_name
        self.checkpoint_path = self.checkpoint_dir / "rqvae_checkpoint.pt"
        self.checkpoint_best_path = self.checkpoint_dir / "rqvae_best.pt"
        self.code_indices_log_path = self.checkpoint_dir / "code_indices_log.pt"
        self.hub_id = f"{settings.HF_ORG}/{self.run_name}"
        self.model_card_path = self.checkpoint_dir / "README.md"

    def set_run_name(self, new_name: str):
        """Set a new run name and update all derived paths accordingly."""
        self.run_name = new_name
        self._refresh_paths()
        # Ensure directories exist for the new name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fmt_float(v: float) -> str:
        # compact float for names: 0.00001 -> 1e-05, 0.5 -> 0.5
        s = f"{v}"
        # prefer scientific for very small numbers
        if v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e3):
            s = f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
        return s

    def generate_run_name(self, include_defaults: bool = False) -> str:
        """Generate a concise, meaningful run name encoding key config changes.

        Only parameters that differ from sensible defaults are included, unless include_defaults=True.
        Defaults used:
          - dataset_name
          - codebook_num=3
          - vector_num: NYC=32 else 64
          - vector_dim=64
          - vae_hidden_dims=[128,256,512]
          - lr=1e-5, batch_size=128, epoch_num=50
          - div_weight=0.25, commitment_weight=0.5, quant_weight=1.0
          - split_ratios=(0.8,0.1,0.1), split_seed=settings.RANDOM_STATE
        """
        parts: list[str] = ["rqvae", self.dataset_name.lower()]

        # Model capacity
        if include_defaults or self.codebook_num != 3:
            parts.append(f"cb{self.codebook_num}")

        default_vn = 32 if self.dataset_name == "NYC" else 64
        if include_defaults or self.vector_num != default_vn:
            parts.append(f"vn{self.vector_num}")

        if include_defaults or self.vector_dim != 64:
            parts.append(f"vd{self.vector_dim}")

        default_h = [128, 256, 512]
        if include_defaults or list(self.vae_hidden_dims) != default_h:
            hstr = "-".join(str(h) for h in self.vae_hidden_dims)
            parts.append(f"h{hstr}")

        # Optimization
        if include_defaults or self.lr != 1e-5:
            parts.append(f"lr{self._fmt_float(self.lr)}")
        if include_defaults or self.batch_size != 128:
            parts.append(f"bs{self.batch_size}")

        # Loss weights
        if include_defaults or self.div_weight != 0.25:
            parts.append(f"div{self._fmt_float(self.div_weight)}")
        if include_defaults or self.commitment_weight != 0.5:
            parts.append(f"commit{self._fmt_float(self.commitment_weight)}")
        if include_defaults or self.quant_weight != 1.0:
            parts.append(f"q{self._fmt_float(self.quant_weight)}")

        # Training schedule / data
        if include_defaults or self.epoch_num != 50:
            parts.append(f"ep{self.epoch_num}")
        if include_defaults or self.split_seed != settings.RANDOM_STATE:
            parts.append(f"s{self.split_seed}")
        if include_defaults or tuple(self.split_ratios) != (0.8, 0.1, 0.1):
            r = "-".join(str(int(x * 10)) for x in self.split_ratios)  # 8-1-1
            parts.append(f"sr{r}")

        return "-".join(parts)
