import json
import importlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ... import settings


@dataclass
class RQVAEConfig:
    # Training parameters
    dataset_name: str = "TKY"
    batch_size: int = 128
    epoch_num: int = 3000
    diversity_start_epoch: int = 1000
    lr: float = 1e-3
    run_name: str = "rqvae-1"

    num_dataloader_workers: int = 4  # 数据加载并行进程数，可根据 CPU 核数调整（2~8）
    device: Literal["cpu", "cuda", "mps"] = settings.DEVICE

    # Model parameters
    codebook_num: int = 3
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
        self._ensure_local_dataset()

        metadata_path = self.dataset_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found for dataset {self.dataset_name}. "
                "Please ensure the dataset is prepared locally or available on Hugging Face."
            )

        self.metadata = json.loads(metadata_path.read_text())
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
        # - NYC variants: 3 layers × 32 codewords × 64 dims
        # - TKY/GWL variants: 3 layers × 64 codewords × 64 dims
        dataset_name_lower = self.dataset_name.lower()
        if "nyc" in dataset_name_lower:
            self.vector_num = 32
        else:
            self.vector_num = 64
        
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

    def _ensure_local_dataset(self) -> None:
        """
        Ensure required dataset artifacts are available locally.
        If they are missing, download them from the Hugging Face dataset repository.
        """
        dataset_root = settings.DATASETS_DIR / self.dataset_name
        rqvae_dir = dataset_root / "RQVAE Dataset"
        intermediate_dir = dataset_root / "Intermediate Files"

        required_files = {
            rqvae_dir / "metadata.json": "RQVAE Dataset/metadata.json",
            rqvae_dir / "poi_features.pt": "RQVAE Dataset/poi_features.pt",
            intermediate_dir / "pid_mapping.csv": "Intermediate Files/pid_mapping.csv",
        }

        missing_files = [dest for dest in required_files if not dest.exists()]
        if not missing_files:
            return

        if not settings.HF_TOKEN:
            missing = ", ".join(str(path) for path in missing_files)
            raise RuntimeError(
                f"Required dataset files are missing ({missing}) and HF_TOKEN is not set "
                "to download them automatically."
            )

        try:
            snapshot_download = getattr(importlib.import_module("huggingface_hub"), "snapshot_download")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download datasets automatically. "
                "Please install it or ensure dataset files exist locally."
            ) from exc

        snapshot_dir = Path(
            snapshot_download(
                repo_id=f"{settings.HF_ORG}/{self.dataset_name.lower()}",
                repo_type="dataset",
                token=settings.HF_TOKEN,
                allow_patterns=list(required_files.values()),
            )
        )

        for dest_path, rel_path in required_files.items():
            src_path = snapshot_dir / rel_path
            if not src_path.exists():
                raise FileNotFoundError(
                    f"Expected file '{rel_path}' not found in Hugging Face dataset "
                    f"{settings.HF_ORG}/{self.dataset_name}."
                )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
