from typing import Literal

from huggingface_hub import snapshot_download

from ..settings import DATASETS_DIR, HF_ORG


def download_dataset(dataset_name: Literal["NYC", "TKY"]):
    snapshot_download(repo_id=f"{HF_ORG}/{dataset_name}", local_dir=DATASETS_DIR / dataset_name, repo_type="dataset")
