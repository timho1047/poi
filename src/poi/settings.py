import os
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv


# 选择计算设备：优先 CUDA，否则 CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


ROOT_DIR = Path(__file__).parent.parent.parent

load_dotenv(ROOT_DIR / ".env")


DATASETS_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

DEVICE: Literal["cpu", "cuda", "mps"] = os.getenv("DEVICE", get_device())
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 2024))

# ===== Hugging Face =====
HF_ORG = "comp5331poi"
HF_TOKEN = os.getenv("HF_TOKEN")


# Initialize directories
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
