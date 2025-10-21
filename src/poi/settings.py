import os
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv


# 选择计算设备：优先 CUDA，其次 MPS（苹果），否则 CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ROOT_DIR = Path(__file__).parent.parent.parent

load_dotenv(ROOT_DIR / ".env")


DATASETS_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

DEVICE: Literal["cpu", "cuda", "mps"] = os.getenv("DEVICE", get_device())
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 43))

# ===== Hugging Face 上传辅助 =====
# 通过环境变量进行配置：
# - HF_REPO: 目标仓库，格式为 "<username>/<repo_name>"
# - HF_TOKEN: 访问令牌（建议设置为只读或写入token）
HF_ORG = "comp5331poi"
HF_REPO = os.getenv("HF_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_PRIVATE = False


# Initialize directories
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
