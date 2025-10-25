"""
POI (Point of Interest) Recommendation System

This package provides tools for POI recommendation using RQ-VAE and LLM models.
It includes data processing, model training, and inference capabilities.
"""

from . import settings
from . import hf_utils

__version__ = "0.1.0"
__all__ = ["settings", "hf_utils"]


def main() -> None:
    print("Hello from poi!")
