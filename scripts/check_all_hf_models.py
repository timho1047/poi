#!/usr/bin/env python3
"""
Check All Hugging Face Models

This script provides a simple interface for checking and exploring
models stored in the Hugging Face organization.

Usage:
    HF_TOKEN="Your HF Token" uv run scripts/check_all_hf_models.py
    HF_TOKEN="Your HF Token" uv run scripts/check_all_hf_models.py --model rqvae-nyc-div0.1-commit0.5-lr5e-5
    HF_TOKEN=“Your HF Token” uv run scripts/check_all_hf_models.py --filter rqvae
    HF_TOKEN=“Your HF Token” uv run scripts/check_all_hf_models.py --filter llm
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poi.hf_utils import run_model_checker


def main():
    """Main function to handle command line arguments and execute the model checker."""
    parser = argparse.ArgumentParser(
        description="Check and explore Hugging Face models in the organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # List all models
  %(prog)s --model rqvae-nyc-div0.1-commit0.5-lr5e-5  # Check specific model
  %(prog)s --filter rqvae                     # Filter RQ-VAE models
  %(prog)s --filter llm                       # Filter LLM models
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="Check details of a specific model (full model ID)"
    )
    parser.add_argument(
        "--filter", 
        choices=["rqvae", "llm"], 
        help="Filter models by type"
    )
    parser.add_argument(
        "--summary", 
        action="store_true", 
        help="Show only summary information"
    )
    
    args = parser.parse_args()
    
    # Call the main function from the library
    run_model_checker(
        model_id=args.model,
        filter_type=args.filter,
        summary_only=args.summary
    )


if __name__ == "__main__":
    main()
