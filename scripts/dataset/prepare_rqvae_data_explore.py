#!/usr/bin/env python3
"""
RQ-VAE Data Preparation Script (Exploration Version)
Read data from Intermediate Files, generate feature matrices with recency-weighted user embeddings and upload to HF
"""

import os
import sys

from poi.dataset.rqvae_preparer_explore import RQVAEPreparer


def main():
    """Main function"""
    print("🚀 RQ-VAE Data Preparation Script (Exploration Version)")
    print("=" * 50)

    # Check if token is provided
    if len(sys.argv) > 1:
        token = sys.argv[1]
        os.environ["HF_TOKEN"] = token
        print("🔑 Using provided token...")
    else:
        print("⚠️  No token provided. Please provide your HF token as an argument:")
        print("   python scripts/dataset/prepare_rqvae_data_explore.py YOUR_TOKEN_HERE")
        print("   Or set HF_TOKEN environment variable")
        return

    # Get token
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_TOKEN")
    
    # Process NYC dataset
    print("\n📁 Preparing RQ-VAE data for NYC dataset (exploration)...")
    nyc_preparer = RQVAEPreparer("NYC_Exploration", hf_token=token)
    try:
        nyc_preparer.process_dataset()
        print("✅ NYC RQ-VAE data preparation completed!")
    except Exception as e:
        print(f"❌ Error preparing NYC RQ-VAE data: {e}")

    # Process TKY dataset
    print("\n📁 Preparing RQ-VAE data for TKY dataset (exploration)...")
    tky_preparer = RQVAEPreparer("TKY_Exploration", hf_token=token)
    try:
        tky_preparer.process_dataset()
        print("✅ TKY RQ-VAE data preparation completed!")
    except Exception as e:
        print(f"❌ Error preparing TKY RQ-VAE data: {e}")

    print("\n🎉 All RQ-VAE data preparation completed!")
    print("📁 Check your Hugging Face datasets:")
    print("   - https://huggingface.co/datasets/comp5331poi/NYC_Exploration/tree/main/RQVAE%20Dataset")
    print("   - https://huggingface.co/datasets/comp5331poi/TKY_Exploration/tree/main/RQVAE%20Dataset")


if __name__ == "__main__":
    main()

