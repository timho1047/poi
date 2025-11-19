#!/usr/bin/env python3
"""
LLM Dataset Generation Script

Downloads codebook files and data files from Hugging Face, generates JSON datasets 
for LLM training, and uploads them to Hugging Face LLM Dataset directories.

USAGE:
1. Set environment variable:
   export HF_TOKEN="your_huggingface_token_here"

2. Run the script:
   uv run scripts/dataset/generate_llm_datasets.py

3. Or run directly with token:
   HF_TOKEN="your_token" uv run scripts/dataset/generate_llm_datasets.py

FUNCTIONALITY:
- Downloads 10 codebook CSV files from HF
- Downloads NYC and TKY data folders from HF
- Generates all required JSON format LLM training data
- Uploads to comp5331poi/NYC/LLM Dataset/ and comp5331poi/TKY/LLM Dataset/

OUTPUT STRUCTURE:
For each dataset (NYC/TKY):
├── LLM Dataset/
│   ├── ablation without SID/
│   │   ├── train_id.json
│   │   ├── val_id.json
│   │   ├── test_id.json
│   │   └── test_all_id.json
│   ├── ablation without Time/
│   │   ├── train_codebook_notime.json
│   │   ├── val_codebook_notime.json
│   │   ├── test_codebook_notime.json
│   │   └── test_all_codebook_notime.json
│   └── rqvae-{model_name}/
│       ├── train_codebook.json
│       ├── val_codebook.json
│       ├── test_codebook.json
│       └── test_all_codebook.json

TOTAL FILES GENERATED:
- ID files: 2 datasets × 4 files = 8 files
- Codebook files: 10 models × 4 files = 40 files  
- Notime files: 2 models × 4 files = 8 files
- Total: 56 files

NOTES:
- ID files are dataset-level (same for all models)
- Notime files only generated for specific models:
  - rqvae-nyc-div0.25-commit0.5-lr5e-5
  - rqvae-tky-div0.25-commit0.25-lr5e-5
- All files include retry mechanism for network issues
- Requires stable internet connection for HF uploads
"""

import os
from poi.dataset.llm_dataset_generator import LLMDatasetGenerator
from poi.settings import HF_TOKEN


def main():
    """Main function - simple wrapper around LLMDatasetGenerator."""
    # Check HF token
    if not HF_TOKEN:
        print("❌ HF_TOKEN not set! Please set your HF_TOKEN environment variable.")
        return
    
    # Create generator and run
    generator = LLMDatasetGenerator(HF_TOKEN)
    generator.generate_all_datasets()


if __name__ == "__main__":
    main()