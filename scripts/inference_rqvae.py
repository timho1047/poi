#!/usr/bin/env python3
"""
RQ-VAE Batch Inference Script

Performs batch inference on pre-trained RQ-VAE models, encodes POI feature vectors 
into codebook indices (SID), and uploads results to Hugging Face Datasets.

USAGE:
1. Set environment variable:
   export HF_TOKEN="your_huggingface_token_here"

2. Run the script:
   uv run scripts/inference_rqvae.py

3. Or run directly with token:
   HF_TOKEN="your_token" uv run scripts/inference_rqvae.py

FUNCTIONALITY:
- Performs batch inference on 10 pre-trained RQ-VAE models
- Encodes POI feature vectors into codebook indices (SID)
- Generates CSV files containing: Pid, Codebook, Vector
- Uploads results to Hugging Face Datasets

OUTPUT LOCATIONS:
- NYC model results: comp5331poi/nyc/LLM Dataset/Intermediate Files/
- TKY model results: comp5331poi/tky/LLM Dataset/Intermediate Files/

FILE NAMING FORMAT:
- codebooks-rqvae-{dataset}-div{div_weight}-commit{commit_weight}-lr{lr}.csv
- Example: codebooks-rqvae-tky-div0.0-commit0.25-lr5e-5.csv

VERIFICATION METHODS:
1. Check if generated CSV file format is correct
2. Verify PID to codebook correspondence is consistent
3. Compare results from different runs to ensure consistency
4. Check if uploaded files to HF are complete

NOTES:
- Ensure stable network connection for downloading model weights
- Ensure sufficient disk space for temporary files
- Inference process may take several minutes to hours depending on data size
- Uses PID mapping to ensure correct POI-to-codebook correspondence
- Disables data shuffling to maintain consistent order
"""

import os
from poi.rqvae.batch_inference import BatchInferenceManager
from poi.settings import HF_TOKEN


def main():
    """Main function - simple wrapper around BatchInferenceManager."""
    # Check HF token
    if not HF_TOKEN:
        print("❌ HF_TOKEN not set! Please set your HF_TOKEN environment variable.")
        return
    
    # Create manager and process all models
    manager = BatchInferenceManager(HF_TOKEN)
    manager.process_models()


if __name__ == "__main__":
    main()