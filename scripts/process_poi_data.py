#!/usr/bin/env python3
"""
POI Data Processing Script
Download raw data from Hugging Face, process it, and upload to Intermediate Files folder
"""

import os
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from poi.dataset.poi_processor import POIProcessor


def main():
    """Main function"""
    print("🚀 POI Data Processing Script")
    print("=" * 50)
    
    # Check if token is provided
    if len(sys.argv) > 1:
        token = sys.argv[1]
        os.environ['HF_TOKEN'] = token
        print("🔑 Using provided token...")
    else:
        print("⚠️  No token provided. Please provide your HF token as an argument:")
        print("   python scripts/process_poi_data.py YOUR_TOKEN_HERE")
        print("   Or set HF_TOKEN environment variable")
        return
    
    # Process NYC dataset
    print("\n📁 Processing NYC dataset...")
    nyc_processor = POIProcessor("NYC")
    try:
        nyc_processor.process_dataset(
            poi_min_freq=10,
            user_min_freq=10,
            window_size=50,
            step_size=10,
            mask_prob=0.1
        )
        print("✅ NYC dataset processing completed!")
    except Exception as e:
        print(f"❌ Error processing NYC dataset: {e}")
    
    # Process TKY dataset
    print("\n📁 Processing TKY dataset...")
    tky_processor = POIProcessor("TKY")
    try:
        tky_processor.process_dataset(
            poi_min_freq=10,
            user_min_freq=10,
            window_size=50,
            step_size=10,
            mask_prob=0.1
        )
        print("✅ TKY dataset processing completed!")
    except Exception as e:
        print(f"❌ Error processing TKY dataset: {e}")
    
    print("\n🎉 All datasets processed successfully!")
    print("📁 Check your Hugging Face datasets:")
    print("   - https://huggingface.co/datasets/comp5331poi/NYC/tree/main/Intermediate%20Files")
    print("   - https://huggingface.co/datasets/comp5331poi/TKY/tree/main/Intermediate%20Files")


if __name__ == "__main__":
    main()
