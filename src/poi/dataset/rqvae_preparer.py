"""
RQ-VAE Data Preparation Module
Read data from Intermediate Files and generate feature matrices required for RQ-VAE training
"""

import json
import os
import shutil
from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, upload_file


class RQVAEPreparer:
    """RQ-VAE Data Preparer"""

    def __init__(self, dataset_name: str, hf_org: str = "comp5331poi"):
        self.dataset_name = dataset_name
        self.hf_org = hf_org
        self.hf_repo = f"{hf_org}/{dataset_name}"

    def download_from_hf(self, filename: str, local_path: str) -> str:
        """Download file from Hugging Face"""
        try:
            downloaded_path = hf_hub_download(repo_id=self.hf_repo, filename=filename, repo_type="dataset")
            return downloaded_path
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            raise

    def upload_to_hf(self, local_path: str, hf_path: str) -> None:
        """Upload file to Hugging Face"""
        try:
            upload_file(path_or_fileobj=local_path, path_in_repo=hf_path, repo_id=self.hf_repo, repo_type="dataset")
            print(f"✅ Uploaded {local_path} to {hf_path}")
        except Exception as e:
            print(f"❌ Error uploading {hf_path}: {e}")
            raise

    def parse_list_column(self, value) -> list:
        """Safely parse list string"""
        import ast

        if pd.isna(value) or value == "[]":
            return []
        try:
            return ast.literal_eval(value)
        except:
            return []

    def create_onehot(self, index: int, total_dim: int) -> np.ndarray:
        """Create one-hot vector"""
        vec = np.zeros(total_dim, dtype=np.float32)
        if 0 < index <= total_dim:  # Assume index starts from 1
            vec[index - 1] = 1.0
        return vec

    def create_multihot(self, indices: list, total_dim: int, top_k: int = 10) -> np.ndarray:
        """Create multi-hot vector (select top-k items)"""
        vec = np.zeros(total_dim, dtype=np.float32)
        if not indices:
            return vec

        # Count frequencies
        counter = Counter(indices)
        # Take top-k most common items
        top_items = counter.most_common(top_k)

        for item, _ in top_items:
            if 0 < item <= total_dim:  # Assume index starts from 1
                vec[item - 1] = 1.0

        return vec

    def prepare_rqvae_input(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Prepare input data required for RQ-VAE training

        Input format: pe = concat(c, r, t, cu)
        - c (category): one-hot, dimension is the number of categories in the dataset
        - r (region): one-hot, dimension is the number of regions in the dataset
        - t (temporal): 10-hot, dimension 24 (24 hours)
        - cu (collaborative/users): 10-hot, dimension is the number of users in the dataset
        """

        print(f"🔄 Preparing RQ-VAE input for {self.dataset_name}...")

        # Download data from HF
        poi_info_path = self.download_from_hf("Intermediate Files/poi_info.csv", f"/tmp/{self.dataset_name}_poi_info.csv")
        data_path = self.download_from_hf("Intermediate Files/data.csv", f"/tmp/{self.dataset_name}_data.csv")

        # Read data
        poi_info_df = pd.read_csv(poi_info_path)
        data_df = pd.read_csv(data_path)

        # Get dimensions
        num_categories = data_df["Catname"].nunique()
        num_regions = data_df["Region"].nunique()
        num_users = data_df["Uid"].nunique()
        num_pois = len(poi_info_df)

        print("\n📊 Dataset Statistics:")
        print(f"  POIs: {num_pois}")
        print(f"  Categories: {num_categories}")
        print(f"  Regions: {num_regions}")
        print(f"  Users: {num_users}")
        print("  Time bins: 24 (hours)")

        # Calculate total dimension
        total_dim = num_categories + num_regions + 24 + num_users
        print(f"\n🔢 Total feature dimension: {total_dim}")
        print(f"  = {num_categories}(cat) + {num_regions}(reg) + 24(time) + {num_users}(user)")

        # Initialize feature matrix
        poi_features = np.zeros((num_pois, total_dim), dtype=np.float32)

        print("\n🔄 Processing POI features...")
        for idx, row in poi_info_df.iterrows():
            if idx % 500 == 0:
                print(f"  Processing {idx}/{num_pois}...")

            pid = row["Pid"]
            category = row["Catname"]
            region = row["Region"]
            time_list = self.parse_list_column(row["Time"])
            user_list = self.parse_list_column(row["Uid"])

            # Build feature vector
            # 1. Category one-hot
            cat_vec = self.create_onehot(category, num_categories)

            # 2. Region one-hot
            reg_vec = self.create_onehot(region, num_regions)

            # 3. Temporal 10-hot (24 hours)
            time_vec = self.create_multihot(time_list, 24, top_k=10)

            # 4. Collaborative (user) 10-hot
            user_vec = self.create_multihot(user_list, num_users, top_k=10)

            # Concatenate all features
            poi_feature = np.concatenate([cat_vec, reg_vec, time_vec, user_vec])

            # Use pid-1 as index (assume pid starts from 1)
            poi_features[pid - 1] = poi_feature

        print(f"\n✅ Feature matrix shape: {poi_features.shape}")

        # Create POI ID to index mapping
        pid_to_idx = {row["Pid"]: idx for idx, row in poi_info_df.iterrows()}

        # Create metadata
        metadata = {
            "num_pois": num_pois,
            "num_categories": num_categories,
            "num_regions": num_regions,
            "num_users": num_users,
            "num_time_bins": 24,
            "total_dim": total_dim,
            "feature_dims": {"category": num_categories, "region": num_regions, "temporal": 24, "collaborative": num_users},
            "feature_offsets": {
                "category": 0,
                "region": num_categories,
                "temporal": num_categories + num_regions,
                "collaborative": num_categories + num_regions + 24,
            },
        }

        return poi_features, metadata, pid_to_idx

    def save_and_upload(self, poi_features: np.ndarray, metadata: Dict[str, Any], pid_to_idx: Dict[int, int]) -> None:
        """Save data and upload to HF"""
        print("🔄 Saving and uploading data...")

        # Create temporary directory
        temp_dir = f"/tmp/{self.dataset_name}_rqvae"
        os.makedirs(temp_dir, exist_ok=True)

        # Save as numpy format
        npy_path = os.path.join(temp_dir, "poi_features.npy")
        np.save(npy_path, poi_features)
        print(f"💾 Saved: {npy_path}")

        # Save as PyTorch format
        pt_path = os.path.join(temp_dir, "poi_features.pt")
        poi_features_tensor = torch.from_numpy(poi_features)
        torch.save(poi_features_tensor, pt_path)
        print(f"💾 Saved: {pt_path}")

        # Save metadata
        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved: {metadata_path}")

        # Save POI ID mapping
        pid_path = os.path.join(temp_dir, "pid_to_index.csv")
        pd.DataFrame({"Pid": list(pid_to_idx.keys()), "Index": list(pid_to_idx.values())}).to_csv(pid_path, index=False)
        print(f"💾 Saved: {pid_path}")

        # Upload to HF
        print("🔄 Uploading to Hugging Face...")

        # Upload to RQVAE Dataset folder (main files)
        self.upload_to_hf(metadata_path, "RQVAE Dataset/metadata.json")
        self.upload_to_hf(pt_path, "RQVAE Dataset/poi_features.pt")

        # Upload to RQVAE Dataset/Intermedia Files folder (intermediate files)
        self.upload_to_hf(npy_path, "RQVAE Dataset/Intermedia Files/poi_features.npy")
        self.upload_to_hf(pid_path, "RQVAE Dataset/Intermedia Files/pid_to_index.csv")

        print("✅ Data uploaded successfully!")
        print(f"📁 Main files: {self.hf_repo}/RQVAE Dataset/")
        print(f"📁 Intermediate files: {self.hf_repo}/RQVAE Dataset/Intermedia Files/")

        # Clean up temporary files

        shutil.rmtree(temp_dir, ignore_errors=True)

    def process_dataset(self) -> None:
        """Complete RQ-VAE data preparation workflow"""
        print(f"🚀 Starting RQ-VAE data preparation for {self.dataset_name}")
        print("=" * 60)

        try:
            # Prepare data
            poi_features, metadata, pid_to_idx = self.prepare_rqvae_input()

            # Save and upload
            self.save_and_upload(poi_features, metadata, pid_to_idx)

            print(f"\n🎉 RQ-VAE data preparation completed for {self.dataset_name}!")
            print("=" * 60)
            print("\n📊 Final Statistics:")
            print(f"  Feature matrix shape: {poi_features.shape}")
            print(f"  Total POIs: {metadata['num_pois']}")
            print(f"  Feature dimensions: {metadata['total_dim']}")
            print(f"    - Category: {metadata['feature_dims']['category']}")
            print(f"    - Region: {metadata['feature_dims']['region']}")
            print(f"    - Temporal: {metadata['feature_dims']['temporal']}")
            print(f"    - Collaborative: {metadata['feature_dims']['collaborative']}")

            print("\n📁 Files available at:")
            print(f"  - https://huggingface.co/datasets/{self.hf_repo}/tree/main/RQVAE%20Dataset")

        except Exception as e:
            print(f"❌ Error processing {self.dataset_name}: {e}")
            raise
