#!/usr/bin/env python3
"""
RQ-VAE Batch Inference Module

This module provides core functionality for performing batch inference on RQ-VAE models.
It handles model configuration parsing, data loading, inference processing, and result uploads.

Key Components:
- ModelConfigParser: Parses Hugging Face model IDs to extract configuration parameters
- BatchInferenceProcessor: Handles the complete inference pipeline for a single model
- BatchInferenceManager: Manages batch processing of multiple models

Usage:
    from poi.rqvae.batch_inference import BatchInferenceManager
    
    manager = BatchInferenceManager(hf_token="your_token")
    manager.process_models(model_list)
"""

import os
import re
import tempfile
import pandas as pd
from typing import Dict, List, Optional
from huggingface_hub import upload_file
from poi.dataset.rqvae import get_dataloader
from poi.rqvae import RQVAEConfig
from poi.rqvae.inference import load_inference_model, encode_poi_sid_ver2
from poi.settings import DEVICE, HF_ORG


class ModelConfigParser:
    """Parses Hugging Face model IDs to extract RQ-VAE configuration parameters."""
    
    @staticmethod
    def parse_model_name(model_id: str) -> Dict[str, any]:
        """Parse model ID to extract configuration parameters from Hugging Face model name."""
        model_name = model_id.split('/')[-1]
        
        # Extract dataset name
        if 'nyc' in model_name.lower():
            dataset_name = 'NYC'
        elif 'tky' in model_name.lower():
            dataset_name = 'TKY'
        else:
            raise ValueError(f"Cannot determine dataset from model name: {model_name}")
        
        # Extract div_weight
        div_match = re.search(r'div(\d+\.?\d*)', model_name)
        div_weight = float(div_match.group(1)) if div_match else 0.25
        
        # Extract commitment_weight
        commit_match = re.search(r'commit(\d+\.?\d*)', model_name)
        commitment_weight = float(commit_match.group(1)) if commit_match else 0.5
        
        # Extract learning rate
        lr_match = re.search(r'lr(\d+)e-(\d+)', model_name)
        if lr_match:
            lr = float(f"{lr_match.group(1)}e-{lr_match.group(2)}")
        else:
            lr = 5e-5
        
        # Check for special flags
        quant_weight = 0.0 if 'without_L_quant' in model_name else 1.0
        
        return {
            'dataset_name': dataset_name,
            'div_weight': div_weight,
            'commitment_weight': commitment_weight,
            'lr': lr,
            'quant_weight': quant_weight,
            'run_name': model_name
        }


class BatchInferenceProcessor:
    """Handles the complete inference pipeline for a single RQ-VAE model."""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
    
    def process_model(self, model_id: str) -> None:
        """Process a single RQ-VAE model: load, infer, and upload results to Hugging Face."""
        print(f"\n🔄 Processing model: {model_id}")
        print("=" * 60)
        
        # Parse model name and create config
        params = ModelConfigParser.parse_model_name(model_id)
        config = RQVAEConfig(**params)
        print(f"📋 Config: {config.dataset_name}, div={config.div_weight}, commit={config.commitment_weight}, lr={config.lr}, quant={config.quant_weight}")
        
        # Load PID mapping for correct POI-to-codebook correspondence
        pid_mapping_path = config.dataset_path.parent / "Intermediate Files" / "pid_mapping.csv"
        pid_mapping = pd.read_csv(pid_mapping_path)
        print(f"📋 Loaded PID mapping with {len(pid_mapping)} POIs")
        
        # Get data loader WITHOUT shuffle to maintain consistent order
        train_loader = get_dataloader(
            config.dataset_path, 
            batch_size=config.batch_size, 
            num_workers=config.num_dataloader_workers, 
            device=DEVICE,
            shuffle=False  # CRITICAL: No shuffle to maintain consistent order
        )
        
        # Load model from HF Hub
        model = load_inference_model(config, from_hub=True)
        print(f"✅ Model loaded from HF Hub")
        
        # Process all data in batches
        all_results = self._process_batches(model, train_loader, pid_mapping, config)
        
        # Upload results
        self._upload_results(all_results, model_id, config)
        
        print(f"✅ Completed model: {model_id}")
    
    def _process_batches(self, model, train_loader, pid_mapping, config):
        """Process all data batches and return combined results."""
        all_results = []
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)
            
            # Encode POI features to SIDs (start_pid is placeholder, will fix later)
            df = encode_poi_sid_ver2(model, batch, start_pid=1)
            
            # Fix PIDs using the correct mapping to ensure accuracy
            batch_start_idx = batch_idx * config.batch_size
            batch_end_idx = min(batch_start_idx + config.batch_size, len(pid_mapping))
            
            # Get the correct Mapped_Pid values for this batch
            correct_pids = pid_mapping.iloc[batch_start_idx:batch_end_idx]['Mapped_Pid'].values
            
            # Update the Pid column with correct values
            df['Pid'] = correct_pids
            
            all_results.append(df)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1}/{total_batches} batches")
        
        # Combine all results
        final_df = pd.concat(all_results, ignore_index=True)
        print(f"✅ Generated {len(final_df)} POI encodings")
        
        return final_df
    
    def _upload_results(self, final_df, model_id, config):
        """Upload results to Hugging Face."""
        # Generate filename
        model_name = model_id.split('/')[-1]
        filename = f"codebooks-{model_name}.csv"
        
        # Save to temporary file for upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            final_df.to_csv(tmp_file.name, index=False)
            local_path = tmp_file.name
        
        # Upload to correct HF dataset location
        dataset_id = f"{HF_ORG}/{config.dataset_name.lower()}"
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"LLM Dataset/Intermediate Files/{filename}",
            repo_id=dataset_id,
            repo_type="dataset",
            token=self.hf_token
        )
        print(f"✅ Uploaded to {dataset_id}/LLM Dataset/Intermediate Files/{filename}")
        
        # Clean up temporary file
        os.remove(local_path)
        print(f"🧹 Temporary file cleaned up")


class BatchInferenceManager:
    """Manages batch processing of multiple RQ-VAE models."""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.processor = BatchInferenceProcessor(hf_token)
        
        # Define all target models
        self.target_models = [
            "comp5331poi/rqvae-tky-div0.0-commit0.25-lr5e-5",
            "comp5331poi/rqvae-tky-div0.25-commit0.25-lr5e-5", 
            "comp5331poi/rqvae-tky-div0.5-commit0.25-lr5e-5",
            "comp5331poi/rqvae-tky-div0.75-commit0.25-lr5e-5",
            "comp5331poi/rqvae-nyc-div0.75-commit0.5-lr5e-5",
            "comp5331poi/rqvae-nyc-div0.5-commit0.5-lr5e-5",
            "comp5331poi/rqvae-nyc-div0.25-commit0.5-lr5e-5",
            "comp5331poi/rqvae-nyc-div0.0-commit0.5-lr5e-5",
            "comp5331poi/rqvae-nyc-div0.25-commit0.5-lr5e-5-without_L_quant",
            "comp5331poi/rqvae-tky-div0.25-commit0.25-lr5e-5-without_L_quant"
        ]
    
    def process_models(self, model_list: Optional[List[str]] = None) -> None:
        """Process a list of models for batch inference."""
        if model_list is None:
            model_list = self.target_models
        
        print("🚀 Batch RQ-VAE Inference Script")
        print("=" * 60)
        print(f"📊 Processing {len(model_list)} models")
        
        # Check HF token
        if not self.hf_token:
            print("❌ HF_TOKEN not set! Please set your HF_TOKEN environment variable.")
            return
        
        # Process each model
        for i, model_id in enumerate(model_list, 1):
            print(f"\n🎯 Model {i}/{len(model_list)}: {model_id}")
            
            try:
                self.processor.process_model(model_id)
            except Exception as e:
                print(f"❌ Error processing {model_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n🎉 Batch processing completed!")
    
    def process_single_model(self, model_id: str) -> None:
        """Process a single model for testing or debugging."""
        print("🚀 Single Model RQ-VAE Inference")
        print("=" * 60)
        
        if not self.hf_token:
            print("❌ HF_TOKEN not set! Please set your HF_TOKEN environment variable.")
            return
        
        try:
            self.processor.process_model(model_id)
        except Exception as e:
            print(f"❌ Error processing {model_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def get_model_list(self) -> List[str]:
        """Get the list of all target models."""
        return self.target_models.copy()
    
    def get_nyc_models(self) -> List[str]:
        """Get the list of NYC models."""
        return [model for model in self.target_models if 'nyc' in model.lower()]
    
    def get_tky_models(self) -> List[str]:
        """Get the list of TKY models."""
        return [model for model in self.target_models if 'tky' in model.lower()]
