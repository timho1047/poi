#!/usr/bin/env python3
"""
LLM Dataset Generator Module

This module provides core functionality for generating LLM training datasets from
codebook files and POI sequence data. It handles data processing, JSON generation,
and Hugging Face uploads with retry mechanisms.

Key Components:
- DataDownloader: Downloads codebook and data files from Hugging Face
- JSONGenerator: Generates JSON format LLM training data
- HFUploader: Handles file uploads to Hugging Face with retry logic
- LLMDatasetGenerator: Main orchestrator class

Usage:
    from poi.llm_dataset_generator import LLMDatasetGenerator
    
    generator = LLMDatasetGenerator(hf_token="your_token")
    generator.generate_all_datasets()
"""

import os
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from huggingface_hub import upload_file, snapshot_download
from poi.settings import HF_ORG


class DataDownloader:
    """Handles downloading of codebook and data files from Hugging Face."""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
    
    def download_codebook_file(self, dataset_name: str, model_name: str, temp_dir: str) -> str:
        """Download codebook file from Hugging Face."""
        filename = f"codebooks-{model_name}.csv"
        hf_path = f"LLM Dataset/Intermediate Files/{filename}"
        
        # Use snapshot_download method
        local_dir = snapshot_download(
            repo_id=f"{HF_ORG}/{dataset_name.lower()}",
            repo_type='dataset',
            token=self.hf_token,
            allow_patterns=hf_path
        )
        
        # Find the downloaded file
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file == filename:
                    local_path = os.path.join(root, file)
                    print(f"✅ Downloaded codebook: {filename}")
                    return local_path
        
        raise FileNotFoundError(f"File not found: {filename}")
    
    def download_data_files(self, dataset_name: str, temp_dir: str) -> str:
        """Download data files from Hugging Face."""
        data_dir = os.path.join(temp_dir, f"{dataset_name}_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Use snapshot_download to download all data files
        local_dir = snapshot_download(
            repo_id=f"{HF_ORG}/{dataset_name.lower()}",
            repo_type='dataset',
            token=self.hf_token,
            allow_patterns='Intermediate Files/data/*.csv'
        )
        
        # Copy files to target directory
        import shutil
        source_data_dir = os.path.join(local_dir, 'Intermediate Files', 'data')
        for filename in os.listdir(source_data_dir):
            if filename.endswith('.csv'):
                shutil.copy2(os.path.join(source_data_dir, filename), data_dir)
                print(f"✅ Downloaded data file: {filename}")
        
        return data_dir


class JSONGenerator:
    """Generates JSON format LLM training data from codebook and sequence data."""
    
    @staticmethod
    def generate_json(mode: str, target_dataset: str, codebook_path: Optional[str], 
                     data_dir: str, idorcodebook: str = 'codebook', 
                     include_time: bool = True) -> Tuple[str, str]:
        """Generate JSON format LLM training data."""
        
        # Read codebook file (only when needed)
        poi_to_codebook = None
        if codebook_path is not None:
            codebook_df = pd.read_csv(codebook_path)
            codebook_df['Codebook'] = codebook_df['Codebook'].apply(eval)
            poi_to_codebook = dict(zip(codebook_df['Pid'], codebook_df['Codebook']))
        
        # Read data file
        data_file = os.path.join(data_dir, f"{mode}.csv")
        poi_sequence_df = pd.read_csv(data_file)
        
        instructions = []
        inputs = []
        outputs = []
        
        for _, row in poi_sequence_df.iterrows():
            uid = row['Uid']
            poi_sequence = eval(row['Pids'])
            time_sequence = eval(row['Times']) if include_time else None
            target_time = row['Target_time'] if include_time else None
            target = row['Target']
            
            if idorcodebook == 'codebook':
                # Generate SID format: <a_20> <b_21> <c_19> <d_1>
                if poi_to_codebook is None:
                    raise ValueError("codebook_path is required for codebook generation")
                embedded_sequence = [
                    ''.join([f"<{chr(97 + idx)}_{code}>" for idx, code in enumerate(poi_to_codebook[poi])])
                    for poi in poi_sequence
                ]
                target_embedding = ''.join([f"<{chr(97 + idx)}_{code}>" for idx, code in enumerate(poi_to_codebook[target])])
            
            elif idorcodebook == 'id':
                embedded_sequence = [f"<{poi}>" for poi in poi_sequence]
                target_embedding = f"<{target}>"
            
            else:
                raise ValueError("Invalid idorcodebook value. Use 'codebook' or 'id'.")
            
            # Generate input text
            if include_time:
                # Format with time information
                if time_sequence:
                    embedded_sequence_with_time = [
                        seq + f' at {time_sequence[i]}, ' if i < len(poi_sequence) - 1 else
                        seq + f' at {time_sequence[i]}.'
                        for i, seq in enumerate(embedded_sequence)
                    ]
                    input_text = f"User_{uid} visited: " + "".join(embedded_sequence_with_time) + f" When {target_time} user_{uid} is likely to visit:"
                else:
                    input_text = f"User_{uid} visited: " + " ".join(embedded_sequence) + f" When {target_time} user_{uid} is likely to visit:"
                
                instruction = "Here is a record of a user's POI accesses, your task is based on the history to predict the POI that the user is likely to access at the specified time."
            else:
                # Format without time information
                if len(embedded_sequence) == 1:
                    input_text = f"The user_{uid} visited: {embedded_sequence[0]}, and in the next time user_{uid} is likely to visit:"
                else:
                    sequence_str = ", ".join(embedded_sequence) + ","
                    input_text = f"The user_{uid} visited: {sequence_str} and in the next time user_{uid} is likely to visit:"
                
                instruction = "Here is a record of a user's POI accesses, your task is based on the history to predict the next POI that the user is likely to access."
            
            instructions.append(instruction)
            inputs.append(input_text)
            outputs.append(target_embedding)
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'instruction': instructions,
            'input': inputs,
            'output': outputs
        })
        
        # Save as JSON
        json_data = result_df.to_json(orient="records", indent=4)
        
        # Generate filename
        time_suffix = "" if include_time else "_notime"
        filename = f"{mode}_{idorcodebook}{time_suffix}.json"
        
        return json_data, filename


class HFUploader:
    """Handles file uploads to Hugging Face with retry mechanisms."""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.max_retries = 3
    
    def upload_json_to_hf(self, json_data: str, filename: str, dataset_name: str, model_name: str) -> None:
        """Upload JSON file to Hugging Face model folder."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write(json_data)
            tmp_file_path = tmp_file.name
        
        # Upload to HF with retry mechanism
        hf_path = f"LLM Dataset/{model_name}/{filename}"
        self._upload_with_retry(tmp_file_path, hf_path, dataset_name)
        
        # Clean up temporary file
        os.remove(tmp_file_path)
    
    def upload_json_to_hf_notime(self, json_data: str, filename: str, dataset_name: str) -> None:
        """Upload notime JSON file to LLM Dataset/ablation without Time folder."""
        hf_path = f"LLM Dataset/ablation without Time/{filename}"
        self._upload_with_retry(json_data.encode('utf-8'), hf_path, dataset_name)
    
    def upload_json_to_hf_id(self, json_data: str, filename: str, dataset_name: str) -> None:
        """Upload ID JSON file to LLM Dataset/ablation without SID folder."""
        hf_path = f"LLM Dataset/ablation without SID/{filename}"
        self._upload_with_retry(json_data.encode('utf-8'), hf_path, dataset_name)
    
    def _upload_with_retry(self, data, hf_path: str, dataset_name: str) -> None:
        """Upload data to HF with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                upload_file(
                    path_or_fileobj=data,
                    path_in_repo=hf_path,
                    repo_id=f"{HF_ORG}/{dataset_name.lower()}",
                    repo_type="dataset",
                    token=self.hf_token
                )
                print(f"✅ Uploaded: {hf_path}")
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️ Upload failed, retrying {attempt + 1}/{self.max_retries}: {e}")
                    import time
                    time.sleep(5)  # Wait 5 seconds before retry
                else:
                    print(f"❌ Upload failed after {self.max_retries} attempts: {e}")
                    raise


class LLMDatasetGenerator:
    """Main orchestrator class for generating LLM datasets."""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.downloader = DataDownloader(hf_token)
        self.json_generator = JSONGenerator()
        self.uploader = HFUploader(hf_token)
        
        # Define all models
        self.models = {
            'NYC': [
                'rqvae-nyc-div0.0-commit0.5-lr5e-5',
                'rqvae-nyc-div0.25-commit0.5-lr5e-5',
                'rqvae-nyc-div0.25-commit0.5-lr5e-5-without_L_quant',
                'rqvae-nyc-div0.5-commit0.5-lr5e-5',
                'rqvae-nyc-div0.75-commit0.5-lr5e-5'
            ],
            'TKY': [
                'rqvae-tky-div0.0-commit0.25-lr5e-5',
                'rqvae-tky-div0.25-commit0.25-lr5e-5',
                'rqvae-tky-div0.25-commit0.25-lr5e-5-without_L_quant',
                'rqvae-tky-div0.5-commit0.25-lr5e-5',
                'rqvae-tky-div0.75-commit0.25-lr5e-5'
            ]
        }
    
    def generate_id_files(self, dataset_name: str, temp_dir: str) -> None:
        """Generate ID files to ablation without SID folder."""
        print(f"\n🔄 Generating {dataset_name} ID files")
        print("=" * 60)
        
        # Download data files
        data_dir = self.downloader.download_data_files(dataset_name, temp_dir)
        
        # Generate ID JSON files
        modes = ['train', 'val', 'test', 'test_all']
        
        for mode in modes:
            json_data, filename = self.json_generator.generate_json(
                mode, dataset_name, None, data_dir, 
                idorcodebook='id', include_time=True
            )
            self.uploader.upload_json_to_hf_id(json_data, filename, dataset_name)
        
        print(f"✅ Completed {dataset_name} ID files generation")
    
    def process_model(self, dataset_name: str, model_name: str, temp_dir: str) -> None:
        """Process all data for a single model."""
        print(f"\n🔄 Processing model: {model_name}")
        print("=" * 60)
        
        # Download codebook file
        codebook_path = self.downloader.download_codebook_file(dataset_name, model_name, temp_dir)
        
        # Download data files
        data_dir = self.downloader.download_data_files(dataset_name, temp_dir)
        
        # Generate codebook JSON files
        modes = ['train', 'val', 'test', 'test_all']
        
        for mode in modes:
            # Generate codebook JSON with time information
            json_data, filename = self.json_generator.generate_json(
                mode, dataset_name, codebook_path, data_dir, 
                idorcodebook='codebook', include_time=True
            )
            self.uploader.upload_json_to_hf(json_data, filename, dataset_name, model_name)
        
        # Only specific models generate notime files
        if model_name in ['rqvae-nyc-div0.25-commit0.5-lr5e-5', 'rqvae-tky-div0.25-commit0.25-lr5e-5']:
            for mode in modes:
                json_data, filename = self.json_generator.generate_json(
                    mode, dataset_name, codebook_path, data_dir, 
                    idorcodebook='codebook', include_time=False
                )
                self.uploader.upload_json_to_hf_notime(json_data, filename, dataset_name)
        
        print(f"✅ Completed model: {model_name}")
    
    def generate_all_datasets(self) -> None:
        """Generate all LLM datasets for all models."""
        print("🚀 LLM Dataset Generation Script")
        print("=" * 60)
        
        # Check HF token
        if not self.hf_token:
            print("❌ HF_TOKEN not set! Please set your HF_TOKEN environment variable.")
            return
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Temporary directory: {temp_dir}")
            
            # Process each dataset and model
            for dataset_name, model_list in self.models.items():
                print(f"\n📊 Processing dataset: {dataset_name}")
                print(f"   Model count: {len(model_list)}")
                
                # First generate ID files
                try:
                    self.generate_id_files(dataset_name, temp_dir)
                except Exception as e:
                    print(f"❌ Failed to generate ID files for {dataset_name}: {e}")
                
                # Then process each model
                for model_name in model_list:
                    try:
                        self.process_model(dataset_name, model_name, temp_dir)
                    except Exception as e:
                        print(f"❌ Failed to process model {model_name}: {e}")
                        continue
        
        print(f"\n🎉 All LLM datasets generation completed!")
        print(f"📁 File locations:")
        print(f"   - comp5331poi/nyc/LLM Dataset/")
        print(f"   - comp5331poi/tky/LLM Dataset/")
