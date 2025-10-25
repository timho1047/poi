import string

import torch
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

from poi import settings

from ..rqvae import RQVAE, RQVAEConfig


def load_raw_model(config: RQVAEConfig):
    model = RQVAE(
        embedding_dim=config.embedding_dim,
        vae_hidden_dims=config.vae_hidden_dims,
        vector_dim=config.vector_dim,
        vector_num=config.vector_num,
        codebook_num=config.codebook_num,
        commitment_weight=config.commitment_weight,
        random_state=config.random_state,
    ).to(config.device)
    return model


def load_inference_model(config: RQVAEConfig, from_hub: bool = False):
    """
    Load the RQVAE model from the config.
    If from_hub is True, load the model from the Hugging Face hub.
    If from_hub is False, load the model from the local directory.
    """
    model = RQVAE(
        embedding_dim=config.embedding_dim,
        vae_hidden_dims=config.vae_hidden_dims,
        vector_dim=config.vector_dim,
        vector_num=config.vector_num,
        codebook_num=config.codebook_num,
        commitment_weight=config.commitment_weight,
        random_state=config.random_state,
    ).to(config.device)

    if from_hub:
        ckp_path = hf_hub_download(repo_id=config.hub_id, filename=config.checkpoint_best_path.name, token=settings.HF_TOKEN)
        ckp = torch.load(ckp_path, map_location=config.device)
    else:
        ckp = torch.load(config.checkpoint_best_path, map_location=config.device)

    model.load_state_dict(ckp["model_state_dict"])
    model.eval()

    return model


@torch.inference_mode()
def encode_poi_sid(model: RQVAE, batch: torch.Tensor):
    letters = string.ascii_lowercase
    batch_size = batch.size(0)
    all_indices: list[torch.Tensor]

    _, _, all_indices = model.forward(batch)

    codebook_num = len(all_indices)
    assert codebook_num <= len(letters)

    sids = ["".join([f"<{letters[level]}_{all_indices[level][i]}>" for level in range(codebook_num)]) for i in range(batch_size)]

    return sids


@torch.inference_mode()
def encode_poi_sid_ver2(model: RQVAE, batch: torch.Tensor, start_pid: int = 1):
    """
    Encode POI features to codebook indices and generate CSV format data.
    
    This function generates a DataFrame with the same format as the reference CSV file:
    - Pid: POI ID (starting from start_pid)
    - Codebook: List of quantized indices from all codebook layers
    - Vector: Quantized vector representation
    
    Args:
        model: Trained RQ-VAE model
        batch: Batch of POI features
        start_pid: Starting POI ID (default: 1)
        
    Returns:
        pandas.DataFrame: DataFrame with columns ['Pid', 'Codebook', 'Vector']
    """
    model.eval()
    
    with torch.no_grad():
        # Get features and quantized indices
        features = model.encoder(batch)
        quantized, cur_loss, all_indices = model.quantizer(features)
        
        # Convert to numpy for easier processing
        all_indices = [indices.cpu().numpy() for indices in all_indices]
        quantized_np = quantized.cpu().numpy()
        
        # Create DataFrame
        data = []
        for i in range(len(batch)):
            # Combine indices from all codebook layers
            codebook_indices = []
            for layer_indices in all_indices:
                codebook_indices.append(int(layer_indices[i]))
            
            # Get the quantized vector
            vector = quantized_np[i].tolist()
            
            data.append({
                'Pid': start_pid + i,
                'Codebook': codebook_indices,
                'Vector': vector
            })
    
    return pd.DataFrame(data)
