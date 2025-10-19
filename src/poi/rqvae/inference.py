import string

import torch

from ..rqvae import RQVAE, RQVAEConfig


def load_model(config: RQVAEConfig):
    model = RQVAE(
        embedding_dim=config.embedding_dim,
        vae_hidden_dims=config.vae_hidden_dims,
        vector_dim=config.vector_dim,
        vector_num=config.vector_num,
        codebook_num=config.codebook_num,
        commitment_weight=config.commitment_weight,
    ).to(config.device)
    return model

@torch.inference_mode()
def encode_poi_sid(model: RQVAE, batch: torch.Tensor):
    letters = string.ascii_lowercase
    batch_size = batch.size(0)
    all_indices: list[torch.Tensor]

    _, _, all_indices = model.forward(batch.to(model.device))

    codebook_num = len(all_indices)
    assert codebook_num <= len(letters)

    sids = [
        "".join(
            [
                f"<{letters[level]}_{all_indices[level][i]}>"
                for level in range(codebook_num)
            ]
        )
        for i in range(batch_size)
    ]

    return sids
