import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class VectorQuantizer(nn.Module):
    def __init__(self, vector_num, vector_dim, commitment_weight):
        super().__init__()
        self.vector_dim = vector_dim
        self.vector_num = vector_num
        self.commitment_weight = commitment_weight

        self.vectors = nn.Embedding(vector_num, vector_dim)
        self.vectors.weight.data.uniform_(-1 / vector_num, 1 / vector_num)  # to K-means

        self.register_buffer("ema_cluster_size", torch.zeros(vector_num))
        self.register_buffer("ema_w", torch.zeros(vector_num, vector_dim))
        self.decay = 0.99  # CODEBOOK_EMA_DECAY
        self.eps = 1e-5

    def forward(self, x):
        ## Compute the L2 distance
        # x: batch, vector_dim
        # weight: vector_num, vector_dim
        distances = (
            torch.sum(x**2, dim=1, keepdim=True)
            + torch.sum(self.vectors.weight**2, dim=1)
            - 2 * torch.matmul(x, self.vectors.weight.t())
        )
        # distances: batch, vector_num

        ## Get the vector
        vector_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(vector_indices, self.vector_num).float()
        quantized = torch.matmul(encodings, self.vectors.weight)
        # quantized = quantized.view(input_shape)

        cur_loss = {}
        ## Compute the quant loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        quant_loss = q_latent_loss + self.commitment_weight * e_latent_loss

        ## Get the quantized outputs
        quantized = x + (quantized - x).detach()

        ## Compute the diverse loss
        # encodings (one-hot): batch, vector_num
        # counts (whose sum = batch): vector_num
        counts = torch.sum(encodings, dim=0)
        batch_size = x.shape[0]

        util_loss = torch.mean(torch.abs(counts - batch_size / self.vector_num))
        compact_loss = 2 * torch.mean(torch.pdist(self.vectors.weight, p=2))
        # div_loss = util_loss + compact_loss

        cur_loss["quantization"] = quant_loss
        cur_loss["utilization"] = util_loss
        cur_loss["compactness"] = compact_loss

        ## EMA update
        with torch.no_grad():
            cluster_size = torch.sum(encodings, dim=0)
            embed_sum = torch.matmul(encodings.t(), x)

            self.ema_cluster_size.mul_(self.decay).add_(
                cluster_size, alpha=1 - self.decay
            )
            self.ema_w.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = torch.sum(self.ema_cluster_size)
            cluster_size_normalized = (
                (self.ema_cluster_size + self.eps)
                / (n + self.vector_num * self.eps)
                * n
            )
            self.vectors.weight.data.copy_(
                self.ema_w / cluster_size_normalized.unsqueeze(1)
            )

        return quantized, cur_loss, vector_indices


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_num, vector_num, vector_dim, commitment_weight):
        super().__init__()
        self.codebook_num = codebook_num
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(vector_num, vector_dim, commitment_weight)
                for _ in range(codebook_num)
            ]
        )
        self.loss_items = ["quantization", "utilization", "compactness"]

    def forward(self, x):
        quantized = 0
        residual = x
        all_indices = []
        total_loss = {k: 0.0 for k in self.loss_items}

        for _, quantizer in enumerate(self.quantizers):
            cur_quantized, cur_loss, indices = quantizer(residual)

            quantized += cur_quantized
            residual = x - quantized

            all_indices.append(indices)
            total_loss = {k: total_loss[k] + cur_loss[k] for k in total_loss.keys()}

        return quantized, total_loss, all_indices

    def initialize(self, x, random_state):
        quantized = 0
        residual = x
        device = x.device

        for layer_idx, quantizer in enumerate(self.quantizers):
            kmeans = KMeans(
                n_clusters=quantizer.vector_num, random_state=random_state, n_init=10
            )
            kmeans.fit(residual.detach().cpu().numpy())
            # 将 k-means 聚类中心写入该层的 codebook 向量
            centers = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=device
            )
            self.quantizers[layer_idx].vectors.weight.data.copy_(centers)
            cur_quantized, _, _ = quantizer(residual)

            quantized += cur_quantized
            residual = x - quantized

        print("Initialized quantizer using k-means!")
