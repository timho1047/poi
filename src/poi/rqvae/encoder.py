import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        hidden_dims = sorted(hidden_dims, reverse=True)
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.proj = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.layers(x)
        return self.proj(x)
