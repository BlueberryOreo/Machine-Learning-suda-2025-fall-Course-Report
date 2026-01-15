
import torch
from torch import nn

class MLP(nn.Module):
    """A simple MLP block: Linear -> LayerNorm -> GELU -> Dropout, repeated."""
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        assert n_layers >= 1
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleEncoder(nn.Module):
    """
    Encoder for AE:
    
    Return: z
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = MLP(input_dim, hidden_dim, n_layers, dropout)

        self.fc_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        z = self.fc_z(h)
        return z


class SimpleDecoder(nn.Module):
    """Decoder maps latent z back to x reconstruction."""
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        out_activation: str = "identity",  # "identity" | "sigmoid"
    ):
        super().__init__()
        self.backbone = MLP(latent_dim, hidden_dim, n_layers, dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        if out_activation not in ["identity", "sigmoid"]:
            raise ValueError("out_activation must be 'identity' or 'sigmoid'")
        self.out_activation = out_activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.backbone(z)
        x_hat = self.fc_out(h)
        if self.out_activation == "sigmoid":
            x_hat = torch.sigmoid(x_hat)
        return x_hat

