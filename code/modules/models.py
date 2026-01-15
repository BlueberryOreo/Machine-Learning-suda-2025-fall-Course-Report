import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable

from modules.components import SimpleEncoder, SimpleDecoder

# Can add other models here as needed
# Use load_model(model_name) in utils/utils.py to dynamically load models by name

class AE(nn.Module):
    """
    A simple AE implementation.

    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_hidden: int = 1024,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        recon_loss: str = "mse",     # "mse" | "bce"
        out_activation: str = "identity",  # use "sigmoid" if recon_loss="bce"
        beta_kl: float = 1.0,        # beta-VAE
        contrastive_loss: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.recon_loss = recon_loss
        self.beta_kl = beta_kl
        self.contrastive_loss = contrastive_loss or (lambda z: 0.0)

        self.encoder = SimpleEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=n_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
        )
        self.decoder = SimpleDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=n_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            out_activation=out_activation,
        )

        if recon_loss not in ["mse", "bce"]:
            raise ValueError("recon_loss must be 'mse' or 'bce'")

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with key "z": latent representation
        """
        z = self.encoder(x)
        return {"z": z}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returns a dict for convenience in training loops.
        """
        enc = self.encode(x)
        z = enc["z"]
        x_hat = self.decode(z)

        out = {"x_hat": x_hat, "z": z}

        return out

    def loss(self, x: torch.Tensor, lambda_contrastive: float, out: Optional[Dict[str, torch.Tensor]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute losses:
            recon + lambda_contrastive * contrastive_loss

        Returns tuple of scalar tensors: total, recon, contrastive_loss
        """
        if out is None:
            out = self.forward(x)

        x_hat = out["x_hat"]

        if self.recon_loss == "mse":
            recon = F.mse_loss(x_hat, x, reduction="mean")
        else:  # bce
            # For BCE, x should be in [0, 1] and decoder activation typically sigmoid.
            recon = F.binary_cross_entropy(x_hat, x, reduction="mean")

        contrastive_loss = torch.tensor(0.0, device=x.device)
        if lambda_contrastive > 0:
            z = out["z"]
            z_aug = out.get("z_aug", None)
            contrastive_loss = lambda_contrastive * self.contrastive_loss(z, z_aug)

        total = recon + contrastive_loss
        return total, recon, contrastive_loss