import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from modules.components import Encoder, DecoderSCVI, SimpleEncoder, SimpleDecoder


class VAE(nn.Module):
    """
    A simple AE/VAE implementation.

    - is_vae=True: Variational Autoencoder
    - is_vae=False: Autoencoder (deterministic)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        is_vae: bool = True,
        recon_loss: str = "mse",     # "mse" | "bce"
        out_activation: str = "identity",  # use "sigmoid" if recon_loss="bce"
        beta_kl: float = 1.0,        # beta-VAE
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_vae = is_vae
        self.recon_loss = recon_loss
        self.beta_kl = beta_kl

        self.encoder = SimpleEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=n_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            is_vae=is_vae,
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
          mu, logvar, z
        1. If is_vae=True, z is sampled from N(mu, var)
        2. If is_vae=False, z = fc_z(backbone(x))
        """
        return self.encoder(x)

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
        if self.is_vae:
            out["mu"] = enc["mu"]
            out["logvar"] = enc["logvar"]
        return out

    def sample(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        For compatibility with your original API: 'sample' means encode.
        """
        return self.encode(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """
        For compatibility with your original API: decode latent to x.
        """
        return self.decode(z)

    def sample_prior(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample z ~ N(0, I) and decode to x."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    def loss(self, x: torch.Tensor, out: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute losses:
          recon + beta * kl (if VAE)
        Returns dict of scalar tensors: total, recon, kl
        """
        if out is None:
            out = self.forward(x)

        x_hat = out["x_hat"]

        if self.recon_loss == "mse":
            recon = F.mse_loss(x_hat, x, reduction="mean")
        else:  # bce
            # For BCE, x should be in [0, 1] and decoder activation typically sigmoid.
            recon = F.binary_cross_entropy(x_hat, x, reduction="mean")

        kl = torch.tensor(0.0, device=x.device)
        if self.is_vae:
            mu, logvar = out["mu"], out["logvar"]
            # KL(q(z|x)||p(z)) for diagonal Gaussian
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

        total = recon + self.beta_kl * kl
        return {"total": total, "recon": recon, "kl": kl}


class ZINBVAE(nn.Module):
    """
    A Variational Autoencoder (VAE) with Zero-Inflated Negative Binomial (ZINB) likelihood for single-cell RNA-seq data.
    """
    def __init__(self, input_dim, latent_dim):
        super(ZINBVAE, self).__init__()
        # Encoder
        