import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Iterable, Callable, Literal

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
        n_hidden: int = 1024,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        is_vae: bool = True,
        recon_loss: str = "mse",     # "mse" | "bce"
        out_activation: str = "identity",  # use "sigmoid" if recon_loss="bce"
        beta_kl: float = 1.0,        # beta-VAE
        contrastive_loss: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_vae = is_vae
        self.recon_loss = recon_loss
        self.beta_kl = beta_kl
        self.contrastive_loss = contrastive_loss or (lambda z: 0.0)

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
          If VAE: {"mu": mu, "logvar": logvar, "z": z}
          If AE : {"z": z}
        """
        if self.is_vae:
            mu, logvar, z = self.encoder(x)
            return {"mu": mu, "logvar": logvar, "z": z}
        else:
            _, _, z = self.encoder(x)
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
        if self.is_vae:
            out["mu"] = enc["mu"]
            out["logvar"] = enc["logvar"]
        return out

    def sample_prior(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample z ~ N(0, I) and decode to x."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    def loss(self, x: torch.Tensor, lambda_contrastive: float, out: Optional[Dict[str, torch.Tensor]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute losses:
          recon + beta * kl (if VAE)
        Returns tuple of scalar tensors: total, recon, kl
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

        if lambda_contrastive > 0:
            z = out["z"]
            recon += lambda_contrastive * self.contrastive_loss(z)

        total = recon + self.beta_kl * kl
        return total, recon, kl


class ZINBVAE(nn.Module):
    """
    A Variational Autoencoder (VAE) with Zero-Inflated Negative Binomial (ZINB) likelihood for single-cell RNA-seq data.
    """
    def __init__(self, 
            n_input: int,
            n_output: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            latent_distribution: Literal["normal", "ln"] = "normal",
            deeply_inject_covariates: bool = True,
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
            log_variational: bool = True,
            var_activation: Callable | None = None,
            use_size_factor_key: bool = False,
            **kwargs,
        ):
        super(ZINBVAE, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_cat_list = n_cat_list
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.log_variational = log_variational

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Encoder
        self.encoder = Encoder(
            n_input=n_input,
            n_output=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=n_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        # Decoder
        self.decoder = DecoderSCVI(
            n_output,
            n_input,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )


if __name__ == "__main__":
    ae = VAE(
        input_dim=2000,
        latent_dim=128,
        n_hidden=1024,
        n_layers=2,
        dropout_rate=0.1,
        is_vae=False,
        recon_loss="mse",
        out_activation="identity",
        beta_kl=1.0,
    )

    x_input = torch.randn(16, 2000)
    out = ae(x_input)
    print(out["z"].shape)
