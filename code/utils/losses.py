
import torch
import torch.nn.functional as F

@torch.no_grad()
def augment_x(
    x: torch.Tensor,
    gene_dropout_p: float = 0.2,
    noise_std: float = 0.05,
    scale_low: float = 0.9,
    scale_high: float = 1.1,
) -> torch.Tensor:
    """
    Simple scRNA augmentations on input expression.

    Assumes x is already normalized/log-transformed (e.g., log1p normalized counts)
    shape: (B, G)
    """
    # 1) gene (feature) dropout
    if gene_dropout_p > 0:
        mask = torch.rand_like(x) > gene_dropout_p
        x = x * mask

    # 2) gaussian noise (small)
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    # 3) library size scaling (mild)
    if scale_low != 1.0 or scale_high != 1.0:
        s = torch.empty((x.size(0), 1), device=x.device, dtype=x.dtype).uniform_(scale_low, scale_high)
        x = x * s

    return x


def simclr_nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
    normalize: bool = True,
    eps: float = 1e-12,
    return_sim: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Standard SimCLR NT-Xent loss for two views.
    z1, z2: (B, D)
    """
    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # (2B, 2B) similarity
    logits = (z @ z.t()) / max(temperature, eps)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # mask self
    device = z.device
    self_mask = torch.eye(2 * B, device=device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -float("inf"))

    # positives: i in [0,B-1] matches i+B; i in [B,2B-1] matches i-B
    pos = torch.arange(B, device=device)
    pos_idx = torch.cat([pos + B, pos], dim=0)  # (2B,)

    loss = F.cross_entropy(logits, pos_idx)

    if return_sim:
        sim = z1 @ z2.t() / max(temperature, eps)
        return loss, sim
    return loss