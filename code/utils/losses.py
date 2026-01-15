
import torch
import torch.nn.functional as F


def knn_graph_contrastive_loss(
    z: torch.Tensor,
    z_cons: torch.Tensor | None = None,
    k: int = 15,
    temperature: float = 0.2,
    metric: str = "cosine",   # "cosine" or "euclidean"
    normalize: bool = True,   # for cosine metric
    mutual: bool = False,     # keep only mutual kNN edges
    undirected: bool = True,  # treat edges as undirected positives
    exclude_self: bool = True,
    eps: float = 1e-12,
    return_edge_index: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Build kNN graph inside a batch from z, then compute graph contrastive loss
    (multi-positive InfoNCE) where neighbors are positives.

    Args:
        z: (B, D) embeddings.
        k: number of neighbors per node (typical 10~30 for scRNA).
        temperature: softmax temperature.
        metric: "cosine" (recommended) or "euclidean".
        normalize: if True and metric=="cosine", L2 normalize z.
        mutual: if True, keep only mutual kNN edges (i->j and j->i both exist).
        undirected: if True, edges are treated as undirected positives.
        exclude_self: if True, do not allow self as neighbor/positive.
        eps: numerical stability.
        return_edge_index: if True, also return constructed edge_index (2, E).

    Returns:
        loss (scalar), or (loss, edge_index).
    """
    if z.dim() != 2:
        raise ValueError(f"z must be (B, D), got {tuple(z.shape)}")

    device = z.device
    B, D = z.shape
    if B < 2:
        loss = torch.zeros([], device=device, dtype=z.dtype)
        return (loss, torch.zeros((2, 0), device=device, dtype=torch.long)) if return_edge_index else loss

    if k <= 0:
        raise ValueError("k must be > 0")
    # If exclude_self, you can have at most B-1 neighbors
    k_eff = min(k, B - 1) if exclude_self else min(k, B)

    # ---- 1) Compute neighbor scores / distances ----
    if metric not in ("cosine", "euclidean"):
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    if metric == "cosine":
        zz = F.normalize(z, p=2, dim=1) if normalize else z
        # similarity: (B,B)
        sim = zz @ zz.t()
        # forbid self if needed
        if exclude_self:
            sim.fill_diagonal_(-float("inf"))
        # top-k highest similarity
        nn_idx = sim.topk(k_eff, dim=1, largest=True).indices  # (B, k_eff)

    else:  # euclidean
        # distance: (B,B)
        dist = torch.cdist(z, z, p=2)
        if exclude_self:
            dist.fill_diagonal_(float("inf"))
        # top-k smallest distance
        nn_idx = dist.topk(k_eff, dim=1, largest=False).indices  # (B, k_eff)

    # ---- 2) Build directed edge_index i -> nn_idx[i, :] ----
    src = torch.arange(B, device=device).unsqueeze(1).expand(B, k_eff).reshape(-1)  # (B*k_eff,)
    dst = nn_idx.reshape(-1)  # (B*k_eff,)
    edge_index = torch.stack([src, dst], dim=0)  # (2, E)

    # Optional: keep only mutual edges
    if mutual:
        # adjacency boolean from directed edges
        adj = torch.zeros((B, B), device=device, dtype=torch.bool)
        adj[edge_index[0], edge_index[1]] = True
        mutual_adj = adj & adj.t()
        # Extract edges from mutual_adj
        # (keep directed edges that are mutual; can later be treated undirected)
        keep = mutual_adj[edge_index[0], edge_index[1]]
        edge_index = edge_index[:, keep]

    # ---- 3) Graph contrastive loss (multi-positive InfoNCE) ----
    # Use cosine similarity for logits if normalize==True, else dot-product.
    z_for_logits = F.normalize(z, p=2, dim=1) if normalize else z
    logits = (z_for_logits @ z_for_logits.t()) / max(temperature, eps)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Positive mask from edges
    pos_mask = torch.zeros((B, B), device=device, dtype=torch.bool)
    if edge_index.numel() > 0:
        pos_mask[edge_index[0], edge_index[1]] = True
        if undirected:
            pos_mask[edge_index[1], edge_index[0]] = True

    # Exclude self from pos and denominator
    self_mask = torch.eye(B, device=device, dtype=torch.bool)
    pos_mask = pos_mask & (~self_mask)
    denom_mask = ~self_mask

    exp_logits = torch.exp(logits) * denom_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        loss = torch.zeros([], device=device, dtype=z.dtype)
        return (loss, edge_index) if return_edge_index else loss

    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count.float() + eps)
    loss = -mean_log_prob_pos[valid].mean()

    return (loss, edge_index) if return_edge_index else loss


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
    return loss