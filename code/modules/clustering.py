
import os
import torch
import torch.nn.functional as F
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans


def knn_graph_contrastive_loss(
    z: torch.Tensor,
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


def run_cluster(adata: AnnData, config, use_rep: str = "latent"):
    if config.method == "kmeans":
        X = adata.obsm[use_rep]
        kmeans = KMeans(n_clusters=config.n_clusters, n_init=config.n_init, random_state=config.random_state)
        pred_labels = kmeans.fit_predict(X)
        adata.obs[config.method] = pred_labels.astype(str)
        adata.obs[config.method] = adata.obs[config.method].astype("category")
        return pred_labels
    elif config.method == "leiden":
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=config.n_neighbors, metric=config.metric)
        sc.tl.leiden(adata, key_added=config.method, resolution=config.resolution)
        pred_labels = adata.obs[config.method]
        return pred_labels
    else:
        raise NotImplementedError(f"Clustering method '{config.method}' is not implemented.")


def visualize(adata: AnnData, save_path: str, method: str, use_rep: str = "latent", legend_loc: str = "on data", display_gt: bool = False):
    # Generate UMAP and save the plot
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep, metric="cosine")
    sc.tl.umap(adata, random_state=0)

    plot_cols = [method]
    if display_gt:
        plot_cols.insert(0, "celltype")

    axs = sc.pl.umap(
        adata,
        color=plot_cols,
        legend_loc=legend_loc,
        show=False,
    )
    if isinstance(axs, list):
        fig = axs[0].figure
    else:
        fig = axs.figure

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
