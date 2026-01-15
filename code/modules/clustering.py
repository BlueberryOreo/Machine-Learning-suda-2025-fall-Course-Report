# clustering.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans


def run_cluster(adata: AnnData, config, use_rep: str = "latent"):
    """
    Run clustering on the latent representations stored in adata.obsm[use_rep].
    Supported methods: "kmeans", "leiden".

    Args:
      adata: AnnData object with latent representations.
      config: Configuration object with clustering parameters.
      use_rep: Key in adata.obsm to use for clustering.
    Returns:
      pred_labels: np.ndarray of predicted cluster labels.
    """
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
    """
    Generate and save UMAP visualization of clusters.

    Args:
      adata: AnnData object with latent representations.
      save_path: Path to save the UMAP plot.
      method: Clustering method used (for coloring).
      use_rep: Key in adata.obsm to use for UMAP.
      legend_loc: Location of the legend in the plot.
      display_gt: Whether to display ground truth labels.
      
    Returns:
      None
    """
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
