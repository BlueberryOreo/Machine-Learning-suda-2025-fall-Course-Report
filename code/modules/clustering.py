
import os
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans

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
        pred_labels = adata.obs[config.method].to_list()
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
