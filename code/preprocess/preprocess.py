
import os
import scanpy as sc
import numpy as np
import argparse

sc.settings.set_figure_params(dpi=50, facecolor="white")

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess single-cell RNA-seq data using Scanpy.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input .h5ad data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the preprocessed .h5ad data file.")
    parser.add_argument("--nan_to_num", type=float, default=0.0, help="Value to replace NaNs in the data.")
    parser.add_argument("--mitochondrial_prefixes", type=str, nargs='+', default=["MT"], help="Prefixes for mitochondrial genes.")
    parser.add_argument("--ribosomal_prefixes", type=str, nargs='+', default=["RPS", "RPL"], help="Prefixes for ribosomal genes.")
    parser.add_argument("--n_genes_threshold", type=int, default=6000, help="Threshold for filtering cells by number of genes.")
    parser.add_argument("--pct_counts_mt_threshold", type=float, default=0.9, help="Threshold for filtering cells by percentage of mitochondrial counts.")
    parser.add_argument("--hvgs_n", type=int, default=2000, help="Number of highly variable genes to select.")
    parser.add_argument("--batch_key", type=str, default=None, help="Key in adata.obs for batch information (if any).")
    parser.add_argument("--save_violin", action="store_true", help="Whether to save violin plots for QC metrics.")
    return parser.parse_args()

def preprocess(
    data_path: str,
    output_path: str,
    nan_to_num: float = 0.0,
    mitochondrial_prefix: tuple = ("MT",),
    ribosomal_prefixes: tuple = ("RPS", "RPL"),
    n_genes_threshold: int = 6000,
    pct_counts_mt_threshold: float = 0.9,
    hvgs_n: int = 2000,
    batch_key: str = None,
    save_violin: bool = False,
):
    dset_name = os.path.basename(data_path).split(".")[0]
    adata = sc.read(data_path)
    processed_data = np.nan_to_num(adata.X, nan=nan_to_num)
    adata.X = processed_data

    print("Initial data:")
    print(adata)

    # QC
    # Mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith(mitochondrial_prefix)
    # Ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(ribosomal_prefixes)

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], log1p=True, inplace=True)

    if save_violin:
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            multi_panel=True,
            show=False,
            save="violin.png",
        )

    # Filter cells
    adata = adata[adata.obs.n_genes_by_counts < n_genes_threshold, :]
    adata = adata[adata.obs.pct_counts_mt < pct_counts_mt_threshold, :]

    # Normalization
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Highly variable genes
    adata.raw = adata # keep full dimension safe
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=hvgs_n,
        flavor="seurat_v3",
        layer="counts",
        subset=True,
        batch_key=batch_key
    )

    print("Preprocessed data:")
    print(adata)

    adata.write_h5ad(os.path.join(output_path, f"{dset_name}_processed.h5ad"))
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        data_path=args.data_path,
        output_path=args.output_path,
        nan_to_num=args.nan_to_num,
        mitochondrial_prefix=tuple(args.mitochondrial_prefixes),
        ribosomal_prefixes=tuple(args.ribosomal_prefixes),
        n_genes_threshold=args.n_genes_threshold,
        pct_counts_mt_threshold=args.pct_counts_mt_threshold,
        hvgs_n=args.hvgs_n,
        batch_key=args.batch_key,
        save_violin=args.save_violin,
    )
