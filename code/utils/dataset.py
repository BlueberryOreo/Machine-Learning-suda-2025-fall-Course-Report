
import os
import torch
from torch.utils.data import Dataset
import scanpy as sc
import anndata as ad

class H5adDataset(Dataset):
    def __init__(self, data_path: str):
        self.adata = sc.read(data_path)
        self.X = torch.tensor(self.adata.X.toarray(), dtype=torch.float32) # (n_cells, n_genes)
        self.n_cells, self.n_genes = self.X.shape

        self.cell_types = self.adata.obs["celltype"].tolist()
    
    def __len__(self):
        return self.n_cells
    
    def __getitem__(self, index):
        return self.X[index], self.cell_types[index]


if __name__ == "__main__":
    data_path = "../../data/Tosches_turtle_processed.h5ad"
    data = H5adDataset(data_path)
    print(f"Dataset size: {len(data)} cells, {data.n_genes} genes")
    print(data[0:5])
