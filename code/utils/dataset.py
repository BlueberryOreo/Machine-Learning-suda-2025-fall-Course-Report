
import os
import torch
from torch.utils.data import Dataset
import scanpy as sc
import anndata as ad

class H5adDataset(Dataset):
    def __init__(self, data_path: str):
        self.adata = sc.read(data_path)
        self.X = torch.tensor(self.adata.X, dtype=torch.float32) # (n_cells, n_genes)
        self.n_cells, self.n_genes = self.X.shape

        self.cell_types = self.adata.obs["celltype"].tolist()
    
    def __len__(self):
        return self.n_cells
    
    def __getitem__(self, index):
        return self.X[index]