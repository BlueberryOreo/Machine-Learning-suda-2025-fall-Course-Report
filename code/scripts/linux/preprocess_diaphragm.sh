
python preprocess/preprocess.py \
    --data_path ../data/Quake_Diaphragm.h5ad \
    --output_path ../data \
    --nan_to_num 0.0 \
    --n_genes_threshold 6000 \
    --pct_counts_mt_threshold 0.9 \
    --hvgs_n 2000 \
    --save_violin
    # --mitochondrial_prefixes MT \
    # --ribosomal_prefixes RPS RPL \
    
