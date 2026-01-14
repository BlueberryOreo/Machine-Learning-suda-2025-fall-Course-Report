
python preprocess/preprocess.py \
    --data_path ../data/Tosches_turtle.h5ad \
    --output_path ../data \
    --nan_to_num 0.0 \
    --mitochondrial_prefixes MT \
    --ribosomal_prefixes RPS RPL \
    --n_genes_threshold 6000 \
    --pct_counts_mt_threshold 0.9 \
    --save_violin
