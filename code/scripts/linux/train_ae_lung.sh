
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/lung_ae.yaml --out_dir output/lung_ae
# --eval --resume ./output/lung_ae/202601151613/model_final.pt