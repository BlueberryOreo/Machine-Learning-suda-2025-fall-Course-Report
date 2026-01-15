
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/diaphragm_ae.yaml --out_dir output/diaphragm_ae
# --eval --resume ./output/diaphragm_ae/202601151613/model_final.pt