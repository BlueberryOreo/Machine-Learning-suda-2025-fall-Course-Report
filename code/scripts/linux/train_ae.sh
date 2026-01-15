
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/turtle_ae.yaml --out_dir output/turtle_ae
# --eval --resume ./output/turtle_ae/202601151510/model_final.pt