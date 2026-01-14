
import os
import yaml
import torch
import argparse
import datetime
from torch.utils.data import DataLoader
from modules.models import ZINBVAE, VAE

from utils.dataset import H5adDataset
from utils.utils import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE/ZINBVAE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save trained model and logs")
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    
    # Create output directory with timestamp
    nowtime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    args.out_dir = os.path.join(args.out_dir, f"{nowtime}")
    os.makedirs(args.out_dir, exist_ok=True)

    return args

def train_epoch(model, dataloader, optimizer, device, config):
    pass

def train(model, train_loader, device, config):

    pass

def main(args):
    # Load model class
    model = load_model(args.model_name)(args.vae)
    pass


if __name__ == "__main__":
    args = parse_args()
    
    print(args)
