
import os
from ruamel.yaml import YAML
import torch
import argparse
import datetime
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dataset import H5adDataset
from utils.utils import load_model, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE/ZINBVAE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save trained model and logs")
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = YAML().load(f)
        for key, value in config.items():
            setattr(args, key, edict(value) if isinstance(value, dict) else value)
    
    # Create output directory with timestamp
    nowtime = datetime.datetime.now().strftime("%Y%m%d%H%M")
    args.train.out_dir = os.path.join(args.out_dir, f"{nowtime}")
    os.makedirs(args.train.out_dir, exist_ok=True)

    return args


def train_epoch(model, dataloader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, type) in enumerate(tqdm(dataloader, desc="Training Batches", disable=True)):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss, _, _ = model.loss(data, output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(model, train_loader, device, config):
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses = []
    bar = tqdm(range(1, config.num_epochs + 1))

    for epoch in bar:
        avg_loss = train_epoch(model, train_loader, optimizer, device, config)
        bar.desc = f"Epoch {epoch}/{config.num_epochs} - Loss: {avg_loss:.4f}"
        losses.append(avg_loss)
        
        # Save model at intervals
        if epoch % config.save_interval == 0 and epoch < config.num_epochs:
            model_path = os.path.join(config.out_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            # print(f"Saved model checkpoint at {model_path}")
    
    torch.save(model.state_dict(), os.path.join(config.out_dir, "model_final.pt"))
    print("Training complete. Saved final model.")

    if config.save_loss_curve:
        plt.figure()
        plt.plot(range(1, config.num_epochs + 1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        loss_curve_path = os.path.join(config.out_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        print(f"Saved loss curve at {loss_curve_path}")

def main(args):
    # Seed everything for reproducibility
    seed_everything(args.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model class
    model = load_model(args.vae.model_name)(**args.vae).to(device)
    print(model)

    # Load dataset
    dataset = H5adDataset(args.dataset.data_path)
    train_loader = DataLoader(dataset, batch_size=args.train.batch_size, shuffle=True)

    train(model, train_loader, device, args.train)


if __name__ == "__main__":
    args = parse_args()
    main(args)
