
import os
from ruamel.yaml import YAML
import torch
import argparse
import datetime
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.clustering import run_cluster, visualize
from utils.dataset import H5adDataset
from utils.utils import load_model, seed_everything
from utils.evaluate import evaluate_clustering


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE/ZINBVAE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save trained model and logs")
    parser.add_argument("--eval", action='store_true', help="Whether to run evaluation after training")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training")
    args = parser.parse_args()

    assert os.path.exists(args.config), f"Config file {args.config} does not exist."
    if args.eval:
        assert args.resume is not None, "Evaluation requires a trained model checkpoint. Please provide --resume."
        assert os.path.exists(args.resume), f"Checkpoint file {args.resume} does not exist."

    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = YAML().load(f)
        for key, value in config.items():
            setattr(args, key, edict(value) if isinstance(value, dict) else value)
    
    # Create output directory with timestamp
    if args.resume:
        args.train.out_dir = os.path.dirname(args.resume)
    else:
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


def train(model, optimizer, train_loader, device, config):
    losses = []
    bar = tqdm(range(config.start_epoch, config.num_epochs + 1))

    for epoch in bar:
        avg_loss = train_epoch(model, train_loader, optimizer, device, config)
        bar.desc = f"Epoch {epoch}/{config.num_epochs} - Loss: {avg_loss:.4f}"
        losses.append(avg_loss)
        
        # Save model at intervals
        if epoch % config.save_interval == 0 and epoch < config.num_epochs:
            model_path = os.path.join(config.out_dir, f"model_epoch_{epoch}.pt")
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state_dict, model_path)
            # print(f"Saved model checkpoint at {model_path}")
    
    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state_dict, os.path.join(config.out_dir, "model_final.pt"))
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


@torch.no_grad()
def eval(model, dataset, device, config):
    eval_config = config.train
    cluster_config = config.cluster

    # Use cluster to eval the latent representations
    model.eval()

    # Inference all cells to get latent representations
    eval_loader = DataLoader(dataset, batch_size=eval_config.batch_size, shuffle=False)
    all_z = []

    for batch_idx, (data, type) in enumerate(tqdm(eval_loader, desc="Evaluating Batches", disable=False)):
        data = data.to(device)
        output = model(data)
        z = output["z"]
        all_z.append(z.detach())
    
    all_z = torch.cat(all_z, dim=0).cpu().numpy()
    dataset.adata.obsm["latent"] = all_z

    # Cluster
    cluster_config.n_clusters = len(set(dataset.cell_types)) if cluster_config.n_clusters == "auto" else cluster_config.n_clusters
    pred_labels = run_cluster(dataset.adata, config=cluster_config, use_rep="latent")
    if cluster_config.save_cluster_results:
        dataset.adata.obs["predicted_labels"] = pred_labels.astype(str)
        cluster_results_path = os.path.join(eval_config.out_dir, "cluster_results.csv")
        dataset.adata.obs.to_csv(cluster_results_path)
        print(f"Saved clustering results at {cluster_results_path}")
    
    # Evaluate clustering
    results = evaluate_clustering(true_labels=dataset.cell_types, pred_labels=pred_labels)
    print("Clustering Evaluation Results:", results)

    # Save visualization
    if cluster_config.visualize_clusters:
        print("Generating UMAP visualization of clusters...")
        output_path = os.path.join(eval_config.out_dir, f"{cluster_config.method}_umap_clusters.{cluster_config.file_type}")
        visualize(dataset.adata, output_path, method=cluster_config.method, use_rep="latent", legend_loc="on data", display_gt=cluster_config.display_gt)


def main(args):
    # Seed everything for reproducibility
    seed_everything(args.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and set up optimizer
    model = load_model(args.vae.model_name)(**args.vae)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        args.train.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed model from {args.resume}, epoch={checkpoint['epoch']}")
    else:
        args.train.start_epoch = 1
    
    model.to(device)

    # Load dataset
    dataset = H5adDataset(args.dataset.data_path)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.train.batch_size, 
        num_workers=args.train.num_workers, 
        shuffle=True
    )

    if args.eval:
        eval(model, dataset, device, args)
    else:
        train(model, optimizer, train_loader, device, args.train)
        eval(model, dataset, device, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
