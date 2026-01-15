
import os
import importlib
import torch
import random
import numpy as np


class LinearWarmupLambda:
    """
    Linear warm-up scheduler for contrastive loss weight lambda.

    Typical usage:
        lambda_scheduler = LinearWarmupLambda(
            lambda_max=0.1,
            warmup_epochs=20,
            ramp_epochs=30
        )

        for epoch in range(num_epochs):
            lambda_gcl = lambda_scheduler(epoch)
            loss = recon_loss + lambda_gcl * gcl_loss
    """
    def __init__(
        self,
        lambda_max: float,
        warmup_epochs: int,
        ramp_epochs: int,
        **kwargs,
    ):
        assert lambda_max >= 0
        assert warmup_epochs >= 0
        assert ramp_epochs > 0

        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            return self.lambda_max * progress
        else:
            return self.lambda_max


def load_model(model_name: str) -> torch.nn.Module:
    """
    Dynamically load a model class from modules.models by name.
    Args:
      model_name: Name of the model class to load (e.g., "VAE", "ZINBVAE")
    Returns:
      The model class.
    Raises:
      ValueError: If the model class is not found or is not a subclass of torch.nn.Module.
    """
    module = importlib.import_module("modules.models")
    if not hasattr(module, model_name):
        raise ValueError(f"Model {model_name} not found in modules.models")
    if not issubclass(getattr(module, model_name), torch.nn.Module):
        raise ValueError(f"{model_name} is not a subclass of torch.nn.Module")
    model_class = getattr(module, model_name)
    return model_class


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark=False


def parameter_count(model: torch.nn.Module) -> dict:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }