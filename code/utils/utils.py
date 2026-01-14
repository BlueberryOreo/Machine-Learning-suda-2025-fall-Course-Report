
import os
import importlib
import torch

def load_model(model_name: str):
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
