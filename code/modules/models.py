import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.components import Encoder, DecoderSCVI


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        