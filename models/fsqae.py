from tqdm.auto import trange

import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange
from utils.builder import MODELS 
levels=[8,6,5]
from vector_quantize_pytorch import FSQ

@MODELS.register_module()
class SimpleFSQAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, len(levels), kernel_size=1),
                FSQ(levels),
                nn.Conv2d(len(levels), 32, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            ]
        )
        return

    def forward(self, x):
        '''
        x [bs, n, p, c]
        '''
        x = rearrange(x,'b n p c -> b c n p')
        for layer in self.layers:
            if isinstance(layer, FSQ):
                x, indices = layer(x)
            else:
                x = layer(x)

        return x.clamp(-1, 1)

