import torch
import numpy as np 
from loguru import logger
import importlib
import torch.nn as nn  
from models.distributions import Normal
from utils import utils as helper 
from utils.builder import MODELS 
from einops import rearrange
from models import *
from models.voxelization import Voxelization
from vector_quantize_pytorch import FSQ

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(oc),
                nn.ReLU(True),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)

@MODELS.register_module()
class VoxelUNet(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, config):
        super(VoxelUNet, self).__init__()
        self.resolution = 6
        self.voxelization = Voxelization(self.resolution,normalize=False)
        self.unet_encoder = UNet3D(3,128)
        self.unet_decoder = UNet3D(128,3)
        self.final_dim = (self.resolution ** 3)
        # self.mlp = SharedMLP(self.final_dim,512,dim=2)
        self.mlp = nn.Sequential(
            nn.Linear(self.final_dim,512)
        )

    def forward(self, points):
        """
        points: [B, N, P, 3]
        """
        points = rearrange(points,'b n p c -> b n c p')
        features = torch.stack([self.voxelization(points[:,i],points[:,i])[0] for i in range(32)],dim=2) #[B,C,N,R,R,R]
        features = rearrange(features,'b c (n1 n2 n3) r1 r2 r3 -> b c (n1 r1) (n2 r2) (n3 r3)',n1=2,n2=2,n3=8)
        z = self.unet_encoder(features)
        decoded = self.unet_decoder(z)
        decoded =  rearrange(features,'b c (n1 r1) (n2 r2) (n3 r3) -> b c (n1 n2 n3) (r1 r2 r3)',n1=2,n2=2,n3=8)
        rec = self.mlp(decoded)
        rec = rearrange(rec,'b c n p-> b n p c')
        return rec
