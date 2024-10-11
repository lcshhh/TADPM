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

def patchify(features):
    return rearrange(features,'b c (n1 n2 n3) (p1 p2 p3) -> b c (n1 p1) (n2 p2) (n3 p3)',n1=2,n2=2,p1=4,p2=4).flatten(2).transpose(1,2)

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
        self.local_encoders = nn.ModuleList([PointNetEncoder(config.local_encoder) for _ in range(32)])
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
        features = torch.stack([self.local_encoders.encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        features = patchify(features)
        print(features.shape)
        z = self.unet_encoder(features)
        decoded = self.unet_decoder(z)
        decoded =  rearrange(features,'b c (n1 r1) (n2 r2) (n3 r3) -> b c (n1 n2 n3) (r1 r2 r3)',n1=2,n2=2,n3=8)
        rec = self.mlp(decoded)
        rec = rearrange(rec,'b c n p-> b n p c')
        return rec
