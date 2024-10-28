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
from vector_quantize_pytorch import FSQ, LFQ, ResidualVQ

def patchify(features,resolution):
    return rearrange(features,'b c (n1 n2 n3) (p1 p2 p3) -> b c (n1 p1) (n2 p2) (n3 p3)',n1=2,n2=2,n3=8,p1=resolution,p2=resolution,p3=resolution)

def unpatchify(features,resolution):
    return rearrange(features,'b c (n1 p1) (n2 p2) (n3 p3) -> b (n1 n2 n3) (p1 p2 p3 c)',n1=2,n2=2,n3=8,p1=resolution,p2=resolution,p3=resolution)

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
        self.resolution = config.resolution
        self.local_encoders = nn.ModuleList([PointNetEncoder(config.local_encoder) for _ in range(32)])
        self.unet_encoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False)
        self.quantizer = LFQ(
            codebook_size = 4096,      # codebook size, must be a power of 2
            dim = config.zdim,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = 1.,  # how much weight to place on entropy loss
            diversity_gamma = 1.,        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
            experimental_softplus_entropy_loss=True
        )
        # self.quantizer = ResidualVQ(
        #     dim = 120,
        #     codebook_size = 4096,
        #     num_quantizers = 4,
        #     kmeans_init = True,   # set to True
        #     kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
        # )
        self.unet_decoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False)
        self.final_dim = (self.resolution ** 3) * config.zdim
        self.mlps = nn.ModuleList([nn.Linear(self.final_dim,512*3) for _ in range(32)])
        self.mask_predictor = nn.ModuleList([nn.Linear(self.final_dim,1) for _ in range(32)])

    def forward(self, points):
        """
        points: [B, N, P, 3]
        """
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        features = patchify(features,self.resolution)
        z = self.unet_encoder(features)
        z = rearrange(z,'b c w h d -> b (w h d) c')
        z, indices, entropy_aux_loss = self.quantizer(z) 
        z = rearrange(z,'b (w h d) c -> b c w h d',w=2*self.resolution,h=2*self.resolution,d=8*self.resolution)
        # entropy_aux_loss = torch.FloatTensor([0.]).cuda()
        decoded = unpatchify(self.unet_decoder(z),self.resolution)
        # rec = self.mlp(decoded)
        rec = torch.stack([self.mlps[i](decoded[:,i]) for i in range(32)],dim=1)
        masks = torch.stack([self.mask_predictor[i](decoded[:,i]) for i in range(32)],dim=1).squeeze(2)
        rec = rearrange(rec,'b n (p c)-> b n p c',c=3)
        return rec, entropy_aux_loss, masks
    
    def encode(self, points):
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        features = patchify(features,self.resolution)
        z = self.unet_encoder(features)
        return z

    def decode(self,z):
        z = rearrange(z,'b c w h d -> b (w h d) c')
        z, indices, entropy_aux_loss = self.quantizer(z) 
        z = rearrange(z,'b (w h d) c -> b c w h d',w=2*self.resolution,h=2*self.resolution,d=8*self.resolution)
        decoded = unpatchify(self.unet_decoder(z),self.resolution)
        rec = torch.stack([self.mlps[i](decoded[:,i]) for i in range(32)],dim=1)
        masks = torch.stack([self.mask_predictor[i](decoded[:,i]) for i in range(32)],dim=1).squeeze(2)
        rec = rearrange(rec,'b n (p c)-> b n p c',c=3)
        return rec, masks

