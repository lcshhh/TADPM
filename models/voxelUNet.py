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
from models.uvit import Block
from vector_quantize_pytorch import FSQ, LFQ, ResidualVQ

def patchify(features,resolution):
    return rearrange(features,'b c (n1 n2 n3) (p1 p2 p3) -> b c (n1 p1) (n2 p2) (n3 p3)',n1=2,n2=2,n3=8,p1=resolution,p2=resolution,p3=resolution)

def unpatchify(features,resolution):
    return rearrange(features,'b c (n1 p1) (n2 p2) (n3 p3) -> b (n1 n2 n3) (p1 p2 p3 c)',n1=2,n2=2,n3=8,p1=resolution,p2=resolution,p3=resolution)

class MLP(nn.Module):
    def __init__(self, final_dim, npoint):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(final_dim,final_dim),
            nn.GELU(),
            nn.Linear(final_dim,final_dim),
            nn.GELU(),
            nn.Linear(final_dim,npoint*3)
        )
    
    def forward(self, x):
        return self.model(x)


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
        if self.resolution==3:
            self.unet_encoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False,dropout_prob=0.7,num_levels=3)
        else:
            self.unet_encoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False,dropout_prob=0.7,num_levels=4)
        self.npoint = config.npoint
        # self.bn1 = nn.BatchNorm3d(config.zdim)
        # self.bn2 = nn.BatchNorm3d(config.zdim)
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
        if self.resolution == 3:
            self.unet_decoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False,dropout_prob=0.7,num_levels=3)
        else:
            self.unet_decoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False,dropout_prob=0.7,num_levels=4)
        self.final_dim = (self.resolution ** 3) * config.zdim
        self.mlps = nn.ModuleList([nn.Linear(self.final_dim,self.npoint*3) for _ in range(32)])
        # self.mlps = nn.ModuleList([MLP(self.final_dim,self.npoint) for _ in range(32)])
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
        decoded = unpatchify(self.unet_decoder(z),self.resolution)
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


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_dense = config.num_dense
        self.latent_dim = config.latent_dim
        self.grid_size = config.grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)
    

    def forward(self, feature_global):
        B = feature_global.shape[0]
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        # print('coarse',coarse.shape)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense).to(point_feat.device)                                          # (B, 2, num_fine)
        

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
        
    
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return fine.transpose(1, 2).contiguous()


@MODELS.register_module()
class VoxelPCNUNet(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, config):
        super(VoxelPCNUNet, self).__init__()
        self.resolution = config.resolution
        self.local_encoders = nn.ModuleList([PointNetEncoder(config.local_encoder) for _ in range(32)])
        self.unet_encoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False,num_levels=3)
        self.npoint = config.npoint
        self.quantizer = LFQ(
            codebook_size = 4096,      # codebook size, must be a power of 2
            dim = config.zdim,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = 1.,  # how much weight to place on entropy loss
            diversity_gamma = 1.,        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
            experimental_softplus_entropy_loss=True
        )
        self.unet_decoder = ResidualUNetSE3D(config.zdim,config.zdim,final_sigmoid=False,is_segmentation=False,num_levels=3)
        self.final_dim = (self.resolution ** 3) * config.zdim
        # self.mlps = nn.ModuleList([nn.Linear(self.final_dim,self.npoint*3) for _ in range(32)])
        self.generator = nn.ModuleList([PCN(config) for _ in range(32)])
        self.mask_predictor = nn.ModuleList([nn.Linear(self.final_dim,1) for _ in range(32)])

    def forward(self, points):
        """
        points: [B, N, P, 3]
        """
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        features = patchify(features,self.resolution)
        # features = self.bn1(features)
        z = self.unet_encoder(features)
        z = rearrange(z,'b c w h d -> b (w h d) c')
        z, indices, entropy_aux_loss = self.quantizer(z) 
        z = rearrange(z,'b (w h d) c -> b c w h d',w=2*self.resolution,h=2*self.resolution,d=8*self.resolution)
        # entropy_aux_loss = torch.FloatTensor([0.]).cuda()
        # z = self.bn2(features)
        decoded = unpatchify(self.unet_decoder(z),self.resolution)
        # rec = self.mlp(decoded)
        rec = torch.stack([self.generator[i](decoded[:,i]) for i in range(32)],dim=1)
        masks = torch.stack([self.mask_predictor[i](decoded[:,i]) for i in range(32)],dim=1).squeeze(2)
        # rec = rearrange(rec,'b n (p c)-> b n p c',c=3)
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


