import torch
import numpy as np 
from loguru import logger
import importlib
import torch.nn as nn  
from models.distributions import Normal
from utils import utils as helper 
from utils.builder import MODELS 
from einops import rearrange, repeat
from models import *
from models.voxelization import Voxelization
from vector_quantize_pytorch import FSQ, LFQ
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


def patchify(features):
    return rearrange(features,'b c (n1 n2 n3) (p1 p2 p3) -> (n1 p1) (n2 p2) (n3 p3) b c',n1=2,n2=2,n3=8,p1=2,p2=2,p3=2).flatten(0,-3)

def unpatchify(features):
    return rearrange(features,'b (n1 p1 n2 p2 n3 p3) c -> b (n1 n2 n3) (p1 p2 p3 c)',n1=2,n2=2,n3=8,p1=2,p2=2,p3=2)

@MODELS.register_module()
class TeethMAE(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, config):
        super(TeethMAE, self).__init__()
        self.resolution = 2
        self.local_encoders = nn.ModuleList([PointNetEncoder(config.local_encoder) for _ in range(32)])
        self.shuffle = PatchShuffle(0.3)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(256, 1, 512))
        self.decoder_pos_embedding = torch.nn.Parameter(torch.zeros(1, 256, 512))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 512))
        # self.unet_encoder = ResidualUNetSE3D(128,32,final_sigmoid=False,is_segmentation=False)
        self.encoder = torch.nn.Sequential(*[Block(512, 8) for _ in range(24)])
        self.decoder = torch.nn.Sequential(*[Block(512, 8) for _ in range(24)])
        self.layer_norm = torch.nn.LayerNorm(512)
        self.quantizer = LFQ(
            codebook_size = 4096,      # codebook size, must be a power of 2
            dim = 512,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = 1.,  # how much weight to place on entropy loss
            diversity_gamma = 1.,        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
            experimental_softplus_entropy_loss=True
        )
        # self.unet_decoder = ResidualUNetSE3D(32,3,final_sigmoid=False,is_segmentation=False)
        self.final_dim = (self.resolution ** 3) * 512
        self.mlp = nn.Sequential(
            nn.Linear(self.final_dim,512*3),
        )
        trunc_normal_(self.decoder_pos_embedding, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, points):
        """
        points: [B, N, P, 3]
        """
        ## encode
        B,N,P,_ = points.shape
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        patches = patchify(features)
        patches = patches + self.pos_embedding
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.encoder(patches))
        patches = rearrange(patches, 'b t c -> t b c')

        ## pad
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = rearrange(features, 't b c -> b t c')
        ## vq
        features, indices, entropy_aux_loss = self.quantizer(features) 

        ##decode
        features = features + self.decoder_pos_embedding
        features = self.decoder(features)
        features = unpatchify(features)
        rec = self.mlp(features).reshape(B,N,P,-1)
        return rec, entropy_aux_loss

    def reconstruct(self, points):
        B,N,P,_ = points.shape
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        patches = patchify(features)
        patches = patches + self.pos_embedding
        # patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.encoder(patches))

        ## pad
        # features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        # features = take_indexes(features, backward_indexes)
        # features = rearrange(features, 't b c -> b t c')

        ## vq
        features, indices, entropy_aux_loss = self.quantizer(features) 

        ##decode
        features = features + self.decoder_pos_embedding
        features = self.decoder(features)
        features = unpatchify(features)
        rec = self.mlp(features).reshape(B,N,P,-1)
        return rec, entropy_aux_loss

    
    def encode(self, points):
        features = torch.stack([self.local_encoders[i].encode(points[:,i]) for i in range(32)],dim=2)  # [B,C,N,P]
        features = patchify(features)
        z = self.unet_encoder(features)
        return z

    def decode(self,z):
        z, indices, entropy_aux_loss = self.quantizer(z) 
        decoded = unpatchify(self.unet_decoder(z))
        rec = self.mlp(decoded)
        rec = rearrange(rec,'b c n p-> b n p c')
        return rec

