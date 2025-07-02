import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from collections import OrderedDict
from models.diffusion import diffusion
import numpy as np
import math
from functools import partial
from models.MeshMAE import Mesh_encoder
import torch
import timm.models.vision_transformer
from models.pointnet import PointNetEncoder
from timm.models.vision_transformer import PatchEmbed, Block
from utils.builder import MODELS 
# from util import SharedMLP, LinearMLP

@MODELS.register_module()
class TADPM(nn.Module):
    def __init__(self,config):
        super(TADPM, self).__init__()
        self.encoder = Mesh_encoder()
        # self.encoder.requires_grad_(False)
        embed_dim = config.dim
        self.embed_dim = embed_dim
        self.global_encoder = PointNetEncoder()
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, embed_dim+32))
        self.centroid_embedding = nn.Linear(3,32)
        self.blocks = nn.ModuleList([
            Block(embed_dim+32, config.num_heads, qkv_bias=True)
            for i in range(config.depth)])
        final_dim = embed_dim + 32 + 1024 + 3
        self.ln = nn.LayerNorm(final_dim)
        self.regressor = diffusion(final_dim)
        self.initialze_weights()   
        if config.args.encoder_ckpts != '':
            checkpoint = torch.load(config.args.encoder_ckpts)['base_model']
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            self.encoder.load_state_dict(new_state_dict, strict=False)
    
    def initialze_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self,faces, feats, centers, Fs, cordinates, centroid, points, gt_6dof=None):
        '''
        points [bs,32,2048,3]
        '''
        n = faces.shape[1]
        encodings = []
        for i in range(n):
            encoding = self.encoder(faces[:,i],feats[:,i],centers[:,i],Fs[:,i],cordinates[:,i])
            encodings.append(encoding)
        embedding = torch.stack(encodings,dim=1)  #[bs,32,768]
        embedding = torch.cat([self.centroid_embedding(centroid),embedding],dim=-1) #[bs,32,800]
        trans_feature = embedding + self.pos_embedding
        for blk in self.blocks:
            trans_feature = blk(trans_feature)
        global_points = rearrange(points,'b n p c -> b c (n p)')
        global_embedding = self.global_encoder(global_points).unsqueeze(1).repeat(1,n,1) #[bs,32,1024]
        embedding = torch.cat([global_embedding,trans_feature,centroid],dim=-1)
        embedding = self.ln(embedding)
        dofs = self.regressor(gt_6dof,embedding)
        return dofs

 