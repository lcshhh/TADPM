import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from chamfer_dist import ChamferDistanceL1
import copy
from collections import OrderedDict
from models.diffusion import diffuse
import numpy as np
import math
from functools import partial
from models.mesh_mae import Mesh_encoder
import torch
import timm.models.vision_transformer
from models.pointnet import PointNetEncoder
from timm.models.vision_transformer import PatchEmbed, Block
class TADPM(nn.Module):
    def __init__(self,args):
        super(TADPM, self).__init__()
        self.encoder = Mesh_encoder(
            decoder_embed_dim=args.decoder_dim,
            masking_ratio=args.mask_ratio,
            encoder_depth=args.encoder_depth,
            num_heads=args.heads,
            channels=args.channels,
            patch_size=args.patch_size,
            embed_dim=args.dim,
            decoder_num_heads=args.decoder_num_heads,
            decoder_depth=args.decoder_depth
        )
        self.regressor = diffuse(1827)
        embed_dim = args.dim
        self.embed_dim = embed_dim
        self.use_pointnet = args.use_pointnet
        self.global_encoder = PointNetEncoder()
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, embed_dim+32))
        self.centroid_embedding = nn.Linear(3,32)
        self.ln = nn.LayerNorm(1827)
        self.bn1 = nn.BatchNorm1d(800)
        self.bn2 = nn.BatchNorm1d(1827)
        self.blocks = nn.ModuleList([
            Block(embed_dim+32, 8, qkv_bias=True)
            for i in range(12)])
        self.initialze_weights()
        if args.encoder_checkpoint != '':
            checkpoint = torch.load(args.encoder_checkpoint)['model']
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            self.encoder.load_state_dict(new_state_dict, strict=False)
    
    def initialze_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)
            if isinstance(layer, nn.Conv2d):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Conv1d):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self,faces, feats, centers, Fs, cordinates, centroid, points, gt_6dof=None):
        '''
        points [bs,32,2048,3]
        '''
        bs = faces.shape[0]
        n = faces.shape[1]
        encodings = []
        for i in range(n):
            encoding = self.encoder(faces[:,i],feats[:,i],centers[:,i],Fs[:,i],cordinates[:,i])
            encodings.append(encoding)
        embedding = torch.stack(encodings,dim=1)  #[bs,32,768]
        embedding = torch.cat([self.centroid_embedding(centroid),embedding],dim=-1)
        trans_feature = embedding + self.pos_embedding
        for blk in self.blocks:
            trans_feature = blk(trans_feature)
        global_points = rearrange(points,'b n p c -> b c (n p)')
        global_embedding = self.global_encoder(global_points).unsqueeze(1).repeat(1,n,1)
        embedding = torch.cat([global_embedding,trans_feature,centroid],dim=-1)
        embedding = self.ln(embedding)
        # embedding = torch.mean(embedding,dim=1)

        # embedding = self.bn2(embedding.transpose(1,2)).transpose(1,2)
        # embedding = embedding.mean(dim=1)
        dofs = self.regressor(gt_6dof,embedding)
        return dofs

 