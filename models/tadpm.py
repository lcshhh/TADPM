import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from chamfer_dist import ChamferDistanceL1
import copy
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
        self.encoder.requires_grad_(False)
        self.use_mlp = args.use_mlp
        if self.use_mlp:
            # self.regressor = nn.Sequential(
            #     nn.Linear(1827, 512),
            #     nn.Dropout(0.3),
            #     nn.GELU(),
            #     nn.Linear(512, 256),
            #     nn.Dropout(0.3),
            #     nn.GELU(),
            #     nn.Linear(256, 9)
            # )
            self.regressor = nn.Sequential(
                nn.Linear(1827, 1024),
                nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(512, 32*9)
            )
        else:
            self.regressor = diffuse(1888,use_ae=args.use_ae)
        embed_dim = args.dim
        self.embed_dim = embed_dim
        self.use_pointnet = args.use_pointnet
        self.global_encoder = PointNetEncoder()
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, embed_dim+32))
        self.centroid_embedding = nn.Linear(3,32)
        # self.ln = nn.LayerNorm(1024)
        self.ln = nn.LayerNorm(1827)
        self.bn1 = nn.BatchNorm1d(800)
        self.bn2 = nn.BatchNorm1d(1827)
        # self.ln = nn.BatchNorm1d(1827)
        self.blocks = nn.ModuleList([
            Block(embed_dim+32, 8, qkv_bias=True, drop_path=0.1)
            for i in range(12)])
        self.initialze_weights()
        if args.encoder_checkpoint != '':
            self.encoder.load_state_dict(torch.load(args.encoder_checkpoint), strict=False)
    
    def initialze_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
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
        embedding = self.bn1(embedding.transpose(1,2)).transpose(1,2)
        trans_feature = embedding + self.pos_embedding
        for blk in self.blocks:
            trans_feature = blk(trans_feature)
        trans_feature = embedding
        global_points = rearrange(points,'b n p c -> b c (n p)')
        global_embedding = self.global_encoder(global_points).unsqueeze(1).repeat(1,n,1)
        # global_embedding = self.ln(global_embedding)
        embedding = torch.cat([global_embedding,trans_feature,centroid],dim=-1)
        # embedding = self.ln(embedding)
        # embedding = self.bn2(embedding.transpose(1,2)).transpose(1,2)
        embedding = embedding.mean(dim=1)
        embedding = self.bn2(embedding)
        if True:
            dofs = self.regressor(embedding).view(bs,n,-1)
        return dofs