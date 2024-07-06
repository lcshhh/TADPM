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

class AxisPredictor(nn.Module):
    def __init__(self,args):
        super(AxisPredictor, self).__init__()
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
        self.regressor = nn.Sequential(
                nn.Linear(768, 512),
                # nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(512, 256),
                # nn.Dropout(0.3),
                nn.GELU(),
                nn.Linear(256,128),
                nn.GELU(),
                nn.Linear(128, 8)
        )
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
                nn.init.zeros_(layer.bias)
            if isinstance(layer, nn.Conv2d):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Conv1d):
                # 使用 Xavier 初始化
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self,faces, feats, centers, Fs, cordinates):
        '''
        points [bs,32,2048,3]
        '''
        bs = faces.shape[0]
        n = faces.shape[1]
        encoding = self.encoder(faces,feats,centers,Fs,cordinates)
        axis = self.regressor(encoding)
        return axis