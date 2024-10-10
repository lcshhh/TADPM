# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch.nn as nn 
from loguru import logger 
from models.pvcnn2 import create_pointnet2_sa_components 
from utils.builder import MODELS 
from models import *
import torch

# implement the global encoder for VAE model 

class PointNetPlusEncoder(nn.Module):
    sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 64]]]
        ]
    force_att = 0 # add attention to all layers  
    # def __init__(self, zdim, input_dim, extra_feature_channels=0, args={}):
    def __init__(self, config):
        super().__init__()
        sa_blocks = self.sa_blocks 
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=config.input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp = nn.Linear(channels_sa_features, config.zdim) 
        self.zdim = config.zdim 
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]

    def forward(self, x):
        """
        Args: 
            x: B,N,3 
        Returns: 
            mu, sigma: B,D
        """
        # output = {} 
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
       
        features = features.max(-1)[0]
        features = self.mlp(features)

        return features


@MODELS.register_module()
class PointNetEncoder(nn.Module):
    sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
        [[32, 1, 16], [256, 0.2, 32, [32, 64]]]
        ]
    force_att = 0 # add attention to all layers  
    # def __init__(self, zdim, input_dim, extra_feature_channels=0, args={}):
    def __init__(self, config):
        super().__init__()
        sa_blocks = self.sa_blocks 
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels=0, input_dim=config.input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        self.mlp = nn.Linear(channels_sa_features, config.zdim) 
        self.zdim = config.zdim 
        # logger.info('[Encoder] zdim={}, out_sigma={}; force_att: {}', config.zdim, True, self.force_att) 
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]

        self.num_dense = config.num_dense

        self.mlp2 = nn.Sequential(
            nn.Linear(config.zdim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_dense)
        )

    def forward(self, x):
        """
        Args: 
            x: B,N,3 
        Returns: 
            mu, sigma: B,D
        """
        # output = {} 
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
        features = features.max(-1)[0]
        features = self.mlp(features)

        ## decode
        B = features.shape[0]
        outputs = self.mlp2(features).reshape(-1, self.num_dense, 3)                    # (B, num_coarse, 3), coarse point cloud
        # print('coarse',coarse.shape)
        return outputs
    
    def encode(self,x):
        x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer_id, layer in enumerate(self.layers):
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
       
        features = features.max(-1)[0]
        features = self.mlp(features)
        return features

    def decode(self,features):
        B = features.shape[0]
        outputs = self.mlp2(features).reshape(-1, self.num_dense, 3)                    # (B, num_coarse, 3), coarse point cloud
        # print('coarse',coarse.shape)
        return outputs

