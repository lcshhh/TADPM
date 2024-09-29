from models import PointNetPlusEncoder
import torch
# with open('valid.txt') as f:
#     indexes = [int(i.strip()) for i in f.readlines()]
# root = '/data3/leics/dataset/type2/single_before'
from models.pvcnn2_ada import create_mlp_components
layers, _ = create_mlp_components(
                in_channels=channels_fp_features, 
                out_channels=[128, dropout, num_classes], # was 0.5
                classifier=True, dim=2, width_multiplier=width_multiplier,
                cfg=cfg)
a = torch.rand(12,4,3)
print(mlp(a).shape)
