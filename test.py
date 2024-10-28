# from models import PointNetPlusEncoder
import torch
# import trimesh
# from models import UNet3D
# with open('valid.txt') as f:
#     indexes = [int(i.strip()) for i in f.readlines()]
# root = '/data3/leics/dataset/type2/single_before'
# from utils import dist_utils, misc
# from models.pvcnn2_ada import create_mlp_components
# from models.voxelization import Voxelization
# v = Voxelization(1,False)
# misc.set_random_seed(42)
# features = torch.randn(32,25,50).cuda()
# cords = torch.randn(32,3,50).cuda()/1000 + 0.5
# agg_features, norm_cords = v(features,cords)
# features = features.mean(dim=-1)
# agg_features = agg_features.flatten(start_dim=-4,end_dim=-1)
# print(features[2,10])
# print(agg_features[2,10])

# from utils.pointcloud import write_pointcloud
# mesh = trimesh.load_mesh('/data3/leics/dataset/type2/single_before/0_2.obj')
# points = trimesh.sample.volume_mesh(mesh,8192)
# write_pointcloud(points,'/data3/leics/dataset/sample.ply')
# print(points.shape)
# from models.pointnet_plus_encoder import PointNetEncoder
# x = torch.randn((32,3,20,20,20),dtype=torch.float32).cuda()
# model = PointNetEncoder()
# output = model(x)
# print(output.shape)

import torch
from vector_quantize_pytorch import LFQ,ResidualVQ

quantizer = ResidualVQ(
            dim = 128,
            codebook_size = 4096,
            num_quantizers = 4,
            kmeans_init = True,   # set to True
            kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
 )

seq = torch.randn(32, 16, 128)
quantized, indices, loss = quantizer(seq)
print(loss)
