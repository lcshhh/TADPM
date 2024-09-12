# from loguru import logger
# from pathlib import Path
# import random
# with open('train.txt') as f:
#     train_indexes = [int(i.strip()) for i in f.readlines()]

# with open('test.txt') as f:
#     test_indexes = [int(i.strip()) for i in f.readlines()]
# dataroot = '/data3/leics/dataset/teeth_full/single_normed_after_pointcloud513'
# dataroot = Path(dataroot)
# mesh_paths = []
# for tooth in dataroot.iterdir():
#     mesh_paths.append(tooth)
        
# random.shuffle(mesh_paths)
# split_point = int(len(mesh_paths) * 0.9)
# train_objs = mesh_paths[:split_point]
# test_objs = mesh_paths[split_point:]
# with open('train.txt','w') as f:
#     for obj in train_objs:
#         f.write(str(obj)+'\n')
# with open('test.txt','w') as f:
#     for obj in test_objs:
#         f.write(str(obj)+'\n')
from einops import rearrange
from pytorch3d.transforms import euler_angles_to_matrix
import torch
import open3d as o3d
import numpy as np
import os
import random

# def read_pointcloud(path):
#     pcd = o3d.io.read_point_cloud(path)
#     xyz = np.asarray(pcd.points)
#     return torch.from_numpy(xyz)

# def transform_vertices(vertices,centroids,dofs):
#     '''
#     vertices: [bs, 32, pt_num, 3]
#     centroids: [bs, 32, 3]
#     dofs: [bs, 32, 6]
#     '''
#     angles = rearrange(dofs[:,:,3:]*torch.pi/6,'b n c -> (b n) c')
#     move = rearrange(dofs[:,:,:3]/5,'b n c -> (b n) c').unsqueeze(1) #[b*n,1,3]
#     centroids = rearrange(centroids, 'b n c -> (b n) c').unsqueeze(1)
#     R = euler_angles_to_matrix(angles,'XYZ')
#     vertices = rearrange(vertices,'b n pn c -> (b n) pn c')
#     vertices = torch.bmm(vertices - centroids,R) + centroids + move
#     return vertices

# vertices = read_pointcloud('/data/lcs/dataset/teeth_full/pointcloud_before512/1178_21.ply')
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(vertices)
# o3d.io.write_point_cloud('/data/lcs/dataset/homework/before.ply', pcd)
# transform_vertices(vertices)
# with open('valid.txt') as f:
#     indexes = [int(i.strip()) for i in f.readlines()]
# random.shuffle(indexes)
# train_ratio = 0.9
# split_point = int(len(indexes) * train_ratio)
# train_objs = indexes[:split_point]
# test_objs = indexes[split_point:]
# with open('train.txt','w') as f:
#     for index in train_objs:
#         f.write(str(index)+'\n')

# with open('test.txt','w') as f:
#     for index in test_objs:
#         f.write(str(index)+'\n')
def create_attn_mask(masks):
    attn_mask = torch.mm(masks.unsqueeze(1),masks.unsqueeze(0))
    return attn_mask == 0
