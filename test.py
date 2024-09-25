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
import torch.nn as nn
import random

# tooth = '/data3/leics/dataset/teeth_full/single_normed_after'
# index = 161
# axis = f'/data3/leics/dataset/teeth_full/single_normed_after_axis/{index}.npy'
# outputpath = f'/data3/leics/dataset/tmp'
# os.system(f'cp {axis} {outputpath}')
# for i in range(32):
#     tooth_path = os.path.join(tooth,f'{index}_{i}.obj')
#     if os.path.exists(tooth_path):
#         os.system(f'cp {tooth_path} {outputpath}')
import trimesh
with open('valid.txt') as f:
    indexes = [int(i.strip()) for i in f.readlines()]
num = 0
for index in indexes:
    for i in range(16):
        mesh_path = f'/data3/leics/dataset/teeth_full/single_normed_after/{index}_{i}.obj'
        if os.path.exists(mesh_path):
            mesh = trimesh.load_mesh(mesh_path)
            if mesh.centroid[2] < 0:
                print(index)
                num+=1
                break
print(num)
    


