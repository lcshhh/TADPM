from pytorch3d.transforms import se3_exp_map
import numpy as np
import os
import pandas as pd
from glob import glob
import vedo
from pathlib import Path
import torch
from multiprocessing import Pool
from pytorch3d.transforms import *
import random
import scipy
import trimesh

def chamfer_loss(before_points,after_points,outputs):
    '''
    outputs: [bs,16,6]
    before_points: [bs,16,2048,3]
    '''
    bs = before_points.shape[0]
    loss = torch.FloatTensor([0]).cuda()
    for i in range(bs):
        trans_matrix = se3_exp_map(outputs[i]).transpose(1,2)      # [16,4,4]    
        riged_tar = Transform3d(matrix=trans_matrix.transpose(1,2)).transform_points(before_points[i])
        # N,P,C = before_points.shape
        # before_points_pre = torch.cat([before_points[i],torch.ones(N,P,1).cuda()],dim=-1).permute(0,2,1)
        # riged_tar = torch.bmm(trans_matrix,before_points_pre).permute(0,2,1)
        # riged_tar = riged_tar[:,:,:3]
        tmp,_ = chamfer_distance(after_points[i], riged_tar, point_reduction="sum", norm=1)
        loss += tmp
    return loss/bs

# def transform_mesh(mesh,dof):
#     trans_matrix = se3_exp_map(dof.unsqueeze(0)).transpose(1,2)[0]
#     mesh.apply_transform(trans_matrix)
#     return mesh

def transform_mesh(mesh,dof):
    '''
    centroid [bs,16,3]
    '''
    bs = dof.shape[0]
    trans_matrix = se3_exp_map(dof.unsqueeze(0)).transpose(2,1) # [16,4,4]
    vertices = torch.from_numpy(mesh.vertices).to(dof.device).float()
    predicted_vertices = Transform3d(matrix=trans_matrix.transpose(2,1)[0]).transform_points(vertices)
    mesh.vertices = predicted_vertices
    return mesh

def merge_mesh(index):
    meshes = []
    for i in range(32):
        path = f'/data/lcs/dataset/created/single_normed_before/{index}_{i}.obj'
        if os.path.exists(path):
            mesh = vedo.Mesh(path)
            meshes.append(mesh)
    mesh = vedo.merge(meshes)
    vedo.write(mesh,f'/data/lcs/dataset/{index}.obj')

paramroot = '/data3/leics/dataset/mesh/param'

# dof = torch.load(os.path.join(paramroot,f'1014.pkl')).float()
# meshes = []
# before_meshes = []
# for i in range(32):
#     path = f'/data3/leics/dataset/mesh/single_before/1014_{i}.obj'
#     if not os.path.exists(path):
#         continue
#     mesh = trimesh.load_mesh(f'/data3/leics/dataset/mesh/single_before/1014_{i}.obj')
#     before_mesh = trimesh.load_mesh(path)
#     predicetd_mesh = transform_mesh(mesh,dof[i])
#     meshes.append(vedo.trimesh2vedo(predicetd_mesh))
#     before_meshes.append(vedo.trimesh2vedo(before_mesh))
#     # predicetd_mesh.export('/data/lcs/dataset/after.obj')
#     # gt_mesh = trimesh.load_mesh('/data/lcs/dataset/created/single_normed_after/2_3.obj')
#     # gt_mesh.export('/data/lcs/dataset/gt.obj')
# mesh = vedo.merge(meshes)
# before_mesh = vedo.merge(before_meshes)
# vedo.write(mesh,'/data3/leics/dataset/mesh/after.obj')
# vedo.write(before_mesh,'/data3/leics/dataset/mesh/before.obj')
# merge_mesh(177)
path = Path(paramroot)
for pkl in path.iterdir():
    dof = torch.load(os.path.join(paramroot,f'1014.pkl')).float()
    print(dof)