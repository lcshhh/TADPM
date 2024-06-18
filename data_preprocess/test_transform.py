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

mesh = trimesh.load_mesh('/data/lcs/dataset/created/single_normed_before/2_3.obj')
mesh.export('/data/lcs/dataset/before.obj')
paramroot = Path('/data/lcs/dataset/created/params')
dof = torch.load(os.path.join(paramroot,f'6dof_{2}.pkl')).float()
predicetd_mesh = transform_mesh(mesh,dof[3])
predicetd_mesh.export('/data/lcs/dataset/after.obj')
gt_mesh = trimesh.load_mesh('/data/lcs/dataset/created/single_normed_after/2_3.obj')
gt_mesh.export('/data/lcs/dataset/gt.obj')
# merge_mesh(177)