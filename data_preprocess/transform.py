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
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# def transform_mesh(vtp_path,outputroot):
#     num = vtp_path.name.split('_')[0]
#     mesh = vedo.Mesh(str(vtp_path))
#     labels = mesh.celldata['Label']
#     dofs = torch.normal(mean=0.0,std=0.01,size=(32,6))
#     trans = se3_exp_map(dofs).transpose(1,2)
#     teeth = []
#     for i in range(32):
#         if i in labels:
#             tmp = vedo.vedo2trimesh(mesh).submesh([np.where(labels==i)[0]], append=True)
#             tmp = vedo.trimesh2vedo(tmp)
#             tmp.apply_transform(trans[i])
#             tmp.celldata['Label'] = torch.ones(len(tmp.faces()))*i
#             teeth.append(tmp)
#     after_mesh =  vedo.merge(teeth)
#     vedo.write(after_mesh,os.path.join(outputroot,f'{num}.obj'))
    # mesh2 = vedo.Mesh(str(origin_path))
    # vedo.write(mesh2,os.path.join(outputroot,f'{num}_after_upper.vtp'))

def paste(index,dataroot,outputroot,num):
    before_mesh_lower = vedo.Mesh(os.path.join(dataroot,f'{index}_before_lower.vtp'))
    after_mesh_lower = vedo.Mesh(os.path.join(dataroot,f'{index}_after_lower.vtp'))
    before_mesh_upper = vedo.Mesh(os.path.join(dataroot,f'{index}_before_upper.vtp'))
    after_mesh_upper = vedo.Mesh(os.path.join(dataroot,f'{index}_after_upper.vtp'))
    vedo.write(before_mesh_lower,os.path.join(outputroot,f'{num}_before_lower.vtp'))
    vedo.write(after_mesh_lower,os.path.join(outputroot,f'{num}_after_lower.vtp'))
    vedo.write(before_mesh_upper,os.path.join(outputroot,f'{num}_before_upper.vtp'))
    vedo.write(after_mesh_upper,os.path.join(outputroot,f'{num}_after_upper.vtp'))
    print(num)

def  get_mesh(dataroot,outputroot_before,outputroot_after,paramroot,index,num):
    seed_torch(num)
    dofs = torch.cat([torch.normal(mean=0.,std=0.02,size=(32,3)),torch.normal(mean=0.,std=0.05,size=(32,3))],dim=1)
    trans = se3_exp_map(dofs).transpose(1,2)
    ran = range(32)
    for i in ran:
        path = os.path.join(dataroot,f'{index}_{i}.obj')
        if os.path.exists(path):
            mesh = trimesh.load_mesh(path)
            mesh.export(os.path.join(outputroot_after,f'{num}_{i}.obj'))
            mesh.apply_transform(trans[i])
            mesh.export(os.path.join(outputroot_before,f'{num}_{i}.obj'))
    trans_matrix = torch.inverse(trans)
    torch.save(trans_matrix,os.path.join(paramroot,f'matrix_{num}.pkl'))
    dofs_gt = se3_log_map(trans_matrix.transpose(1,2))
    torch.save(dofs_gt,os.path.join(paramroot,f'{num}.pkl'))

def get_mesh2(dataroot,outputroot_before,outputroot_after,paramroot,index,num):
    seed_torch(num)
    dofs = torch.cat([torch.normal(mean=0.,std=0.02,size=(32,3)),torch.normal(mean=0.,std=0.05,size=(32,3))],dim=1)
    trans = se3_exp_map(dofs).transpose(1,2)
    ran = range(32)
    for i in ran:
        path = os.path.join(dataroot,f'{index}_{i}.obj')
        remesh_path = os.path.join('',f'')
        if os.path.exists(path):
            mesh = trimesh.load_mesh(path)
            mesh.export(os.path.join(outputroot_after,f'{num}_{i}.obj'))
            mesh.apply_transform(trans[i])
            mesh.export(os.path.join(outputroot_before,f'{num}_{i}.obj'))
    trans_matrix = torch.inverse(trans)
    torch.save(trans_matrix,os.path.join(paramroot,f'matrix_{num}.pkl'))
    dofs_gt = se3_log_map(trans_matrix.transpose(1,2))
    torch.save(dofs_gt,os.path.join(paramroot,f'{num}.pkl'))
    

# dataroot = Path('/data/lcs/first_upper/upper_centered_normed')
# outputroot = Path('/data/lcs/first_upper/transformed_centered_normed')
# os.makedirs(outputroot,exist_ok=True)
# # for obj in dataroot.iterdir():
# n_variance = 10
# for _ in range(n_variance):
#     pool = Pool(processes=64)
#     num = len(glob(os.path.join(outputroot,f'*.vtp')))//2
#     for i,index in enumerate(useful_lst):
#         obj = os.path.join(dataroot,f'{index}_before_upper.vtp')
#         origin_path = os.path.join(dataroot,f'{index}_after_upper.vtp')
#         pool.apply_async(
#             transform_mesh,
#             (obj,origin_path,outputroot,num+i)
#         )
#     pool.close()
#     pool.join()
# pool = Pool(processes=64)
# num = len(glob(os.path.join(outputroot,f'*.vtp')))//4
# for i,index in enumerate(useful_lst):
#         pool.apply_async(
#             paste,
#             (index,dataroot,outputroot,num+i)
#         )
# pool.close()
# pool.join()
dataroot = Path('/data3/leics/dataset/mesh/single_before')
outputroot_before = Path('/data3/leics/dataset/created/single_before')
os.makedirs(outputroot_before,exist_ok=True)
outputroot_after = Path('/data3/leics/dataset/created/single_after')
os.makedirs(outputroot_after,exist_ok=True)
paramroot = Path('/data3/leics/dataset/created/params')
os.makedirs(paramroot,exist_ok=True)
with open('train.txt') as f:
     indexes = [int(i.strip()) for i in f.readlines()]
pool = Pool(processes=64)
n_variance = 10
for i in range(n_variance):
# if True:
    # num = len(glob(os.path.join(outputroot,f'*.vtp')))//2
    for j,index in enumerate(indexes):
            # obj = os.path.join(dataroot,f'{i}_after_lower.vtp')
            num = i * len(indexes) + j
            pool.apply_async(
                get_mesh,
                (dataroot,outputroot_before,outputroot_after,paramroot,index,num)
            )
pool.close()
pool.join()