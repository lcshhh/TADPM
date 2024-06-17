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
def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def transform_mesh(vtp_path,outputroot):
    num = vtp_path.name.split('_')[0]
    mesh = vedo.Mesh(str(vtp_path))
    labels = mesh.celldata['Label']
    dofs = torch.normal(mean=0.0,std=0.01,size=(32,6))
    trans = se3_exp_map(dofs).transpose(1,2)
    teeth = []
    for i in range(32):
        if i in labels:
            tmp = vedo.vedo2trimesh(mesh).submesh([np.where(labels==i)[0]], append=True)
            tmp = vedo.trimesh2vedo(tmp)
            tmp.apply_transform(trans[i])
            tmp.celldata['Label'] = torch.ones(len(tmp.faces()))*i
            teeth.append(tmp)
    after_mesh =  vedo.merge(teeth)
    vedo.write(after_mesh,os.path.join(outputroot,f'{num}.obj'))
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

def get_mesh(obj_path,outputroot,paramroot,num,upper=True):
    seed_torch(num)
    mesh = vedo.Mesh(str(obj_path))
    labels = mesh.celldata['Label']
    dofs = torch.cat([torch.normal(mean=0.,std=0.02,size=(16,3)),torch.normal(mean=0.,std=0.05,size=(16,3))],dim=1)
    # dofs = torch.cat([torch.normal(mean=0.0,std=0.02,size=(16,3)),torch.normal(mean=0.0,std=0.05,size=(16,1)),torch.zeros(16,1),torch.normal(mean=0.0,std=0.05,size=(16,1))],dim=1)
    trans = se3_exp_map(dofs).transpose(1,2)
    teeth = []
    ran = range(16) if upper else range(16,32)
    for i in ran:
        if i in labels:
            tmp = vedo.vedo2trimesh(mesh).submesh([np.where(labels==i)[0]], append=True)
            tmp = vedo.trimesh2vedo(tmp)
            if upper:
                tmp.apply_transform(trans[i])
            else:
                tmp.apply_transform(trans[i-16])
            tmp.celldata['Label'] = torch.ones(len(tmp.faces()))*i
            teeth.append(tmp)
    after_mesh =  vedo.merge(teeth)
    vedo.write(after_mesh,os.path.join(outputroot,f'{num}_before_lower.vtp'))
    vedo.write(mesh,os.path.join(outputroot,f'{num}_after_lower.vtp'))
    trans_matrix = torch.inverse(trans)
    torch.save(trans_matrix,os.path.join(paramroot,f'matrix_{num}.pkl'))
    dofs_gt = se3_log_map(trans_matrix.transpose(1,2))
    torch.save(dofs_gt,os.path.join(paramroot,f'6dof_{num}.pkl'))
    

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
dataroot = Path('/data/lcs/upper_jaw/after_without_gingiva')
outputroot = Path('/data/lcs/upper_jaw/simulated_before')
# paramroot = Path('/data/lcs/first_lower/trans_param_centered')
os.makedirs(outputroot,exist_ok=True)
# os.makedirs(paramroot,exist_ok=True)
# for obj in dataroot.iterdir():
n_variance = 10
# for _ in range(n_variance):
if True:
    pool = Pool(processes=64)
    # num = len(glob(os.path.join(outputroot,f'*.vtp')))//2
    for path in dataroot.iterdir():
            # obj = os.path.join(dataroot,f'{i}_after_lower.vtp')
            pool.apply_async(
                get_mesh,
                (path,outputroot,index)
            )
    pool.close()
    pool.join()