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
import copy
from scipy.spatial.transform import Rotation

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

def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-20,20) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1, rotation

def get_mesh(obj_path,outputroot,axis_outputroot,num):
    seed_torch(num)
    mesh = trimesh.load_mesh(obj_path)
    index = int(obj_path.name.split('_')[0])
    fdi = int(obj_path.name.split('_')[1].split('.')[0])
    rotated_mesh, rotation = randomize_mesh_orientation(mesh)
    rotated_mesh.export(os.path.join(outputroot,f'{num}.obj'))
    axis = np.load(f'/data3/leics/dataset/mesh/single_after_axis_revert/{index}.npy')[fdi]
    axis = np.reshape(axis,(3,3))
    after_axis = rotation.apply(axis)
    np.save(os.path.join(axis_outputroot,f'{num}.npy'),after_axis)

    

dataroot = Path('/data3/leics/dataset/mesh/remesh_after_centered')
outputroot = Path('/data3/leics/dataset/mesh/remesh_after_centered_aug')
axis_outputroot = Path('/data3/leics/dataset/mesh/axis_aug')
os.makedirs(outputroot,exist_ok=True)
pool = Pool(processes=64)
n_variance = 3
for i in range(n_variance):
# if True:
    # num = len(glob(os.path.join(outputroot,f'*.vtp')))//2
    for j,index in enumerate(dataroot.iterdir()):
            # obj = os.path.join(dataroot,f'{i}_after_lower.vtp')
            num = i * 821 + j
            pool.apply_async(
                get_mesh,
                (dataroot,outputroot_before,outputroot_after,paramroot,index,num)
            )
pool.close()
pool.join()