from pathlib import Path
import os
import trimesh
import json
import numpy as np
from multiprocessing import Pool
import vedo
import torch
from glob import glob
from pytorch3d.transforms import *

def registration(before_path,after_path,outputroot,index):
    trans_matrix = torch.zeros(32,4,4)
    for i in range(32):
        before = os.path.join(before_path,f'{index}_{i}.obj')
        after = os.path.join(after_path,f'{index}_{i}.obj')
        if os.path.exists(before) and os.path.exists(after):
            mesh_before = trimesh.load_mesh(before)
            mesh_after = trimesh.load_mesh(after)
            # mesh1 = vedo.vedo2trimesh(mesh_before)
            # mesh2 = vedo.vedo2trimesh(mesh_after)
            points = mesh_before.sample(2048)
            matrix = trimesh.registration.icp(points,mesh_after,scale=False)[0]
            trans_matrix[i] = torch.from_numpy(matrix)
    trans_6dof = se3_log_map(trans_matrix.transpose(1,2))
    torch.save(trans_matrix,os.path.join(outputroot,f'matrix_{index}.pkl'))
    torch.save(trans_6dof,os.path.join(outputroot,f'6dof_{index}.pkl'))

with open('check.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    indexes = [int(i) for i in lines]
before_path = Path('/data/lcs/finetuned_teeth/single_registered_before')
after_path = Path('/data/lcs/finetuned_teeth/single_registered_after')
outputroot = '/data/lcs/finetuned_teeth/register_param'
os.makedirs(outputroot,exist_ok=True)
pool = Pool(processes=64)
for i in range(790):
     pool.apply_async(
          registration,
          (before_path,after_path,outputroot,i)
     )
pool.close()
pool.join()
