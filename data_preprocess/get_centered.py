from pathlib import Path
import os
import trimesh
import json
import numpy as np
import vedo
import torch
from multiprocessing import Pool
def get_center(dataroot, outputroot, index):
    mesh = vedo.Mesh(os.path.join(dataroot,f'{index}.vtp'))
    tri_mesh = vedo.vedo2trimesh(mesh)
    centroid = tri_mesh.centroid
    tri_mesh.vertices = tri_mesh.vertices - centroid
    m = np.max(np.sqrt(np.sum(tri_mesh.vertices ** 2, axis=1)))
    tri_mesh.vertices = tri_mesh.vertices / m
    altered_after = vedo.trimesh2vedo(tri_mesh)
    altered_after.celldata['Label'] = np.array(mesh.celldata['Label'],dtype=int)
    vedo.write(altered_after,os.path.join(outputroot,f'{index}.vtp'))

def single_center(dataroot, outputroot, obj_path):
    mesh = vedo.Mesh(str(obj_path))
    tri_mesh = vedo.vedo2trimesh(mesh)
    centroid = tri_mesh.centroid
    tri_mesh.vertices = tri_mesh.vertices - centroid
    m = np.max(np.sqrt(np.sum(tri_mesh.vertices ** 2, axis=1)))
    tri_mesh.vertices = tri_mesh.vertices / m
    name = obj_path.name.split('.')[0]+'.obj'
    # tri_mesh.export(os.path.join(outputroot,obj_path.name))
    tri_mesh.export(os.path.join(outputroot,name))

with open('check2.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    indexes = [int(i) for i in lines]
dataroot = Path('/data/lcs/finetuned_teeth/merged_after')
outputroot = '/data/lcs/finetuned_teeth/centered_after'   #
os.makedirs(outputroot,exist_ok=True)
pool = Pool(processes=64)
for i in indexes:
    pool.apply_async(
        get_center,
        (dataroot,outputroot,i)
    )
    # get_center(dataroot,outputroot,obj_path)
pool.close()
pool.join()
