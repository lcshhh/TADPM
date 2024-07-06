from pathlib import Path
import os
import trimesh
import json
import numpy as np
import vedo
import torch
from multiprocessing import Pool

def exam(path):
    try:
        mesh = trimesh.load_mesh(path)
        len(mesh.faces)
    except:
        print(path)

# pool = Pool(processes=32)
# outputroot_before = Path('/data3/leics/dataset/created/remesh_before')
# indexes = set()
# for path in outputroot_before.iterdir():
#     # pool.apply_async(
#     #     exam(path)
#     # )
#     index = int(path.name.split('_')[0])
#     indexes.add(index)
# # pool.close()
# # pool.join()
# with open('train.txt','w') as f:
#     for index in indexes:
#         f.write(str(index)+'\n')
# with open('wrong.txt') as f:
#     lines = f.readlines()
# for line in lines:
#     try:
#         mesh = trimesh.load_mesh(Path(f'/data3/leics/dataset/created/single_before/364_24.obj'))
#         print(len(mesh.faces))
#     except:
#         print('error')
#         continue

# meshes = []
# for i in range(32):
#     path = os.path.join(outputroot_before,f'0_{i}.obj')
#     if not os.path.exists(path):
#         continue
#     mesh = vedo.Mesh(path)
#     meshes.append(mesh)
# mesh = vedo.merge(meshes)
# vedo.write(mesh,'/data3/leics/dataset/created/0.obj')

# def get_center(path,remesh_path,outputroot):
#     mesh = trimesh.load_mesh(path)
#     mesh.vertices = mesh.vertices - mesh.centroid
#     mesh.export(os.path.join(outputroot,path.name))
#     remesh_mesh = 

# path = Path('/data3/leics/dataset/mesh/single_after')
# remesh_path = Path('')
# outputroot = Path('/data3/leics/dataset/mesh/single_after_centered')
# os.makedirs(outputroot,exist_ok=True)
# pool = Pool(processes=16)
# for path in path.iterdir():
#     pool.apply_async(
#         get_center,
#         (path,outputroot)
#     )
# pool.close()
# pool.join()


