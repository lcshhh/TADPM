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
import time
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def test(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = torch.nn.functional.normalize(x_raw,dim=-1)  # batch*3
    z = torch.cross(x, y_raw, dim=-1)  # batch*3
    z = torch.nn.functional.normalize(z,dim=-1)  # batch*3
    y = torch.cross(z,x,dim=-1)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

a = torch.rand(3,9)
t1 = time.time()
b = robust_compute_rotation_matrix_from_ortho6d(a)
t2 = time.time()
c = robust_compute_rotation_matrix_from_ortho6d(b.view(3,9))
t3 = time.time()
# print(t2-t1)
# print(t3-t2)
print(torch.bmm(c,c.transpose(1,2)))
print(b)
print(c)
print(torch.abs(b-c).sum())

