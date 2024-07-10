from pathlib import Path
import os
import trimesh
import json
import numpy as np
import vedo
import torch
from multiprocessing import Pool

import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import copy
import open3d as o3d

def exam(path):
    try:
        mesh = trimesh.load_mesh(path)
        len(mesh.faces)
    except:
        print(path)

def randomize_mesh_orientation2():
    # mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-15,15) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    # mesh1.vertices = rotation.apply(mesh1.vertices)
    return rotation

# a = torch.rand(3,9)
# t1 = time.time()
# b = robust_compute_rotation_matrix_from_ortho6d(a)
# t2 = time.time()
# c = robust_compute_rotation_matrix_from_ortho6d(b.view(3,9))
# t3 = time.time()
# # print(t2-t1)
# # print(t3-t2)
# print(torch.bmm(c,c.transpose(1,2)))
# print(b)
# print(c)
# print(torch.abs(b-c).sum())
vertices = np.random.random((1,3))
rotation = randomize_mesh_orientation2()
after_vertices = rotation.apply(vertices)
matrix1 = rotation.as_matrix().transpose()
# after_vertices2 = np.matmul(vertices,matrix.transpose())
rotation2 = randomize_mesh_orientation2()
before_vertices = rotation2.apply(vertices)
matrix = np.matmul(rotation2.as_matrix(),matrix1)
after_vertices2 = np.matmul(before_vertices,matrix)
print(after_vertices)
print(after_vertices2)

