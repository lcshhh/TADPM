import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import CubicSpline
import os
import math
import trimesh
from pathlib import Path
from multiprocessing import Pool
import torch
import vedo
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import *
def register(after_dataroot,before_dataroot,trans_root,index):
    after_axis = np.load(os.path.join(after_dataroot,f'{index}.npy'))   
    dofs = torch.load(os.path.join(trans_root,f'{index}.pkl'))
    trans_matrix = se3_exp_map(dofs).transpose(2,1) # [32,4,4]
    trans_matrix = torch.inverse(trans_matrix)
    rot_matrix = trans_matrix[:,:3,:3]
    # rotation = Rotation.from_matrix(rot_matrix)
    after_axis = torch.from_numpy(np.reshape(after_axis,(32,3,3)))
    # after_axis  = rotation.apply(after_axis)
    after_axis = torch.bmm(after_axis.float(),rot_matrix.transpose(2,1)).numpy()
    after_axis = np.reshape(after_axis,(32,9))
    np.save(os.path.join(before_dataroot,f'{index}.npy'),after_axis)
    # vertices = torch.from_numpy(mesh.vertices).to(dof.device).float()
    # predicted_vertices = Transform3d(matrix=trans_matrix.transpose(2,1)[0]).transform_points(vertices)

after_dataroot = Path('/data3/leics/dataset/mesh/single_after_axis_revert')
before_dataroot = Path('/data3/leics/dataset/mesh/single_before_axis')
trans_root = Path('/data3/leics/dataset/mesh/param')
indexes = []
with open('train.txt') as f:
    idx = [int(i) for i in f.readlines()]
for i in idx:
    indexes.append(i)
with open('val.txt') as f:
    idx = [int(i) for i in f.readlines()]
for i in idx:
    indexes.append(i)
pool = Pool(processes=32)
for index in indexes:
     pool.apply_async(
          register,
          (after_dataroot,before_dataroot,trans_root,index)
     )
    # register(after_dataroot,before_dataroot,trans_root,index)
    # get_pointcloud_with_center(outputroot,obj)
    # exit()
pool.close()
pool.join()

