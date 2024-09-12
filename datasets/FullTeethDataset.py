
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
from utils.builder import DATASETS
import open3d as o3d

def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points)
    return xyz

def  randomize_mesh_orientation():
    # mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-30,30) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    translation = np.array([random.uniform(-0.04,0.04),random.uniform(-0.04,0.04),random.uniform(-0.01,0.01)])
    # mesh1.vertices = rotation.apply(mesh1.vertices - centroid) + centroid + translation
    return rotation, translation

@DATASETS.register_module()
class FullTeethDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.after_path = config.after_path
        self.npoint = config.npoint
        self.axis_path = config.axis_path
        with open(config.file) as f:
            self.indexes = [int(i.strip()) for i in f.readlines()]
        self.train = config.train
    
    def __getitem__(self, idx):
        point_num = self.npoint
        after_points = np.zeros((32,point_num,3))
        after_centroid = np.zeros((32,3))
        index = self.indexes[idx]
        masks = np.zeros((32),dtype=np.int32)
        axis_path = os.path.join(self.axis_path,f'{index}.npy')
        axis = np.load(str(axis_path))
        for i in range(32):
            after_path = os.path.join(self.after_path,f'{index}_{i}.ply')
            if os.path.exists(after_path) and os.path.exists(axis_path):
                masks[i] = 1
                after = read_pointcloud(after_path)
                after_points[i] = after[:point_num]
                after_centroid[i] = after[point_num] 
                if self.train and np.random.rand() < 0.3:
                    rotation, translation = randomize_mesh_orientation()
                    after_points[i] = rotation.apply(after_points[i] - np.expand_dims(after_centroid[i],0)) + np.expand_dims(after_centroid[i] + translation,0)
                    after_centroid[i] = after_centroid[i] + translation
        return   index,after_points,after_centroid,axis,masks


    def __len__(self):
        return len(self.indexes)