
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
class AxisDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.after_path = config.after_path
        self.npoint = config.npoint
        self.after_axis_path = config.after_axis_path
        with open(config.file) as f:
            self.indexes = [int(i.strip()) for i in f.readlines()]
        self.train = config.train
    
    def __getitem__(self, idx):
        point_num = self.npoint
        after_points = np.zeros((32,point_num,3))
        after_centroid = np.zeros((32,3))
        index = self.indexes[idx]
        masks = np.zeros((32),dtype=np.int32)
        after_axis_path = os.path.join(self.after_axis_path,f'{index}.npy')
        after_axis = np.load(str(after_axis_path))
        for i in range(32):
            after_path = os.path.join(self.after_path,f'{index}_{i}.ply')
            if os.path.exists(after_path):
                masks[i] = 1
                after = read_pointcloud(after_path)
                after_points[i] = after[:point_num]
                after_centroid[i] = after[point_num] 
                if self.train and np.random.rand() < 0.5:
                    rotation, translation = randomize_mesh_orientation()
                    after_points[i] = rotation.apply(after_points[i] - np.expand_dims(after_centroid[i],0)) + np.expand_dims(after_centroid[i] + translation,0)
                    after_centroid[i] = after_centroid[i] + translation
                    after_axis[i][3:6] = rotation.apply(after_axis[i][3:6])
                    after_axis[i][6:] = rotation.apply(after_axis[i][6:])
        return   index,after_points,after_centroid,after_axis,masks


    def __len__(self):
        return len(self.indexes)

@DATASETS.register_module()
class AxisTestDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.before_path = config.before_path
        self.npoint = config.npoint
        with open(config.file) as f:
            self.indexes = [int(i.strip()) for i in f.readlines()]
        self.train = config.train
    
    def __getitem__(self, idx):
        point_num = self.npoint
        before_points = np.zeros((32,point_num,3))
        before_centroid = np.zeros((32,3))
        index = self.indexes[idx]
        masks = np.zeros((32),dtype=np.int32)
        for i in range(32):
            before_path = os.path.join(self.before_path,f'{index}_{i}.ply')
            if os.path.exists(before_path):
                masks[i] = 1
                before = read_pointcloud(before_path)
                before_points[i] = before[:point_num]
                before_centroid[i] = before[point_num] 
        return   index,before_points,before_centroid,masks


    def __len__(self):
        return len(self.indexes)