
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

def randomize_orientation(vertices):
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-30,30) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    vertices = rotation.apply(vertices)
    return vertices

@DATASETS.register_module()
class teethDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.train = config.train
        with open(config.file) as f:
            self.mesh_paths = [l.strip() for l in f.readlines()]

    def __len__(self):
        return len(self.mesh_paths)
    
    def __getitem__(self, idx):
        path = self.mesh_paths[idx]
        name = path.split('/')[-1]
        label = int(name.split('.')[0].split('_')[1])
        points = read_pointcloud(path)
        if self.train:
            points = randomize_orientation(points)
        return   points,label