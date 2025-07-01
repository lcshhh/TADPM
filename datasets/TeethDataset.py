
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
from utils.builder import DATASETS
import copy
import open3d as o3d

def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-30,30) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1

def  randomize_mesh_orientation2(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-30,30) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    translation = np.array([random.uniform(-0.04,0.04),random.uniform(-0.04,0.04),random.uniform(-0.01,0.01)])
    centroid = mesh.centroid
    mesh1.vertices = rotation.apply(mesh1.vertices - centroid) + centroid + translation
    return mesh1, rotation, translation

def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    return xyz

def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh

def load_mesh_shape(mesh, augments=[], request=[], seed=None):

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]

    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch

    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=int)), 0)
    Fs_patcha = np.array(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha


@DATASETS.register_module()
class teethDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()

        with open(config.file) as f:
            self.mesh_paths = [i for i in f.readlines()]
        self.train = config.train
        self.augments = ['scale', 'orient'] if self.train else []
        self.feats = ['area', 'face_angles', 'curvs', 'normal']

    
    def __len__(self):
        return len(self.mesh_paths)
    
    def __getitem__(self, idx):
        mesh_path = self.mesh_paths[idx].strip()
        mesh = trimesh.load_mesh(mesh_path)
        feats, center, cordinates, faces, Fs = load_mesh_shape(mesh, augments=self.augments,
                                                             request=self.feats)

        return   feats, center, cordinates, faces, Fs