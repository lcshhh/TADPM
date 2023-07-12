
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import pygem
from glob import glob
from pygem import FFD
import copy
import csv

def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 90, 180, 270]) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    vertices = mesh1.vertices - mesh1.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh1.vertices = vertices
    return mesh1


def mesh_deformation(mesh: trimesh.Trimesh):
    ffd = FFD([2, 2, 2])
    random = np.random.rand(6) * 0.1
    ffd.array_mu_x[1, 1, 1] = random[0]
    ffd.array_mu_y[1, 1, 1] = random[1]
    ffd.array_mu_z[1, 1, 1] = random[2]
    ffd.array_mu_x[0, 0, 0] = random[3]
    ffd.array_mu_y[0, 0, 0] = random[4]
    ffd.array_mu_z[0, 0, 0] = random[5]
    vertices = mesh.vertices
    new_vertices = ffd(vertices)
    mesh.vertices = new_vertices
    return mesh


def load_mesh_shape(path, augments=[], request=[], seed=None):

    mesh = trimesh.load_mesh(path, process=False)
    m = np.max(np.sqrt(np.sum(mesh.vertices ** 2, axis=1)))

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)

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

class TeethDataManager:
    def __init__(self, dataroot, train_ratio=0.9,augment=None):
        super().__init__()
        self.dataroot = Path(dataroot)
        self.augments = augment
        self.train_ratio = train_ratio
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.mesh_paths = []
        # with open('/data/lcs/batch2_merged_final/useful.txt','r',encoding='utf-8') as f:
        #     lines = f.readlines()
        #     self.indexes = [int(i) for i in lines]
        self.browse_dataroot()
    
    def browse_dataroot(self):
        for tooth in self.dataroot.iterdir():
            self.mesh_paths.append(tooth)
        
        random.shuffle(self.mesh_paths)
        split_point = int(len(self.mesh_paths) * self.train_ratio)
        self.train_objs = self.mesh_paths[:split_point]
        self.test_objs = self.mesh_paths[split_point:]
    
    def train_dataset(self):
        dataset = teethDataset(self.train_objs,True,self.augments)
        return dataset

    def test_dataset(self):
        dataset = teethDataset(self.test_objs,False,self.augments)
        return dataset


class teethDataset(data.Dataset):
    def __init__(self, objs, train=True, augment=None):
        super().__init__()

        self.augments = []
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.mesh_paths = objs

        if train and augment:
            self.augments = augment
    
    def __len__(self):
        return len(self.mesh_paths)
    
    def __getitem__(self, idx):
        label = 0
        feats, center, cordinates, faces, Fs = load_mesh_shape(self.mesh_paths[idx], augments=self.augments,
                                                             request=self.feats)

        return   feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx])


class TeethRegressorDataManager:
    def __init__(self, dataroot, paramroot, train_ratio=0.9):
        super().__init__()
        self.dataroot = Path(dataroot)
        self.train_ratio = train_ratio
        self.paramroot = paramroot
        self.useful_lst = [i for i in range(2120)]
        self.browse_dataroot()
    
    def browse_dataroot(self):
        random.shuffle(self.useful_lst)
        split_point = int(len(self.useful_lst) * self.train_ratio)
        self.train_objs = self.useful_lst[:split_point]
        self.test_objs = self.useful_lst[split_point:]
    
    def train_dataset(self):
        dataset = teethRegressorDataset(self.dataroot, self.paramroot, self.train_objs,True)
        return dataset

    def test_dataset(self):
        dataset = teethRegressorDataset(self.dataroot, self.paramroot, self.test_objs,False)
        return dataset

class teethRegressorDataset(data.Dataset):
    def __init__(self, dataroot, paramroot, objs, train=True):
        super().__init__()
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.dataroot = dataroot
        self.paramroot = paramroot
        self.indexes = objs
        self.train = train
    
    def __getitem__(self, idx):
        feats = np.zeros((32,10,256,64))
        center = np.zeros((32,256,64,3))
        cordinates = np.zeros((32,256,64,9))
        faces = np.zeros((32,256,64,3))
        Fs = np.zeros(32)
        before_points = np.zeros((32,2048,3))
        after_points = np.zeros((32,2048,3))
        centroid = np.zeros((32,3))
        after_centroid = np.zeros((32,3))
        index = self.indexes[idx]
        before_points_centered = np.zeros((32,2048,3))
        trans_6dof = torch.load(os.path.join(self.paramroot,f'6dof_{index}.pkl'))
        trans_matrix = torch.load(os.path.join(self.paramroot,f'matrix_{index}.pkl'))
        for i in range(1,33):
            obj_path = os.path.join(self.dataroot,f'{index}_{i}.obj')
            if os.path.exists(obj_path):
                # masks[i] = 1
                feats[i-1], center[i-1], cordinates[i-1], faces[i-1], Fs[i-1]= load_mesh_shape(obj_path, 
                                                                request=self.feats)
            before_path = os.path.join('/data/lcs/finetuned_teeth/single_after_before',f'{index}_{i}.obj')
            after_path = os.path.join('/data/lcs/finetuned_teeth/single_after_after',f'{index}_{i}.obj')
            if os.path.exists(before_path):
                after_mesh = trimesh.load_mesh(after_path)
                before_mesh = trimesh.load_mesh(before_path)
                before_points[i-1] = np.array(trimesh.sample.sample_surface_even(before_mesh,count=2048)[0])
                after_points[i-1] = np.array(trimesh.sample.sample_surface_even(after_mesh,count=2048)[0])
                centroid[i-1] = before_mesh.centroid
                # before_points_centered[i] = before_points[i] - np.array(centroid[i]).unsqueeze(0).repeat((2048,1))
                after_centroid[i-1] = after_mesh.centroid


        return   feats,center,cordinates,faces,Fs,trans_6dof,trans_matrix,index,before_points,after_points,centroid,after_centroid,before_points_centered



    def __len__(self):
        return len(self.indexes)
