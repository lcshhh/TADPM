import numpy as np
import open3d as o3d
from pathlib import Path
import os
import trimesh
import json
import numpy as np
from multiprocessing import Pool
import vedo
import torch
def farthest_point_sample(point, npoint):
        """
        Input:
            xyz: pointcloud data, [N, D]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint, D]
        """
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point, centroids

def get_pointcloud(outputroot,obj,indexes):
    index = int(obj.name.split('_')[0])
    if index not in indexes:
         return
    point_num = 512
    mesh = trimesh.load_mesh(obj)
    points = farthest_point_sample(mesh.vertices,point_num)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    new_name = obj.name.split('.')[0]+'.ply'
    o3d.io.write_point_cloud(os.path.join(outputroot,new_name), pcd)

def get_pointcloud_with_center(before_outputroot,after_outputroot,after_dataroot,obj):
    index = int(obj.name.split('_')[0])
    new_name = obj.name.split('.')[0]+'.ply'
    if os.path.exists(os.path.join(before_outputroot,new_name)):
         return
    point_num = 256
    mesh = trimesh.load_mesh(obj)
    points, centroids = farthest_point_sample(mesh.vertices,point_num)
    points = np.concatenate([points,torch.tensor(mesh.centroid).unsqueeze(0).numpy()],axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(before_outputroot,new_name), pcd)
    
    after_mesh = trimesh.load_mesh(os.path.join(after_dataroot,obj.name))
    if os.path.exists(os.path.join(after_outputroot,new_name)):
         return
    points = np.array(after_mesh.vertices)
    points = points[centroids.astype(np.int32)]
    points = np.concatenate([points,torch.tensor(after_mesh.centroid).unsqueeze(0).numpy()],axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(after_outputroot,new_name), pcd)

def read_pointcloud():
    pcd = o3d.io.read_point_cloud("/data/lcs/dataset/teeth_full/pointcloud_after512/0_19.ply")
    xyz = np.asarray(pcd.points)
    return xyz

before_dataroot = Path('/data/lcs/dataset/teeth_full/single_normed_before')
after_dataroot = Path('/data/lcs/dataset/teeth_full/single_normed_after')
before_outputroot = '/data3/leics/dataset/teeth_full/single_pointcloud_before256'   #
after_outputroot = '/data3/leics/dataset/teeth_full/single_pointcloud_after256'   #
# with open('valid.txt') as f:
#      indexes = [int(i.strip()) for i in f.readlines()]
os.makedirs(before_outputroot,exist_ok=True)
os.makedirs(after_outputroot,exist_ok=True)
pool = Pool(processes=32)
for obj in before_dataroot.iterdir():
     if not os.path.exists(os.path.join(after_dataroot,obj.name)):
          continue
     pool.apply_async(
          get_pointcloud_with_center,
          (before_outputroot,after_outputroot,after_dataroot,obj)
     )
    # get_pointcloud_with_center(outputroot,obj)
pool.close()
pool.join()
