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

def get_pointcloud_with_center(before_outputroot,after_outputroot,after_dataroot,obj,sample_num):
    try:
        index = int(obj.name.split('_')[0])
        new_name = obj.name.split('.')[0]+'.ply'
        if os.path.exists(os.path.join(before_outputroot,new_name)):
            return
        point_num = sample_num
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
    except:
        # pass
        print(mesh.vertices.shape)
        print(after_mesh.vertices.shape)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--before_dataroot', type=str, required=True)
parser.add_argument('--after_dataroot', type=str, required=True)
parser.add_argument('--before_output', type=str, required=True)
parser.add_argument('--after_output', type=str, required=True)
parser.add_argument('--sample_num', type=int, default=512)
args = parser.parse_args()

before_dataroot = Path(args.before_dataroot)
after_dataroot = Path(args.after_dataroot)
before_output = args.before_output   #
after_output = args.after_output
os.makedirs(before_output,exist_ok=True)
os.makedirs(after_output,exist_ok=True)
pool = Pool(processes=16)
for obj in before_dataroot.iterdir():
     if not os.path.exists(os.path.join(after_dataroot,obj.name)):
          continue
     pool.apply_async(
          get_pointcloud_with_center,
          (before_output,after_output,after_dataroot,obj,args.sample_num)
     )
pool.close()
pool.join()
