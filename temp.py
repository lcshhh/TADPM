from einops import rearrange
from pytorch3d.transforms import euler_angles_to_matrix
import torch
import open3d as o3d
import numpy as np
import os

def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    return torch.from_numpy(xyz)

def transform_vertices(vertices,centroids,dofs):
    '''
    vertices: [bs, 32, pt_num, 3]
    centroids: [bs, 32, 3]
    dofs: [bs, 32, 6]
    '''
    angles = rearrange(dofs[:,:,3:]*torch.pi/6,'b n c -> (b n) c')
    move = rearrange(dofs[:,:,:3]/5,'b n c -> (b n) c').unsqueeze(1) #[b*n,1,3]
    centroids = rearrange(centroids, 'b n c -> (b n) c').unsqueeze(1)
    R = euler_angles_to_matrix(angles,'XYZ')
    vertices = rearrange(vertices,'b n pn c -> (b n) pn c')
    vertices = torch.bmm(vertices - centroids,R) + centroids + move
    return vertices

vertices = read_pointcloud('/data/lcs/dataset/teeth_full/pointcloud_before512/1178_21.ply')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
o3d.io.write_point_cloud('/data/lcs/dataset/homework/before.ply', pcd)
transform_vertices(vertices)