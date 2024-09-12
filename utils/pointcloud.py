import numpy as np
import open3d as o3d

def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    return xyz 

def write_pointcloud(points, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)