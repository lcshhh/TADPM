import numpy as np
from pathlib import Path
import numpy as np
import os
import torch
from multiprocessing import Pool
from utils.pointcloud import read_pointcloud
import argparse
import trimesh
def kabsch_algorithm(A, B):
    assert A.shape == B.shape
    
    A_mean = A - np.mean(A, axis=0)
    B_mean = B - np.mean(B, axis=0)

    H = np.dot(A_mean.T, B_mean)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R

def register(before_path,after_path,outputroot,index):
    rot_param = np.tile(np.eye(3), (32, 1, 1))
    for j in range(32):
        before_mesh_path = before_path / f'{index}_{j}.obj'
        after_mesh_path = after_path / f'{index}_{j}.obj'
        if os.path.exists(before_mesh_path) and os.path.exists(after_mesh_path):
            before_mesh = trimesh.load(before_mesh_path)
            after_mesh = trimesh.load(after_mesh_path)
            before_vertices = before_mesh.vertices - before_mesh.centroid
            after_vertices = after_mesh.vertices - after_mesh.centroid
            try:
                R = kabsch_algorithm(before_vertices, after_vertices)
            except:
                matrix, _, _ = trimesh.registration.icp(before_vertices, after_vertices, scale=False, reflection=False)

                R = matrix[:3, :3]
            # Save the rotation matrix
            rot_param[j] = R.T
    np.save(outputroot / f'{index}.npy', rot_param)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_before', type=str, required=True)
    parser.add_argument('--single_after', type=str, required=True)
    parser.add_argument('--outputroot', type=str, required=True)
    parser.add_argument('--index_file', type=str, required=True)
    args = parser.parse_args()

    before_path = Path(args.single_before)
    after_path = Path(args.single_after)
    output_root = Path(args.outputroot)
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    pool = Pool(processes=16)
    with open(args.index_file, 'r') as f:
        indexes = [line.strip() for line in f if line.strip()]
    for index in indexes:
        pool.apply_async(
            register,
            (before_path,after_path,output_root,index),
        )
    pool.close()
    pool.join()