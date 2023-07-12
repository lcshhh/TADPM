import vedo
import trimesh
import os
import numpy as np
from multiprocessing import Pool
def registration(index,save_path):
    mesh1_path = f'/data/lcs/finetuned_teeth/merged/{index}_after.vtp'
    mesh2_path = f'/data/lcs/finetuned_teeth/merged/{index}_before.vtp'
    mesh_1 = vedo.Mesh(mesh1_path)
    mesh_2 = vedo.Mesh(mesh2_path)
    mesh_1 = vedo.vedo2trimesh(mesh_1)
    mesh_2 = vedo.vedo2trimesh(mesh_2)
    points = mesh_1.sample(2048)
    matrix = trimesh.registration.icp(points,mesh_2,scale=False,max_iterations=50)[0]
    mesh_1 = vedo.trimesh2vedo(mesh_1)
    mesh_2 = vedo.trimesh2vedo(mesh_2)
    mesh_1.apply_transform(matrix)
    vedo.write(mesh_1,os.path.join(save_path,f'{index}_after.vtp'))
    vedo.write(mesh_2,os.path.join(save_path,f'{index}_before.vtp'))

inputroot = '/data/lcs/MOXING'
outputroot = '/data/lcs/finetuned_teeth/registered'
os.makedirs(outputroot,exist_ok=True)
with open('check.txt','r',encoding='utf-8') as file:
     lines = file.readlines()
indexes = [int(i) for i in lines]     
pool = Pool(processes=64)
for index in indexes:
     pool.apply_async(
          registration,
          (index,outputroot)
     )
pool.close()
pool.join()
    