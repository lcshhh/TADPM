import vedo
import trimesh
import os
import numpy as np
import torch
from multiprocessing import Pool
from pytorch3d.transforms import se3_log_map
def registration(index,save_path):
    mesh1_path = f'/data/lcs/finetuned_teeth/merged/{index}_after.vtp'
    mesh2_path = f'/data/lcs/finetuned_teeth/merged/{index}_before.vtp'
    mesh_1 = vedo.Mesh(mesh1_path)
    mesh_2 = vedo.Mesh(mesh2_path)
    label_1 = mesh_1.celldata['Label']
    label_2 = mesh_2.celldata['Label']
    mesh_1 = vedo.vedo2trimesh(mesh_1)
    mesh_2 = vedo.vedo2trimesh(mesh_2)
    points = mesh_1.sample(2048)
    matrix = trimesh.registration.icp(points,mesh_2,scale=False,max_iterations=50)[0]
    mesh_1 = vedo.trimesh2vedo(mesh_1)
    mesh_2 = vedo.trimesh2vedo(mesh_2)
    mesh_1.celldata['Label'] = label_1
    mesh_2.celldata['Label'] = label_2
    mesh_1.apply_transform(matrix)
    vedo.write(mesh_1,os.path.join(save_path,f'{index}_after.vtp'))
    vedo.write(mesh_2,os.path.join(save_path,f'{index}_before.vtp'))

def register(index,dataroot1,dataroot2,outputroot):
     # dofs = torch.zeros(32,6)
     rot_matrix = torch.zeros(32,3,3)
     for i in range(32):
        obj_path1 = os.path.join(dataroot1,f'{292}_{i}.obj')
        obj_path2 = os.path.join(dataroot2,f'{292}_{i}.obj')
        if os.path.exists(obj_path1) and os.path.exists(obj_path2):
            mesh1 = trimesh.load_mesh(obj_path1)
            mesh2 = trimesh.load_mesh(obj_path2)
            points = mesh1.sample(2048)
            matrix = trimesh.registration.icp(points,mesh2,scale=False,max_iterations=20)[0]
          #   dofs[i] = se3_log_map(torch.from_numpy(matrix).transpose(0,1).unsqueeze(0))
          #   matrix = matrix[:6]
            rot_matrix[i] = torch.from_numpy(matrix[:3,:3])
            print(matrix)
          #   exit()
     # torch.save(dofs,os.path.join(outputroot,f'{index}.pkl'))
     torch.save(rot_matrix,os.path.join(outputroot,f'{292}.pkl'))



dataroot1 = '/data3/leics/dataset/rot_matrix/single_before_centered'
dataroot2 = '/data3/leics/dataset/rot_matrix/single_after_centered'
with open('val.txt','r') as f:
     indexes = [int(i.strip()) for i in f.readlines()]
outputroot = '/data3/leics/dataset/rot_matrix/param'
os.makedirs(outputroot,exist_ok=True)
with open('train.txt','r') as f:
     for i in f.readlines():
         indexes.append(int(i.strip()))
# with open('check.txt','r',encoding='utf-8') as file:
#      lines = file.readlines()
# indexes = [int(i) for i in lines]     
pool = Pool(processes=64)
for index in indexes:
     # pool.apply_async(
     #      register,
     #      (index,dataroot1,dataroot2,outputroot)
     # )
    register(index,dataroot1,dataroot2,outputroot)
    exit()
    #  register(obj_path1,obj_path2,outputroot)
    #  exit()
pool.close()
pool.join()
    