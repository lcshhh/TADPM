import json
import numpy as np
import vedo
import os
import trimesh
import numpy as np
from multiprocessing import Pool
import vedo

def merge(index,inputroot,outputroot):
    if not os.path.exists(os.path.join(inputroot,f'upper_after_{index}.vtp')):
         return
    if not os.path.exists(os.path.join(inputroot,f'lower_after_{index}.vtp')):
         return
    mesh_upper = vedo.Mesh(os.path.join(inputroot,f'upper_after_{index}.vtp'))
    mesh_lower = vedo.Mesh(os.path.join(inputroot,f'lower_after_{index}.vtp'))
    label = np.array(mesh_lower.celldata['Label'],dtype=int)
    label += 16
    label[label==16] = 0
    mesh_lower.celldata['Label'] = label
    label = np.array(mesh_upper.celldata['Label'],dtype=int)
    mesh_upper.celldata['Label'] = label
    mesh = vedo.merge(mesh_lower,mesh_upper)
    mesh.delete_cells_by_point_index(list(set(np.asarray(mesh.cells())[np.where(mesh.celldata["Label"]==0)[0]].flatten())))
    vedo.write(mesh,os.path.join(outputroot,f'{index}.vtp'))


# mesh1.AddPosition(10,0,0)bj
inputroot = '/data/lcs/finetuned_teeth/after'
outputroot = '/data/lcs/finetuned_teeth/merged_after'
os.makedirs(outputroot,exist_ok=True)
with open('check.txt','r',encoding='utf-8') as file:
     lines = file.readlines()
indexes = [int(i) for i in lines]     
pool = Pool(processes=64)
for index in range(236):
     pool.apply_async(
          merge,
          (index,inputroot,outputroot)
     )
    # merge()
    # segment(index,outputroot)
    # extract_tooth(dataroot,outputroot,obj)
pool.close()
pool.join()