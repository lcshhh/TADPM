from pathlib import Path
import os
import trimesh
import json
import numpy as np
from multiprocessing import Pool
import vedo
def extract_tooth(dataroot,outputroot,index):
    mesh = vedo.Mesh(os.path.join(dataroot,f'before_{index}.vtp'))
    labels = mesh.celldata['Label']
    num = index
    for i in range(1,33):
        if i in labels:
            tmp = vedo.vedo2trimesh(mesh).submesh([np.where(labels==i)[0]], append=True)
            tmp = vedo.trimesh2vedo(tmp)
            # tmp.export(os.path.join(outputroot,f'{num}-{i}.vtp'))
            vedo.write(tmp,os.path.join(outputroot,f'{num}_{i}.obj'))


with open('check.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    indexes = [int(i) for i in lines]
dataroot = Path('/data/lcs/finetuned_teeth/transformed_centered_after')
outputroot = '/data/lcs/finetuned_teeth/single_after_before'   #
os.makedirs(outputroot,exist_ok=True)
pool = Pool(processes=64)
for i in range(2120):
     pool.apply_async(
          extract_tooth,
          (dataroot,outputroot,i)
     )
pool.close()
pool.join()