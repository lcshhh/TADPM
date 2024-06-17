from pathlib import Path
import os
import trimesh
import json
import numpy as np
from multiprocessing import Pool
import vedo
def extract_tooth(dataroot,outputroot,index):
    mesh = vedo.Mesh(os.path.join(dataroot,f'{index}.vtp'))
    labels = mesh.celldata['Label']
    num = index
    for i in range(1,33):
        if i in labels:
            tmp = vedo.vedo2trimesh(mesh).submesh([np.where(labels==i)[0]], append=True)
            tmp = vedo.trimesh2vedo(tmp)
            vedo.write(tmp,os.path.join(outputroot,f'{num}_{i-1}.obj'))


with open('valid.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    indexes = [int(i) for i in lines]
dataroot = Path('/data/lcs/dataset/teeth_full/centered_after')
outputroot = '/data/lcs/dataset/no_norm/single_after'   #
os.makedirs(outputroot,exist_ok=True)
pool = Pool(processes=16)
for i in indexes:
     pool.apply_async(
          extract_tooth,
          (dataroot,outputroot,i)
     )
pool.close()
pool.join()