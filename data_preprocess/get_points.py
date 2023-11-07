from pathlib import Path
import os
import trimesh
import json
import numpy as np
from multiprocessing import Pool
import vedo
def get_points(dataroot,outputroot,index):
    with open(f'/data/lcs/batch2_modified/pat_{index}/0/upper_segment.json','r',encoding='utf-8') as f:
        file = json.load(f)
        print(file)


with open('check.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    indexes = [int(i) for i in lines]
dataroot = Path('/data/lcs/pure_test_teeth/whole_gt_centered')
outputroot = '/data/lcs/pure_test_teeth/single_gt_centered'   #
os.makedirs(outputroot,exist_ok=True)
pool = Pool(processes=8)
for i in indexes:
     pool.apply_async(
          get_points,
          (dataroot,outputroot,i)
     )
pool.close()
pool.join()