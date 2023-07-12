import json
import numpy as np
import vedo
import os
from pathlib import Path
import trimesh
import numpy as np
from multiprocessing import Pool
from glob import glob
import vedo
def check(index,dataroot):
    with open('/data/lcs/check_upper.txt','a',encoding='utf-8') as f:
        # mesh_before = vedo.Mesh(os.path.join(dataroot,f'before_{index}.vtp'))
        # mesh_after = vedo.Mesh(os.path.join(dataroot,f'after_{index}.vtp'))
        mesh_before = vedo.Mesh(os.path.join(dataroot,f'upper_before_{index}.vtp'))
        mesh_after = vedo.Mesh(os.path.join(dataroot,f'upper_after_{index}.vtp'))
        before_lst = []
        after_lst = []
        for label in mesh_before.celldata['Label']:
            if label not in before_lst:
                before_lst.append(label)
        for label in mesh_after.celldata['Label']:
            if label not in after_lst:
                after_lst.append(label)
        # for label in before_lst:
        #     if label not in after_lst:
        #         return
        # for label in after_lst:
        #     if label not in before_lst:
        #         return
        if len(before_lst) != len(after_lst):
            return
        f.write(str(index)+'\n')

# dataroot = '/data/lcs/batch2_merged'
# pool = Pool(processes=64)
# for index in range(1,236):
#      pool.apply_async(
#           check,
#           (index,dataroot)
#      )
#     # segment(index,outputroot)
#     # extract_tooth(dataroot,outputroot,obj)
# pool.close()
# pool.join()
with open('/data/lcs/check_lower.txt','r',encoding='utf-8') as f:
    lines1 = f.readlines()
    lines1 = sorted([int(i) for i in lines1])

with open('/data/lcs/check_upper.txt','r',encoding='utf-8') as f:
    lines2 = f.readlines()
    lines2 = sorted([int(i) for i in lines2])



with open('/data/lcs/check_num.txt','w',encoding='utf-8') as f:
    for line in lines1:
        if line in lines2:
            f.write(str(line)+'\n')

# for i in range(1,236):
#     if i not in lines:
#         print(i)
# root = Path('/data/lcs/batch2_merged_final/before_without_gingiva')
# # files = glob(os.path.join(root,'*.vtp'))
# with open('/data/lcs/batch2_merged_final/useful.txt','w',encoding='utf-8') as f:
#     for file in root.iterdir():
#         num = file.name.split('.')[0]
#         f.write(num+'\n')