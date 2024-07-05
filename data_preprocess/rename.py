import os
import trimesh
import json
import numpy as np
import vedo
import torch
from multiprocessing import Pool
from pathlib import Path
def rearrange(npary: np.ndarray)->np.ndarray:
    """
    Rearrange the oder of the label of the input numpy array.
    :param npary: Input numpy array.
    :return:
    """
    npary[npary == 11] = 8
    npary[npary == 12] = 7
    npary[npary == 13] = 6
    npary[npary == 14] = 5
    npary[npary == 15] = 4
    npary[npary == 16] = 3
    npary[npary == 17] = 2
    npary[npary == 18] = 1
    npary[npary == 21] = 9
    npary[npary == 22] = 10
    npary[npary == 23] = 11
    npary[npary == 24] = 12
    npary[npary == 25] = 13
    npary[npary == 26] = 14
    npary[npary == 27] = 15
    npary[npary == 28] = 16
    npary[npary == 31] = 9+16
    npary[npary == 32] = 10+16
    npary[npary == 33] = 11+16
    npary[npary == 34] = 12+16
    npary[npary == 35] = 13+16
    npary[npary == 36] = 14+16
    npary[npary == 37] = 15+16
    npary[npary == 38] = 16+16
    npary[npary == 41] = 8+16
    npary[npary == 42] = 7+16
    npary[npary == 43] = 6+16
    npary[npary == 44] = 5+16
    npary[npary == 45] = 4+16
    npary[npary == 46] = 3+16
    npary[npary == 47] = 2+16
    npary[npary == 48] = 1+16
    return npary

def rearrange_index(index):
    index = int(index)
    a = rearrange(np.array([index]))
    return a[0] - 1

# train_path = Path('/data/lcs/dataset/mesh/scan_train')
# val_path = Path('/data/lcs/dataset/mesh/scan_eval')
# test_path = Path('/data/lcs/dataset/mesh/scan_test')
# out_before_path = Path('/data3/leics/dataset/mesh/single_before')
# out_after_path = Path('/data3/leics/dataset/mesh/single_after')
# os.makedirs(out_before_path,exist_ok=True)
# os.makedirs(out_after_path,exist_ok=True)
# train_list = []
# val_list = []
# test_list = []
# num = 1082
# with open('train.txt') as f:
#      train_list = [int(i.strip()) for i in f.readlines()]
# with open('val.txt') as f:
#      val_list = [int(i.strip()) for i in f.readlines()]
# with open('test.txt') as f:
#      test_list = [int(i.strip()) for i in f.readlines()]
# for i,index in enumerate(train_list):
#     before_path = Path(os.path.join(train_path,f'{index}_all32_start'))
#     after_path = Path(os.path.join(train_path,f'{index}_all32_end'))
#     for obj_path in before_path.iterdir():
#           fdi = rearrange_index(obj_path.name.split('.')[0])
#           mesh = trimesh.load_mesh(obj_path)
#           mesh.vertices = mesh.vertices / 40
#           mesh.export(os.path.join(out_before_path,f'{num}_{fdi}.obj'))
#     for obj_path in after_path.iterdir():
#           fdi = rearrange_index(obj_path.name.split('.')[0])
#           mesh = trimesh.load_mesh(obj_path)
#           mesh.vertices = mesh.vertices / 40
#           mesh.export(os.path.join(out_after_path,f'{num}_{fdi}.obj'))
#     num += 1
# print(num)

# for i,index in enumerate(val_list):
#     before_path = Path(os.path.join(val_path,f'{index}_all32_start'))
#     after_path = Path(os.path.join(val_path,f'{index}_all32_end'))
#     for obj_path in before_path.iterdir():
#           fdi = rearrange_index(obj_path.name.split('.')[0])
#           mesh = trimesh.load_mesh(obj_path)
#           mesh.vertices = mesh.vertices / 40
#           mesh.export(os.path.join(out_before_path,f'{num}_{fdi}.obj'))
#     for obj_path in after_path.iterdir():
#           fdi = rearrange_index(obj_path.name.split('.')[0])
#           mesh = trimesh.load_mesh(obj_path)
#           mesh.vertices = mesh.vertices / 40
#           mesh.export(os.path.join(out_after_path,f'{num}_{fdi}.obj'))
#     num += 1
# print(num)
# with open('train.txt','w') as f:
#     for i in range(959):
#          f.write(str(i)+'\n')
with open('val.txt','w') as f:
    for i in range(959,1082):
         f.write(str(i)+'\n')
with open('test.txt','w') as f:
    for i in range(1082,1208):
         f.write(str(i)+'\n')
     

# for i,index in enumerate(test_list):
#     before_path = Path(os.path.join(test_path,f'{index}_all32_start'))
#     after_path = Path(os.path.join(test_path,f'{index}_all32_end'))
#     for obj_path in before_path.iterdir():
#           fdi = rearrange_index(obj_path.name.split('.')[0])
#           mesh = trimesh.load_mesh(obj_path)
#           mesh.vertices = mesh.vertices / 40
#           mesh.export(os.path.join(out_before_path,f'{num}_{fdi}.obj'))
#     # for obj_path in after_path.iterdir():
#     #       fdi = rearrange_index(obj_path.name.split('.')[0])
#     #       mesh = trimesh.load_mesh(obj_path)
#     #       mesh.vertices = mesh.vertices / 40
#     #       mesh.export(os.path.join(out_after_path,f'{num}_{fdi}.obj'))
#     num += 1
# print(num)