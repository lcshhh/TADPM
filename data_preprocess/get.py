from pathlib import Path
import os
import trimesh
import json
import numpy as np
import vedo
import torch
from multiprocessing import Pool
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
    # if not upper:
    #     npary[npary == 0] = 16
    #     npary -= 16
    return npary

train_path = Path('/data/lcs/dataset/mesh/scan_train')
val_path = Path('/data/lcs/dataset/mesh/scan_eval')
test_path = Path('/data/lcs/dataset/mesh/scan_test')
out_path = Path('/data/lcs/dataset/mesh/single')
os.makedirs(out_path,exist_ok=True)
train_list = []
for path in train_path.iterdir():
    for stl_path in path.iterdir():
        train_list.append(stl_path)
for path in val_path.iterdir():
    for stl_path in path.iterdir():
        train_list.append(stl_path)
for path in test_path.iterdir():
    for stl_path in path.iterdir():
        train_list.append(stl_path)
# with open('single.txt','w') as f:
#     for path in train_list:
#         f.write(str(path)+'\n')
for i,path in enumerate(train_list):
    mesh = trimesh.load_mesh(path)
    mesh.vertices = mesh.vertices / 40
    mesh.export(os.path.join(out_path,f'{i}.obj'))

