# from pathlib import Path
# dataroot = Path('/data3/leics/dataset/type2_process_final/single_before')
# indexes = set()
# for obj in dataroot.iterdir():
#     index = int(obj.stem.split('_')[0])
#     indexes.add(index)
# with open('files/second_stage/all.txt','w') as f:
#     for index in indexes:
#         f.write(str(index) + '\n')

# import random

# # 设置随机种子以确保可复现
# random.seed(42)

# # 读取所有索引
# with open('files/second_stage/all.txt', 'r') as f:
#     all_indices = [line.strip() for line in f if line.strip()]

# # 打乱顺序
# random.shuffle(all_indices)

# # 设定划分比例
# split_ratio = 0.9
# split_point = int(len(all_indices) * split_ratio)

# # 划分为训练集和测试集
# train_indices = all_indices[:split_point]
# test_indices = all_indices[split_point:]

# # 写入文件
# with open('files/second_stage/train.txt', 'w') as f:
#     for idx in train_indices:
#         f.write(f"{idx}\n")

# with open('files/second_stage/test.txt', 'w') as f:
#     for idx in test_indices:
#         f.write(f"{idx}\n")

# print(f"训练集数量: {len(train_indices)}, 测试集数量: {len(test_indices)}")
from pathlib import Path
import numpy as np
import trimesh
import os
single_before = Path('/data3/leics/dataset/type2_process_final/single_before')
single_after = Path('/data3/leics/dataset/type2_process_final/single_after')
param_path = Path('/data3/leics/dataset/type2_process_final/param')
param = np.load(param_path / '0.npy')
before_meshes = []
after_meshes = []
for j in range(32):
    before_mesh_path = single_before / f'0_{j}.obj'
    after_mesh_path = single_after / f'0_{j}.obj'
    if os.path.exists(before_mesh_path) and os.path.exists(after_mesh_path):
        before_mesh = trimesh.load(before_mesh_path)
        after_mesh = trimesh.load(after_mesh_path)
        after_centroid = after_mesh.centroid
        before_vertices = before_mesh.vertices - before_mesh.centroid
        before_mesh.vertices = before_vertices @ param[j] + after_centroid
        before_meshes.append(before_mesh)
        after_meshes.append(after_mesh)
before_mesh = trimesh.util.concatenate(before_meshes)
after_mesh = trimesh.util.concatenate(after_meshes)
before_mesh.export('/data3/leics/dataset/type2_process_final/before.obj')
after_mesh.export('/data3/leics/dataset/type2_process_final/after.obj')
        
    