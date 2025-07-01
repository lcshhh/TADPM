import random
from pathlib import Path
# 定义一个函数，用于分割数据集
def split_dataset(dataroot,train_ratio=0.8):
    mesh_paths = []
    for tooth in dataroot.iterdir():
            # 将每一个子目录的路径添加到mesh_paths列表中
            mesh_paths.append(tooth)
        
    random.shuffle(mesh_paths)
    split_point = int(len(mesh_paths) * train_ratio)
    train_objs = mesh_paths[:split_point]
    test_objs = mesh_paths[split_point:]
    with open('files/pretrain/train.txt', 'w') as f:
        for obj in train_objs:
            f.write(f"{obj}\n")
    with open('files/pretrain/test.txt', 'w') as f:
        for obj in test_objs:
            f.write(f"{obj}\n")

if __name__ == '__main__':
    dataroot = Path('/data3/leics/dataset/type2_process_final/remesh_before')
    split_dataset(dataroot)

