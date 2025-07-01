import vedo
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import os
import json
import argparse
import glob
def rearrange(npary):
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

def pointLabel2cellLabel(mesh,pointLabel):
    labels = np.zeros(mesh.ncells,dtype=int)
    faces = mesh.cells
    for i in range(mesh.ncells):
        three_label = pointLabel[faces[i]]
        labels[i] = int(np.argmax(np.bincount(three_label)))
    return labels

def get_labels(mesh_path,label_path,upper=True):
    mesh = vedo.Mesh(mesh_path)
    if upper:
        teeth = ['11','12','13','14','15','16','17','18','21','22','23','24','25','26','27','28']
    else:
        teeth = ['31','32','33','34','35','36','37','38','41','42','43','44','45','46','47','48']
    labels = [0 for _ in range(len(mesh.vertices)) ]
    point_labels = np.array(labels,dtype=int)
    with open(label_path,'r',encoding='utf-8') as f:
        lines = json.load(f)
        for tooth in teeth:
            if tooth in lines['segmentation']:
                for point in lines['segmentation'][tooth]['vertices']:
                    index = mesh.closest_point(point,return_point_id=True)
                    point_labels[index] = int(tooth)
    cell_labels = rearrange(pointLabel2cellLabel(mesh,point_labels))
    mesh.celldata['Label'] = cell_labels
    return mesh

def get_mesh(upper_mesh_path,upper_label_path,lower_mesh_path,lower_label_path,outputroot,index):
    upper_mesh = get_labels(upper_mesh_path,upper_label_path)
    lower_mesh = get_labels(lower_mesh_path,lower_label_path,False)
    mesh = vedo.merge([upper_mesh,lower_mesh])
    labels = mesh.celldata['Label']
    mesh = vedo.vedo2trimesh(mesh)
    mesh.vertices = (mesh.vertices - mesh.centroid)/40.0

    num = index
    for i in range(1,33):
        if i in labels:
            tmp = mesh.submesh([np.where(labels==i)[0]], append=True)
            tmp.export(os.path.join(outputroot,f'{num}_{i-1}.obj'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--single_before', type=str, required=True)
    parser.add_argument('--single_after', type=str, required=True)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    single_before = args.single_before
    single_after = args.single_after
    os.makedirs(single_before,exist_ok=True)
    os.makedirs(single_after,exist_ok=True)
    indexes = set()
    pool = Pool(processes=32)
    for index,patient in enumerate(dataroot.iterdir()):
        path = os.path.join(patient,'ori')
        try:
            try:
                upper_mesh_path = glob.glob(os.path.join(path,f'*U*.stl'))[0]
                lower_mesh_path = glob.glob(os.path.join(path,f'*L*.stl'))[0]
            except:
                upper_mesh_path = glob.glob(os.path.join(path,f'*upper*.stl'))[0]
                lower_mesh_path = glob.glob(os.path.join(path,f'*lower*.stl'))[0]
            try:
                upper_label_path = glob.glob(os.path.join(path,f'upper_pred_landmarks.json'))[0]
                lower_label_path = glob.glob(os.path.join(path,f'lower_pred_landmarks.json'))[0]
            except:
                upper_label_path = glob.glob(os.path.join(path,f'*U*.json'))[0]
                lower_label_path = glob.glob(os.path.join(path,f'*L*.json'))[0]

            pool.apply_async(
                get_mesh,
                (upper_mesh_path,upper_label_path,lower_mesh_path,lower_label_path,single_before,index)
            )
        except:
            print(index)
            continue
        path = os.path.join(patient,'final')
        try:
            try:
                upper_mesh_path = glob.glob(os.path.join(path,f'*U*.stl'))[0]
                lower_mesh_path = glob.glob(os.path.join(path,f'*L*.stl'))[0]
            except:
                upper_mesh_path = glob.glob(os.path.join(path,f'*upper*.stl'))[0]
                lower_mesh_path = glob.glob(os.path.join(path,f'*lower*.stl'))[0]
            try:
                upper_label_path = glob.glob(os.path.join(path,f'upper_pred_landmarks.json'))[0]
                lower_label_path = glob.glob(os.path.join(path,f'lower_pred_landmarks.json'))[0]
            except:
                upper_label_path = glob.glob(os.path.join(path,f'*U*.json'))[0]
                lower_label_path = glob.glob(os.path.join(path,f'*L*.json'))[0]
            pool.apply_async(
                get_mesh,
                (upper_mesh_path,upper_label_path,lower_mesh_path,lower_label_path,single_after,index)
            )
        except:
            print(index)
            continue
        indexes.add(index)
    pool.close()
    pool.join()
    with open('valid.txt','w') as f:
        for index in indexes:
            f.write(str(index)+'\n')
