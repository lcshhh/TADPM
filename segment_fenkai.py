import json
import numpy as np
import vedo
import os
import trimesh
import numpy as np
from multiprocessing import Pool
import vedo

labels = ['11','12','13','14','15','16','17','18']
labels2 = ['21','22','23','24','25','26','27','28']
labels3 = ['31','32','33','34','35','36','37','38']
labels4 = ['41','42','43','44','45','46','47','48']

def rearrange(npary: np.ndarray,upper=True)->np.ndarray:
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
    if not upper:
        npary[npary == 0] = 16
        npary -= 16
    return npary

def pointLabel2cellLabel(mesh,pointLabel):
    labels = np.zeros(mesh.ncells,dtype=int)
    faces = mesh.faces()
    for i in range(mesh.ncells):
        three_label = pointLabel[faces[i]]
        labels[i] = int(np.argmax(np.bincount(three_label)))
    return labels

def segment(index,outputroot):
    mesh = vedo.Mesh(f'/data/lcs/batch2_modified/pat_{index}/0/upper_model.stl')
    label = np.zeros(len(mesh.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2_modified/pat_{index}/0/upper_segment.json','r',encoding='utf-8') as f:
        file = json.load(f)
        # print(file['11']['vertices'][0])
        # exit()
        for tooth in labels:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
        for tooth in labels2:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
    mesh.celldata['Label'] = rearrange(pointLabel2cellLabel(mesh,label))
    vedo.write(mesh,os.path.join(outputroot,f'upper_before_{index}.vtp'))

    mesh1 = vedo.Mesh(f'/data/lcs/batch2_modified/pat_{index}/0/lower_model.stl')
    label = np.zeros(len(mesh1.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2_modified/pat_{index}/0/lower_segment.json','r',encoding='utf-8') as f:
        file = json.load(f)
        # print(file['11']['vertices'][0])
        # exit()
        for tooth in labels3:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh1.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
        for tooth in labels4:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh1.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
    mesh1.celldata['Label'] = rearrange(pointLabel2cellLabel(mesh1,label),False)
    vedo.write(mesh1,os.path.join(outputroot,f'lower_before_{index}.vtp'))

    # merge_mesh = vedo.merge(mesh,mesh1)
    # vedo.write(merge_mesh,os.path.join(outputroot,f'after_{index}.vtp'))

    mesh = vedo.Mesh(f'/data/lcs/batch2_modified/pat_{index}/1/upper_model.stl')
    label = np.zeros(len(mesh.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2_modified/pat_{index}/1/upper_segment.json','r',encoding='utf-8') as f:
        file = json.load(f)
        # print(file['11']['vertices'][0])
        # exit()
        for tooth in labels:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
        for tooth in labels2:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
    mesh.celldata['Label'] = rearrange(pointLabel2cellLabel(mesh,label))
    vedo.write(mesh,os.path.join(outputroot,f'upper_after_{index}.vtp'))

    mesh1 = vedo.Mesh(f'/data/lcs/batch2_modified/pat_{index}/1/lower_model.stl')
    label = np.zeros(len(mesh1.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2_modified/pat_{index}/1/lower_segment.json','r',encoding='utf-8') as f:
        file = json.load(f)
        # print(file['11']['vertices'][0])
        # exit()
        for tooth in labels3:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh1.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
        for tooth in labels4:
            if tooth in file:
                for face in file[tooth]['vertices']:
                    out = mesh1.closest_point(face,n=1,return_point_id=True)
                    label[out] = int(tooth)
    mesh1.celldata['Label'] = rearrange(pointLabel2cellLabel(mesh1,label),False)

    # merge_mesh = vedo.merge(mesh,mesh1)
    vedo.write(mesh1,os.path.join(outputroot,f'lower_after_{index}.vtp'))


# mesh1.AddPosition(10,0,0)bj
outputroot = '/data/lcs/batch2_rearrange'
os.makedirs(outputroot,exist_ok=True)
pool = Pool(processes=64)
for index in range(1,236):
     pool.apply_async(
          segment,
          (index,outputroot)
     )
    # segment(index,outputroot)
    # extract_tooth(dataroot,outputroot,obj)
pool.close()
pool.join()
    

