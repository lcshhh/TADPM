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

def pointLabel2cellLabel(mesh,pointLabel):
    labels = np.zeros(mesh.ncells,dtype=int)
    faces = mesh.faces()
    for i in range(mesh.ncells):
        three_label = pointLabel[faces[i]]
        labels[i] = int(np.argmax(np.bincount(three_label)))
    return labels

def segment(index,outputroot):
    mesh = vedo.Mesh(f'/data/lcs/batch2/pat_{index}/0/upper_model.stl')
    label = np.zeros(len(mesh.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2/pat_{index}/0/upper_segment.json','r',encoding='utf-8') as f:
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
    mesh.celldata['Label'] = pointLabel2cellLabel(mesh,label)

    mesh1 = vedo.Mesh(f'/data/lcs/batch2/pat_{index}/0/lower_model.stl')
    label = np.zeros(len(mesh1.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2/pat_{index}/0/lower_segment.json','r',encoding='utf-8') as f:
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
    mesh1.celldata['Label'] = pointLabel2cellLabel(mesh1,label)

    merge_mesh = vedo.merge(mesh,mesh1)
    vedo.write(merge_mesh,os.path.join(outputroot,f'after_{index}.vtp'))

    mesh = vedo.Mesh(f'/data/lcs/batch2/pat_{index}/2/upper_model.stl')
    label = np.zeros(len(mesh.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2/pat_{index}/2/upper_segment.json','r',encoding='utf-8') as f:
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
    mesh.celldata['Label'] = pointLabel2cellLabel(mesh,label)

    mesh1 = vedo.Mesh(f'/data/lcs/batch2/pat_{index}/2/lower_model.stl')
    label = np.zeros(len(mesh1.points()),dtype=np.int32)
    with open(f'/data/lcs/batch2/pat_{index}/2/lower_segment.json','r',encoding='utf-8') as f:
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
    mesh1.celldata['Label'] = pointLabel2cellLabel(mesh1,label)

    merge_mesh = vedo.merge(mesh,mesh1)
    vedo.write(merge_mesh,os.path.join(outputroot,f'before_{index}.vtp'))


# mesh1.AddPosition(10,0,0)bj
outputroot = '/data/lcs/batch2_merged'
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
    

