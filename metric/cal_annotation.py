# This is a sample Python script.
import os
from pathlib import Path
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import trimesh
import json
import vedo
import numpy as np
from pytorch3d.transforms import Transform3d
from multiprocessing import Pool
from pytorch3d.transforms import se3_exp_map,se3_log_map
def transform_points(points,trans_6dof):
    '''
    points [3]
    '''
    trans_matrix = se3_exp_map(trans_6dof.unsuqeeze(0)).transpose(1,2)      # [16,4,4]    
    after_points = Transform3d(matrix=trans_matrix.transpose(1,2)).transform_points(points.view(1,1,-1))
    return after_points

def distance(v1,v2):
    distance = np.linalg.norm(np.array(v1)-np.array(v2))
    return distance

# Press the green button in the gutter to run the script.
def cal_annotation(upper_path,lower_path):
    with open(lower_path) as f:
            segment_simplify_L = json.load(f)
    with open(upper_path) as f:
            segment_simplify_U = json.load(f)
    landmarks_u = np.array(segment_simplify_U['landmarks'])
    landmarks_l = np.array(segment_simplify_L['landmarks'])
    pt0_u = []
    pt2_u = []
    pt3_u = []
    pt6_u = []

    pt0_l = []
    pt2_l = []
    pt3_l = []
    pt6_l = []
    pt7_l = []
    for i in range(len(segment_simplify_U['landmarks'])):
        if i == 0:
            continue
        tooth_index = segment_simplify_U['landmarks'][i].keys()
        tooth_index = int(str(tooth_index)[12:14])
        value = segment_simplify_U['landmarks'][i].values()
        for v in value:
            if 'Pt0' in v.keys():
                pt0_u.append([v['Pt0'],tooth_index])
            if 'Pt2' in v.keys():
                pt2_u.append([v['Pt2'],tooth_index])
            if 'Pt3' in v.keys():
                pt3_u.append([v['Pt3'],tooth_index])
            if 'Pt6' in v.keys():
                pt6_u.append([v['Pt6'],tooth_index])

    pt0_u = sorted(pt0_u, key=lambda x: x[1])
    pt2_u = sorted(pt2_u, key=lambda x: x[1])
    pt3_u = sorted(pt3_u, key=lambda x: x[1])
    pt6_u = sorted(pt6_u, key=lambda x: x[1])

    for i in range(len(segment_simplify_L['landmarks'])):
        if i == 0:
            continue
        tooth_index = segment_simplify_L['landmarks'][i].keys()
        tooth_index = int(str(tooth_index)[12:14])
        value = segment_simplify_L['landmarks'][i].values()
        for v in value:
            if 'Pt0' in v.keys():
                pt0_l.append([v['Pt0'],tooth_index])
            if 'Pt2' in v.keys():
                pt2_l.append([v['Pt2'],tooth_index])
            if 'Pt3' in v.keys():
                pt3_l.append([v['Pt3'],tooth_index])
            if 'Pt6' in v.keys():
                pt6_l.append([v['Pt6'],tooth_index])
            if 'Pt7' in v.keys():
                pt7_l.append([v['Pt7'],tooth_index])
    pt0_l = sorted(pt0_l, key=lambda x: x[1])
    pt2_l = sorted(pt2_l, key=lambda x: x[1])
    pt3_l = sorted(pt3_l, key=lambda x: x[1])
    pt6_l = sorted(pt6_l, key=lambda x: x[1])
    pt7_l = sorted(pt7_l, key=lambda x: x[1])
    align = 0
    Anterior_occlusion = 0
    Posterior_occlusion = 0
    for i in range(len(pt2_u)):
        adjcent =999
        if pt2_u[i][1] == 11:
            for j in range(0,len(pt2_u),1):
                if pt2_u[j][1] ==21:
                    adjcent = distance(pt2_u[i][0],pt2_u[j][0])
                    continue
        elif pt2_u[i][1] ==21:
            continue
        else:
            adjcent = distance(pt2_u[i][0],pt3_u[0][0])
            for j in range(1,len(pt3_u),1):
                if pt2_u[i][1]==pt3_u[j][1]:
                    continue
                if distance(pt2_u[i][0],pt3_u[j][0]) <adjcent:
                    adjcent = distance(pt2_u[i][0],pt3_u[j][0])
                    # print(pt2_u[i][1],pt3_u[j][1],adjcent,align)
        align+=adjcent
    for i in range(len(pt2_l)):
        if pt2_l[i][1] == 31:
            for j in range(0,len(pt2_l),1):
                if pt2_l[j][1] ==41:
                    adjcent = distance(pt2_l[i][0],pt2_l[j][0])
                    continue
        if pt2_l[i][1] ==41:
            continue
        adjcent = distance(pt2_l[i][0],pt3_l[0][0])
        for j in range(1,len(pt3_l),1):
            if pt2_l[i][1]==pt3_l[j][1]:
                continue
            if distance(pt2_l[i][0],pt3_l[j][0]) <adjcent:
                adjcent = distance(pt2_l[i][0],pt3_l[j][0])
                # print(pt2_l[i][1], pt3_l[j][1], adjcent, align)
        align+=adjcent
    # print('align\n',align)
    for i in range(len(pt0_l)):
        if pt0_l[i][1] in [33, 34, 35, 43, 44, 45]:
            try:
                mid = (np.array(pt0_l[i][0]) + np.array(pt0_l[i+1][0])) / 2
                adjcent = distance(pt0_u[0][0], mid)
                for j in range(1, len(pt0_u) - 1, 1):


                    if distance(pt0_u[j][0], mid) < adjcent:
                        adjcent = distance(pt0_u[j][0], mid)
                        # print(pt0_u[j][1], pt0_l[i][1], pt0_l[i + 1][1], adjcent, Posterior_occlusion)

                Posterior_occlusion += adjcent
            except:
                continue
    for i in range(len(pt0_u)):

        if pt0_u[i][1] in [16,17,26,27]:
            left = 0
            right = 0
            adjcent = distance(pt0_u[i][0], pt7_l[0][0])
            for j in range(1,len(pt7_l),1):
                if distance(pt0_u[i][0], pt7_l[j][0]) < adjcent:
                    adjcent = distance(pt0_u[i][0], pt7_l[j][0])
                    # print(pt0_u[i][1],pt7_l[j][1], adjcent, Posterior_occlusion)
    # print('Posterior_occlusion',Posterior_occlusion)
    for i in range(len(pt0_u)):
        if pt0_u[i][1] not in [21,11,12,22]:
            continue
        adjcent = distance(pt0_u[i][0], pt0_l[0][0])
        for j in range(0,len(pt0_l),1):
            if distance(pt0_u[i][0],pt0_l[j][0])<adjcent:
                adjcent = distance(pt0_u[i][0],pt0_l[j][0])
                # print(pt0_u[i][1],pt0_l[j][1],adjcent,Anterior_occlusion)
        Anterior_occlusion += adjcent
    # print('Anterior_occlusion',Anterior_occlusion)
    with open('align.txt','a') as f:
        f.write(str(align)+'\n')
    with open('post.txt','a') as f:
        f.write(str(Posterior_occlusion)+'\n')
    with open('ante.txt','a') as f:
        f.write(str(Anterior_occlusion)+'\n')

def rearrange(npary):
    """
    Rearrange the oder of the label of the input numpy array.
    :param npary: Input numpy array.
    :return:
    """
    npary = np.array(npary)
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
    npary -= 1
    return npary

def center_and_norm(index,outputroot):
    upper_path = f'/data/lcs/dataset/teeth_full/landmark/{index}/ori/uppper_mark.json'
    lower_path = f'/data/lcs/dataset/teeth_full/landmark/{index}/ori/lower_mark.json'
    with open('data_preprocess/mapping.json') as f:
        maps = json.load(f)
    try:
        new_index = maps[str(int(index))]
    except:
        return
    nums = np.load('/data/lcs/dataset/m.npy')
    try:
        mesh = vedo.Mesh(str(f'/data/lcs/dataset/teeth_full/before/{new_index}.vtp'))
    except:
        return
    tri_mesh = vedo.vedo2trimesh(mesh)
    centroid = tri_mesh.centroid
    try:
        with open(lower_path) as f:
                segment_simplify_L = json.load(f)
    except:
        return
    try:
        with open(upper_path) as f:
                segment_simplify_U = json.load(f)
    except:
        with open(f'/data/lcs/dataset/teeth_full/landmark/{index}/ori/upper_pred_landmarks.json') as f:
                segment_simplify_U = json.load(f)
    # for i in range(len(segment_simplify_U['landmarks'])):
    #     if i == 0:
    #         continue
    #     original_index = segment_simplify_U['landmarks'][i].keys()
    #     tooth_index = int(str(original_index)[12:14])
    #     value = segment_simplify_U['landmarks'][i].values()
    #     for v in value:
    #         if 'Pt0' in v.keys():
    #             # pt0_u.append(v['Pt0'])
    #             for j in range(3):
    #                 # segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt0'][j] = (v['Pt0'][j] - centroid[j])/nums[new_index]
    #                 segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt0'][j] = (v['Pt0'][j] - centroid[j])/nums[new_index]
    #         if 'Pt2' in v.keys():
    #             # pt2_u.append(v['Pt2'])
    #             for j in range(3):
    #                 segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt2'][j] = (v['Pt2'][j] - centroid[j])/nums[new_index]
    #         if 'Pt3' in v.keys():
    #             # pt3_u.append(v['Pt3'])
    #             for j in range(3):
    #                 segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt3'][j] = (v['Pt3'][j] - centroid[j])/nums[new_index]
    #         if 'Pt6' in v.keys():
    #             for j in range(3):
    #                 segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt6'][j] = (v['Pt6'][j] - centroid[j])/nums[new_index]
    #             # pt6_u.append(v['Pt6'])

    # for i in range(len(segment_simplify_L['landmarks'])):
    #     if i == 0:
    #         continue
    #     original_index = segment_simplify_L['landmarks'][i].keys()
    #     tooth_index = int(str(original_index)[12:14])
    #     value = segment_simplify_L['landmarks'][i].values()
    #     for v in value:
    #         pt0_l = []
    #         pt2_l = []
    #         pt3_l = []
    #         pt6_l = []
    #         pt7_l = []
    #         if 'Pt0' in v.keys():
    #             # pt0_l.append(v['Pt0'])
    #             for j in range(3):
    #                 segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt0'][j] = (v['Pt0'][j] - centroid[j])/nums[new_index]
    #         if 'Pt2' in v.keys():
    #             # pt2_l.append(v['Pt2'])
    #             for j in range(3):
    #                 segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt2'][j] = (v['Pt2'][j] - centroid[j])/nums[new_index]
    #         if 'Pt3' in v.keys():
    #             # pt3_l.append(v['Pt3'])
    #             for j in range(3):
    #                 segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt3'][j] = (v['Pt3'][j] - centroid[j])/nums[new_index]
    #         if 'Pt6' in v.keys():
    #             # pt6_l.append(v['Pt6'])
    #             for j in range(3):
    #                 segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt6'][j] = (v['Pt6'][j] - centroid[j])/nums[new_index]
    #         if 'Pt7' in v.keys():
    #             # pt7_l.append(v['Pt7'])
    #             for j in range(3):
    #                 segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt7'][j] = (v['Pt7'][j] - centroid[j])/nums[new_index]
    #         pt0_l = np.array(pt0_l)
    #         pt2_l = np.array(pt2_l)
    #         pt3_l = np.array(pt3_l)
    #         pt6_l = np.array(pt6_l)
    #         pt7_l = np.array(pt7_l)
    with open(f'{outputroot}/{new_index}_upper.json','w') as f:
        f.write(json.dumps(segment_simplify_U))
    with open(f'{outputroot}/{new_index}_lower.json','w') as f:
        f.write(json.dumps(segment_simplify_L))

def cal_annot(index,trans6dof,outputroot):
    '''
    这里的index是我的index
    '''
    upper_path = f'/data/lcs/dataset/teeth_full/all_landmark/centered_normed_landmark/{index}_upper.json'
    lower_path = f'/data/lcs/dataset/teeth_full/all_landmark/centered_normed_landmark/{index}_lower.json'

    try:
        with open(lower_path) as f:
                segment_simplify_L = json.load(f)
        with open(upper_path) as f:
                segment_simplify_U = json.load(f)
    except:
        return
    for i in range(len(segment_simplify_U['landmarks'])):
        if i == 0:
            continue
        original_index = segment_simplify_U['landmarks'][i].keys()
        tooth_index = int(str(original_index)[12:14])
        value = segment_simplify_U['landmarks'][i].values()
        for v in value:
            if 'Pt0' in v.keys():
                new_point = transform_points(v['Pt0'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt0'][j] = new_point[j]
            if 'Pt2' in v.keys():
                # pt2_u.append(v['Pt2'])
                new_point = transform_points(v['Pt2'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt2'][j] = new_point[j]
            if 'Pt3' in v.keys():
                # pt3_u.append(v['Pt3'])
                new_point = transform_points(v['Pt3'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt3'][j] = new_point[j]
            if 'Pt6' in v.keys():
                new_point = transform_points(v['Pt6'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_U['landmarks'][i][str(tooth_index)]['Pt6'][j] = new_point[j]
                # pt6_u.append(v['Pt6'])

    for i in range(len(segment_simplify_L['landmarks'])):
        if i == 0:
            continue
        original_index = segment_simplify_L['landmarks'][i].keys()
        tooth_index = int(str(original_index)[12:14])
        value = segment_simplify_L['landmarks'][i].values()
        for v in value:
            pt0_l = []
            pt2_l = []
            pt3_l = []
            pt6_l = []
            pt7_l = []
            if 'Pt0' in v.keys():
                # pt0_l.append(v['Pt0'])
                new_point = transform_points(v['Pt0'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt0'][j] = new_point[j]
            if 'Pt2' in v.keys():
                # pt2_l.append(v['Pt2'])
                new_point = transform_points(v['Pt2'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt2'][j] = new_point[j]
            if 'Pt3' in v.keys():
                # pt3_l.append(v['Pt3'])
                new_point = transform_points(v['Pt3'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt3'][j] = new_point[j]
            if 'Pt6' in v.keys():
                # pt6_l.append(v['Pt6'])
                new_point = transform_points(v['Pt6'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt6'][j] = new_point[j]
            if 'Pt7' in v.keys():
                # pt7_l.append(v['Pt7'])
                new_point = transform_points(v['Pt7'],trans6dof[rearrange(tooth_index)])
                for j in range(3):
                    segment_simplify_L['landmarks'][i][str(tooth_index)]['Pt7'][j] = new_point[j]
            pt0_l = np.array(pt0_l)
            pt2_l = np.array(pt2_l)
            pt3_l = np.array(pt3_l)
            pt6_l = np.array(pt6_l)
            pt7_l = np.array(pt7_l)
    with open(f'{outputroot}/{index}_upper.json','w') as f:
        f.write(json.dumps(segment_simplify_U))
    with open(f'{outputroot}/{index}_lower.json','w') as f:
        f.write(json.dumps(segment_simplify_L))

if __name__ == '__main__':
    # pool = Pool(processes=64)
    # for path in dataroot.iterdir():
    #     pool.apply_async(
    #         single_center,
    #         (dataroot,outputroot,path)
    #     )
    #     # get_center(dataroot,outputroot,obj_path)
    # pool.close()
    # pool.join()
    # pool = Pool(processes=64)
    # path = Path('/data/lcs/dataset/teeth_full/landmark')
    dataroot =  Path('/data/lcs/dataset/teeth_full/all_landmark/centered_normed_landmark')
    # outputroot = '/data/lcs/dataset/teeth_full/all_landmark/before' 
    # os.makedirs(outputroot,exist_ok=True)
    indexes = set([int(p.name.split('_')[0]) for p in dataroot.iterdir()])
    print(indexes)
    with open('annot.txt','w') as f:
        for i in indexes:
            f.write(str(i)+'\n')
    # # indexes = [p.name for p in path.iterdir()]
    # for index in indexes:
    #     upper_path = os.path.join(dataroot,f'{index}_upper.json')
    #     lower_path = os.path.join(dataroot,f'{index}_lower.json')
    #     pool.apply_async(
    #         cal_annotation,
    #         (upper_path,lower_path)
    #     )
    #     # center_and_norm(upper_path,lower_path)
    #     # center_and_norm(index,outputroot)
    # pool.close()
    # pool.join()
    # with open('data_preprocess/mapping.json') as f:
    #     maps = json.load(f)
    # map2 = {}
    # for key, value in maps.items():
    #     map2[int(value)] = int(key)
    # with open('data_preprocess/invert_mapping.json','w') as f:
    #     f.write(json.dumps(map2))




