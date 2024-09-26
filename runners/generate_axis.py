import torch
import torch.nn as nn
from utils import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from pytorch3d.loss import chamfer_distance
from utils.pointcloud import write_pointcloud
import numpy as np
import vedo
import trimesh
import os
from einops import rearrange
from pytorch3d.transforms import matrix_to_quaternion, Transform3d
from torchvision import transforms
import torch.nn.functional as F

def get_axis(x):
    '''
    x:[bs,32,8]
    '''
    normal1 = x[:,:,:3]
    normal2 = torch.zeros_like(normal1).to(normal1.device)
    normal2[:,:,:2] = x[:,:,3:]
    normal2[:,:,2] = -(normal1[:,:,0]*x[:,:,3]+normal1[:,:,1]*x[:,:,4])/normal1[:,:,2]
    return normal1,normal2

def rotation_matrix(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    # a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)


    bs = vec1.shape[0]
    a = nn.functional.normalize(vec1,dim=-1)
    b = nn.functional.normalize(vec2,dim=-1)
    n_vector = torch.cross(a, b, dim=-1)
    c = torch.sum(a*b,dim=-1)
    s = torch.norm(n_vector,dim=-1)
    # s.masked_fill_(masks.flatten(),1)
    n_matrix = torch.zeros(bs,3,3).cuda()
    for i in range(bs):
        n_matrix[i] = torch.tensor([[0, -n_vector[i,2], n_vector[i,1]],
                             [n_vector[i,2], 0, -n_vector[i,0]],
                             [-n_vector[i,1], n_vector[i,0], 0]], dtype=torch.float32)
    I = torch.stack([torch.eye(3) for _ in range(bs)],dim=0).cuda()
    rotation_matrix = I + n_matrix + ((torch.bmm(n_matrix,n_matrix) * ((1 - c.view(-1,1,1)) / (s.view(-1,1,1) ** 2))))
    return rotation_matrix

def rotation_matrix_with_axis(theta, v):
    """
    创建绕任意轴旋转的旋转矩阵
    :param theta: 旋转角度（弧度）
    :param v: 旋转轴的单位向量
    :return: 旋转矩阵
    """
    bs = theta.shape[0]
    K = torch.zeros(bs,3,3).cuda()
    for i in range(bs):
        K[i] = torch.tensor([[0, -v[i,2], v[i,1]],
                        [v[i,2], 0, -v[i,0]],
                        [-v[i,1], v[i,0], 0]]).cuda()
    I = torch.stack([torch.eye(3) for _ in range(bs)],dim=0).cuda()
    R = I + torch.sin(theta).view(-1,1,1) * K + (1 - torch.cos(theta).view(-1,1,1)) * torch.bmm(K, K)
    return R

def align_axis(normal1,normal2,gt_normal1,gt_normal2):
    rot_matrix = rotation_matrix(gt_normal1.view(-1,3),normal1.view(-1,3)).transpose(2,1) #将z轴对齐

    after_axis = torch.bmm(gt_normal2.view(-1,3).unsqueeze(1),rot_matrix)
    rho = torch.cross(after_axis.squeeze(1), normal2.view(-1,3),dim=-1)
    eps = 1e-7
    theta = torch.acos(torch.clamp(torch.sum(normal2.view(-1,3)*after_axis.squeeze(1),dim=-1),min=-1+eps,max=1-eps)).cuda()
    theta = -torch.sign(torch.sum(normal1.view(-1,3)*rho,dim=-1)) * theta
    R = rotation_matrix_with_axis(theta,normal1.view(-1,3))
    RR = torch.bmm(rot_matrix,R)
    return RR

offset=40

def cal_add(mesh1,mesh2):
    ADD = np.mean(np.sqrt(np.sum(((mesh1.vertices - mesh2.vertices)**2),axis=1))) * offset
    return ADD

def cal_pa_add(mesh1,mesh2):
    if mesh1.vertices.shape[0]!=mesh2.vertices.shape[0]:
        return
    mesh1.vertices = mesh1.vertices * offset
    mesh2.vertices = mesh2.vertices * offset
    matrix = trimesh.registration.icp(mesh1.vertices,mesh2.vertices,scale=False,max_iterations=20)[0]
    predicted_vertices = Transform3d(matrix=torch.tensor(matrix).transpose(0,1).float()).transform_points(torch.tensor(mesh1.vertices).float()).numpy()
    PA_ADD = np.mean(np.sqrt(np.sum(((predicted_vertices - mesh2.vertices)**2),axis=1)))
    with open('PA_ADD.txt','a') as f:
        f.write(str(PA_ADD)+'\n')

def cal_me_rot(mesh1,mesh2):
    mesh1.vertices = (mesh1.vertices - mesh1.centroid)*offset
    mesh2.vertices = (mesh2.vertices - mesh2.centroid)*offset
    matrix = trimesh.registration.icp(mesh1.vertices,mesh2.vertices,translation=False,scale=False,max_iterations=50)[0]
    matrix = torch.tensor(matrix)[:3,:3]
    angles = matrix_to_quaternion(matrix)
    angle = torch.acos(angles[0])*180/torch.pi
    return angle

def cal_CSA(before_mesh,after_mesh,gt_mesh):
    before_mesh.vertices = before_mesh.vertices*offset
    after_mesh.vertices = after_mesh.vertices*offset
    gt_mesh.vertices = gt_mesh.vertices*offset
    matrix1 = trimesh.registration.icp(before_mesh.vertices,after_mesh.vertices,scale=False,max_iterations=20)[0]
    matrix2 = trimesh.registration.icp(before_mesh.vertices,gt_mesh.vertices,scale=False,max_iterations=20)[0]
    matrix1 = torch.tensor(matrix1).view((1,16))
    matrix2 = torch.tensor(matrix2).view((1,16))
    cos_sim = (F.cosine_similarity(matrix1, matrix2)+1)/2
    return cos_sim

def cal_FD(mesh1,mesh2):
    return np.sqrt(np.square(mesh1.centroid - mesh2.centroid).sum())*offset

def transform_teeth(index,before_path,after_path,RR,predicted_centers):
    after_meshes = []
    before_meshes = []
    gt_meshes = []
    for i in range(32):
        mesh_path = os.path.join(before_path,f'{index}_{i}.obj')
        gt_path = os.path.join(after_path,f'{index}_{i}.obj')
        if os.path.exists(mesh_path) and os.path.exists(gt_path):
            before_mesh = trimesh.load_mesh(mesh_path)
            after_mesh = trimesh.load_mesh(mesh_path)
            gt_mesh = trimesh.load_mesh(gt_path)
            vertices = torch.from_numpy(after_mesh.vertices).cuda().float()
            centroid = torch.from_numpy(after_mesh.centroid).cuda().float()
            vertices = torch.mm(vertices - centroid,RR[i]) + predicted_centers[i]
            before_meshes.append(vedo.trimesh2vedo(before_mesh))
            after_mesh.vertices = vertices.cpu().numpy()
            after_meshes.append(vedo.trimesh2vedo(after_mesh))
            gt_meshes.append(vedo.trimesh2vedo(gt_mesh))

            # try:
            #     ADD = cal_add(after_mesh,gt_mesh)
            #     ME_ROT = cal_me_rot(after_mesh.copy(),gt_mesh.copy())
            #     CSA = cal_CSA(before_mesh.copy(),after_mesh.copy(),gt_mesh.copy())
            #     FD = cal_FD(after_mesh.copy(),gt_mesh.copy())
            #     with open('ADD.txt','a') as f:
            #         f.write(str(ADD)+'\n')
            #     with open('CSA.txt','a') as f:
            #         f.write(str(CSA.item())+'\n')
            #     with open('ROT.txt','a') as f:
            #         f.write(str(ME_ROT.item())+'\n')
            #     with open('FD.txt','a') as f:
            #         f.write(str(FD.item())+'\n')
            # except:
            #     continue

    before_mesh = vedo.merge(before_meshes)
    after_mesh = vedo.merge(after_meshes)
    gt_mesh = vedo.merge(gt_meshes)
    vedo.write(before_mesh,f'/data3/leics/dataset/tmp/{index}_before.obj')
    vedo.write(after_mesh,f'/data3/leics/dataset/tmp/{index}_after.obj')
    vedo.write(gt_mesh,f'/data3/leics/dataset/tmp/{index}_gt.obj')
    # cal_pa_add(after_mesh,gt_mesh)

def cal_average(file_name):
    with open(file_name) as f:
        lines = [float(i.strip()) for i in f.readlines()]
    val = torch.tensor(lines).mean()
    os.system(f'rm {file_name}')
    return val.item()

def generate_axis(args, config, logger):
    # build dataset
    config.model.args = args
    (_, test_dataloader)= builder.dataset_builder(args, config.dataset.test)
    # build model
    base_model = builder.model_builder(config.model)
    

    # resume ckpts
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model_from_ckpt(base_model,args.ckpts)
 
    logger.info('Using Data parallel ...' , logger = logger)
    device = torch.device('cuda') 
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    logger.info(f"[TEST] Start test")
    base_model.eval()  # set model to eval mode

    losses = AverageMeter(['loss','rec','kl'])
    with torch.no_grad():
        for idx, (index,points,after_centers,masks) in enumerate(test_dataloader):
            points = points.cuda().float()
            after_centers = after_centers.cuda().float()
            masks = masks.cuda()
            predicted_axis = base_model(points)

            # diffusion loss
            num = 2
            normal1, normal2 = get_axis(predicted_axis)
            axis = torch.cat([after_centers,normal1,normal2],dim=-1)
            idx = index[num]
            # meshes = []
            # for i in range(32):
            #     mesh_path = f'/data3/leics/dataset/type2/single_after/{idx}_{i}.obj'
            #     if os.path.exists(mesh_path):
            #         meshes.append(vedo.Mesh(mesh_path))
            # mesh = vedo.merge(meshes)
            # vedo.write(mesh,'/data3/leics/dataset/test.obj')
            save_path = '/data3/leics/dataset/type2/single_before_axis'
            os.makedirs(save_path,exist_ok=True)
            for i in range(index.shape[0]):
                np.save(f'{save_path}/{index[i]}.npy',axis[i].cpu().numpy())
            # np.save('/data3/leics/dataset/test.npy',after_axis[num].cpu().numpy())

        

            # rec loss
            

    # ADD = cal_average('ADD.txt')
    # CSA = cal_average('CSA.txt')
    # PA_ADD = cal_average('PA_ADD.txt')
    # ROT = cal_average('ROT.txt')
    # FD = cal_average('FD.txt')
    # print('ADD:',ADD)
    # print('PA_ADD:',PA_ADD)
    # print('CSA:',CSA)
    # print('ME_ROT:',ROT)
    # print('FD:',FD)



    # Add testing results to TensorBoard