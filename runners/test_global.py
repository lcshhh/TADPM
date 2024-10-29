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
from einops import rearrange
from torchvision import transforms

def get_center_and_axis(x):
    '''
    x:[bs,32,8]
    '''
    centers = x[:,:,:3]
    normal1 = x[:,:,3:6]
    normal2 = torch.zeros_like(normal1).to(normal1.device)
    normal2[:,:,:2] = x[:,:,6:]
    normal2[:,:,2] = -(normal1[:,:,0]*x[:,:,6]+normal1[:,:,1]*x[:,:,7])/normal1[:,:,2]
    return centers,normal1,normal2

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

def test_global(args, config, logger):
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

    losses = AverageMeter(['loss','rec'])
    with torch.no_grad():
        for idx, (index,point,centers,axis,masks) in enumerate(test_dataloader):
            point = point.cuda().float()
            centers = centers.cuda().float()
            axis = axis.cuda().float()
            masks = masks.cuda()
            # outputs = base_model(point)
            # outputs = rearrange(outputs,'b c n p -> b n p c')
            outputs, l, predicted_masks = base_model(point)
            rec_loss = torch.stack([chamfer_distance(point[:,i],outputs[:,i],point_reduction='sum',batch_reduction=None)[0] for i in range(32)],dim=1)
            rec_loss = (rec_loss * masks).mean()
            loss = rec_loss
            losses.update([loss.item(),rec_loss.item()])
            print(predicted_masks[0])
            exit()
            print(rec_loss)
            outputs = outputs * masks.unsqueeze(2).unsqueeze(3)
            outputs = rearrange(outputs,'b n p c -> b (n p) c')
            point = rearrange(point,'b n p c -> b (n p) c')
            for i in range(point.shape[0]):
                write_pointcloud(point[i].cpu().numpy(),f'/data3/leics/dataset/test_global/before{i}.ply')
                write_pointcloud(outputs[i].cpu().numpy(),f'/data3/leics/dataset/test_global/after{i}.ply')

        logger.info('[TEST] Loss rec_loss kl_loss = %s' % (['%.4f' % l for l in losses.avg()]))


    # Add testing results to TensorBoard